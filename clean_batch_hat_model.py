import cv2
import numpy as np
import torch
import requests
import os
import sys
import time

# Проверка наличия spandrel
try:
    from spandrel import ModelLoader
except ImportError:
    print("ОШИБКА: Библиотека 'spandrel' не найдена.")
    print("Пожалуйста, установите её: pip install spandrel")
    sys.exit(1)

def download_weights(url, save_path):
    if not os.path.exists(save_path):
        print(f"Скачиваю веса модели в {save_path}...")
        try:
            response = requests.get(url, stream=True)
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Скачивание завершено.")
        except Exception as e:
            print(f"Ошибка скачивания: {e}")
            if os.path.exists(save_path):
                os.remove(save_path)
            sys.exit(1)

def setup_model(device):
    # === НАСТРОЙКИ МОДЕЛИ ===
    # Используем HAT x4 (Standard SR)
    model_name = 'HAT_SRx2_ImageNet-pretrain.pth'
    # Ссылка на официальный релиз
    model_url = f'https://github.com/XPixelGroup/HAT/releases/download/v1.0/{model_name}'
    
    download_weights(model_url, model_name)

    print(f"Загрузка архитектуры модели из {model_name}...")
    
    try:
        loader = ModelLoader()
        model = loader.load_from_file(model_name)
    except Exception as e:
        print(f"Ошибка загрузки через Spandrel: {e}")
        sys.exit(1)
    
    model.eval()
    model = model.to(device)
    
    print(f"Модель загружена успешно. Архитектура: {model.architecture}, Scale: {model.scale}")
    return model

def process_image_tiled(img_tensor, model, device, tile_size=512, tile_overlap=32):
    """
    Функция обработки по тайлам (кускам) с учетом масштабирования (Scale).
    """
    scale = model.scale  # Получаем коэффициент увеличения (например, 4)
    b, c, h, w = img_tensor.shape
    
    # Размер выходного изображения с учетом скейла
    h_out, w_out = h * scale, w * scale
    
    output_tensor = torch.zeros((b, c, h_out, w_out), device=device, dtype=torch.float32)
    weights_tensor = torch.zeros((b, c, h_out, w_out), device=device, dtype=torch.float32)

    stride = tile_size - tile_overlap
    
    h_idx_list = list(range(0, h - tile_size, stride))
    if not h_idx_list or h_idx_list[-1] != h - tile_size:
        h_idx_list.append(h - tile_size)
        
    w_idx_list = list(range(0, w - tile_size, stride))
    if not w_idx_list or w_idx_list[-1] != w - tile_size:
        w_idx_list.append(w - tile_size)

    # Проходим по тайлам
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            # Вырезаем кусок из входного изображения
            in_patch = img_tensor[..., h_idx:h_idx+tile_size, w_idx:w_idx+tile_size]
            
            with torch.no_grad():
                out_patch = model(in_patch)
            
            # Определяем координаты для вставки в выходной тензор (с учетом scale)
            out_h_idx = h_idx * scale
            out_w_idx = w_idx * scale
            out_tile_size = tile_size * scale
            
            # Вставляем обработанный кусок
            output_tensor[..., out_h_idx:out_h_idx+out_tile_size, out_w_idx:out_w_idx+out_tile_size] += out_patch
            weights_tensor[..., out_h_idx:out_h_idx+out_tile_size, out_w_idx:out_w_idx+out_tile_size] += 1.0

    # Усредняем перекрытия
    weights_tensor[weights_tensor == 0] = 1.0
    output_tensor = output_tensor / weights_tensor
    return output_tensor

def process_image(img_path, save_path, model, device):
    img_lq = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_lq is None:
        print(f"  -> Не удалось прочитать {img_path}")
        return

    # Нормализация
    img_lq = img_lq.astype(np.float32) / 255.
    # BGR -> RGB и HWC -> CHW
    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))
    
    img_tensor = torch.from_numpy(img_lq).unsqueeze(0).to(device)

    # Паддинг (добавление краев), чтобы изображение делилось на window_size
    # Для HAT окно обычно больше, берем безопасное значение 64
    window_size = 64 
    _, _, h_old, w_old = img_tensor.size()
    h_pad = (window_size - h_old % window_size) % window_size
    w_pad = (window_size - w_old % window_size) % window_size
    img_tensor = torch.nn.functional.pad(img_tensor, (0, w_pad, 0, h_pad), 'reflect')

    try:
        # Уменьшил tile_size до 400 для HAT, так как он ест больше памяти, чем SwinIR
        # Если видеокарта мощная (>=12GB VRAM), можно вернуть 512
        output = process_image_tiled(img_tensor, model, device, tile_size=400, tile_overlap=32)
    except RuntimeError as e:
        print(f"  -> ОШИБКА VRAM: {e}")
        torch.cuda.empty_cache()
        return

    # Обрезка паддинга (с учетом скейла!)
    scale = model.scale
    output = output[..., :h_old*scale, :w_old*scale]
    
    output = output.data.squeeze().float().cpu().numpy()
    output = np.clip(output, 0, 1)

    if output.ndim == 3:
        # RGB -> BGR и CHW -> HWC
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    
    output = (output * 255.0).round().astype(np.uint8)

    ext = os.path.splitext(save_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        cv2.imwrite(save_path, output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    elif ext == '.png':
        cv2.imwrite(save_path, output, [int(cv2.IMWRITE_PNG_COMPRESSION), 3]) 
    else:
        cv2.imwrite(save_path, output)

def main():
    input_folder = 'input_images'
    output_folder = 'output_images'
    
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"Создана папка {input_folder}. Положите туда изображения.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    print("Инициализация модели HAT...")
    model = setup_model(device)
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    
    print(f"Найдено файлов: {len(files)}")
    print("-" * 30)

    for i, filename in enumerate(files):
        print(f"[{i+1}/{len(files)}] Обработка: {filename}...", end=" ", flush=True)
        
        input_path = os.path.join(input_folder, filename)
        filename_no_ext = os.path.splitext(filename)[0]
        # Добавляем суффикс, так как разрешение меняется
        output_path = os.path.join(output_folder, f"hat_{filename_no_ext}.png")
        
        start_time = time.time()
        
        process_image(input_path, output_path, model, device)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Готово ({duration:.2f} сек)")
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("-" * 30)
    print("Все задачи выполнены!")

if __name__ == '__main__':
    main()