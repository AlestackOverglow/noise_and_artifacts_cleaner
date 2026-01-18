import cv2
import numpy as np
import torch
import requests
import os
import sys
import time  # <--- Добавлена библиотека для работы со временем

# Проверка наличия файла архитектуры
try:
    from network_swinir import SwinIR
except ImportError:
    print("ОШИБКА: Не найден файл 'network_swinir.py'.")
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
    model_name = '005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth'
    model_url = f'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{model_name}'
    
    download_weights(model_url, model_name)

    # Инициализация архитектуры
    model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                   img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                   num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2, upsampler='', resi_connection='1conv')
    
    param_key_g = 'params'
    if os.path.exists(model_name):
        try:
            pretrained_dict = torch.load(model_name, map_location=device, weights_only=False)
            model.load_state_dict(pretrained_dict[param_key_g] if param_key_g in pretrained_dict else pretrained_dict)
        except Exception as e:
            print(f"Ошибка загрузки весов: {e}")
            sys.exit(1)
    
    model.eval()
    model = model.to(device)
    return model

def process_image_tiled(img_tensor, model, device, tile_size=512, tile_overlap=32):
    b, c, h, w = img_tensor.shape
    
    output_tensor = torch.zeros((b, c, h, w), device=device, dtype=torch.float32)
    weights_tensor = torch.zeros((b, c, h, w), device=device, dtype=torch.float32)

    stride = tile_size - tile_overlap
    
    h_idx_list = list(range(0, h - tile_size, stride))
    if not h_idx_list or h_idx_list[-1] != h - tile_size:
        h_idx_list.append(h - tile_size)
        
    w_idx_list = list(range(0, w - tile_size, stride))
    if not w_idx_list or w_idx_list[-1] != w - tile_size:
        w_idx_list.append(w - tile_size)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img_tensor[..., h_idx:h_idx+tile_size, w_idx:w_idx+tile_size]
            
            with torch.no_grad():
                out_patch = model(in_patch)
            
            output_tensor[..., h_idx:h_idx+tile_size, w_idx:w_idx+tile_size] += out_patch
            weights_tensor[..., h_idx:h_idx+tile_size, w_idx:w_idx+tile_size] += 1.0

    weights_tensor[weights_tensor == 0] = 1.0
    output_tensor = output_tensor / weights_tensor
    return output_tensor

def process_image(img_path, save_path, model, device):
    img_lq = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_lq is None:
        print(f"  -> Не удалось прочитать {img_path}")
        return

    img_lq = img_lq.astype(np.float32) / 255.
    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))
    
    img_tensor = torch.from_numpy(img_lq).unsqueeze(0).to(device)

    window_size = 8
    _, _, h_old, w_old = img_tensor.size()
    h_pad = (window_size - h_old % window_size) % window_size
    w_pad = (window_size - w_old % window_size) % window_size
    img_tensor = torch.nn.functional.pad(img_tensor, (0, w_pad, 0, h_pad), 'reflect')

    try:
        output = process_image_tiled(img_tensor, model, device, tile_size=512, tile_overlap=32)
    except RuntimeError as e:
        print(f"  -> ОШИБКА: {e}")
        torch.cuda.empty_cache()
        return

    output = output[..., :h_old, :w_old]
    output = output.data.squeeze().float().cpu().numpy()
    
    output = np.clip(output, 0, 1)

    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    
    output = (output * 255.0).round().astype(np.uint8)

    ext = os.path.splitext(save_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        cv2.imwrite(save_path, output, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    elif ext == '.png':
        cv2.imwrite(save_path, output, [int(cv2.IMWRITE_PNG_COMPRESSION), 1]) 
    else:
        cv2.imwrite(save_path, output)

def main():
    input_folder = 'input_images'
    output_folder = 'output_images'
    
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    print("Загрузка модели...")
    model = setup_model(device)
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    
    print(f"Найдено файлов: {len(files)}")
    print("-" * 30)

    for i, filename in enumerate(files):
        print(f"[{i+1}/{len(files)}] Обработка: {filename}...", end=" ", flush=True)
        
        input_path = os.path.join(input_folder, filename)
        filename_no_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"clean_{filename_no_ext}.png")
        
        # === ЗАМЕР ВРЕМЕНИ ===
        start_time = time.time()
        
        process_image(input_path, output_path, model, device)
        
        end_time = time.time()
        duration = end_time - start_time
        # =====================
        
        print(f"Готово ({duration:.2f} сек)")
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("-" * 30)
    print("Готово!")

if __name__ == '__main__':
    main()