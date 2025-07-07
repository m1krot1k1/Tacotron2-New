import numpy as np
from scipy.io.wavfile import read
import torch
import cv2
import math
import os
import re
import importlib.util
from pathlib import Path



def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def get_drop_frame_mask_from_lengths(lengths, drop_frame_rate):
    batch_size = lengths.size(0)
    max_len = torch.max(lengths).item()
    mask = get_mask_from_lengths(lengths).float()
    drop_mask = torch.empty([batch_size, max_len], device=lengths.device).uniform_(0., 1.) < drop_frame_rate
    drop_mask = drop_mask.float() * mask
    return drop_mask


def dropout_frame(mels, global_mean, mel_lengths, drop_frame_rate):
    drop_mask = get_drop_frame_mask_from_lengths(mel_lengths, drop_frame_rate)
    
    # Если global_mean не задан, используем нулевые значения
    if global_mean is None:
        global_mean = torch.zeros(mels.size(1), device=mels.device)
    
    dropped_mels = (mels * (1.0 - drop_mask).unsqueeze(1) +
                    global_mean[None, :, None] * drop_mask.unsqueeze(1))
    return dropped_mels

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def guide_attention_slow(text_lengths, mel_lengths, max_txt=None, max_mel=None):
    b = len(text_lengths)
    if max_txt is None:
        max_txt= np.max(text_lengths)
    if max_mel is None:
        max_mel = np.max(mel_lengths)
    guide = np.ones((b, max_txt, max_mel), dtype=np.float32)
    mask = np.zeros((b, max_txt, max_mel), dtype=np.float32)
    for i in range(b):
        W = guide[i]
        M = mask[i]
        N = float(text_lengths[i])
        T = float(mel_lengths[i])
        for n in range(max_txt):
            for t in range(max_mel):
                if n < N and t < T:
                    W[n][t] = 1.0 - np.exp(-(float(n) / N - float(t) / T) ** 2 / (2.0 * (0.2 ** 2)))
                    M[n][t] = 1.0
                elif t >= T and n < N:
                    W[n][t] = 1.0 - np.exp(-((float(n - N - 1) / N)** 2 / (2.0 * (0.2 ** 2))))
    if len(guide) == 1:
        cv2.imwrite('messigray2.png',(guide[0]*255).astype(np.uint8))
        return guide[0], mask[0]
    return guide, mask

def rotate_image(image, angle, center=(0,25)):
  rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def diagonal_guide(text_len, mel_len, g=0.2):
    # Проверяем на нулевые значения для предотвращения деления на ноль
    if text_len <= 0 or mel_len <= 0:
        return np.ones((mel_len, text_len), dtype=np.float32)
    
    grid_text = torch.linspace(0., 1. - 1. / text_len, text_len)  # (T)
    grid_mel = torch.linspace(0., 1. - 1. / mel_len, mel_len)  # (M)
    grid = grid_text.view(1, -1) - grid_mel.view(-1, 1)  # (M, T)
    W = 1 - torch.exp(-grid ** 2 / (2 * g ** 2))
    return W.numpy()

def linear_guide(text_len, mel_len, g=0.2):
    # Проверяем на нулевые значения для предотвращения деления на ноль
    if text_len <= 0:
        return np.ones(text_len, dtype=np.float32)
    
    a = np.linspace(-1., -1./text_len, text_len)  # (T)
    W = 1 - np.exp(-a ** 2 / (2 * g ** 2))
    return W

def guide_attention_fast(txt_len, mel_len, max_txt, max_mel, g=0.20):
    h,w = max_txt, max_mel
    mask = np.ones((h,w), dtype=np.float32)

    diag = diagonal_guide(txt_len, mel_len, g=g)
    mask[:txt_len,:mel_len] = np.transpose(diag,(1,0))

    linear = linear_guide(txt_len,mel_len).reshape(-1,1)
    mask[:txt_len,mel_len:] = linear.repeat(max_mel-mel_len,axis=-1)
    return mask


# res = guide_attention_fast(150,700,200,1000)
# cv2.imwrite('test.png', (res*255).astype(np.uint8))

def find_latest_checkpoint(checkpoint_path: str) -> str:
    """Находит последний чекпоинт в указанной директории."""
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_*"))
    if not checkpoints:
        return None
        
    latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split('_')[1]))
    return str(latest_checkpoint)

def load_hparams(hparams_path: str):
    """Загружает гиперпараметры из файла hparams.py."""
    spec = importlib.util.spec_from_file_location("hparams", hparams_path)
    hparams_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hparams_module)
    hparams = hparams_module.create_hparams()
    return hparams

def save_hparams(hparams_path: str, hparams_dict: dict):
    """
    Сохраняет измененные гиперпараметры обратно в hparams.py.
    Использует регулярные выражения для безопасной замены значений.
    """
    with open(hparams_path, 'r', encoding='utf-8') as f:
        content = f.read()

    for key, value in hparams_dict.items():
        # Формируем корректное представление значения (строка в кавычках, остальное как есть)
        if isinstance(value, str):
            value_str = f"'{value}'"
        else:
            value_str = str(value)
        
        # Регулярное выражение для поиска "key=value"
        # Оно ищет ключ, окруженный пробелами или началом строки/скобкой,
        # за которым следует знак равенства и любое значение до запятой или новой строки.
        pattern = re.compile(f"({key}\\s*=\\s*)[^,\\n)]*")
        
        # Заменяем найденное значение на новое
        new_content, count = pattern.subn(f"\\g<1>{value_str}", content)
        if count > 0:
            content = new_content
        else:
            # Если параметр не найден, возможно, его нужно добавить.
            # Для простоты пока будем только обновлять существующие.
            print(f"Warning: a chave de hiperparâmetro '{key}' não foi encontrada em {hparams_path} e não foi atualizada.")


    with open(hparams_path, 'w', encoding='utf-8') as f:
        f.write(content)

def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Загружает чекпоинт для модели и оптимизатора.
    
    Args:
        checkpoint_path: Путь к файлу чекпоинта
        model: Модель PyTorch
        optimizer: Оптимизатор PyTorch
    
    Returns:
        Tuple[model, optimizer, learning_rate, iteration]
    """
    assert os.path.isfile(checkpoint_path), f"Файл чекпоинта не найден: {checkpoint_path}"
    print(f"Загрузка чекпоинта '{checkpoint_path}'")
    
    # Добавляем weights_only=False для совместимости с PyTorch 2.6+
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    
    print(f"Чекпоинт загружен с итерации {iteration}")
    return model, optimizer, learning_rate, iteration