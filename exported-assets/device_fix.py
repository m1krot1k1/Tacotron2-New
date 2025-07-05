
# ФАЙЛ: data_utils.py
# ПРОБЛЕМА: AttributeError: 'tuple' object has no attribute 'device' в train.py:897

import torch
from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    '''
    Исправленная collate_fn для правильной обработки device transfer
    '''
    # Проверяем, что batch не пустой
    if not batch:
        return torch.tensor([]), torch.tensor([])

    # Разделяем данные и метки
    data_batch = []
    target_batch = []

    for sample in batch:
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            data, target = sample[0], sample[1]

            # Убеждаемся, что data это тензор
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)

            # Убеждаемся, что target это тензор
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)

            data_batch.append(data)
            target_batch.append(target)

    # Стекируем тензоры
    try:
        data_tensor = torch.stack(data_batch)
        target_tensor = torch.stack(target_batch)
    except RuntimeError as e:
        # Если размеры не совпадают, используем pad_sequence
        from torch.nn.utils.rnn import pad_sequence
        data_tensor = pad_sequence(data_batch, batch_first=True)
        target_tensor = pad_sequence(target_batch, batch_first=True)

    return data_tensor, target_tensor

# ИСПРАВЛЕНИЕ В train.py строка 897:
def train_step_fixed(model, data_loader, device):
    for batch_idx, batch_data in enumerate(data_loader):
        # БЫЛО: device = x.device  # где x может быть tuple
        # СТАЛО:
        if isinstance(batch_data, (list, tuple)):
            x, y = batch_data
            # Проверяем, что x - это тензор
            if isinstance(x, torch.Tensor):
                device = x.device
            else:
                # Если x не тензор, используем переданный device
                x = x.to(device) if hasattr(x, 'to') else torch.tensor(x).to(device)
                y = y.to(device) if hasattr(y, 'to') else torch.tensor(y).to(device)
        else:
            x = batch_data.to(device)

        # Продолжаем обучение...
