import torch
import torch.nn.functional as F

class DynamicPaddingCollator:
    """
    Динамический collator для батчей переменной длины.
    Выравнивает тензоры только до максимальной длины в текущем батче.
    """
    def __init__(self, pad_value=0.0):
        self.pad_value = pad_value

    def __call__(self, batch):
        # Сортируем по длине для оптимизации
        batch = sorted(batch, key=lambda x: x.size(-1), reverse=True)
        max_len = batch[0].size(-1)
        padded_batch = []
        lengths = []
        for item in batch:
            current_len = item.size(-1)
            if current_len < max_len:
                pad_size = max_len - current_len
                padded_item = F.pad(item, (0, pad_size), value=self.pad_value)
            else:
                padded_item = item
            padded_batch.append(padded_item)
            lengths.append(current_len)
        return torch.stack(padded_batch), torch.tensor(lengths) 