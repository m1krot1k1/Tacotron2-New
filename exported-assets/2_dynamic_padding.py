
# Решение 2: Динамический padding для переменных длин
import torch
from torch.nn.utils.rnn import pad_sequence

class DynamicPaddingCollator:
    def __init__(self, pad_value=0.0):
        self.pad_value = pad_value

    def __call__(self, batch):
        """
        Динамический collator для батчей переменной длины
        Args:
            batch: список тензоров разной длины
        """
        # Сортируем по длине для оптимизации
        batch = sorted(batch, key=lambda x: x.size(-1), reverse=True)

        # Получаем максимальную длину в текущем батче
        max_len = batch[0].size(-1)

        # Pad только до максимума в текущем батче
        padded_batch = []
        lengths = []

        for item in batch:
            current_len = item.size(-1)
            if current_len < max_len:
                # Pad справа нулями
                pad_size = max_len - current_len
                padded_item = F.pad(item, (0, pad_size), value=self.pad_value)
            else:
                padded_item = item

            padded_batch.append(padded_item)
            lengths.append(current_len)

        return torch.stack(padded_batch), torch.tensor(lengths)
