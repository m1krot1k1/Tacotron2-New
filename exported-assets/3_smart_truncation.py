
# Решение 3: Умное обрезание с сохранением важной информации
import torch
import torch.nn as nn

class SmartTruncationDDC(nn.Module):
    def __init__(self, preserve_ratio=0.8, attention_threshold=0.1):
        super(SmartTruncationDDC, self).__init__()
        self.preserve_ratio = preserve_ratio
        self.attention_threshold = attention_threshold

    def find_important_region(self, attention_weights):
        """Находит наиболее важную область attention"""
        # Суммируем attention по encoder dimension
        attention_sum = attention_weights.sum(dim=-1)  # [batch, time_steps]

        # Находим пики attention
        important_mask = attention_sum > self.attention_threshold

        # Находим границы важной области
        for batch_idx in range(attention_weights.size(0)):
            mask = important_mask[batch_idx]
            if mask.any():
                start_idx = mask.nonzero()[0].item()
                end_idx = mask.nonzero()[-1].item() + 1
            else:
                # Если нет явных пиков, берем центральную часть
                total_len = attention_weights.size(1)
                start_idx = total_len // 4
                end_idx = 3 * total_len // 4

        return start_idx, end_idx

    def forward(self, coarse_attention, fine_attention):
        """
        DDC loss с умным обрезанием
        """
        coarse_len = coarse_attention.size(1)
        fine_len = fine_attention.size(1)

        if coarse_len == fine_len:
            return F.mse_loss(coarse_attention, fine_attention)

        # Определяем целевую длину
        target_len = min(coarse_len, fine_len)
        target_len = int(target_len * self.preserve_ratio)

        # Находим важные области в обеих последовательностях
        coarse_start, coarse_end = self.find_important_region(coarse_attention)
        fine_start, fine_end = self.find_important_region(fine_attention)

        # Извлекаем важные части
        coarse_important = coarse_attention[:, coarse_start:coarse_start+target_len, :]
        fine_important = fine_attention[:, fine_start:fine_start+target_len, :]

        # Если длины все еще не совпадают, используем интерполяцию
        if coarse_important.size(1) != fine_important.size(1):
            min_len = min(coarse_important.size(1), fine_important.size(1))
            coarse_important = coarse_important[:, :min_len, :]
            fine_important = fine_important[:, :min_len, :]

        return F.mse_loss(coarse_important, fine_important)
