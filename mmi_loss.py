import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MMI_loss(nn.Module):
    """
    Maximum Mutual Information (MMI) Loss для улучшения качества TTS моделей.
    
    MMI loss помогает улучшить соответствие между входным текстом и выходными mel-спектрограммами,
    увеличивая взаимную информацию между ними.
    
    Основан на принципах из статей:
    - "Maximum Mutual Information Training for End-to-End Speech Recognition"
    - "Improving Neural Text-to-Speech Synthesis with Maximum Mutual Information"
    """
    
    def __init__(self, mmi_map=None, mmi_weight=1.0, reduction='mean', temperature=1.0):
        """
        Args:
            mmi_map: Карта для MMI вычислений (может быть None для автоматического вычисления)
            mmi_weight (float): Вес MMI loss в общей функции потерь
            reduction (str): Тип редукции ('mean', 'sum', 'none')
            temperature (float): Температура для softmax в MMI вычислениях
        """
        super(MMI_loss, self).__init__()
        self.mmi_map = mmi_map
        self.mmi_weight = mmi_weight
        self.reduction = reduction
        self.temperature = temperature
        
        # Кэш для хранения вычислений
        self._cache = {}
        
    def _compute_mutual_information(self, mel_outputs, mel_targets):
        """
        Вычисляет взаимную информацию между предсказанными и целевыми mel-спектрограммами.
        
        Args:
            mel_outputs (Tensor): Предсказанные mel-спектрограммы [B, T, mel_dim]
            mel_targets (Tensor): Целевые mel-спектрограммы [B, T, mel_dim]
            
        Returns:
            mi_loss (Tensor): MMI loss
        """
        batch_size, seq_len, mel_dim = mel_outputs.shape
        
        # Нормализация входов
        mel_outputs = F.normalize(mel_outputs.reshape(-1, mel_dim), p=2, dim=1)
        mel_targets = F.normalize(mel_targets.reshape(-1, mel_dim), p=2, dim=1)
        
        # Вычисление косинусного сходства
        similarity_matrix = torch.mm(mel_outputs, mel_targets.t()) / self.temperature
        
        # Создание положительных пар (диагональные элементы)
        batch_indices = torch.arange(batch_size * seq_len, device=mel_outputs.device)
        positive_similarities = similarity_matrix[batch_indices, batch_indices]
        
        # Вычисление InfoNCE loss (контрастивная версия MMI)
        exp_similarities = torch.exp(similarity_matrix)
        sum_exp_similarities = torch.sum(exp_similarities, dim=1)
        
        # MMI loss = -log(exp(positive) / sum(exp(all)))
        mmi_loss = -torch.log(torch.exp(positive_similarities) / sum_exp_similarities)
        
        # Применение редукции
        if self.reduction == 'mean':
            return torch.mean(mmi_loss)
        elif self.reduction == 'sum':
            return torch.sum(mmi_loss)
        else:
            return mmi_loss
    
    def _compute_frame_level_mmi(self, mel_outputs, mel_targets):
        """
        Вычисляет MMI на уровне фреймов для более детального контроля.
        
        Args:
            mel_outputs (Tensor): Предсказанные mel-спектрограммы [B, T, mel_dim]
            mel_targets (Tensor): Целевые mel-спектрограммы [B, T, mel_dim]
            
        Returns:
            frame_mmi_loss (Tensor): Frame-level MMI loss
        """
        batch_size, seq_len, mel_dim = mel_outputs.shape
        
        frame_losses = []
        
        for t in range(seq_len):
            # Извлекаем фреймы в момент времени t
            output_frame = mel_outputs[:, t, :]  # [B, mel_dim]
            target_frame = mel_targets[:, t, :]  # [B, mel_dim]
            
            # Нормализация
            output_frame = F.normalize(output_frame, p=2, dim=1)
            target_frame = F.normalize(target_frame, p=2, dim=1)
            
            # Вычисление сходства
            frame_similarity = torch.sum(output_frame * target_frame, dim=1)  # [B]
            
            # Контрастивные пары (все остальные фреймы как негативные примеры)
            all_similarities = torch.mm(output_frame, target_frame.t())  # [B, B]
            
            # Применение температуры
            all_similarities = all_similarities / self.temperature
            frame_similarity = frame_similarity / self.temperature
            
            # InfoNCE для данного фрейма
            exp_all = torch.exp(all_similarities)
            sum_exp = torch.sum(exp_all, dim=1)
            frame_loss = -torch.log(torch.exp(frame_similarity) / sum_exp)
            
            frame_losses.append(frame_loss)
        
        # Объединяем потери по всем фреймам
        frame_mmi_loss = torch.stack(frame_losses, dim=1)  # [B, T]
        
        if self.reduction == 'mean':
            return torch.mean(frame_mmi_loss)
        elif self.reduction == 'sum':
            return torch.sum(frame_mmi_loss)
        else:
            return frame_mmi_loss
    
    def forward(self, mel_outputs, mel_targets, use_frame_level=False):
        """
        Вычисляет MMI loss между предсказанными и целевыми mel-спектрограммами.
        
        Args:
            mel_outputs (Tensor): Предсказанные mel-спектрограммы [B, T, mel_dim]
            mel_targets (Tensor): Целевые mel-спектрограммы [B, T, mel_dim]
            use_frame_level (bool): Использовать ли frame-level MMI
            
        Returns:
            weighted_mmi_loss (Tensor): Взвешенный MMI loss
        """
        # Проверка размерностей
        if mel_outputs.shape != mel_targets.shape:
            raise ValueError(f"Shape mismatch: outputs {mel_outputs.shape} vs targets {mel_targets.shape}")
        
        # Обеспечиваем, что тензоры требуют градиенты
        if not mel_outputs.requires_grad:
            mel_outputs = mel_outputs.requires_grad_(True)
            
        # Выбираем метод вычисления MMI
        if use_frame_level:
            mmi_loss = self._compute_frame_level_mmi(mel_outputs, mel_targets)
        else:
            mmi_loss = self._compute_mutual_information(mel_outputs, mel_targets)
        
        # Применяем вес MMI
        weighted_mmi_loss = self.mmi_weight * mmi_loss
        
        return weighted_mmi_loss
    
    def get_weight(self):
        """Возвращает текущий вес MMI loss."""
        return self.mmi_weight
    
    def set_weight(self, weight):
        """Устанавливает новый вес MMI loss."""
        self.mmi_weight = weight
        
    def decay_weight(self, decay_factor=0.99):
        """Уменьшает вес MMI loss."""
        self.mmi_weight *= decay_factor
        
    def reset_weight(self, weight=1.0):
        """Сбрасывает вес MMI loss."""
        self.mmi_weight = weight


class AdaptiveMMI_loss(MMI_loss):
    """
    Адаптивная версия MMI loss с динамическим изменением веса в процессе обучения.
    """
    
    def __init__(self, mmi_map=None, initial_weight=1.0, min_weight=0.1, 
                 decay_steps=10000, decay_factor=0.95, **kwargs):
        """
        Args:
            initial_weight (float): Начальный вес MMI loss
            min_weight (float): Минимальный вес MMI loss
            decay_steps (int): Количество шагов для одного цикла decay
            decay_factor (float): Фактор уменьшения веса
        """
        super().__init__(mmi_map, initial_weight, **kwargs)
        self.initial_weight = initial_weight
        self.min_weight = min_weight
        self.decay_steps = decay_steps
        self.decay_factor = decay_factor
        self.step_count = 0
        
    def step(self):
        """Обновляет вес MMI loss на основе количества шагов обучения."""
        self.step_count += 1
        
        if self.step_count % self.decay_steps == 0:
            new_weight = max(self.mmi_weight * self.decay_factor, self.min_weight)
            self.set_weight(new_weight)
            
    def reset_schedule(self):
        """Сбрасывает расписание адаптации веса."""
        self.step_count = 0
        self.mmi_weight = self.initial_weight


# Псевдоним для обратной совместимости
MMI_Loss = MMI_loss 