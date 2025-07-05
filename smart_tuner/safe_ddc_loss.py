#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SafeDDCLoss - Безопасная обработка DDC Loss для Tacotron2-New
Основан на анализе проблем "DDC loss: размеры не совпадают"

Особенности:
- Безопасное выравнивание размеров тензоров
- Правильная обработка масок
- Детальное логирование
- Интеграция с Smart Tuner
"""

import torch
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Dict, Any
from smart_tuner.smart_truncation_ddc import SmartTruncationDDC
from smart_tuner.memory_efficient_ddc import MemoryEfficientDDC

class SafeDDCLoss:
    """
    Безопасный DDC Loss с интеллектуальной обработкой размеров.
    
    Решает проблемы:
    - Размеры тензоров не совпадают
    - Потеря информации при обрезании
    - Неправильная обработка масок
    """
    
    def __init__(self, weight=1.0, use_masking=True, log_warnings=True, mode='safe'):
        self.weight = weight
        self.use_masking = use_masking
        self.log_warnings = log_warnings
        self.mode = mode  # 'safe', 'smart_truncation', 'memory_efficient'
        
        # Статистика
        self.total_calls = 0
        self.size_mismatches = 0
        self.masking_applied = 0
        self.errors = 0
        
        # Логирование
        self.logger = logging.getLogger(__name__)
        
        self.smart_trunc = SmartTruncationDDC() if mode == 'smart_truncation' else None
        self.memory_efficient = MemoryEfficientDDC() if mode == 'memory_efficient' else None
        
    def __call__(self, pred: torch.Tensor, target: torch.Tensor, 
                 input_lengths: Optional[torch.Tensor] = None, 
                 target_lengths: Optional[torch.Tensor] = None,
                 step: int = 0) -> torch.Tensor:
        """
        Безопасное вычисление DDC loss с обработкой размеров.
        
        Args:
            pred: Предсказанные mel спектрограммы (B, n_mels, T_pred)
            target: Целевые mel спектрограммы (B, n_mels, T_target)
            input_lengths: Длины входных последовательностей
            target_lengths: Длины целевых последовательностей
            step: Текущий шаг обучения
            
        Returns:
            DDC loss tensor
        """
        self.total_calls += 1
        
        try:
            if self.mode == 'smart_truncation' and self.smart_trunc is not None:
                return self.smart_trunc(pred, target)
            if self.mode == 'memory_efficient' and self.memory_efficient is not None:
                return self.memory_efficient(pred, target)
            
            # Проверяем входные данные
            if pred is None or target is None:
                self.logger.warning("❌ DDC Loss: pred или target равны None")
                return torch.tensor(0.0, requires_grad=True, device=pred.device if pred is not None else 'cpu')
            
            # Проверяем размерности
            if pred.dim() != 3 or target.dim() != 3:
                self.logger.warning(f"❌ DDC Loss: Неправильные размерности pred={pred.shape}, target={target.shape}")
                return torch.tensor(0.0, requires_grad=True, device=pred.device)
            
            batch_size = pred.size(0)
            
            # Определяем минимальные размеры
            min_seq_len = min(pred.size(-1), target.size(-1))
            min_feat_len = min(pred.size(1), target.size(1))
            
            # Проверяем необходимость обрезания
            needs_trimming = (pred.size(-1) != min_seq_len or 
                            target.size(-1) != min_seq_len or
                            pred.size(1) != min_feat_len or 
                            target.size(1) != min_feat_len)
            
            if needs_trimming:
                self.size_mismatches += 1
                if self.log_warnings:
                    self.logger.info(f"🔧 DDC Loss: Выравнивание размеров до {min_feat_len}x{min_seq_len} "
                                   f"(было pred={pred.shape}, target={target.shape})")
            
            # Безопасное обрезание
            pred_safe = pred[:, :min_feat_len, :min_seq_len]
            target_safe = target[:, :min_feat_len, :min_seq_len]
            
            # Создание и применение масок
            if self.use_masking and input_lengths is not None and target_lengths is not None:
                mask = self._create_advanced_mask(batch_size, min_seq_len, min_feat_len,
                                                input_lengths, target_lengths, pred_safe.device)
                pred_safe = pred_safe * mask
                target_safe = target_safe * mask
                self.masking_applied += 1
            
            # Вычисление loss с проверкой на NaN/Inf
            loss = F.mse_loss(pred_safe, target_safe, reduction='mean')
            
            # Проверка на валидность loss
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.error(f"❌ DDC Loss: NaN/Inf в результате loss={loss}")
                self.errors += 1
                return torch.tensor(0.0, requires_grad=True, device=pred.device)
            
            # Применяем вес
            weighted_loss = loss * self.weight
            
            return weighted_loss
            
        except Exception as e:
            self.errors += 1
            self.logger.error(f"❌ DDC Loss error: {e}")
            return torch.tensor(0.0, requires_grad=True, device=pred.device if pred is not None else 'cpu')
    
    def _create_advanced_mask(self, batch_size: int, seq_len: int, feat_len: int,
                            input_lengths: torch.Tensor, target_lengths: torch.Tensor,
                            device: torch.device) -> torch.Tensor:
        """
        Создает продвинутую маску для правильного вычисления loss.
        
        Особенности:
        - Учитывает оба типа длин (input и target)
        - Правильная обработка batch размерности
        - Защита от выхода за границы
        """
        # Создаем маску размером (batch_size, feat_len, seq_len)
        mask = torch.ones(batch_size, feat_len, seq_len, device=device)
        
        # Приводим длины к правильному формату
        if input_lengths.dim() == 0:
            input_lengths = input_lengths.unsqueeze(0)
        if target_lengths.dim() == 0:
            target_lengths = target_lengths.unsqueeze(0)
        
        # Расширяем до batch_size если нужно
        if input_lengths.size(0) == 1 and batch_size > 1:
            input_lengths = input_lengths.expand(batch_size)
        if target_lengths.size(0) == 1 and batch_size > 1:
            target_lengths = target_lengths.expand(batch_size)
        
        # Применяем маскирование для каждого элемента batch
        for i in range(min(batch_size, input_lengths.size(0), target_lengths.size(0))):
            # Определяем валидную длину как минимум из двух
            valid_len = min(
                int(input_lengths[i].item()) if input_lengths[i] is not None else seq_len,
                int(target_lengths[i].item()) if target_lengths[i] is not None else seq_len,
                seq_len
            )
            
            # Применяем маску
            if valid_len < seq_len:
                mask[i, :, valid_len:] = 0.0
        
        return mask
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику работы SafeDDCLoss."""
        if self.total_calls == 0:
            return {
                'total_calls': 0,
                'size_mismatches': 0,
                'masking_applied': 0,
                'errors': 0,
                'mismatch_rate': 0.0,
                'error_rate': 0.0
            }
        
        return {
            'total_calls': self.total_calls,
            'size_mismatches': self.size_mismatches,
            'masking_applied': self.masking_applied,
            'errors': self.errors,
            'mismatch_rate': self.size_mismatches / self.total_calls,
            'error_rate': self.errors / self.total_calls
        }
    
    def get_recommendations(self) -> list:
        """Возвращает рекомендации на основе статистики."""
        stats = self.get_statistics()
        recommendations = []
        
        if stats['mismatch_rate'] > 0.5:
            recommendations.append("⚠️ Частые несовпадения размеров DDC - проверить архитектуру модели")
        
        if stats['error_rate'] > 0.1:
            recommendations.append("🚨 Высокая частота ошибок DDC - проверить входные данные")
        
        if stats['masking_applied'] == 0 and self.use_masking:
            recommendations.append("💡 Маскирование DDC не применяется - проверить передачу lengths")
        
        return recommendations
    
    def reset_statistics(self):
        """Сбрасывает статистику."""
        self.total_calls = 0
        self.size_mismatches = 0
        self.masking_applied = 0
        self.errors = 0
        self.logger.info("🔄 Статистика SafeDDCLoss сброшена")


class AdaptiveDDCLoss(SafeDDCLoss):
    """
    Адаптивный DDC Loss с автоматической настройкой веса.
    
    Особенности:
    - Автоматическая настройка веса на основе качества
    - Интеграция с системой мониторинга
    - Адаптация к фазе обучения
    """
    
    def __init__(self, initial_weight=1.0, min_weight=0.1, max_weight=5.0,
                 adaptation_rate=0.01, quality_threshold=0.7):
        super().__init__(weight=initial_weight)
        
        self.initial_weight = initial_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.adaptation_rate = adaptation_rate
        self.quality_threshold = quality_threshold
        
        # История качества для адаптации
        self.quality_history = []
        self.weight_history = []
        
    def adapt_weight(self, quality_score: float, step: int):
        """
        Адаптирует вес DDC loss на основе качества.
        
        Args:
            quality_score: Оценка качества (0-1)
            step: Текущий шаг обучения
        """
        self.quality_history.append(quality_score)
        if len(self.quality_history) > 100:
            self.quality_history.pop(0)
        
        # Вычисляем среднее качество за последние шаги
        if len(self.quality_history) >= 10:
            recent_quality = sum(self.quality_history[-10:]) / 10
            
            # Адаптируем вес
            if recent_quality < self.quality_threshold:
                # Низкое качество - увеличиваем вес DDC
                new_weight = min(self.weight * (1 + self.adaptation_rate), self.max_weight)
            else:
                # Высокое качество - уменьшаем вес DDC
                new_weight = max(self.weight * (1 - self.adaptation_rate), self.min_weight)
            
            # Применяем изменение
            if abs(new_weight - self.weight) > 0.01:
                old_weight = self.weight
                self.weight = new_weight
                self.weight_history.append({
                    'step': step,
                    'old_weight': old_weight,
                    'new_weight': new_weight,
                    'quality': recent_quality
                })
                
                self.logger.info(f"🔄 DDC weight адаптирован: {old_weight:.3f} → {new_weight:.3f} "
                               f"(quality: {recent_quality:.3f})")


# Глобальный экземпляр для интеграции
_global_ddc_loss = None

def get_global_ddc_loss() -> SafeDDCLoss:
    """Возвращает глобальный экземпляр DDC loss."""
    global _global_ddc_loss
    if _global_ddc_loss is None:
        _global_ddc_loss = SafeDDCLoss()
    return _global_ddc_loss

def set_global_ddc_loss(ddc_loss: SafeDDCLoss):
    """Устанавливает глобальный экземпляр DDC loss."""
    global _global_ddc_loss
    _global_ddc_loss = ddc_loss 