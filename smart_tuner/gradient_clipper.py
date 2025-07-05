#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенный Gradient Clipper для Tacotron2-New
Основан на анализе проблем взрыва градиентов (Grad Norm 100k-400k)
"""

import torch
import torch.nn.utils as utils
import numpy as np
import logging
from typing import Tuple, Optional

class AdaptiveGradientClipper:
    """
    Интеллектуальный адаптивный gradient clipper с историей и экстренными режимами.
    
    Особенности:
    - Адаптивный порог на основе истории градиентов
    - Экстренный режим при критических значениях
    - Детальное логирование для диагностики
    - Интеграция с Smart Tuner
    """
    
    def __init__(self, max_norm=1.0, adaptive=True, history_size=1000, 
                 emergency_threshold=1000.0, percentile=95):
        self.max_norm = max_norm
        self.adaptive = adaptive
        self.history_size = history_size
        self.emergency_threshold = emergency_threshold
        self.percentile = percentile
        
        # История градиентов для адаптивного порога
        self.grad_history = []
        self.clip_history = []
        
        # Статистика
        self.total_clips = 0
        self.emergency_clips = 0
        self.max_grad_norm_seen = 0.0
        
        # Логирование
        self.logger = logging.getLogger(__name__)
        
        # Экстренный режим
        self.emergency_mode = False
        self.emergency_mode_steps = 0
        
    def clip_gradients(self, model: torch.nn.Module, step: int = 0) -> Tuple[bool, float, float]:
        """
        Интеллектуальное обрезание градиентов с адаптивным порогом.
        
        Args:
            model: Модель PyTorch
            step: Текущий шаг обучения
            
        Returns:
            (was_clipped, current_norm, clip_threshold)
        """
        # Вычисляем текущую норму градиентов
        total_norm = self._calculate_gradient_norm(model)
        
        # Обновляем статистику
        self.max_grad_norm_seen = max(self.max_grad_norm_seen, total_norm)
        
        # Сохраняем в историю
        self.grad_history.append(total_norm)
        if len(self.grad_history) > self.history_size:
            self.grad_history.pop(0)
        
        # Определяем порог для обрезания
        clip_threshold = self._calculate_adaptive_threshold(total_norm, step)
        
        # Проверяем необходимость обрезания
        should_clip = total_norm > clip_threshold
        
        if should_clip:
            # Применяем обрезание
            utils.clip_grad_norm_(model.parameters(), clip_threshold)
            
            # Обновляем статистику
            self.total_clips += 1
            self.clip_history.append({
                'step': step,
                'original_norm': total_norm,
                'clip_threshold': clip_threshold,
                'emergency': total_norm > self.emergency_threshold
            })
            
            # Проверяем экстренный режим
            if total_norm > self.emergency_threshold:
                self.emergency_clips += 1
                self.emergency_mode = True
                self.emergency_mode_steps += 1
                self.logger.warning(f"🚨 ЭКСТРЕННЫЙ режим: Grad Norm {total_norm:.2f} > {self.emergency_threshold}")
            else:
                self.logger.info(f"✂️ Градиенты обрезаны: {total_norm:.2f} → {clip_threshold:.2f}")
        
        # Выход из экстренного режима
        if self.emergency_mode and total_norm < self.max_norm * 0.5:
            self.emergency_mode = False
            self.emergency_mode_steps = 0
            self.logger.info("✅ Выход из экстренного режима gradient clipping")
        
        return should_clip, total_norm, clip_threshold
    
    def _calculate_gradient_norm(self, model: torch.nn.Module) -> float:
        """Вычисляет общую норму градиентов модели."""
        total_norm = 0.0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count == 0:
            return 0.0
        
        total_norm = total_norm ** (1. / 2)
        
        # Проверка на NaN/Inf
        if np.isnan(total_norm) or np.isinf(total_norm):
            self.logger.error(f"❌ NaN/Inf в градиентах: {total_norm}")
            return float('inf')
        
        return total_norm
    
    def _calculate_adaptive_threshold(self, current_norm: float, step: int) -> float:
        """Вычисляет адаптивный порог для обрезания градиентов."""
        if not self.adaptive or len(self.grad_history) < 10:
            return self.max_norm
        
        # Используем процентиль истории для адаптивного порога
        recent_history = self.grad_history[-100:]  # Последние 100 шагов
        if len(recent_history) < 5:
            return self.max_norm
        
        try:
            adaptive_threshold = np.percentile(recent_history, self.percentile)
            
            # В экстренном режиме используем более строгий порог
            if self.emergency_mode:
                adaptive_threshold = min(adaptive_threshold * 0.5, self.max_norm * 0.5)
            
            # Ограничиваем порог разумными пределами
            final_threshold = min(max(adaptive_threshold, self.max_norm * 0.1), self.max_norm)
            
            return final_threshold
            
        except Exception as e:
            self.logger.warning(f"Ошибка вычисления адаптивного порога: {e}")
            return self.max_norm
    
    def get_statistics(self) -> dict:
        """Возвращает статистику работы clipper."""
        if not self.grad_history:
            return {
                'total_clips': 0,
                'emergency_clips': 0,
                'max_grad_norm': 0.0,
                'avg_grad_norm': 0.0,
                'emergency_mode': False,
                'clip_rate': 0.0
            }
        
        avg_norm = np.mean(self.grad_history)
        clip_rate = self.total_clips / len(self.grad_history) if self.grad_history else 0.0
        
        return {
            'total_clips': self.total_clips,
            'emergency_clips': self.emergency_clips,
            'max_grad_norm': self.max_grad_norm_seen,
            'avg_grad_norm': avg_norm,
            'emergency_mode': self.emergency_mode,
            'emergency_mode_steps': self.emergency_mode_steps,
            'clip_rate': clip_rate,
            'recent_clips': len([c for c in self.clip_history if c['step'] > max(0, len(self.grad_history) - 100)])
        }
    
    def reset_emergency_mode(self):
        """Сбрасывает экстренный режим."""
        self.emergency_mode = False
        self.emergency_mode_steps = 0
        self.logger.info("🔄 Экстренный режим gradient clipping сброшен")
    
    def get_recommendations(self) -> list:
        """Возвращает рекомендации на основе статистики."""
        stats = self.get_statistics()
        recommendations = []
        
        if stats['emergency_clips'] > 0:
            recommendations.append("🚨 Обнаружены экстренные обрезания градиентов - снизить learning rate")
        
        if stats['clip_rate'] > 0.3:
            recommendations.append("⚠️ Частые обрезания градиентов - проверить архитектуру модели")
        
        if stats['avg_grad_norm'] > 10.0:
            recommendations.append("📈 Высокая средняя норма градиентов - увеличить batch size")
        
        if stats['emergency_mode']:
            recommendations.append("🛡️ Активен экстренный режим - обучение может быть нестабильным")
        
        return recommendations


# Глобальный экземпляр для интеграции с Smart Tuner
_global_clipper = None

def get_global_clipper() -> AdaptiveGradientClipper:
    """Возвращает глобальный экземпляр clipper."""
    global _global_clipper
    if _global_clipper is None:
        _global_clipper = AdaptiveGradientClipper()
    return _global_clipper

def set_global_clipper(clipper: AdaptiveGradientClipper):
    """Устанавливает глобальный экземпляр clipper."""
    global _global_clipper
    _global_clipper = clipper 