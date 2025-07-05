#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Learning Rate Adapter для Tacotron2-New
Основан на анализе проблем NaN/Inf в Loss и взрыва градиентов

Особенности:
- Адаптивное изменение learning rate
- Экстренные режимы при критических проблемах
- Интеграция с системой мониторинга
- Автоматическое восстановление
"""

import torch
import math
import logging
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

@dataclass
class LRChangeEvent:
    """Событие изменения learning rate."""
    step: int
    old_lr: float
    new_lr: float
    reason: str
    trigger_value: Optional[float] = None
    emergency: bool = False

class SmartLRAdapter:
    """
    Интеллектуальный адаптер learning rate с экстренными режимами.
    
    Решает проблемы:
    - NaN/Inf в loss
    - Взрыв градиентов
    - Плохая сходимость
    - Нестабильное обучение
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 patience: int = 10, factor: float = 0.5, 
                 min_lr: float = 1e-8, max_lr: float = 1e-3,
                 emergency_factor: float = 0.1,
                 grad_norm_threshold: float = 1000.0,
                 loss_nan_threshold: float = 1e6):
        
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.emergency_factor = emergency_factor
        self.grad_norm_threshold = grad_norm_threshold
        self.loss_nan_threshold = loss_nan_threshold
        
        # Состояние адаптера
        self.bad_epochs = 0
        self.best_loss = float('inf')
        self.emergency_mode = False
        self.emergency_mode_steps = 0
        self.recovery_attempts = 0
        self.max_recovery_attempts = 5
        
        # История
        self.lr_history = []
        self.change_events = []
        self.loss_history = []
        self.grad_norm_history = []
        
        # Логирование
        self.logger = logging.getLogger(__name__)
        
        # Инициализация
        self._record_initial_lr()
    
    def _record_initial_lr(self):
        """Записывает начальный learning rate."""
        initial_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(initial_lr)
        self.logger.info(f"🚀 Smart LR Adapter инициализирован с LR: {initial_lr:.2e}")
    
    def step(self, loss: float, grad_norm: Optional[float] = None, 
             step: int = 0) -> bool:
        """
        Выполняет шаг адаптации learning rate.
        
        Args:
            loss: Текущий loss
            grad_norm: Норма градиентов
            step: Текущий шаг обучения
            
        Returns:
            True если LR был изменен
        """
        current_loss = loss  # Для совместимости с существующим кодом
        # Сохраняем историю
        self.loss_history.append(current_loss)
        if grad_norm is not None:
            self.grad_norm_history.append(grad_norm)
        
        # Ограничиваем размер истории
        if len(self.loss_history) > 1000:
            self.loss_history.pop(0)
        if len(self.grad_norm_history) > 1000:
            self.grad_norm_history.pop(0)
        
        # Проверяем экстренные ситуации
        if self._check_emergency_conditions(current_loss, grad_norm):
            return self._handle_emergency(current_loss, grad_norm, step)
        
        # Обычная адаптация
        return self._handle_normal_adaptation(current_loss, step)
    
    def _check_emergency_conditions(self, current_loss: float, 
                                  grad_norm: Optional[float]) -> bool:
        """Проверяет условия для экстренного режима."""
        # Проверка на NaN/Inf в loss
        if math.isnan(current_loss) or math.isinf(current_loss) or current_loss > self.loss_nan_threshold:
            return True
        
        # Проверка на взрыв градиентов
        if grad_norm is not None:
            if math.isnan(grad_norm) or math.isinf(grad_norm) or grad_norm > self.grad_norm_threshold:
                return True
        
        return False
    
    def _handle_emergency(self, current_loss: float, grad_norm: Optional[float], 
                         step: int) -> bool:
        """Обрабатывает экстренные ситуации."""
        self.emergency_mode = True
        self.emergency_mode_steps += 1
        self.recovery_attempts += 1
        
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Экстренное снижение LR
        if self.recovery_attempts <= self.max_recovery_attempts:
            new_lr = max(current_lr * self.emergency_factor, self.min_lr)
        else:
            # Если слишком много попыток - сбрасываем к минимуму
            new_lr = self.min_lr
        
        # Применяем изменение
        self._apply_lr_change(current_lr, new_lr, step, "EMERGENCY", 
                            trigger_value=current_loss, emergency=True)
        
        # Логирование
        reason = "NaN/Inf loss" if math.isnan(current_loss) or math.isinf(current_loss) else "Gradient explosion"
        self.logger.warning(f"🚨 ЭКСТРЕННЫЙ режим: {reason} - LR {current_lr:.2e} → {new_lr:.2e}")
        
        return True
    
    def _handle_normal_adaptation(self, current_loss: float, step: int) -> bool:
        """Обрабатывает обычную адаптацию LR."""
        # Проверяем улучшение
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.bad_epochs = 0
            
            # Выход из экстренного режима
            if self.emergency_mode:
                self._exit_emergency_mode()
        else:
            self.bad_epochs += 1
        
        # Проверяем необходимость снижения LR
        if self.bad_epochs >= self.patience:
            return self._reduce_lr(step, "PATIENCE")
        
        return False
    
    def _reduce_lr(self, step: int, reason: str) -> bool:
        """Снижает learning rate."""
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = max(current_lr * self.factor, self.min_lr)
        
        # Применяем изменение
        self._apply_lr_change(current_lr, new_lr, step, reason)
        
        # Сбрасываем счетчик
        self.bad_epochs = 0
        
        self.logger.info(f"📉 Снижение LR: {current_lr:.2e} → {new_lr:.2e} (причина: {reason})")
        return True
    
    def _apply_lr_change(self, old_lr: float, new_lr: float, step: int, 
                        reason: str, trigger_value: Optional[float] = None, 
                        emergency: bool = False):
        """Применяет изменение learning rate."""
        # Обновляем все группы параметров
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Записываем в историю
        self.lr_history.append(new_lr)
        
        # Создаем событие изменения
        event = LRChangeEvent(
            step=step,
            old_lr=old_lr,
            new_lr=new_lr,
            reason=reason,
            trigger_value=trigger_value,
            emergency=emergency
        )
        self.change_events.append(event)
        
        # Ограничиваем размер истории событий
        if len(self.change_events) > 100:
            self.change_events.pop(0)
    
    def _exit_emergency_mode(self):
        """Выход из экстренного режима."""
        if self.emergency_mode:
            self.emergency_mode = False
            self.emergency_mode_steps = 0
            self.recovery_attempts = 0
            self.logger.info("✅ Выход из экстренного режима LR")
    
    def increase_lr(self, factor: float = 1.5, step: int = 0, reason: str = "MANUAL"):
        """Увеличивает learning rate."""
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = min(current_lr * factor, self.max_lr)
        
        self._apply_lr_change(current_lr, new_lr, step, reason)
        self.logger.info(f"📈 Увеличение LR: {current_lr:.2e} → {new_lr:.2e} (причина: {reason})")
    
    def set_lr(self, new_lr: float, step: int = 0, reason: str = "MANUAL"):
        """Устанавливает конкретное значение learning rate."""
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Ограничиваем в допустимых пределах
        new_lr = max(min(new_lr, self.max_lr), self.min_lr)
        
        self._apply_lr_change(current_lr, new_lr, step, reason)
        self.logger.info(f"🎯 Установка LR: {current_lr:.2e} → {new_lr:.2e} (причина: {reason})")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику работы адаптера."""
        if not self.lr_history:
            return {
                'current_lr': 0.0,
                'total_changes': 0,
                'emergency_changes': 0,
                'emergency_mode': False,
                'recovery_attempts': 0,
                'avg_loss': 0.0,
                'avg_grad_norm': 0.0
            }
        
        current_lr = self.lr_history[-1]
        emergency_changes = len([e for e in self.change_events if e.emergency])
        
        return {
            'current_lr': current_lr,
            'total_changes': len(self.change_events),
            'emergency_changes': emergency_changes,
            'emergency_mode': self.emergency_mode,
            'emergency_mode_steps': self.emergency_mode_steps,
            'recovery_attempts': self.recovery_attempts,
            'bad_epochs': self.bad_epochs,
            'best_loss': self.best_loss,
            'avg_loss': np.mean(self.loss_history) if self.loss_history else 0.0,
            'avg_grad_norm': np.mean(self.grad_norm_history) if self.grad_norm_history else 0.0,
            'lr_history': self.lr_history[-10:],  # Последние 10 значений
            'recent_changes': self.change_events[-5:]  # Последние 5 изменений
        }
    
    def get_recommendations(self) -> List[str]:
        """Возвращает рекомендации на основе статистики."""
        stats = self.get_statistics()
        recommendations = []
        
        if stats['emergency_changes'] > 0:
            recommendations.append("🚨 Обнаружены экстренные изменения LR - проверить стабильность модели")
        
        if stats['emergency_mode']:
            recommendations.append("🛡️ Активен экстренный режим LR - обучение может быть нестабильным")
        
        if stats['recovery_attempts'] >= self.max_recovery_attempts:
            recommendations.append("⚠️ Достигнут лимит попыток восстановления - проверить архитектуру")
        
        if stats['avg_grad_norm'] > 10.0:
            recommendations.append("📈 Высокая средняя норма градиентов - увеличить batch size")
        
        if stats['current_lr'] <= self.min_lr:
            recommendations.append("💡 LR достиг минимума - проверить сходимость модели")
        
        return recommendations
    
    def reset(self):
        """Сбрасывает состояние адаптера."""
        self.bad_epochs = 0
        self.best_loss = float('inf')
        self.emergency_mode = False
        self.emergency_mode_steps = 0
        self.recovery_attempts = 0
        self.logger.info("🔄 Smart LR Adapter сброшен")


# Глобальный экземпляр для интеграции
_global_lr_adapter = None

def get_global_lr_adapter() -> Optional[SmartLRAdapter]:
    """Возвращает глобальный экземпляр LR адаптера."""
    return _global_lr_adapter

def set_global_lr_adapter(adapter: SmartLRAdapter):
    """Устанавливает глобальный экземпляр LR адаптера."""
    global _global_lr_adapter
    _global_lr_adapter = adapter 