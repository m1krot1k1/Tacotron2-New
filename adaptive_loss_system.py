"""
🎯 ADAPTIVE LOSS SYSTEM - Интеллектуальная система адаптивных loss функций
========================================================================

Замена простого AdaptiveLossController на полноценную интеллектуальную систему:

1. 🧮 Dynamic Tversky Loss - адаптивная loss для unbalanced data
2. 📊 Intelligent Weight Manager - умное управление весами на основе контекста  
3. 🔄 Context-Based Loss Scaling - масштабирование по фазам обучения
4. 📈 Phase-Aware Loss Optimization - оптимизация параметров по фазам

Версия: 1.0.0
Автор: Enhanced Tacotron2 AI System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
from collections import deque


class LossPhase(Enum):
    """Фазы обучения для адаптации loss функций"""
    PRE_ALIGNMENT = "pre_alignment"        # Начальная фаза - фокус на attention
    ALIGNMENT_LEARNING = "alignment"       # Фаза обучения alignment
    REFINEMENT = "refinement"              # Фаза улучшения качества
    CONVERGENCE = "convergence"            # Фаза конвергенции


@dataclass
class LossContext:
    """Контекст для адаптации loss функций"""
    phase: LossPhase
    global_step: int
    attention_quality: float               # Диагональность attention
    gate_accuracy: float                   # Точность gate предсказаний
    mel_consistency: float                 # Консистентность mel spectrogram
    gradient_norm: float                   # Норма градиентов
    loss_stability: float                  # Стабильность loss (std)
    learning_rate: float                   # Текущий learning rate


class DynamicTverskyLoss(nn.Module):
    """
    🧮 Dynamic Tversky Loss - адаптивная loss функция для unbalanced data
    
    Основано на исследованиях Focal Loss и Tversky Loss (2017-2024):
    - Автоматическая адаптация alpha/beta параметров
    - Контекстная настройка на основе качества модели
    - Фокус на трудных примерах (hard negative mining)
    """
    
    def __init__(self, 
                 initial_alpha: float = 0.7,
                 initial_beta: float = 0.3,
                 adapt_rate: float = 0.01,
                 min_alpha: float = 0.1,
                 max_alpha: float = 0.9):
        super().__init__()
        
        self.alpha = initial_alpha                    # Вес False Positives
        self.beta = initial_beta                      # Вес False Negatives  
        self.adapt_rate = adapt_rate                  # Скорость адаптации
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        
        # История для адаптации
        self.accuracy_history = deque(maxlen=100)
        self.adaptation_history = []
        
        print(f"🧮 DynamicTverskyLoss инициализирован: α={self.alpha:.3f}, β={self.beta:.3f}")
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                context: Optional[LossContext] = None) -> torch.Tensor:
        """
        Вычисление Dynamic Tversky Loss с адаптивными параметрами
        
        Args:
            predictions: Предсказания модели [B, ...]
            targets: Целевые значения [B, ...]
            context: Контекст обучения для адаптации
            
        Returns:
            torch.Tensor: Вычисленная loss
        """
        # Применяем sigmoid для gate predictions
        if predictions.dim() == targets.dim():
            probs = torch.sigmoid(predictions)
        else:
            probs = predictions
            
        # Clamp для численной стабильности
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        targets = torch.clamp(targets, min=0.0, max=1.0)
        
        # True Positives, False Positives, False Negatives
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        
        # Tversky Index
        tversky_index = tp / (tp + self.alpha * fp + self.beta * fn + 1e-7)
        
        # Tversky Loss (1 - Tversky Index)
        loss = 1 - tversky_index
        
        # Адаптация параметров на основе контекста
        if context is not None:
            self._adapt_parameters(context)
        
        return loss
    
    def _adapt_parameters(self, context: LossContext):
        """Адаптация α и β параметров на основе контекста обучения"""
        
        # Адаптация на основе gate accuracy
        current_accuracy = context.gate_accuracy
        self.accuracy_history.append(current_accuracy)
        
        if len(self.accuracy_history) >= 10:
            recent_accuracy = np.mean(list(self.accuracy_history)[-10:])
            
            # Если accuracy низкая - увеличиваем штраф за FP (увеличиваем alpha)
            if recent_accuracy < 0.7:
                target_alpha = min(self.max_alpha, self.alpha + self.adapt_rate)
            # Если accuracy высокая - увеличиваем штраф за FN (уменьшаем alpha)
            elif recent_accuracy > 0.85:
                target_alpha = max(self.min_alpha, self.alpha - self.adapt_rate)
            else:
                target_alpha = self.alpha
            
            # Плавная адаптация
            self.alpha = 0.9 * self.alpha + 0.1 * target_alpha
            self.beta = 1.0 - self.alpha
            
            # Логируем изменения
            if abs(target_alpha - self.alpha) > 0.01:
                self.adaptation_history.append({
                    'step': context.global_step,
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'accuracy': recent_accuracy,
                    'phase': context.phase.value
                })


class IntelligentWeightManager(nn.Module):
    """
    📊 Intelligent Weight Manager - умное управление весами loss компонентов
    
    Ключевые возможности:
    - Фазово-зависимая адаптация весов
    - Реакция на качество обучения  
    - Предотвращение катастрофических сбоев
    - Адаптивное расписание изменений
    """
    
    def __init__(self, hparams):
        super().__init__()
        
        # Базовые веса из hparams
        self.base_weights = {
            'mel': getattr(hparams, 'mel_loss_weight', 1.0),
            'gate': getattr(hparams, 'gate_loss_weight', 1.0),
            'guided_attention': getattr(hparams, 'guide_loss_weight', 2.0),
            'spectral': getattr(hparams, 'spectral_loss_weight', 0.3),
            'perceptual': getattr(hparams, 'perceptual_loss_weight', 0.2),
            'style': getattr(hparams, 'style_loss_weight', 0.1),
            'monotonic': getattr(hparams, 'monotonic_loss_weight', 0.1)
        }
        
        # Текущие веса (начинаем с базовых)
        self.current_weights = self.base_weights.copy()
        
        # Ограничения безопасности (предотвращение взрывов как в AutoFixManager)
        self.weight_limits = {
            'mel': (0.1, 3.0),
            'gate': (0.1, 2.5),
            'guided_attention': (0.1, 15.0),        # Вместо 200 в AutoFixManager!
            'spectral': (0.0, 1.0),
            'perceptual': (0.0, 1.0),
            'style': (0.0, 0.5),
            'monotonic': (0.0, 0.5)
        }
        
        # Параметры адаптации
        self.adaptation_rate = getattr(hparams, 'weight_adaptation_rate', 0.02)
        self.stability_threshold = getattr(hparams, 'loss_stability_threshold', 2.0)
        
        # История для обучения
        self.weight_history = []
        self.performance_history = deque(maxlen=50)
        
        print(f"📊 IntelligentWeightManager инициализирован с весами: {self.current_weights}")
    
    def get_adaptive_weights(self, context: LossContext) -> Dict[str, float]:
        """
        Получение адаптивных весов на основе контекста обучения
        
        Args:
            context: Контекст текущего состояния обучения
            
        Returns:
            Dict[str, float]: Адаптированные веса для каждого компонента loss
        """
        # Создаем копию текущих весов
        adapted_weights = self.current_weights.copy()
        
        # 1. Фазово-зависимая адаптация
        adapted_weights = self._adapt_by_phase(adapted_weights, context)
        
        # 2. Адаптация на основе качества модели
        adapted_weights = self._adapt_by_quality(adapted_weights, context)
        
        # 3. Стабилизация при проблемах
        adapted_weights = self._stabilize_weights(adapted_weights, context)
        
        # 4. Применение ограничений безопасности
        adapted_weights = self._apply_safety_limits(adapted_weights)
        
        # Обновляем текущие веса плавно
        self._smooth_weight_update(adapted_weights, context)
        
        return self.current_weights
    
    def _adapt_by_phase(self, weights: Dict[str, float], context: LossContext) -> Dict[str, float]:
        """Адаптация весов в зависимости от фазы обучения"""
        
        if context.phase == LossPhase.PRE_ALIGNMENT:
            # Фокус на guided attention и gate для хорошего alignment
            weights['guided_attention'] *= 1.5
            weights['gate'] *= 1.2
            weights['mel'] *= 0.8
            
        elif context.phase == LossPhase.ALIGNMENT_LEARNING:
            # Баланс между alignment и качеством
            weights['guided_attention'] *= 1.2  
            weights['mel'] *= 1.0
            weights['spectral'] *= 1.1
            
        elif context.phase == LossPhase.REFINEMENT:
            # Фокус на качество mel и perceptual
            weights['mel'] *= 1.3
            weights['spectral'] *= 1.4
            weights['perceptual'] *= 1.5
            weights['guided_attention'] *= 0.8
            
        elif context.phase == LossPhase.CONVERGENCE:
            # Максимальное качество, минимальный guided attention
            weights['mel'] *= 1.5
            weights['perceptual'] *= 2.0
            weights['style'] *= 1.5
            weights['guided_attention'] *= 0.4
            
        return weights
    
    def _adapt_by_quality(self, weights: Dict[str, float], context: LossContext) -> Dict[str, float]:
        """Адаптация весов на основе качества модели"""
        
        # Реакция на плохое attention alignment
        if context.attention_quality < 0.3:
            weights['guided_attention'] *= 1.8
            weights['monotonic'] *= 1.5
            
        # Реакция на плохую gate accuracy
        if context.gate_accuracy < 0.6:
            weights['gate'] *= 1.4
            
        # Реакция на нестабильность gradients
        if context.gradient_norm > 5.0:
            # Снижаем все веса для стабилизации
            for key in weights:
                weights[key] *= 0.9
                
        return weights
    
    def _stabilize_weights(self, weights: Dict[str, float], context: LossContext) -> Dict[str, float]:
        """Стабилизация весов при обнаружении проблем"""
        
        # Если loss нестабилен, возвращаемся к базовым весам
        if context.loss_stability > self.stability_threshold:
            stabilization_factor = 0.7  # Плавный возврат
            for key in weights:
                weights[key] = (
                    stabilization_factor * self.base_weights[key] + 
                    (1 - stabilization_factor) * weights[key]
                )
                
        return weights
    
    def _apply_safety_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Применение ограничений безопасности"""
        
        for key, (min_val, max_val) in self.weight_limits.items():
            if key in weights:
                weights[key] = torch.clamp(torch.tensor(weights[key]), min_val, max_val).item()
                
        return weights
    
    def _smooth_weight_update(self, target_weights: Dict[str, float], context: LossContext):
        """Плавное обновление текущих весов"""
        
        smoothing_factor = 0.95  # Сильное сглаживание для стабильности
        
        for key in self.current_weights:
            if key in target_weights:
                self.current_weights[key] = (
                    smoothing_factor * self.current_weights[key] + 
                    (1 - smoothing_factor) * target_weights[key]
                )
        
        # Логируем значительные изменения
        if context.global_step % 100 == 0:
            self.weight_history.append({
                'step': context.global_step,
                'weights': self.current_weights.copy(),
                'phase': context.phase.value,
                'attention_quality': context.attention_quality
            })


class ContextBasedLossScaler(nn.Module):
    """
    🔄 Context-Based Loss Scaling - динамическое масштабирование loss
    
    Функциональность:
    - Автоматическое масштабирование на основе градиентов  
    - Предотвращение gradient explosion/vanishing
    - Адаптивное расписание масштабирования
    - Контекстная адаптация по фазам обучения
    """
    
    def __init__(self, 
                 initial_scale: float = 1.0,
                 target_grad_norm: float = 2.0,
                 adaptation_rate: float = 0.1):
        super().__init__()
        
        self.current_scale = initial_scale
        self.target_grad_norm = target_grad_norm
        self.adaptation_rate = adaptation_rate
        
        # История градиентов для анализа
        self.gradient_history = deque(maxlen=20)
        self.scale_history = []
        
        print(f"🔄 ContextBasedLossScaler инициализирован: scale={self.current_scale}")
    
    def get_loss_scale(self, context: LossContext) -> float:
        """
        Получение адаптивного масштаба loss
        
        Args:
            context: Контекст обучения
            
        Returns:
            float: Масштаб для применения к loss
        """
        # Добавляем текущую норму градиента в историю
        self.gradient_history.append(context.gradient_norm)
        
        # Адаптация масштаба на основе норм градиентов
        if len(self.gradient_history) >= 5:
            recent_grad_norm = np.mean(list(self.gradient_history)[-5:])
            
            # Если градиенты слишком большие - уменьшаем scale
            if recent_grad_norm > self.target_grad_norm * 1.5:
                target_scale = self.current_scale * 0.8
            # Если градиенты слишком маленькие - увеличиваем scale  
            elif recent_grad_norm < self.target_grad_norm * 0.5:
                target_scale = self.current_scale * 1.2
            else:
                target_scale = self.current_scale
            
            # Ограничиваем диапазон масштаба
            target_scale = np.clip(target_scale, 0.1, 10.0)
            
            # Плавная адаптация
            self.current_scale = (
                (1 - self.adaptation_rate) * self.current_scale + 
                self.adaptation_rate * target_scale
            )
        
        # Фазово-зависимые корректировки
        phase_multiplier = self._get_phase_multiplier(context.phase)
        final_scale = self.current_scale * phase_multiplier
        
        # Логируем изменения
        if context.global_step % 200 == 0:
            self.scale_history.append({
                'step': context.global_step,
                'scale': final_scale,
                'grad_norm': context.gradient_norm,
                'phase': context.phase.value
            })
        
        return final_scale
    
    def _get_phase_multiplier(self, phase: LossPhase) -> float:
        """Получение множителя масштаба для фазы обучения"""
        
        phase_multipliers = {
            LossPhase.PRE_ALIGNMENT: 1.2,        # Больше scale в начале
            LossPhase.ALIGNMENT_LEARNING: 1.0,   # Базовый scale
            LossPhase.REFINEMENT: 0.9,           # Меньше scale для точности
            LossPhase.CONVERGENCE: 0.8           # Минимальный scale для стабильности
        }
        
        return phase_multipliers.get(phase, 1.0)


class PhaseAwareLossOptimizer(nn.Module):
    """
    📈 Phase-Aware Loss Optimization - оптимизация параметров loss по фазам
    
    Координирует все компоненты адаптивной системы:
    - Dynamic Tversky Loss
    - Intelligent Weight Manager  
    - Context-Based Loss Scaler
    - Мониторинг и диагностика
    """
    
    def __init__(self, hparams):
        super().__init__()
        
        self.hparams = hparams
        
        # Инициализация компонентов системы
        self.tversky_loss = DynamicTverskyLoss()
        self.weight_manager = IntelligentWeightManager(hparams)
        self.loss_scaler = ContextBasedLossScaler()
        
        # Мониторинг и диагностика
        self.optimization_history = []
        self.performance_metrics = deque(maxlen=100)
        
        print("📈 PhaseAwareLossOptimizer инициализирован - интеллектуальная система активна!")
    
    def optimize_loss_computation(self, 
                                  loss_components: Dict[str, torch.Tensor],
                                  context: LossContext) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Основная функция оптимизации loss с адаптивными компонентами
        
        Args:
            loss_components: Словарь компонентов loss {'mel': tensor, 'gate': tensor, ...}
            context: Контекст текущего состояния обучения
            
        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: (оптимизированная loss, диагностика)
        """
        # 1. Получаем адаптивные веса
        adaptive_weights = self.weight_manager.get_adaptive_weights(context)
        
        # 2. Получаем адаптивный масштаб
        loss_scale = self.loss_scaler.get_loss_scale(context)
        
        # 3. Применяем Dynamic Tversky к gate loss если доступна
        if 'gate' in loss_components and 'gate_predictions' in loss_components:
            tversky_gate_loss = self.tversky_loss(
                loss_components['gate_predictions'],
                loss_components['gate_targets'], 
                context
            )
            loss_components['gate'] = tversky_gate_loss
        
        # 4. Вычисляем итоговую оптимизированную loss
        total_loss = torch.tensor(0.0, device=next(iter(loss_components.values())).device)
        
        component_contributions = {}
        for component, loss_value in loss_components.items():
            if component in adaptive_weights and loss_value is not None:
                weighted_loss = adaptive_weights[component] * loss_value
                total_loss += weighted_loss
                component_contributions[component] = weighted_loss.item()
        
        # 5. Применяем адаптивное масштабирование
        scaled_loss = total_loss * loss_scale
        
        # 6. Собираем диагностическую информацию
        diagnostics = {
            'adaptive_weights': adaptive_weights,
            'loss_scale': loss_scale,
            'component_contributions': component_contributions,
            'total_loss_unscaled': total_loss.item(),
            'total_loss_scaled': scaled_loss.item(),
            'tversky_params': {
                'alpha': self.tversky_loss.alpha,
                'beta': self.tversky_loss.beta
            },
            'optimization_phase': context.phase.value
        }
        
        # 7. Обновляем историю производительности
        self._update_performance_metrics(context, scaled_loss.item(), diagnostics)
        
        return scaled_loss, diagnostics
    
    def _update_performance_metrics(self, 
                                    context: LossContext, 
                                    final_loss: float, 
                                    diagnostics: Dict[str, Any]):
        """Обновление метрик производительности для анализа"""
        
        metric = {
            'step': context.global_step,
            'phase': context.phase.value,
            'final_loss': final_loss,
            'attention_quality': context.attention_quality,
            'gate_accuracy': context.gate_accuracy,
            'gradient_norm': context.gradient_norm,
            'loss_stability': context.loss_stability,
            'adaptive_weights': diagnostics['adaptive_weights'].copy(),
            'loss_scale': diagnostics['loss_scale']
        }
        
        self.performance_metrics.append(metric)
        
        # Периодически сохраняем в историю оптимизации
        if context.global_step % 500 == 0:
            self.optimization_history.append({
                'step': context.global_step,
                'recent_performance': list(self.performance_metrics)[-10:],
                'summary': self._compute_performance_summary()
            })
    
    def _compute_performance_summary(self) -> Dict[str, float]:
        """Вычисление сводки производительности системы"""
        
        if not self.performance_metrics:
            return {}
        
        recent_metrics = list(self.performance_metrics)[-20:]  # Последние 20 шагов
        
        return {
            'avg_loss': np.mean([m['final_loss'] for m in recent_metrics]),
            'loss_std': np.std([m['final_loss'] for m in recent_metrics]),
            'avg_attention_quality': np.mean([m['attention_quality'] for m in recent_metrics]),
            'avg_gate_accuracy': np.mean([m['gate_accuracy'] for m in recent_metrics]),
            'avg_gradient_norm': np.mean([m['gradient_norm'] for m in recent_metrics]),
            'system_stability': 1.0 / (1.0 + np.std([m['final_loss'] for m in recent_metrics]))
        }
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Получение полной диагностики системы"""
        
        return {
            'tversky_loss': {
                'current_alpha': self.tversky_loss.alpha,
                'current_beta': self.tversky_loss.beta,
                'adaptation_history': self.tversky_loss.adaptation_history[-10:],
                'accuracy_trend': list(self.tversky_loss.accuracy_history)[-10:]
            },
            'weight_manager': {
                'current_weights': self.weight_manager.current_weights,
                'base_weights': self.weight_manager.base_weights,
                'weight_history': self.weight_manager.weight_history[-5:],
                'safety_limits': self.weight_manager.weight_limits
            },
            'loss_scaler': {
                'current_scale': self.loss_scaler.current_scale,
                'target_grad_norm': self.loss_scaler.target_grad_norm,
                'gradient_history': list(self.loss_scaler.gradient_history)[-10:],
                'scale_history': self.loss_scaler.scale_history[-5:]
            },
            'performance_summary': self._compute_performance_summary(),
            'optimization_history_length': len(self.optimization_history)
        }


def create_adaptive_loss_system(hparams) -> PhaseAwareLossOptimizer:
    """
    Фабричная функция для создания адаптивной системы loss функций
    
    Args:
        hparams: Гиперпараметры модели
        
    Returns:
        PhaseAwareLossOptimizer: Настроенная система адаптивных loss функций
    """
    print("🎯 Создание Enhanced Adaptive Loss System...")
    
    system = PhaseAwareLossOptimizer(hparams)
    
    print("✅ Enhanced Adaptive Loss System успешно создана!")
    print("🔧 Компоненты: Dynamic Tversky Loss, Intelligent Weight Manager, Context-Based Loss Scaler")
    
    return system


# Утилиты для интеграции с существующей системой

def convert_training_phase_to_loss_phase(training_phase: str) -> LossPhase:
    """Конвертация фазы из Context-Aware Manager в LossPhase"""
    
    phase_mapping = {
        'PRE_ALIGNMENT': LossPhase.PRE_ALIGNMENT,
        'ALIGNMENT_LEARNING': LossPhase.ALIGNMENT_LEARNING,
        'REFINEMENT': LossPhase.REFINEMENT,
        'CONVERGENCE': LossPhase.CONVERGENCE
    }
    
    return phase_mapping.get(training_phase, LossPhase.ALIGNMENT_LEARNING)


def create_loss_context_from_metrics(training_metrics: Dict[str, Any], 
                                     current_phase: str,
                                     global_step: int) -> LossContext:
    """Создание LossContext из метрик обучения"""
    
    return LossContext(
        phase=convert_training_phase_to_loss_phase(current_phase),
        global_step=global_step,
        attention_quality=training_metrics.get('attention_quality', 0.5),
        gate_accuracy=training_metrics.get('gate_accuracy', 0.5),
        mel_consistency=training_metrics.get('mel_consistency', 0.5),
        gradient_norm=training_metrics.get('gradient_norm', 1.0),
        loss_stability=training_metrics.get('loss_stability', 1.0),
        learning_rate=training_metrics.get('learning_rate', 1e-3)
    )


if __name__ == "__main__":
    # Тестирование системы
    print("🧪 Тестирование Enhanced Adaptive Loss System...")
    
    class MockHParams:
        mel_loss_weight = 1.0
        gate_loss_weight = 1.0
        guide_loss_weight = 2.0
        spectral_loss_weight = 0.3
        perceptual_loss_weight = 0.2
        style_loss_weight = 0.1
        monotonic_loss_weight = 0.1
    
    hparams = MockHParams()
    system = create_adaptive_loss_system(hparams)
    
    # Тестовый контекст
    context = LossContext(
        phase=LossPhase.ALIGNMENT_LEARNING,
        global_step=1000,
        attention_quality=0.4,
        gate_accuracy=0.7,
        mel_consistency=0.6,
        gradient_norm=3.2,
        loss_stability=1.8,
        learning_rate=1e-3
    )
    
    # Тестовые loss компоненты
    loss_components = {
        'mel': torch.tensor(2.5),
        'gate': torch.tensor(0.8),
        'guided_attention': torch.tensor(1.2),
        'spectral': torch.tensor(0.4)
    }
    
    # Тест оптимизации
    optimized_loss, diagnostics = system.optimize_loss_computation(loss_components, context)
    
    print(f"✅ Тест завершен успешно!")
    print(f"   Оптимизированная loss: {optimized_loss.item():.4f}")
    print(f"   Адаптивные веса: {diagnostics['adaptive_weights']}")
    print(f"   Loss scale: {diagnostics['loss_scale']:.3f}") 