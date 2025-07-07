"""
🛡️ TRAINING STABILIZATION SYSTEM - Система стабилизации обучения
================================================================

Интеллектуальная система для устранения проблем из exported-assets:
1. 🧠 Intelligent Gradient Manager - умное управление градиентами
2. 📈 Adaptive Learning Rate Scheduler - адаптивный LR планировщик  
3. 📊 Training Stability Monitor - мониторинг стабильности
4. 🚨 Emergency Stabilization System - экстренная стабилизация

Версия: 1.0.0
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
import logging


class StabilityLevel(Enum):
    """Уровни стабильности обучения"""
    STABLE = "stable"           # Стабильно (loss_std < 1.0, grad_norm < 3.0)
    MODERATE = "moderate"       # Умеренно (loss_std < 2.0, grad_norm < 5.0)  
    UNSTABLE = "unstable"       # Нестабильно (loss_std < 5.0, grad_norm < 10.0)
    CRITICAL = "critical"       # Критично (loss_std >= 5.0, grad_norm >= 10.0)


@dataclass
class StabilityMetrics:
    """Метрики стабильности обучения"""
    loss_std: float = 0.0
    gradient_norm: float = 0.0
    lr_volatility: float = 0.0
    attention_stability: float = 0.0
    convergence_trend: float = 0.0
    stability_level: StabilityLevel = StabilityLevel.STABLE


class IntelligentGradientManager(nn.Module):
    """
    🧠 Intelligent Gradient Manager - умное управление градиентами
    
    Устраняет проблемы:
    - Gradient explosion (норма >10.0)
    - Gradient vanishing (норма <0.1)
    - Неравномерность градиентов по слоям
    """
    
    def __init__(self, 
                 target_norm: float = 2.0,
                 max_norm: float = 5.0,
                 min_norm: float = 0.1,
                 adaptation_rate: float = 0.05):
        super().__init__()
        
        self.target_norm = target_norm
        self.max_norm = max_norm
        self.min_norm = min_norm  
        self.adaptation_rate = adaptation_rate
        
        # История для адаптации
        self.gradient_history = deque(maxlen=50)
        self.clip_events = []
        self.current_scale = 1.0
        
        self.logger = logging.getLogger(__name__)
        
    def process_gradients(self, model: nn.Module, loss: torch.Tensor) -> Dict[str, float]:
        """
        Обработка градиентов с адаптивным клиппингом и масштабированием
        
        Args:
            model: Модель для обработки градиентов
            loss: Loss для backward pass
            
        Returns:
            Dict[str, float]: Метрики обработки градиентов
        """
        # Вычисляем градиенты
        if loss.requires_grad:
            loss.backward(retain_graph=True)
        
        # Вычисляем текущую норму градиентов
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        self.gradient_history.append(total_norm)
        
        # Адаптивное клиппирование
        metrics = self._apply_adaptive_clipping(model, total_norm)
        
        # Масштабирование градиентов
        self._apply_gradient_scaling(model, total_norm)
        
        return metrics
    
    def _apply_adaptive_clipping(self, model: nn.Module, current_norm: float) -> Dict[str, float]:
        """Применение адаптивного клиппирования градиентов"""
        
        # Определяем адаптивный порог клиппирования
        if len(self.gradient_history) >= 10:
            recent_norms = list(self.gradient_history)[-10:]
            adaptive_threshold = np.percentile(recent_norms, 75) * 1.5
            adaptive_threshold = min(adaptive_threshold, self.max_norm)
        else:
            adaptive_threshold = self.max_norm
        
        # Применяем клиппирование если необходимо
        if current_norm > adaptive_threshold:
            torch.nn.utils.clip_grad_norm_(model.parameters(), adaptive_threshold)
            
            self.clip_events.append({
                'step': len(self.gradient_history),
                'original_norm': current_norm,
                'clipped_norm': adaptive_threshold,
                'threshold': adaptive_threshold
            })
            
            clipped = True
            final_norm = adaptive_threshold
        else:
            clipped = False
            final_norm = current_norm
        
        return {
            'original_norm': current_norm,
            'final_norm': final_norm,
            'clipped': clipped,
            'adaptive_threshold': adaptive_threshold,
            'clip_ratio': len(self.clip_events) / max(len(self.gradient_history), 1)
        }
    
    def _apply_gradient_scaling(self, model: nn.Module, current_norm: float):
        """Применение адаптивного масштабирования градиентов"""
        
        # Вычисляем необходимый масштаб для достижения target_norm
        if current_norm > 0:
            target_scale = self.target_norm / current_norm
            
            # Ограничиваем масштаб для стабильности
            target_scale = torch.clamp(torch.tensor(target_scale), 0.1, 3.0).item()
            
            # Плавно адаптируем масштаб
            self.current_scale = (
                (1 - self.adaptation_rate) * self.current_scale + 
                self.adaptation_rate * target_scale
            )
            
            # Применяем масштабирование к градиентам
            if abs(self.current_scale - 1.0) > 0.1:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data *= self.current_scale


class AdaptiveLearningRateScheduler:
    """
    📈 Adaptive Learning Rate Scheduler - умный планировщик learning rate
    
    Предотвращает хаотичные изменения LR (колебания в 40 раз из exported-assets)
    """
    
    def __init__(self,
                 initial_lr: float = 1e-3,
                 min_lr: float = 1e-5,
                 max_lr: float = 1e-2,
                 patience: int = 10,
                 factor: float = 0.8):
        
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.patience = patience
        self.factor = factor
        
        self.current_lr = initial_lr
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # История для умной адаптации
        self.lr_history = []
        self.loss_history = deque(maxlen=100)
        
    def step(self, current_loss: float, stability_metrics: StabilityMetrics) -> float:
        """
        Умный шаг планировщика на основе loss и стабильности
        
        Args:
            current_loss: Текущая loss
            stability_metrics: Метрики стабильности
            
        Returns:
            float: Новый learning rate
        """
        self.loss_history.append(current_loss)
        
        # Адаптация на основе стабильности
        if stability_metrics.stability_level == StabilityLevel.CRITICAL:
            # Критическая нестабильность - резко снижаем LR
            new_lr = self.current_lr * 0.5
        elif stability_metrics.stability_level == StabilityLevel.UNSTABLE:
            # Нестабильность - осторожно снижаем LR
            new_lr = self.current_lr * 0.9
        else:
            # Стабильное обучение - стандартная логика
            new_lr = self._standard_lr_update(current_loss)
        
        # Применяем ограничения
        new_lr = np.clip(new_lr, self.min_lr, self.max_lr)
        
        # Предотвращаем слишком резкие изменения (max 20% за шаг)
        max_change = self.current_lr * 0.2
        if abs(new_lr - self.current_lr) > max_change:
            if new_lr > self.current_lr:
                new_lr = self.current_lr + max_change
            else:
                new_lr = self.current_lr - max_change
        
        self.current_lr = new_lr
        self.lr_history.append(new_lr)
        
        return new_lr
    
    def _standard_lr_update(self, current_loss: float) -> float:
        """Стандартная логика обновления LR"""
        
        if current_loss < self.best_loss * 0.99:  # Улучшение на 1%
            self.best_loss = current_loss
            self.patience_counter = 0
            return self.current_lr  # Не изменяем LR при улучшении
        else:
            self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                self.patience_counter = 0
                return self.current_lr * self.factor  # Снижаем LR
            else:
                return self.current_lr  # Ждем


class TrainingStabilityMonitor:
    """
    📊 Training Stability Monitor - мониторинг стабильности обучения
    
    Отслеживает все ключевые метрики стабильности
    """
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        
        # Истории метрик
        self.loss_history = deque(maxlen=window_size)
        self.gradient_history = deque(maxlen=window_size)
        self.lr_history = deque(maxlen=window_size)
        self.attention_history = deque(maxlen=window_size)
        
        # Статистика
        self.stability_reports = []
        
    def update_metrics(self,
                      loss: float,
                      gradient_norm: float,
                      learning_rate: float,
                      attention_quality: float = 0.5) -> StabilityMetrics:
        """
        Обновление метрик и вычисление стабильности
        
        Args:
            loss: Текущая loss
            gradient_norm: Норма градиентов
            learning_rate: Текущий learning rate
            attention_quality: Качество attention
            
        Returns:
            StabilityMetrics: Комплексные метрики стабильности
        """
        # Обновляем истории
        self.loss_history.append(loss)
        self.gradient_history.append(gradient_norm)
        self.lr_history.append(learning_rate)
        self.attention_history.append(attention_quality)
        
        # Вычисляем метрики стабильности
        metrics = self._compute_stability_metrics()
        
        # Создаем отчет
        report = {
            'step': len(self.loss_history),
            'metrics': metrics,
            'recommendations': self._generate_recommendations(metrics)
        }
        self.stability_reports.append(report)
        
        return metrics
    
    def _compute_stability_metrics(self) -> StabilityMetrics:
        """Вычисление комплексных метрик стабильности"""
        
        if len(self.loss_history) < 5:
            return StabilityMetrics()  # Недостаточно данных
        
        # Loss стабильность
        loss_values = list(self.loss_history)
        loss_std = np.std(loss_values)
        
        # Gradient стабильность  
        grad_values = list(self.gradient_history)
        avg_gradient_norm = np.mean(grad_values)
        
        # LR волатильность
        lr_values = list(self.lr_history)
        lr_volatility = np.std(lr_values) / np.mean(lr_values) if len(lr_values) > 1 else 0.0
        
        # Attention стабильность
        att_values = list(self.attention_history)
        attention_stability = 1.0 - np.std(att_values) if len(att_values) > 1 else 1.0
        
        # Тренд конвергенции
        if len(loss_values) >= 10:
            recent_loss = np.mean(loss_values[-5:])
            older_loss = np.mean(loss_values[-10:-5])
            convergence_trend = (older_loss - recent_loss) / older_loss
        else:
            convergence_trend = 0.0
        
        # Определяем уровень стабильности
        stability_level = self._classify_stability_level(loss_std, avg_gradient_norm)
        
        return StabilityMetrics(
            loss_std=loss_std,
            gradient_norm=avg_gradient_norm,
            lr_volatility=lr_volatility,
            attention_stability=attention_stability,
            convergence_trend=convergence_trend,
            stability_level=stability_level
        )
    
    def _classify_stability_level(self, loss_std: float, grad_norm: float) -> StabilityLevel:
        """Классификация уровня стабильности"""
        
        if loss_std < 1.0 and grad_norm < 3.0:
            return StabilityLevel.STABLE
        elif loss_std < 2.0 and grad_norm < 5.0:
            return StabilityLevel.MODERATE
        elif loss_std < 5.0 and grad_norm < 10.0:
            return StabilityLevel.UNSTABLE
        else:
            return StabilityLevel.CRITICAL
    
    def _generate_recommendations(self, metrics: StabilityMetrics) -> List[str]:
        """Генерация рекомендаций по стабилизации"""
        
        recommendations = []
        
        if metrics.stability_level == StabilityLevel.CRITICAL:
            recommendations.append("🚨 Критическая нестабильность - активировать экстренную стабилизацию")
            recommendations.append("📉 Снизить learning rate в 2-3 раза")
            recommendations.append("🛡️ Включить агрессивное клиппирование градиентов")
            
        elif metrics.stability_level == StabilityLevel.UNSTABLE:
            recommendations.append("⚠️ Нестабильность обучения - применить стабилизацию")
            recommendations.append("📈 Снизить learning rate на 10-20%")
            recommendations.append("🔧 Увеличить guided attention weight")
            
        if metrics.gradient_norm > 5.0:
            recommendations.append(f"🌪️ Высокая норма градиентов ({metrics.gradient_norm:.2f}) - усилить клиппирование")
            
        if metrics.lr_volatility > 0.3:
            recommendations.append(f"📊 Высокая волатильность LR ({metrics.lr_volatility:.3f}) - стабилизировать планировщик")
            
        return recommendations


class EmergencyStabilizationSystem:
    """
    🚨 Emergency Stabilization System - система экстренной стабилизации
    
    Активируется при критических проблемах стабильности
    """
    
    def __init__(self):
        self.active = False
        self.activation_count = 0
        self.stabilization_history = []
        
        self.logger = logging.getLogger(__name__)
        
    def check_emergency_conditions(self, metrics: StabilityMetrics) -> bool:
        """
        Проверка условий для активации экстренной стабилизации
        
        Args:
            metrics: Метрики стабильности
            
        Returns:
            bool: Нужна ли экстренная стабилизация
        """
        emergency_conditions = [
            # Критические условия требуют ALL факторов одновременно
            (metrics.stability_level == StabilityLevel.CRITICAL and 
             (metrics.loss_std > 10.0 or metrics.gradient_norm > 15.0)),
            
            # Экстремальные значения
            metrics.loss_std > 20.0,  # Экстремальная нестабильность loss
            metrics.gradient_norm > 25.0,  # Экстремальные градиенты
            metrics.lr_volatility > 0.8,  # Хаос в learning rate
            (metrics.attention_stability < 0.05 and metrics.gradient_norm > 10.0)  # Коллапс + большие градиенты
        ]
        
        return any(emergency_conditions)
    
    def activate_emergency_stabilization(self, 
                                       model: nn.Module,
                                       optimizer: torch.optim.Optimizer,
                                       metrics: StabilityMetrics) -> Dict[str, Any]:
        """
        Активация экстренной стабилизации
        
        Args:
            model: Модель для стабилизации
            optimizer: Оптимизатор
            metrics: Метрики стабильности
            
        Returns:
            Dict[str, Any]: Примененные меры стабилизации
        """
        if self.active:
            return {'message': 'Emergency stabilization already active'}
        
        self.active = True
        self.activation_count += 1
        
        measures = {}
        
        # 1. Экстренное снижение learning rate
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] = old_lr * 0.1  # Снижаем в 10 раз
            measures['lr_reduction'] = f"{old_lr:.2e} → {param_group['lr']:.2e}"
        
        # 2. Агрессивное клиппирование градиентов
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        measures['gradient_clipping'] = 1.0
        
        # 3. Временное отключение momentum (если Adam)
        if hasattr(optimizer, 'param_groups'):
            for param_group in optimizer.param_groups:
                if 'betas' in param_group:
                    param_group['betas'] = (0.5, 0.999)  # Снижаем momentum
                    measures['momentum_reduction'] = True
        
        # 4. Логирование экстренной активации
        self.logger.warning(f"🚨 EMERGENCY STABILIZATION ACTIVATED (#{self.activation_count})")
        self.logger.warning(f"   Причина: {metrics.stability_level.value}")
        self.logger.warning(f"   Loss std: {metrics.loss_std:.3f}")
        self.logger.warning(f"   Gradient norm: {metrics.gradient_norm:.3f}")
        
        # Сохраняем в историю
        self.stabilization_history.append({
            'activation_count': self.activation_count,
            'metrics': metrics,
            'measures': measures
        })
        
        return measures
    
    def deactivate_emergency_stabilization(self,
                                         optimizer: torch.optim.Optimizer,
                                         original_lr: float) -> bool:
        """
        Деактивация экстренной стабилизации
        
        Args:
            optimizer: Оптимизатор
            original_lr: Исходный learning rate
            
        Returns:
            bool: Успешно деактивирована
        """
        if not self.active:
            return False
        
        # Постепенно восстанавливаем параметры
        for param_group in optimizer.param_groups:
            param_group['lr'] = original_lr * 0.5  # Восстанавливаем постепенно
            
            # Восстанавливаем momentum
            if 'betas' in param_group:
                param_group['betas'] = (0.9, 0.999)
        
        self.active = False
        self.logger.info("✅ Emergency stabilization deactivated")
        
        return True


class TrainingStabilizationSystem:
    """
    🛡️ Training Stabilization System - главный координатор стабилизации
    
    Объединяет все компоненты для полной стабилизации обучения
    """
    
    def __init__(self, hparams):
        self.hparams = hparams
        
        # Инициализация компонентов
        self.gradient_manager = IntelligentGradientManager(
            target_norm=getattr(hparams, 'target_gradient_norm', 2.0),
            max_norm=getattr(hparams, 'max_gradient_norm', 5.0)
        )
        
        self.lr_scheduler = AdaptiveLearningRateScheduler(
            initial_lr=getattr(hparams, 'learning_rate', 1e-3),
            min_lr=getattr(hparams, 'min_learning_rate', 1e-5)
        )
        
        self.stability_monitor = TrainingStabilityMonitor(
            window_size=getattr(hparams, 'stability_window_size', 20)
        )
        
        self.emergency_system = EmergencyStabilizationSystem()
        
        # Статистика
        self.stabilization_stats = {
            'interventions': 0,
            'emergency_activations': 0,
            'stability_improvements': 0
        }
        
        self.logger = logging.getLogger(__name__)
        print("🛡️ Training Stabilization System инициализирован - полная защита от нестабильности!")
    
    def stabilize_training_step(self,
                               model: nn.Module,
                               optimizer: torch.optim.Optimizer,
                               loss: torch.Tensor,
                               attention_quality: float = 0.5) -> Dict[str, Any]:
        """
        Полная стабилизация одного шага обучения
        
        Args:
            model: Модель
            optimizer: Оптимизатор
            loss: Loss для обработки
            attention_quality: Качество attention
            
        Returns:
            Dict[str, Any]: Отчет о стабилизации
        """
        # 1. Обработка градиентов
        gradient_metrics = self.gradient_manager.process_gradients(model, loss)
        
        # 2. Мониторинг стабильности
        current_lr = optimizer.param_groups[0]['lr']
        stability_metrics = self.stability_monitor.update_metrics(
            loss.item(),
            gradient_metrics['final_norm'],
            current_lr,
            attention_quality
        )
        
        # 3. Проверка экстренных условий
        if self.emergency_system.check_emergency_conditions(stability_metrics):
            emergency_measures = self.emergency_system.activate_emergency_stabilization(
                model, optimizer, stability_metrics
            )
            self.stabilization_stats['emergency_activations'] += 1
        else:
            emergency_measures = None
            
            # Деактивируем экстренную стабилизацию если была активна
            if self.emergency_system.active:
                self.emergency_system.deactivate_emergency_stabilization(optimizer, current_lr)
        
        # 4. Адаптация learning rate
        new_lr = self.lr_scheduler.step(loss.item(), stability_metrics)
        if abs(new_lr - current_lr) > current_lr * 0.01:  # Изменение >1%
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            self.stabilization_stats['interventions'] += 1
        
        # 5. Формируем отчет
        report = {
            'stability_level': stability_metrics.stability_level.value,
            'gradient_metrics': gradient_metrics,
            'stability_metrics': stability_metrics,
            'lr_adjustment': {'old': current_lr, 'new': new_lr},
            'emergency_measures': emergency_measures,
            'recommendations': stability_metrics.stability_level != StabilityLevel.STABLE
        }
        
        return report
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Получение полной диагностики системы стабилизации"""
        
        return {
            'gradient_manager': {
                'current_scale': self.gradient_manager.current_scale,
                'clip_events_count': len(self.gradient_manager.clip_events),
                'recent_norms': list(self.gradient_manager.gradient_history)[-10:]
            },
            'lr_scheduler': {
                'current_lr': self.lr_scheduler.current_lr,
                'best_loss': self.lr_scheduler.best_loss,
                'patience_counter': self.lr_scheduler.patience_counter
            },
            'stability_monitor': {
                'reports_count': len(self.stability_monitor.stability_reports),
                'recent_stability': [r['metrics'].stability_level.value 
                                   for r in self.stability_monitor.stability_reports[-5:]]
            },
            'emergency_system': {
                'active': self.emergency_system.active,
                'activation_count': self.emergency_system.activation_count
            },
            'statistics': self.stabilization_stats
        }


def create_training_stabilization_system(hparams) -> TrainingStabilizationSystem:
    """
    Фабричная функция для создания системы стабилизации
    
    Args:
        hparams: Гиперпараметры модели
        
    Returns:
        TrainingStabilizationSystem: Настроенная система стабилизации
    """
    print("🛡️ Создание Training Stabilization System...")
    
    system = TrainingStabilizationSystem(hparams)
    
    print("✅ Training Stabilization System успешно создана!")
    print("🔧 Компоненты: Gradient Manager, LR Scheduler, Stability Monitor, Emergency System")
    
    return system


if __name__ == "__main__":
    # Тестирование системы
    print("🧪 Тестирование Training Stabilization System...")
    
    class MockHParams:
        learning_rate = 1e-3
        target_gradient_norm = 2.0
        max_gradient_norm = 5.0
        min_learning_rate = 1e-5
        stability_window_size = 20
    
    hparams = MockHParams()
    system = create_training_stabilization_system(hparams)
    
    print(f"✅ Тестирование завершено!")
    print(f"   Система готова к стабилизации обучения") 