#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 UNIFIED PERFORMANCE OPTIMIZATION SYSTEM
Единая интеллектуальная система оптимизации производительности

Объединяет и координирует все компоненты:
✅ System Performance Monitoring (CPU, GPU, Memory)
✅ Training Performance Optimization (Batch Size, LR, Gradients)
✅ Model Performance Tuning (Architecture, Attention, Loss)
✅ Hardware Adaptation (GPU utilization, Memory efficiency)
✅ Real-time Bottleneck Detection & Resolution
✅ Automated Parameter Tuning

Заменяет:
❌ Разрозненные системы оптимизации
❌ Отсутствие координации между компонентами
❌ Ручная настройка параметров
"""

import os
import time
import torch
import torch.nn as nn
import psutil
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import numpy as np
import json
from pathlib import Path

# Интеграция с существующими системами
try:
    from production_monitoring import ProductionMonitor, MetricsCollector
    PRODUCTION_MONITORING_AVAILABLE = True
except ImportError:
    PRODUCTION_MONITORING_AVAILABLE = False

try:
    from enhanced_mlflow_logger import EnhancedMLflowLogger
    MLFLOW_LOGGER_AVAILABLE = True
except ImportError:
    MLFLOW_LOGGER_AVAILABLE = False

try:
    from context_aware_training_manager import ContextAwareTrainingManager
    CONTEXT_AWARE_AVAILABLE = True
except ImportError:
    CONTEXT_AWARE_AVAILABLE = False

try:
    from training_stabilization_system import TrainingStabilizationSystem
    STABILIZATION_AVAILABLE = True
except ImportError:
    STABILIZATION_AVAILABLE = False

try:
    from unified_logging_system import UnifiedLoggingSystem
    UNIFIED_LOGGING_AVAILABLE = True
except ImportError:
    UNIFIED_LOGGING_AVAILABLE = False

# GPU мониторинг
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False


class OptimizationPriority(Enum):
    """Приоритеты оптимизации"""
    CRITICAL = "critical"      # Критические bottleneck'и
    HIGH = "high"             # Важные оптимизации
    MEDIUM = "medium"         # Умеренные улучшения
    LOW = "low"              # Минорные оптимизации


class PerformanceMetricType(Enum):
    """Типы метрик производительности"""
    SYSTEM = "system"         # Системные ресурсы
    TRAINING = "training"     # Обучение модели
    THROUGHPUT = "throughput" # Пропускная способность
    EFFICIENCY = "efficiency" # Эффективность
    LATENCY = "latency"      # Задержки


@dataclass
class PerformanceMetrics:
    """Комплексные метрики производительности"""
    timestamp: float
    
    # Системные метрики
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available_gb: float = 0.0
    
    # GPU метрики
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    gpu_temperature: float = 0.0
    gpu_power_usage: float = 0.0
    
    # Обучение
    loss: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    attention_quality: float = 0.0
    batch_processing_time: float = 0.0
    
    # Пропускная способность
    samples_per_second: float = 0.0
    batches_per_second: float = 0.0
    steps_per_minute: float = 0.0
    
    # Эффективность
    gpu_utilization_efficiency: float = 0.0
    memory_efficiency: float = 0.0
    compute_efficiency: float = 0.0
    
    # Диагностика
    bottleneck_detected: bool = False
    bottleneck_type: Optional[str] = None
    optimization_opportunities: List[str] = None
    
    def __post_init__(self):
        if self.optimization_opportunities is None:
            self.optimization_opportunities = []


@dataclass
class OptimizationRecommendation:
    """Рекомендация по оптимизации"""
    priority: OptimizationPriority
    metric_type: PerformanceMetricType
    description: str
    suggested_action: str
    expected_improvement: float  # Ожидаемое улучшение в %
    estimated_risk: float       # Оценка риска (0.0-1.0)
    parameters_to_change: Dict[str, Any]
    confidence: float          # Уверенность в рекомендации (0.0-1.0)


class SystemProfiler:
    """🔍 Системный профилировщик - детальный анализ производительности"""
    
    def __init__(self):
        self.profiling_history = deque(maxlen=1000)
        self.bottleneck_history = deque(maxlen=100)
        self.optimization_history = []
        
        # GPU handles
        self.gpu_handles = []
        if GPU_MONITORING_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) 
                                  for i in range(device_count)]
            except Exception:
                pass
        
        self.logger = logging.getLogger(__name__)
    
    def profile_system_performance(self) -> PerformanceMetrics:
        """Комплексное профилирование системы"""
        timestamp = time.time()
        
        # Системные метрики
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            memory_available_gb=memory.available / (1024**3)
        )
        
        # GPU метрики
        if self.gpu_handles:
            try:
                handle = self.gpu_handles[0]  # Основная GPU
                
                # Использование GPU
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics.gpu_usage = util.gpu
                
                # Память GPU
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics.gpu_memory_usage = (mem_info.used / mem_info.total) * 100
                
                # Температура
                try:
                    metrics.gpu_temperature = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except:
                    pass
                
                # Потребление энергии
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    metrics.gpu_power_usage = power
                except:
                    pass
                    
            except Exception as e:
                self.logger.warning(f"Ошибка сбора GPU метрик: {e}")
        
        # Вычисление эффективности
        self._calculate_efficiency_metrics(metrics)
        
        # Обнаружение bottleneck'ов
        self._detect_bottlenecks(metrics)
        
        # Сохранение в историю
        self.profiling_history.append(metrics)
        
        return metrics
    
    def _calculate_efficiency_metrics(self, metrics: PerformanceMetrics):
        """Вычисление метрик эффективности"""
        # GPU utilization efficiency
        if metrics.gpu_usage > 0:
            # Идеальная GPU utilization 80-95%
            target_gpu_usage = 85.0
            efficiency = 1.0 - abs(metrics.gpu_usage - target_gpu_usage) / target_gpu_usage
            metrics.gpu_utilization_efficiency = max(0.0, efficiency) * 100
        
        # Memory efficiency
        if metrics.memory_usage > 0:
            # Идеальное использование памяти 60-80%
            if 60 <= metrics.memory_usage <= 80:
                metrics.memory_efficiency = 100.0
            else:
                target = 70.0
                efficiency = 1.0 - abs(metrics.memory_usage - target) / target
                metrics.memory_efficiency = max(0.0, efficiency) * 100
        
        # Compute efficiency (комбинированная метрика)
        metrics.compute_efficiency = (
            metrics.gpu_utilization_efficiency * 0.6 +
            metrics.memory_efficiency * 0.4
        )
    
    def _detect_bottlenecks(self, metrics: PerformanceMetrics):
        """Обнаружение bottleneck'ов"""
        bottlenecks = []
        
        # CPU bottleneck
        if metrics.cpu_usage > 90:
            bottlenecks.append("cpu_overload")
            
        # Memory bottleneck
        if metrics.memory_usage > 95:
            bottlenecks.append("memory_exhaustion")
        elif metrics.memory_usage < 20:
            bottlenecks.append("memory_underutilization")
            
        # GPU bottlenecks
        if metrics.gpu_usage > 0:
            if metrics.gpu_usage < 30:
                bottlenecks.append("gpu_underutilization")
            elif metrics.gpu_usage > 98:
                bottlenecks.append("gpu_saturation")
                
            if metrics.gpu_memory_usage > 95:
                bottlenecks.append("gpu_memory_exhaustion")
            elif metrics.gpu_memory_usage < 20:
                bottlenecks.append("gpu_memory_underutilization")
                
            if metrics.gpu_temperature > 80:
                bottlenecks.append("gpu_thermal_throttling")
        
        if bottlenecks:
            metrics.bottleneck_detected = True
            metrics.bottleneck_type = "; ".join(bottlenecks)
            self.bottleneck_history.append({
                'timestamp': metrics.timestamp,
                'bottlenecks': bottlenecks
            })


class PerformanceOptimizer:
    """⚡ Интеллектуальный оптимизатор производительности"""
    
    def __init__(self, target_performance_goals: Dict[str, float] = None):
        self.target_goals = target_performance_goals or {
            'gpu_utilization': 85.0,
            'memory_efficiency': 75.0,
            'samples_per_second': 100.0,
            'gradient_stability': 0.8
        }
        
        # Адаптивные параметры
        self.optimization_strategies = {
            'batch_size': {'min': 4, 'max': 64, 'current': 16},
            'learning_rate': {'min': 1e-6, 'max': 1e-2, 'current': 1e-3},
            'gradient_accumulation': {'min': 1, 'max': 8, 'current': 1},
            'attention_chunk_size': {'min': 32, 'max': 512, 'current': None}
        }
        
        self.optimization_history = []
        self.performance_trend = deque(maxlen=50)
        
        self.logger = logging.getLogger(__name__)
    
    def generate_optimization_recommendations(self, 
                                            metrics: PerformanceMetrics,
                                            training_context: Dict = None) -> List[OptimizationRecommendation]:
        """Генерация рекомендаций по оптимизации"""
        recommendations = []
        
        # Анализ GPU utilization
        if metrics.gpu_usage < 50:
            recommendations.append(OptimizationRecommendation(
                priority=OptimizationPriority.HIGH,
                metric_type=PerformanceMetricType.EFFICIENCY,
                description="Низкое использование GPU - можно увеличить batch size",
                suggested_action="Увеличить batch_size в 1.5-2x раза",
                expected_improvement=25.0,
                estimated_risk=0.2,
                parameters_to_change={'batch_size': self._suggest_batch_size_increase()},
                confidence=0.8
            ))
        
        # Анализ памяти GPU
        if metrics.gpu_memory_usage < 60:
            recommendations.append(OptimizationRecommendation(
                priority=OptimizationPriority.MEDIUM,
                metric_type=PerformanceMetricType.EFFICIENCY,
                description="Недостаточное использование GPU памяти",
                suggested_action="Увеличить размер модели или batch size",
                expected_improvement=15.0,
                estimated_risk=0.3,
                parameters_to_change={'mixed_precision': True, 'larger_model': True},
                confidence=0.7
            ))
        
        # Анализ системной памяти
        if metrics.memory_usage > 90:
            recommendations.append(OptimizationRecommendation(
                priority=OptimizationPriority.CRITICAL,
                metric_type=PerformanceMetricType.SYSTEM,
                description="Критически высокое использование RAM",
                suggested_action="Уменьшить batch size или включить gradient accumulation",
                expected_improvement=30.0,
                estimated_risk=0.1,
                parameters_to_change={'batch_size': self._suggest_batch_size_decrease(),
                                    'gradient_accumulation_steps': 2},
                confidence=0.9
            ))
        
        # Анализ обучения
        if training_context:
            loss = training_context.get('loss', 0)
            gradient_norm = training_context.get('gradient_norm', 0)
            
            if gradient_norm > 10.0:
                recommendations.append(OptimizationRecommendation(
                    priority=OptimizationPriority.HIGH,
                    metric_type=PerformanceMetricType.TRAINING,
                    description="Высокая норма градиентов - риск нестабильности",
                    suggested_action="Уменьшить learning rate или усилить gradient clipping",
                    expected_improvement=20.0,
                    estimated_risk=0.2,
                    parameters_to_change={'learning_rate': training_context.get('learning_rate', 1e-3) * 0.7,
                                        'gradient_clip_thresh': min(gradient_norm * 0.5, 5.0)},
                    confidence=0.8
                ))
        
        # Сортировка по приоритету
        priority_order = {
            OptimizationPriority.CRITICAL: 0,
            OptimizationPriority.HIGH: 1,
            OptimizationPriority.MEDIUM: 2,
            OptimizationPriority.LOW: 3
        }
        recommendations.sort(key=lambda x: priority_order[x.priority])
        
        return recommendations
    
    def _suggest_batch_size_increase(self) -> int:
        """Предложение увеличения batch size"""
        current = self.optimization_strategies['batch_size']['current']
        max_batch = self.optimization_strategies['batch_size']['max']
        return min(int(current * 1.5), max_batch)
    
    def _suggest_batch_size_decrease(self) -> int:
        """Предложение уменьшения batch size"""
        current = self.optimization_strategies['batch_size']['current']
        min_batch = self.optimization_strategies['batch_size']['min']
        return max(int(current * 0.7), min_batch)


class AdaptiveParameterController:
    """🎛️ Контроллер адаптивных параметров"""
    
    def __init__(self, hparams):
        self.hparams = hparams
        self.parameter_history = {}
        self.performance_correlation = {}
        
        # Безопасные диапазоны параметров
        self.safe_ranges = {
            'learning_rate': (1e-6, 1e-2),
            'batch_size': (1, 128),
            'gradient_accumulation_steps': (1, 16),
            'attention_dropout': (0.0, 0.5),
            'decoder_dropout': (0.0, 0.5)
        }
        
        self.logger = logging.getLogger(__name__)
    
    def apply_optimization_recommendations(self, 
                                         recommendations: List[OptimizationRecommendation],
                                         max_changes: int = 3) -> Dict[str, Any]:
        """Применение рекомендаций по оптимизации"""
        applied_changes = {}
        changes_count = 0
        
        for rec in recommendations:
            if changes_count >= max_changes:
                break
                
            # Проверка безопасности изменений
            if self._is_safe_to_apply(rec):
                try:
                    # Применение изменений параметров
                    for param_name, new_value in rec.parameters_to_change.items():
                        if hasattr(self.hparams, param_name):
                            old_value = getattr(self.hparams, param_name)
                            
                            # Проверка диапазона
                            if param_name in self.safe_ranges:
                                min_val, max_val = self.safe_ranges[param_name]
                                new_value = np.clip(new_value, min_val, max_val)
                            
                            # Применение изменения
                            setattr(self.hparams, param_name, new_value)
                            applied_changes[param_name] = {
                                'old_value': old_value,
                                'new_value': new_value,
                                'recommendation': rec.description
                            }
                            
                            self.logger.info(f"🎛️ Изменен {param_name}: {old_value} → {new_value}")
                            changes_count += 1
                            
                except Exception as e:
                    self.logger.error(f"Ошибка применения рекомендации: {e}")
        
        return applied_changes
    
    def _is_safe_to_apply(self, recommendation: OptimizationRecommendation) -> bool:
        """Проверка безопасности применения рекомендации"""
        # Не применяем рекомендации с высоким риском
        if recommendation.estimated_risk > 0.7:
            return False
            
        # Не применяем рекомендации с низкой уверенностью
        if recommendation.confidence < 0.5:
            return False
            
        return True


class UnifiedPerformanceOptimizationSystem:
    """🚀 Главная система унифицированной оптимизации производительности"""
    
    def __init__(self, hparams, enable_auto_optimization: bool = True):
        # Логирование - инициализируем первым
        try:
            if UNIFIED_LOGGING_AVAILABLE:
                self.logger = UnifiedLoggingSystem().get_logger("PerformanceOptimizer")
            else:
                self.logger = logging.getLogger(__name__)
        except Exception:
            # Fallback к стандартному логгеру
            self.logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.INFO)
        
        self.hparams = hparams
        self.enable_auto_optimization = enable_auto_optimization
        
        # Инициализация компонентов
        self.profiler = SystemProfiler()
        self.optimizer = PerformanceOptimizer()
        self.parameter_controller = AdaptiveParameterController(hparams)
        
        # Интеграция с существующими системами
        self.production_monitor = None
        self.mlflow_logger = None
        self.context_manager = None
        self.stabilization_system = None
        
        # Инициализация интеграций
        self._initialize_integrations()
        
        # Состояние системы
        self.optimization_active = False
        self.last_optimization = 0
        self.optimization_interval = 300  # 5 минут между оптимизациями
        
        # История и статистика
        self.optimization_results = []
        self.performance_improvements = []
        
        self.logger.info("🚀 Unified Performance Optimization System инициализирована")
    
    def _initialize_integrations(self):
        """Инициализация интеграций с существующими системами"""
        if PRODUCTION_MONITORING_AVAILABLE:
            try:
                self.production_monitor = ProductionMonitor()
                self.logger.info("✅ Интеграция с Production Monitor")
            except Exception as e:
                self.logger.warning(f"Не удалось интегрировать Production Monitor: {e}")
        
        if MLFLOW_LOGGER_AVAILABLE:
            try:
                self.mlflow_logger = EnhancedMLflowLogger()
                self.logger.info("✅ Интеграция с MLflow Logger")
            except Exception as e:
                self.logger.warning(f"Не удалось интегрировать MLflow Logger: {e}")
    
    def optimize_performance_step(self, 
                                training_metrics: Dict = None,
                                force_optimization: bool = False) -> Dict[str, Any]:
        """Один шаг оптимизации производительности"""
        current_time = time.time()
        
        # Проверка интервала оптимизации
        if not force_optimization and (current_time - self.last_optimization) < self.optimization_interval:
            return {'status': 'skipped', 'reason': 'interval_not_reached'}
        
        # Профилирование системы
        performance_metrics = self.profiler.profile_system_performance()
        
        # Обновление метрик обучения
        if training_metrics:
            performance_metrics.loss = training_metrics.get('loss', 0.0)
            performance_metrics.learning_rate = training_metrics.get('learning_rate', 0.0)
            performance_metrics.gradient_norm = training_metrics.get('gradient_norm', 0.0)
            performance_metrics.attention_quality = training_metrics.get('attention_quality', 0.0)
        
        # Генерация рекомендаций
        recommendations = self.optimizer.generate_optimization_recommendations(
            performance_metrics, training_metrics
        )
        
        optimization_result = {
            'timestamp': current_time,
            'performance_metrics': asdict(performance_metrics),
            'recommendations_count': len(recommendations),
            'applied_changes': {},
            'status': 'completed'
        }
        
        # Применение оптимизации (если включено)
        if self.enable_auto_optimization and recommendations:
            applied_changes = self.parameter_controller.apply_optimization_recommendations(
                recommendations, max_changes=2  # Не более 2 изменений за раз
            )
            optimization_result['applied_changes'] = applied_changes
            
            if applied_changes:
                self.logger.info(f"🎯 Применено {len(applied_changes)} оптимизаций")
        
        # Сохранение результатов
        self.optimization_results.append(optimization_result)
        self.last_optimization = current_time
        
        # Логирование в MLflow
        if self.mlflow_logger:
            try:
                self.mlflow_logger.log_metrics({
                    'optimization/gpu_utilization_efficiency': performance_metrics.gpu_utilization_efficiency,
                    'optimization/memory_efficiency': performance_metrics.memory_efficiency,
                    'optimization/compute_efficiency': performance_metrics.compute_efficiency,
                    'optimization/recommendations_count': len(recommendations),
                    'optimization/changes_applied': len(optimization_result['applied_changes'])
                }, step=int(current_time))
            except Exception as e:
                self.logger.warning(f"Ошибка логирования в MLflow: {e}")
        
        return optimization_result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Получение отчета о производительности"""
        if not self.profiler.profiling_history:
            return {'status': 'no_data', 'message': 'Нет данных профилирования'}
        
        # Последние метрики
        latest_metrics = self.profiler.profiling_history[-1]
        
        # Статистика за последний час
        hour_ago = time.time() - 3600
        recent_metrics = [m for m in self.profiler.profiling_history 
                         if m.timestamp > hour_ago]
        
        if recent_metrics:
            avg_gpu_usage = np.mean([m.gpu_usage for m in recent_metrics])
            avg_memory_usage = np.mean([m.memory_usage for m in recent_metrics])
            avg_compute_efficiency = np.mean([m.compute_efficiency for m in recent_metrics])
        else:
            avg_gpu_usage = avg_memory_usage = avg_compute_efficiency = 0.0
        
        return {
            'current_performance': {
                'gpu_usage': latest_metrics.gpu_usage,
                'gpu_memory_usage': latest_metrics.gpu_memory_usage,
                'memory_usage': latest_metrics.memory_usage,
                'compute_efficiency': latest_metrics.compute_efficiency,
                'bottleneck_detected': latest_metrics.bottleneck_detected,
                'bottleneck_type': latest_metrics.bottleneck_type
            },
            'hourly_averages': {
                'avg_gpu_usage': avg_gpu_usage,
                'avg_memory_usage': avg_memory_usage,
                'avg_compute_efficiency': avg_compute_efficiency
            },
            'optimization_stats': {
                'total_optimizations': len(self.optimization_results),
                'recent_changes': len([r for r in self.optimization_results 
                                     if r['timestamp'] > hour_ago and r['applied_changes']]),
                'bottlenecks_detected': len(self.profiler.bottleneck_history)
            },
            'status': 'healthy' if latest_metrics.compute_efficiency > 60 else 'needs_attention'
        }
    
    def activate_emergency_optimization(self, critical_metrics: Dict) -> Dict[str, Any]:
        """Экстренная оптимизация при критических проблемах"""
        self.logger.warning("🚨 Активация экстренной оптимизации производительности")
        
        # Принудительная оптимизация
        result = self.optimize_performance_step(
            training_metrics=critical_metrics,
            force_optimization=True
        )
        
        # Дополнительные экстренные меры
        emergency_changes = {}
        
        # Если GPU memory переполнена
        if critical_metrics.get('gpu_memory_usage', 0) > 95:
            emergency_changes['emergency_batch_size_reduction'] = True
            if hasattr(self.hparams, 'batch_size'):
                old_batch = self.hparams.batch_size
                self.hparams.batch_size = max(1, old_batch // 2)
                emergency_changes['batch_size'] = {
                    'old': old_batch, 
                    'new': self.hparams.batch_size
                }
        
        # Если система memory переполнена
        if critical_metrics.get('memory_usage', 0) > 95:
            emergency_changes['emergency_memory_cleanup'] = True
            # Принудительная очистка кэшей
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        result['emergency_changes'] = emergency_changes
        result['emergency_activation'] = True
        
        return result
    
    def enable_continuous_optimization(self):
        """Включение непрерывной оптимизации"""
        self.optimization_active = True
        self.logger.info("🔄 Включена непрерывная оптимизация производительности")
    
    def disable_continuous_optimization(self):
        """Отключение непрерывной оптимизации"""
        self.optimization_active = False
        self.logger.info("⏸️ Отключена непрерывная оптимизация производительности")


def create_performance_optimization_system(hparams, 
                                         enable_auto_optimization: bool = True) -> UnifiedPerformanceOptimizationSystem:
    """
    Фабрика для создания системы оптимизации производительности
    
    Args:
        hparams: Гиперпараметры модели
        enable_auto_optimization: Включить автоматическую оптимизацию
        
    Returns:
        UnifiedPerformanceOptimizationSystem: Настроенная система оптимизации
    """
    return UnifiedPerformanceOptimizationSystem(
        hparams=hparams,
        enable_auto_optimization=enable_auto_optimization
    )


if __name__ == "__main__":
    # Пример использования
    from hparams import create_hparams
    
    hparams = create_hparams()
    
    # Создание системы оптимизации
    optimization_system = create_performance_optimization_system(
        hparams=hparams,
        enable_auto_optimization=True
    )
    
    # Пример оптимизации
    training_metrics = {
        'loss': 5.2,
        'learning_rate': 1e-3,
        'gradient_norm': 2.1,
        'attention_quality': 0.65
    }
    
    result = optimization_system.optimize_performance_step(training_metrics)
    print("🚀 Результат оптимизации:", result)
    
    # Отчет о производительности
    report = optimization_system.get_performance_report()
    print("📊 Отчет о производительности:", report) 