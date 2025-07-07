#!/usr/bin/env python3
"""
🔧 Context-Aware Training Manager с Unified Logging Integration

Модификация оригинального ContextAwareTrainingManager для использования
централизованной unified logging system вместо множественных логгеров.

Решает проблемы:
❌ Множественные конфликтующие логгеры → ✅ Unified ComponentLogger
❌ Дублирование метрик → ✅ Priority-based logging
❌ Конфликты MLflow/TensorBoard → ✅ Централизованное управление
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import time

# Unified logging imports
try:
    from logging_integration_patches import get_unified_component_logger
    from unified_logging_system import MetricPriority, LogLevel
    UNIFIED_LOGGING_AVAILABLE = True
except ImportError:
    UNIFIED_LOGGING_AVAILABLE = False
    print("⚠️ Unified logging недоступна - используется fallback")

# Оригинальные импорты
try:
    from context_aware_training_manager import (
        ContextAnalyzer, EnhancedLossIntegrator, IntelligentParameterManager,
        TrainingPhase, AdaptationStrategy
    )
    ORIGINAL_COMPONENTS_AVAILABLE = True
except ImportError:
    ORIGINAL_COMPONENTS_AVAILABLE = False
    print("⚠️ Оригинальные компоненты недоступны")


class UnifiedContextAwareTrainingManager:
    """
    🧠 Context-Aware Training Manager с Unified Logging
    
    Замена оригинального ContextAwareTrainingManager с интеграцией
    unified logging system для устранения конфликтов логирования.
    """

    def __init__(self, config: dict):
        self.config = config
        
        # Unified Logging Setup
        if UNIFIED_LOGGING_AVAILABLE:
            self.logger = get_unified_component_logger("context_aware_training_manager")
            if self.logger is None:
                # Fallback к обычному логированию
                self.logger = self._setup_fallback_logger()
                self._unified_available = False
            else:
                self._unified_available = True
                self.logger.info("🔥 Context-Aware Manager инициализирован с Unified Logging")
        else:
            self.logger = self._setup_fallback_logger()
            self._unified_available = False

        # Инициализация оригинальных компонентов
        if ORIGINAL_COMPONENTS_AVAILABLE:
            self._initialize_original_components()
        else:
            self._initialize_fallback_components()
        
        # Unified logging метрики
        self.step_counter = 0
        self.session_start_time = time.time()
        
        # Метрики для unified system
        self.metrics_history = []
        
        self.logger.info("✅ Unified Context-Aware Training Manager инициализирован")

    def _setup_fallback_logger(self) -> logging.Logger:
        """Fallback логгер если unified недоступен"""
        logger = logging.getLogger("ContextAwareTrainer_Fallback")
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger

    def _initialize_original_components(self):
        """Инициализация оригинальных компонентов с unified logging"""
        try:
            self.context_analyzer = ContextAnalyzer(
                history_size=self.config.get('history_size', 100)
            )
            self.loss_controller = EnhancedLossIntegrator(
                initial_guided_weight=self.config.get('initial_guided_weight', 4.5)
            )
            self.param_manager = IntelligentParameterManager(
                initial_lr=self.config.get('initial_lr', 1e-3)
            )
            
            # Патчим логирование в компонентах
            self._patch_component_logging()
            
            self.logger.info("✅ Оригинальные компоненты инициализированы с unified logging")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации оригинальных компонентов: {e}")
            self._initialize_fallback_components()

    def _initialize_fallback_components(self):
        """Fallback компоненты если оригинальные недоступны"""
        self.logger.warning("⚠️ Используются fallback компоненты")
        
        # Минимальные заглушки
        self.context_analyzer = None
        self.loss_controller = None
        self.param_manager = None

    def _patch_component_logging(self):
        """Патчинг логирования в sub-компонентах"""
        if not self._unified_available:
            return
        
        try:
            # Патчим ContextAnalyzer
            if hasattr(self.context_analyzer, 'logger'):
                original_log = self.context_analyzer.logger
                
                class UnifiedLoggerProxy:
                    def __init__(self, unified_logger, component_name):
                        self.unified_logger = unified_logger
                        self.component_name = component_name
                    
                    def info(self, msg, *args):
                        self.unified_logger.info(f"[{self.component_name}] {msg}")
                    
                    def warning(self, msg, *args):
                        self.unified_logger.warning(f"[{self.component_name}] {msg}")
                    
                    def error(self, msg, *args):
                        self.unified_logger.error(f"[{self.component_name}] {msg}")
                    
                    def debug(self, msg, *args):
                        self.unified_logger.debug(f"[{self.component_name}] {msg}")
                
                self.context_analyzer.logger = UnifiedLoggerProxy(self.logger, "ContextAnalyzer")
                
            # Аналогично для других компонентов
            if hasattr(self.loss_controller, 'logger'):
                self.loss_controller.logger = UnifiedLoggerProxy(self.logger, "LossController")
                
            if hasattr(self.param_manager, 'logger'):
                self.param_manager.logger = UnifiedLoggerProxy(self.logger, "ParamManager")
                
            self.logger.info("🔧 Логирование sub-компонентов унифицировано")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка патчинга sub-компонентов: {e}")

    def analyze_training_step(self, metrics: Dict[str, Any], step: int) -> Dict[str, Any]:
        """
        Анализ шага обучения с unified logging
        
        Args:
            metrics: Метрики текущего шага
            step: Номер шага
            
        Returns:
            Рекомендации и адаптации
        """
        start_time = time.time()
        
        try:
            # Обновляем step counter
            self.step_counter = step
            
            # Анализ через оригинальные компоненты
            if self.context_analyzer:
                context_info = self.context_analyzer.analyze_step(metrics, step)
            else:
                context_info = self._fallback_context_analysis(metrics, step)
            
            # Управление loss
            if self.loss_controller:
                loss_adaptations = self.loss_controller.update_weights(metrics, context_info)
            else:
                loss_adaptations = self._fallback_loss_control(metrics)
            
            # Управление параметрами
            if self.param_manager:
                param_adaptations = self.param_manager.adapt_parameters(metrics, context_info)
            else:
                param_adaptations = self._fallback_param_management(metrics)
            
            # Объединяем результаты
            recommendations = {
                **context_info,
                **loss_adaptations,
                **param_adaptations,
                'processing_time': time.time() - start_time,
                'step': step,
                'timestamp': time.time()
            }
            
            # Unified logging метрик
            self._log_unified_metrics(recommendations, step)
            
            # Сохраняем в историю
            self.metrics_history.append(recommendations)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка анализа шага {step}: {e}")
            return self._create_safe_fallback_response(step)

    def _log_unified_metrics(self, recommendations: Dict[str, Any], step: int):
        """Логирование метрик через unified system"""
        if not self._unified_available:
            return
        
        try:
            # Основные метрики с высоким приоритетом
            essential_metrics = {}
            if 'loss' in recommendations:
                essential_metrics['total_loss'] = recommendations['loss']
            if 'attention_diagonality' in recommendations:
                essential_metrics['attention_diagonality'] = recommendations['attention_diagonality']
            if 'guided_attention_weight' in recommendations:
                essential_metrics['guided_attention_weight'] = recommendations['guided_attention_weight']
            
            if essential_metrics:
                self.logger.log_metrics(
                    essential_metrics,
                    step=step
                )
            
            # Дополнительные метрики с меньшим приоритетом
            additional_metrics = {}
            for key, value in recommendations.items():
                if key not in essential_metrics and isinstance(value, (int, float)):
                    additional_metrics[key] = value
            
            if additional_metrics:
                # Используем внутренний метод для логирования с другим приоритетом
                if hasattr(self.logger, 'unified_system'):
                    self.logger.unified_system.log_metrics(
                        additional_metrics,
                        component="context_aware_manager",
                        step=step,
                        priority=MetricPriority.USEFUL
                    )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка unified logging метрик: {e}")

    def _fallback_context_analysis(self, metrics: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Fallback анализ контекста"""
        return {
            'training_phase': 'unknown',
            'adaptation_strategy': 'conservative',
            'context_stability': 0.5,
            'requires_intervention': False
        }

    def _fallback_loss_control(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback управление loss"""
        current_loss = metrics.get('loss', 20.0)
        
        return {
            'guided_attention_weight': max(1.0, min(8.0, 4.5)),
            'loss_scaling_factor': 1.0,
            'loss_trend': 'stable'
        }

    def _fallback_param_management(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback управление параметрами"""
        return {
            'learning_rate_multiplier': 1.0,
            'gradient_clip_threshold': 1.0,
            'parameter_adaptation': 'none'
        }

    def _create_safe_fallback_response(self, step: int) -> Dict[str, Any]:
        """Безопасный fallback ответ при ошибках"""
        return {
            'step': step,
            'training_phase': 'error_recovery',
            'adaptation_strategy': 'conservative',
            'guided_attention_weight': 4.5,
            'learning_rate_multiplier': 1.0,
            'gradient_clip_threshold': 1.0,
            'requires_intervention': False,
            'error_mode': True,
            'recommendations': ['Использовать консервативные настройки']
        }

    def get_session_summary(self) -> Dict[str, Any]:
        """Получение сводки сессии"""
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        
        # Статистика метрик
        total_steps = len(self.metrics_history)
        
        if total_steps > 0:
            recent_metrics = self.metrics_history[-10:]  # Последние 10 шагов
            
            avg_loss = np.mean([m.get('loss', 0) for m in recent_metrics if 'loss' in m])
            avg_attention = np.mean([m.get('attention_diagonality', 0) for m in recent_metrics if 'attention_diagonality' in m])
        else:
            avg_loss = 0
            avg_attention = 0
        
        summary = {
            'session_duration_minutes': session_duration / 60,
            'total_steps_processed': total_steps,
            'unified_logging_enabled': self._unified_available,
            'original_components_available': ORIGINAL_COMPONENTS_AVAILABLE,
            'average_recent_loss': float(avg_loss),
            'average_recent_attention': float(avg_attention),
            'last_step': self.step_counter,
            'components_status': {
                'context_analyzer': self.context_analyzer is not None,
                'loss_controller': self.loss_controller is not None,
                'param_manager': self.param_manager is not None
            }
        }
        
        return summary

    def finalize_session(self):
        """Завершение сессии с unified logging"""
        try:
            # Финальная сводка
            summary = self.get_session_summary()
            
            if self._unified_available:
                self.logger.info(f"📊 Завершение Context-Aware Manager сессии")
                self.logger.log_metrics(summary, step=self.step_counter)
            
            # Очистка ресурсов
            self.metrics_history.clear()
            
            self.logger.info("🏁 Context-Aware Training Manager сессия завершена")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка завершения сессии: {e}")


# Обратная совместимость
ContextAwareTrainingManager = UnifiedContextAwareTrainingManager


# Convenience функции
def create_unified_context_manager(config: Dict[str, Any]) -> UnifiedContextAwareTrainingManager:
    """
    Создание unified context manager с автоматической настройкой
    
    Args:
        config: Конфигурация manager'а
        
    Returns:
        Настроенный UnifiedContextAwareTrainingManager
    """
    return UnifiedContextAwareTrainingManager(config)


def patch_existing_context_manager(existing_manager):
    """
    Патчинг существующего ContextAwareTrainingManager для unified logging
    
    Args:
        existing_manager: Существующий manager для патчинга
    """
    if not UNIFIED_LOGGING_AVAILABLE:
        print("⚠️ Unified logging недоступна для патчинга")
        return existing_manager
    
    try:
        # Получаем unified logger
        unified_logger = get_unified_component_logger("context_aware_training_manager")
        if unified_logger is None:
            print("⚠️ Не удалось получить unified logger")
            return existing_manager
        
        # Заменяем логгер
        original_logger = existing_manager.logger
        existing_manager.logger = unified_logger
        
        # Добавляем unified методы
        existing_manager._unified_available = True
        existing_manager._log_unified_metrics = UnifiedContextAwareTrainingManager._log_unified_metrics.__get__(existing_manager)
        
        unified_logger.info("🔧 Существующий Context Manager patched для unified logging")
        
        return existing_manager
        
    except Exception as e:
        print(f"❌ Ошибка патчинга existing manager: {e}")
        return existing_manager


if __name__ == "__main__":
    # Демонстрация unified context manager
    print("🧠 Демонстрация Unified Context-Aware Training Manager")
    
    # Пример конфигурации
    config = {
        'history_size': 100,
        'initial_guided_weight': 4.5,
        'initial_lr': 1e-3
    }
    
    # Создание manager'а
    manager = create_unified_context_manager(config)
    
    # Демонстрационные метрики
    test_metrics = {
        'loss': 15.5,
        'mel_loss': 12.0,
        'gate_loss': 0.8,
        'attention_diagonality': 0.045,
        'guided_attention_weight': 8.0,
        'learning_rate': 1e-4,
        'gradient_norm': 2.3
    }
    
    # Анализ нескольких шагов
    for step in range(1, 6):
        # Варьируем метрики
        varied_metrics = test_metrics.copy()
        varied_metrics['loss'] *= (1.0 - step * 0.05)  # Улучшение loss
        varied_metrics['attention_diagonality'] *= (1.0 + step * 0.1)  # Улучшение attention
        
        recommendations = manager.analyze_training_step(varied_metrics, step)
        print(f"Step {step}: {recommendations.get('adaptation_strategy', 'unknown')}")
    
    # Финальная сводка
    summary = manager.get_session_summary()
    print(f"📊 Сводка сессии: {summary}")
    
    # Завершение
    manager.finalize_session()
    print("✅ Демонстрация завершена") 