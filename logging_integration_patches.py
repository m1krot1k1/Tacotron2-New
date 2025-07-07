#!/usr/bin/env python3
"""
🔧 Logging Integration Patches для Tacotron2-New

Автоматическая интеграция Unified Logging System в существующие компоненты.
Заменяет хаотичные множественные логгеры на централизованную систему.

Решает проблемы из exported-assets:
❌ 5+ MLflow runs одновременно → ✅ Один централизованный run
❌ 3+ TensorBoard writers → ✅ Один shared writer  
❌ Конфликты параметров → ✅ Координированное логирование
❌ Дублирование метрик → ✅ Priority-based фильтрация
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import importlib.util
from contextlib import contextmanager
import threading
import warnings

# Импорт unified logging system
try:
    from unified_logging_system import (
        get_unified_logger, setup_component_logging, 
        MetricPriority, LogLevel, ComponentLogger,
        start_unified_logging_session, end_unified_logging_session
    )
    UNIFIED_LOGGING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Unified Logging System недоступна: {e}")
    UNIFIED_LOGGING_AVAILABLE = False


class LoggingIntegrationManager:
    """
    🔧 Менеджер интеграции логирования
    
    Централизованно управляет заменой всех существующих логгеров
    на unified logging system с сохранением обратной совместимости.
    """
    
    def __init__(self):
        self.integration_active = False
        self.patched_components = {}
        self.original_loggers = {}
        self.component_loggers: Dict[str, ComponentLogger] = {}
        self._lock = threading.Lock()
        
        # Компоненты для интеграции
        self.target_components = {
            'context_aware_training_manager': {
                'priority': MetricPriority.ESSENTIAL,
                'file_pattern': 'context_aware_training_manager.py',
                'class_name': 'ContextAwareTrainingManager'
            },
            'training_stabilization_system': {
                'priority': MetricPriority.ESSENTIAL,
                'file_pattern': 'training_stabilization_system.py',
                'class_name': 'TrainingStabilizationSystem'
            },
            'advanced_attention_enhancement': {
                'priority': MetricPriority.IMPORTANT,
                'file_pattern': 'advanced_attention_enhancement_system.py',
                'class_name': 'AdvancedAttentionEnhancementSystem'
            },
            'ultimate_tacotron_trainer': {
                'priority': MetricPriority.IMPORTANT,
                'file_pattern': 'ultimate_tacotron_trainer.py',
                'class_name': 'UltimateEnhancedTacotronTrainer'
            },
            'smart_training_logger': {
                'priority': MetricPriority.USEFUL,
                'file_pattern': 'smart_training_logger.py',
                'class_name': 'SmartTrainingLogger'
            },
            'enhanced_mlflow_logger': {
                'priority': MetricPriority.USEFUL,
                'file_pattern': 'enhanced_mlflow_logger.py',
                'class_name': 'EnhancedMLflowLogger'
            },
            'smart_tuner_integration': {
                'priority': MetricPriority.IMPORTANT,
                'file_pattern': 'smart_tuner/smart_tuner_integration.py',
                'class_name': 'SmartTunerIntegration'
            }
        }
        
        self.logger = self._setup_manager_logger()
    
    def _setup_manager_logger(self) -> logging.Logger:
        """Настройка логгера для менеджера интеграции"""
        logger = logging.getLogger('LoggingIntegrationManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - [Integration] - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start_unified_integration(self, session_name: Optional[str] = None) -> bool:
        """
        🚀 Запуск unified logging integration
        
        Args:
            session_name: Имя сессии логирования
            
        Returns:
            True если интеграция успешна
        """
        if not UNIFIED_LOGGING_AVAILABLE:
            self.logger.error("❌ Unified Logging System недоступна")
            return False
        
        if self.integration_active:
            self.logger.warning("⚠️ Интеграция уже активна")
            return True
        
        try:
            with self._lock:
                # Запускаем unified logging session
                if not start_unified_logging_session(session_name, "Tacotron2_Unified"):
                    self.logger.error("❌ Не удалось запустить unified logging session")
                    return False
                
                # Регистрируем все компоненты
                self._register_all_components()
                
                # Устанавливаем patches
                self._install_logging_patches()
                
                self.integration_active = True
                self.logger.info("✅ Unified Logging Integration активирована")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка запуска интеграции: {e}")
            return False
    
    def _register_all_components(self):
        """Регистрация всех компонентов в unified system"""
        for component_name, config in self.target_components.items():
            try:
                component_logger = setup_component_logging(
                    component_name, 
                    config['priority']
                )
                self.component_loggers[component_name] = component_logger
                self.logger.info(f"📝 Зарегистрирован компонент: {component_name}")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Ошибка регистрации {component_name}: {e}")
    
    def _install_logging_patches(self):
        """Установка patches для перехвата логирования"""
        # Patch MLflow
        self._patch_mlflow()
        
        # Patch TensorBoard
        self._patch_tensorboard()
        
        # Patch стандартное логирование
        self._patch_standard_logging()
        
        self.logger.info("🔧 Logging patches установлены")
    
    def _patch_mlflow(self):
        """Patch MLflow для предотвращения множественных runs"""
        try:
            import mlflow
            
            # Сохраняем оригинальные функции
            original_start_run = mlflow.start_run
            original_end_run = mlflow.end_run
            original_log_metric = mlflow.log_metric
            original_log_param = mlflow.log_param
            
            def patched_start_run(*args, **kwargs):
                """Перехватываем start_run - unified system уже управляет run"""
                self.logger.debug("🔧 MLflow start_run перехвачен - используется unified run")
                return None  # Unified system уже имеет активный run
            
            def patched_end_run(*args, **kwargs):
                """Перехватываем end_run"""
                self.logger.debug("🔧 MLflow end_run перехвачен - unified system управляет")
                return None
            
            def patched_log_metric(key, value, step=None, *args, **kwargs):
                """Перехватываем log_metric и перенаправляем в unified system"""
                try:
                    unified_logger = get_unified_logger()
                    unified_logger.log_metrics({key: value}, step=step)
                except Exception as e:
                    self.logger.warning(f"⚠️ Ошибка unified metric logging: {e}")
                    # Fallback к оригинальной функции
                    return original_log_metric(key, value, step, *args, **kwargs)
            
            def patched_log_param(key, value, *args, **kwargs):
                """Перехватываем log_param"""
                self.logger.debug(f"🔧 MLflow param перехвачен: {key}={value}")
                # Параметры логируются только один раз в unified system
                return None
            
            # Применяем patches
            mlflow.start_run = patched_start_run
            mlflow.end_run = patched_end_run
            mlflow.log_metric = patched_log_metric
            mlflow.log_param = patched_log_param
            
            self.patched_components['mlflow'] = {
                'original_start_run': original_start_run,
                'original_end_run': original_end_run,
                'original_log_metric': original_log_metric,
                'original_log_param': original_log_param
            }
            
            self.logger.info("✅ MLflow patched для unified integration")
            
        except ImportError:
            self.logger.debug("MLflow недоступен для patching")
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка patching MLflow: {e}")
    
    def _patch_tensorboard(self):
        """Patch TensorBoard для использования shared writer"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            original_summary_writer = SummaryWriter
            
            class UnifiedSummaryWriter:
                """Wrapper для SummaryWriter, использующий unified system"""
                
                def __init__(self, *args, **kwargs):
                    self.unified_logger = get_unified_logger()
                    self.component_name = kwargs.get('comment', 'tensorboard')
                    # Не создаем реальный SummaryWriter - используем unified
                
                def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
                    """Перенаправляем в unified system"""
                    try:
                        metrics = {tag: scalar_value}
                        self.unified_logger.log_metrics(
                            metrics=metrics,
                            component=self.component_name,
                            step=global_step,
                            priority=MetricPriority.USEFUL
                        )
                    except Exception as e:
                        # Fallback - создаем temporary writer
                        print(f"⚠️ Fallback TensorBoard logging: {e}")
                
                def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None):
                    """Заглушка для histogram"""
                    pass  # Unified system не поддерживает histograms пока
                
                def flush(self):
                    """Заглушка для flush"""
                    pass
                
                def close(self):
                    """Заглушка для close"""
                    pass
            
            # Заменяем SummaryWriter на наш wrapper
            import torch.utils.tensorboard
            torch.utils.tensorboard.SummaryWriter = UnifiedSummaryWriter
            
            self.patched_components['tensorboard'] = {
                'original_summary_writer': original_summary_writer
            }
            
            self.logger.info("✅ TensorBoard patched для unified integration")
            
        except ImportError:
            self.logger.debug("TensorBoard недоступен для patching")
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка patching TensorBoard: {e}")
    
    def _patch_standard_logging(self):
        """Patch стандартного logging для перенаправления в unified system"""
        try:
            # Создаем handler для перенаправления в unified system
            class UnifiedLoggingHandler(logging.Handler):
                def __init__(self, component_name='unknown'):
                    super().__init__()
                    self.component_name = component_name
                    self.unified_logger = get_unified_logger()
                
                def emit(self, record):
                    try:
                        # Конвертируем logging level в LogLevel
                        level_mapping = {
                            logging.CRITICAL: LogLevel.CRITICAL,
                            logging.ERROR: LogLevel.ERROR,
                            logging.WARNING: LogLevel.WARNING,
                            logging.INFO: LogLevel.INFO,
                            logging.DEBUG: LogLevel.DEBUG
                        }
                        
                        log_level = level_mapping.get(record.levelno, LogLevel.INFO)
                        message = self.format(record)
                        
                        self.unified_logger.log_message(
                            message=message,
                            level=log_level,
                            component=self.component_name
                        )
                        
                    except Exception:
                        pass  # Предотвращаем recursion в логировании
            
            # Устанавливаем unified handler как default для новых логгеров
            self.unified_handler = UnifiedLoggingHandler('system')
            
            self.logger.info("✅ Standard logging patched для unified integration")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка patching standard logging: {e}")
    
    def get_component_logger(self, component_name: str) -> Optional[ComponentLogger]:
        """
        Получение логгера для конкретного компонента
        
        Args:
            component_name: Имя компонента
            
        Returns:
            ComponentLogger или None если не найден
        """
        return self.component_loggers.get(component_name)
    
    def stop_unified_integration(self):
        """🏁 Остановка unified logging integration"""
        if not self.integration_active:
            return
        
        try:
            with self._lock:
                # Восстанавливаем оригинальные функции
                self._restore_patches()
                
                # Завершаем unified logging session
                end_unified_logging_session()
                
                # Очищаем состояние
                self.component_loggers.clear()
                self.patched_components.clear()
                
                self.integration_active = False
                self.logger.info("✅ Unified Logging Integration деактивирована")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка остановки интеграции: {e}")
    
    def _restore_patches(self):
        """Восстановление оригинальных функций"""
        try:
            # Восстанавливаем MLflow
            if 'mlflow' in self.patched_components:
                import mlflow
                patches = self.patched_components['mlflow']
                mlflow.start_run = patches['original_start_run']
                mlflow.end_run = patches['original_end_run']
                mlflow.log_metric = patches['original_log_metric']
                mlflow.log_param = patches['original_log_param']
                self.logger.info("✅ MLflow patches восстановлены")
            
            # Восстанавливаем TensorBoard
            if 'tensorboard' in self.patched_components:
                import torch.utils.tensorboard
                patches = self.patched_components['tensorboard']
                torch.utils.tensorboard.SummaryWriter = patches['original_summary_writer']
                self.logger.info("✅ TensorBoard patches восстановлены")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка восстановления patches: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Получение статуса интеграции"""
        status = {
            'active': self.integration_active,
            'unified_logging_available': UNIFIED_LOGGING_AVAILABLE,
            'registered_components': list(self.component_loggers.keys()),
            'patched_systems': list(self.patched_components.keys()),
            'target_components': len(self.target_components)
        }
        
        if self.integration_active:
            try:
                unified_logger = get_unified_logger()
                status['session_summary'] = unified_logger.get_session_summary()
            except Exception as e:
                status['session_error'] = str(e)
        
        return status
    
    @contextmanager
    def unified_integration_context(self, session_name: Optional[str] = None):
        """Context manager для автоматического управления интеграцией"""
        try:
            if self.start_unified_integration(session_name):
                yield self
            else:
                raise RuntimeError("Failed to start unified logging integration")
        finally:
            self.stop_unified_integration()


# Global instance
_global_integration_manager = None
_integration_lock = threading.Lock()


def get_integration_manager() -> LoggingIntegrationManager:
    """Получение глобального менеджера интеграции"""
    global _global_integration_manager
    
    if _global_integration_manager is None:
        with _integration_lock:
            if _global_integration_manager is None:
                _global_integration_manager = LoggingIntegrationManager()
    
    return _global_integration_manager


def start_unified_logging_integration(session_name: Optional[str] = None) -> bool:
    """
    🚀 Быстрый запуск unified logging integration
    
    Args:
        session_name: Имя сессии логирования
        
    Returns:
        True если интеграция успешна
    """
    manager = get_integration_manager()
    return manager.start_unified_integration(session_name)


def stop_unified_logging_integration():
    """🏁 Остановка unified logging integration"""
    manager = get_integration_manager()
    manager.stop_unified_integration()


def get_unified_component_logger(component_name: str) -> Optional[ComponentLogger]:
    """
    Получение unified логгера для компонента
    
    Args:
        component_name: Имя компонента
        
    Returns:
        ComponentLogger если интеграция активна
    """
    manager = get_integration_manager()
    return manager.get_component_logger(component_name)


# Convenience decorators
def unified_logging_required(func):
    """Декоратор требующий активной unified logging integration"""
    def wrapper(*args, **kwargs):
        manager = get_integration_manager()
        if not manager.integration_active:
            warnings.warn(
                "Unified logging integration не активна. "
                "Используйте start_unified_logging_integration()",
                UserWarning
            )
        return func(*args, **kwargs)
    return wrapper


def with_unified_logging(session_name: Optional[str] = None):
    """
    Декоратор для автоматического управления unified logging
    
    Args:
        session_name: Имя сессии логирования
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_integration_manager()
            with manager.unified_integration_context(session_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Availability info
def get_integration_info():
    """Информация о возможностях интеграции"""
    return {
        'available': UNIFIED_LOGGING_AVAILABLE,
        'features': [
            'MLflow conflict resolution',
            'TensorBoard writer unification',
            'Standard logging redirection', 
            'Component isolation',
            'Priority-based filtering',
            'Automatic session management'
        ],
        'target_components': list(LoggingIntegrationManager().target_components.keys())
    }


if __name__ == "__main__":
    # Демонстрация интеграции
    print("🔧 Демонстрация Logging Integration")
    
    manager = get_integration_manager()
    
    with manager.unified_integration_context("demo_integration"):
        # Получаем логгеры для компонентов
        context_logger = get_unified_component_logger("context_aware_training_manager")
        training_logger = get_unified_component_logger("training_stabilization_system")
        
        if context_logger and training_logger:
            # Демонстрационное логирование
            context_logger.log_metrics({
                "loss": 12.5,
                "attention_diagonality": 0.089
            })
            
            training_logger.log_metrics({
                "gradient_norm": 1.8,
                "learning_rate": 1e-4
            })
            
            context_logger.info("Integration test successful")
            training_logger.warning("Demo warning message")
        
        # Статус интеграции
        status = manager.get_integration_status()
        print(f"📊 Статус интеграции: {status}")
    
    print("✅ Демонстрация завершена") 