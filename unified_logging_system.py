#!/usr/bin/env python3
"""
🔥 Unified Logging System для Tacotron2-New

Решает критические проблемы логирования из exported-assets:
❌ Конфликты MLflow runs между компонентами
❌ Дублирование TensorBoard writers
❌ Несогласованные форматы логирования
❌ Отсутствие приоритизации метрик
❌ Множественные логгеры без координации

✅ Единая точка управления всем логированием
✅ Централизованные MLflow и TensorBoard
✅ Priority-based метрики с умной фильтрацией  
✅ Component isolation с namespace'ами
✅ Graceful fallback при отсутствии сервисов
✅ Thread-safe singleton pattern
"""

import logging
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import time
import json
import yaml
from enum import Enum, auto
from contextlib import contextmanager
import atexit

# Optional imports с graceful fallback
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LogLevel(Enum):
    """Уровни важности логирования с приоритетами"""
    CRITICAL = auto()    # Критические ошибки, сбои системы
    ERROR = auto()       # Ошибки, требующие внимания
    WARNING = auto()     # Предупреждения
    INFO = auto()        # Общая информация  
    DEBUG = auto()       # Детальная отладочная информация
    METRICS = auto()     # Метрики обучения
    SYSTEM = auto()      # Системные метрики (CPU, память)


class MetricPriority(Enum):
    """Приоритеты метрик для фильтрации"""
    ESSENTIAL = auto()   # Критически важные метрики (loss, attention quality)
    IMPORTANT = auto()   # Важные метрики (learning rate, gradient norm)
    USEFUL = auto()      # Полезные метрики (system resources)
    VERBOSE = auto()     # Детальные метрики (detailed breakdowns)


@dataclass
class LogEntry:
    """Структура записи лога"""
    timestamp: datetime
    level: LogLevel
    component: str
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    step: Optional[int] = None
    epoch: Optional[int] = None
    priority: MetricPriority = MetricPriority.USEFUL


@dataclass
class ComponentConfig:
    """Конфигурация компонента логирования"""
    name: str
    enabled: bool = True
    log_level: LogLevel = LogLevel.INFO
    metric_priority: MetricPriority = MetricPriority.USEFUL
    custom_format: Optional[str] = None
    namespace: Optional[str] = None


class UnifiedLoggingSystem:
    """
    🔥 Unified Logging System - Единая точка управления всем логированием
    
    Singleton pattern предотвращает множественные экземпляры и конфликты.
    Централизует MLflow, TensorBoard и file logging.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Предотвращаем повторную инициализацию singleton
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.config = config or self._get_default_config()
        
        # Состояние системы
        self._active = False
        self._mlflow_run = None
        self._tensorboard_writer = None
        self._file_handlers: Dict[str, logging.FileHandler] = {}
        
        # Компоненты
        self._components: Dict[str, ComponentConfig] = {}
        self._loggers: Dict[str, logging.Logger] = {}
        
        # Метрики и история
        self._metrics_history: List[LogEntry] = []
        self._session_start_time = datetime.now()
        
        # Thread safety
        self._logging_lock = threading.Lock()
        
        # Автоматическая очистка при выходе
        atexit.register(self._cleanup)
        
        print("🔥 Unified Logging System инициализирована")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию"""
        return {
            'session_name': f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'base_log_dir': 'unified_logs',
            'enable_mlflow': MLFLOW_AVAILABLE,
            'enable_tensorboard': TENSORBOARD_AVAILABLE,
            'enable_file_logging': True,
            'enable_console_logging': True,
            'max_history_entries': 10000,
            'metric_priority_threshold': MetricPriority.USEFUL,
            'auto_flush_interval': 30,  # секунд
            'components': {
                'context_aware_manager': {'priority': MetricPriority.ESSENTIAL},
                'stabilization_system': {'priority': MetricPriority.ESSENTIAL},
                'attention_enhancement': {'priority': MetricPriority.IMPORTANT},
                'smart_tuner': {'priority': MetricPriority.IMPORTANT},
                'training_monitor': {'priority': MetricPriority.USEFUL},
                'system_monitor': {'priority': MetricPriority.VERBOSE}
            }
        }
    
    def start_session(self, session_name: Optional[str] = None, 
                     experiment_name: Optional[str] = None) -> bool:
        """
        🚀 Запуск unified logging session
        
        Args:
            session_name: Имя сессии логирования
            experiment_name: Имя MLflow эксперимента
            
        Returns:
            True если успешно запущена
        """
        if self._active:
            self._log_system_message("Session уже активна", LogLevel.WARNING)
            return True
        
        try:
            self._session_start_time = datetime.now()
            session_name = session_name or self.config['session_name']
            
            # Создаем базовую директорию
            base_dir = Path(self.config['base_log_dir']) / session_name
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Инициализация file logging
            if self.config['enable_file_logging']:
                self._setup_file_logging(base_dir)
            
            # Инициализация TensorBoard
            if self.config['enable_tensorboard'] and TENSORBOARD_AVAILABLE:
                self._setup_tensorboard(base_dir)
            
            # Инициализация MLflow
            if self.config['enable_mlflow'] and MLFLOW_AVAILABLE:
                self._setup_mlflow(experiment_name or "Tacotron2_Unified")
            
            self._active = True
            self._log_system_message(f"🔥 Unified Logging Session запущена: {session_name}", LogLevel.INFO)
            return True
            
        except Exception as e:
            self._log_system_message(f"❌ Ошибка запуска session: {e}", LogLevel.ERROR)
            return False
    
    def _setup_file_logging(self, base_dir: Path):
        """Настройка file logging с rotation"""
        try:
            # Главный лог файл
            main_log_file = base_dir / "unified.log"
            main_handler = logging.FileHandler(main_log_file, encoding='utf-8')
            main_formatter = logging.Formatter(
                '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
            )
            main_handler.setFormatter(main_formatter)
            
            # Метрики файл
            metrics_log_file = base_dir / "metrics.log"
            metrics_handler = logging.FileHandler(metrics_log_file, encoding='utf-8')
            metrics_formatter = logging.Formatter(
                '%(asctime)s - METRICS - %(message)s'
            )
            metrics_handler.setFormatter(metrics_formatter)
            
            self._file_handlers['main'] = main_handler
            self._file_handlers['metrics'] = metrics_handler
            
            print(f"✅ File logging настроено: {base_dir}")
            
        except Exception as e:
            print(f"❌ Ошибка настройки file logging: {e}")
    
    def _setup_tensorboard(self, base_dir: Path):
        """Настройка централизованного TensorBoard"""
        try:
            tb_dir = base_dir / "tensorboard"
            self._tensorboard_writer = SummaryWriter(str(tb_dir))
            print(f"✅ TensorBoard настроен: {tb_dir}")
            
        except Exception as e:
            print(f"❌ Ошибка настройки TensorBoard: {e}")
            self._tensorboard_writer = None
    
    def _setup_mlflow(self, experiment_name: str):
        """Настройка централизованного MLflow"""
        try:
            # Проверяем и завершаем существующие runs
            try:
                mlflow.end_run()
            except:
                pass  # Нет активного run
            
            # Создаем или используем существующий эксперимент
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                else:
                    experiment_id = experiment.experiment_id
            except Exception:
                experiment_id = mlflow.create_experiment(experiment_name)
            
            # Запускаем новый run
            self._mlflow_run = mlflow.start_run(
                experiment_id=experiment_id,
                run_name=f"unified_{self.config['session_name']}"
            )
            
            # Логируем базовые параметры сессии
            mlflow.log_param("session_name", self.config['session_name'])
            mlflow.log_param("session_start", self._session_start_time.isoformat())
            mlflow.log_param("unified_logging", True)
            
            print(f"✅ MLflow настроен: {experiment_name}")
            
        except Exception as e:
            print(f"❌ Ошибка настройки MLflow: {e}")
            self._mlflow_run = None
    
    def register_component(self, component_name: str, 
                          config: Optional[ComponentConfig] = None) -> 'ComponentLogger':
        """
        📝 Регистрация компонента в unified logging system
        
        Args:
            component_name: Имя компонента
            config: Конфигурация компонента
            
        Returns:
            ComponentLogger для использования компонентом
        """
        if config is None:
            # Используем конфигурацию из config файла или дефолтную
            component_config_dict = self.config['components'].get(component_name, {})
            config = ComponentConfig(
                name=component_name,
                metric_priority=component_config_dict.get('priority', MetricPriority.USEFUL)
            )
        
        self._components[component_name] = config
        
        # Создаем namespace для компонента
        namespace = config.namespace or component_name
        
        # Создаем logger для компонента
        logger = logging.getLogger(f"unified.{namespace}")
        logger.setLevel(self._get_log_level(config.log_level))
        
        # Добавляем handlers если нужно
        if 'main' in self._file_handlers:
            logger.addHandler(self._file_handlers['main'])
        
        self._loggers[component_name] = logger
        
        component_logger = ComponentLogger(self, component_name, config)
        
        self._log_system_message(f"📝 Компонент зарегистрирован: {component_name}", LogLevel.INFO)
        return component_logger
    
    def log_metrics(self, metrics: Dict[str, Any], 
                   component: str = "system",
                   step: Optional[int] = None,
                   priority: MetricPriority = MetricPriority.USEFUL):
        """
        📊 Централизованное логирование метрик
        
        Args:
            metrics: Словарь метрик
            component: Имя компонента
            step: Номер шага (для time series)
            priority: Приоритет метрик
        """
        with self._logging_lock:
            try:
                # Фильтрация по приоритету
                if not self._should_log_priority(priority):
                    return
                
                # Создание log entry
                log_entry = LogEntry(
                    timestamp=datetime.now(),
                    level=LogLevel.METRICS,
                    component=component,
                    message=f"Metrics from {component}",
                    metrics=metrics,
                    step=step,
                    priority=priority
                )
                
                # Добавляем в историю
                self._add_to_history(log_entry)
                
                # Логируем в файл
                self._log_to_file(log_entry)
                
                # Логируем в TensorBoard
                self._log_to_tensorboard(metrics, component, step)
                
                # Логируем в MLflow
                self._log_to_mlflow(metrics, step)
                
            except Exception as e:
                self._log_system_message(f"❌ Ошибка логирования метрик: {e}", LogLevel.ERROR)
    
    def log_message(self, message: str,
                   level: LogLevel = LogLevel.INFO,
                   component: str = "system",
                   **kwargs):
        """Логирование текстового сообщения"""
        with self._logging_lock:
            try:
                log_entry = LogEntry(
                    timestamp=datetime.now(),
                    level=level,
                    component=component,
                    message=message,
                    **kwargs
                )
                
                self._add_to_history(log_entry)
                self._log_to_file(log_entry)
                
                # Консольный вывод для важных сообщений
                if level in [LogLevel.CRITICAL, LogLevel.ERROR, LogLevel.WARNING]:
                    print(f"[{level.name}] {component}: {message}")
                
            except Exception as e:
                print(f"❌ Ошибка логирования сообщения: {e}")
    
    def _should_log_priority(self, priority: MetricPriority) -> bool:
        """Проверка должна ли метрика логироваться на основе приоритета"""
        threshold = self.config['metric_priority_threshold']
        priority_values = {
            MetricPriority.ESSENTIAL: 4,
            MetricPriority.IMPORTANT: 3, 
            MetricPriority.USEFUL: 2,
            MetricPriority.VERBOSE: 1
        }
        return priority_values.get(priority, 1) >= priority_values.get(threshold, 2)
    
    def _add_to_history(self, entry: LogEntry):
        """Добавляет запись в историю с ограничением размера"""
        self._metrics_history.append(entry)
        
        # Ограничиваем размер истории
        max_entries = self.config['max_history_entries']
        if len(self._metrics_history) > max_entries:
            self._metrics_history = self._metrics_history[-max_entries:]
    
    def _log_to_file(self, entry: LogEntry):
        """Логирование в файл"""
        try:
            if entry.level == LogLevel.METRICS and 'metrics' in self._file_handlers:
                handler = self._file_handlers['metrics']
                logger = logging.getLogger("unified.metrics")
                logger.handlers = [handler]
                
                metrics_str = json.dumps(entry.metrics, default=str)
                logger.info(f"[{entry.component}] {metrics_str}")
                
            elif 'main' in self._file_handlers:
                handler = self._file_handlers['main']
                logger = logging.getLogger("unified.main")
                logger.handlers = [handler]
                
                logger.log(
                    self._get_log_level(entry.level),
                    f"[{entry.component}] {entry.message}"
                )
                
        except Exception as e:
            print(f"❌ Ошибка записи в файл: {e}")
    
    def _log_to_tensorboard(self, metrics: Dict[str, Any], component: str, step: Optional[int]):
        """Логирование в TensorBoard"""
        if not self._tensorboard_writer or step is None:
            return
        
        try:
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    full_metric_name = f"{component}/{metric_name}"
                    self._tensorboard_writer.add_scalar(full_metric_name, value, step)
            
            self._tensorboard_writer.flush()
            
        except Exception as e:
            self._log_system_message(f"❌ Ошибка TensorBoard логирования: {e}", LogLevel.WARNING)
    
    def _log_to_mlflow(self, metrics: Dict[str, Any], step: Optional[int]):
        """Логирование в MLflow"""
        if not self._mlflow_run:
            return
        
        try:
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(metric_name, value, step=step)
                    
        except Exception as e:
            self._log_system_message(f"❌ Ошибка MLflow логирования: {e}", LogLevel.WARNING)
    
    def _get_log_level(self, level: LogLevel) -> int:
        """Конвертация LogLevel в стандартный logging level"""
        mapping = {
            LogLevel.CRITICAL: logging.CRITICAL,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.INFO: logging.INFO,
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.METRICS: logging.INFO,
            LogLevel.SYSTEM: logging.DEBUG
        }
        return mapping.get(level, logging.INFO)
    
    def _log_system_message(self, message: str, level: LogLevel):
        """Внутреннее логирование системных сообщений"""
        self.log_message(message, level, "unified_logging_system")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Получение сводки текущей сессии"""
        current_time = datetime.now()
        session_duration = current_time - self._session_start_time
        
        # Статистика по компонентам
        component_stats = {}
        for entry in self._metrics_history:
            comp = entry.component
            if comp not in component_stats:
                component_stats[comp] = {'count': 0, 'last_update': None}
            component_stats[comp]['count'] += 1
            component_stats[comp]['last_update'] = entry.timestamp
        
        # Статистика метрик
        total_metrics = len([e for e in self._metrics_history if e.level == LogLevel.METRICS])
        
        return {
            'session_name': self.config['session_name'],
            'session_start': self._session_start_time.isoformat(),
            'session_duration': str(session_duration),
            'active': self._active,
            'mlflow_active': self._mlflow_run is not None,
            'tensorboard_active': self._tensorboard_writer is not None,
            'total_log_entries': len(self._metrics_history),
            'total_metrics_logged': total_metrics,
            'registered_components': list(self._components.keys()),
            'component_statistics': component_stats
        }
    
    def end_session(self):
        """🏁 Завершение unified logging session"""
        if not self._active:
            return
        
        try:
            # Финальная сводка
            summary = self.get_session_summary()
            self._log_system_message(f"📊 Завершение сессии: {json.dumps(summary, indent=2, default=str)}", LogLevel.INFO)
            
            # Закрытие TensorBoard
            if self._tensorboard_writer:
                self._tensorboard_writer.close()
                self._tensorboard_writer = None
                self._log_system_message("✅ TensorBoard закрыт", LogLevel.INFO)
            
            # Завершение MLflow
            if self._mlflow_run and MLFLOW_AVAILABLE:
                try:
                    # Логируем финальные параметры сессии
                    session_duration = datetime.now() - self._session_start_time
                    mlflow.log_param("session_duration_minutes", session_duration.total_seconds() / 60)
                    mlflow.log_param("total_log_entries", len(self._metrics_history))
                    
                    mlflow.end_run()
                    self._mlflow_run = None
                    self._log_system_message("✅ MLflow run завершен", LogLevel.INFO)
                except Exception as e:
                    self._log_system_message(f"⚠️ Ошибка завершения MLflow: {e}", LogLevel.WARNING)
            
            # Закрытие file handlers
            for handler_name, handler in self._file_handlers.items():
                try:
                    handler.close()
                except:
                    pass
            self._file_handlers.clear()
            
            self._active = False
            print("🏁 Unified Logging Session завершена")
            
        except Exception as e:
            print(f"❌ Ошибка завершения session: {e}")
    
    def _cleanup(self):
        """Автоматическая очистка при выходе"""
        if self._active:
            self.end_session()
    
    @contextmanager
    def session_context(self, session_name: Optional[str] = None):
        """Context manager для автоматического управления сессией"""
        try:
            if self.start_session(session_name):
                yield self
            else:
                raise RuntimeError("Failed to start logging session")
        finally:
            self.end_session()


class ComponentLogger:
    """
    📝 Логгер для конкретного компонента
    
    Предоставляет простой интерфейс для компонентов, скрывая
    сложность unified logging system.
    """
    
    def __init__(self, unified_system: UnifiedLoggingSystem, 
                 component_name: str, config: ComponentConfig):
        self.unified_system = unified_system
        self.component_name = component_name
        self.config = config
        self._step_counter = 0
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Логирование метрик компонента"""
        if step is None:
            step = self._step_counter
            self._step_counter += 1
        
        self.unified_system.log_metrics(
            metrics=metrics,
            component=self.component_name,
            step=step,
            priority=self.config.metric_priority
        )
    
    def info(self, message: str, **kwargs):
        """Информационное сообщение"""
        self.unified_system.log_message(
            message, LogLevel.INFO, self.component_name, **kwargs
        )
    
    def warning(self, message: str, **kwargs):
        """Предупреждение"""
        self.unified_system.log_message(
            message, LogLevel.WARNING, self.component_name, **kwargs
        )
    
    def error(self, message: str, **kwargs):
        """Ошибка"""
        self.unified_system.log_message(
            message, LogLevel.ERROR, self.component_name, **kwargs
        )
    
    def critical(self, message: str, **kwargs):
        """Критическая ошибка"""
        self.unified_system.log_message(
            message, LogLevel.CRITICAL, self.component_name, **kwargs
        )
    
    def debug(self, message: str, **kwargs):
        """Отладочное сообщение"""
        self.unified_system.log_message(
            message, LogLevel.DEBUG, self.component_name, **kwargs
        )


# Global instance
_global_unified_logging = None
_global_lock = threading.Lock()


def get_unified_logger() -> UnifiedLoggingSystem:
    """
    🌍 Получение глобального экземпляра Unified Logging System
    
    Thread-safe singleton access
    """
    global _global_unified_logging
    
    if _global_unified_logging is None:
        with _global_lock:
            if _global_unified_logging is None:
                _global_unified_logging = UnifiedLoggingSystem()
    
    return _global_unified_logging


def setup_component_logging(component_name: str, 
                          priority: MetricPriority = MetricPriority.USEFUL) -> ComponentLogger:
    """
    🔧 Быстрая настройка логирования для компонента
    
    Args:
        component_name: Имя компонента
        priority: Приоритет метрик компонента
        
    Returns:
        ComponentLogger для использования
    """
    unified_system = get_unified_logger()
    
    config = ComponentConfig(
        name=component_name,
        metric_priority=priority
    )
    
    return unified_system.register_component(component_name, config)


# Convenience functions для backward compatibility
def log_training_metrics(metrics: Dict[str, Any], step: Optional[int] = None, component: str = "training"):
    """Convenience функция для логирования метрик обучения"""
    unified_system = get_unified_logger()
    unified_system.log_metrics(
        metrics=metrics,
        component=component,
        step=step,
        priority=MetricPriority.ESSENTIAL
    )


def start_unified_logging_session(session_name: Optional[str] = None, 
                                experiment_name: Optional[str] = None) -> bool:
    """Запуск глобальной сессии логирования"""
    unified_system = get_unified_logger()
    return unified_system.start_session(session_name, experiment_name)


def end_unified_logging_session():
    """Завершение глобальной сессии логирования"""
    unified_system = get_unified_logger()
    unified_system.end_session()


# Availability check
UNIFIED_LOGGING_AVAILABLE = True

def get_unified_logging_info():
    """Информация о unified logging system"""
    return {
        'available': UNIFIED_LOGGING_AVAILABLE,
        'mlflow_available': MLFLOW_AVAILABLE,
        'tensorboard_available': TENSORBOARD_AVAILABLE,
        'torch_available': TORCH_AVAILABLE,
        'features': [
            'Centralized MLflow management',
            'Unified TensorBoard writer',
            'Priority-based metric filtering',
            'Component isolation',
            'Thread-safe logging',
            'Automatic session management',
            'Graceful fallback'
        ]
    }


if __name__ == "__main__":
    # Демонстрация использования
    print("🔥 Демонстрация Unified Logging System")
    
    # Создание unified logger
    unified_logger = get_unified_logger()
    
    # Запуск сессии
    with unified_logger.session_context("demo_session"):
        
        # Регистрация компонентов
        context_logger = setup_component_logging("context_aware_manager", MetricPriority.ESSENTIAL)
        training_logger = setup_component_logging("training_monitor", MetricPriority.IMPORTANT)
        
        # Демонстрационные метрики
        context_logger.log_metrics({
            "loss": 15.5,
            "attention_diagonality": 0.045,
            "guided_attention_weight": 8.0
        })
        
        training_logger.log_metrics({
            "learning_rate": 1e-4,
            "gradient_norm": 2.3
        })
        
        # Сообщения
        context_logger.info("Context analysis completed")
        training_logger.warning("High gradient norm detected")
        
        # Сводка сессии
        summary = unified_logger.get_session_summary()
        print(f"📊 Сводка сессии: {json.dumps(summary, indent=2)}")
    
    print("✅ Демонстрация завершена") 