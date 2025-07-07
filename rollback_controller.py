"""
Rollback Controller для Enhanced Tacotron2 AI System

Модуль автоматического отката неудачных решений с использованием
state machine и checkpointing системы.
"""

import logging
import json
import pickle
import shutil
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import torch
import threading
from collections import deque
import sqlite3

# Настройка логирования
logger = logging.getLogger(__name__)

class SystemStateEncoder(json.JSONEncoder):
    """Кастомный JSON encoder для сериализации SystemState enum"""
    def default(self, obj):
        if isinstance(obj, SystemState):
            return obj.value
        elif isinstance(obj, RollbackTrigger):
            return obj.value
        return super().default(obj)

class SystemState(Enum):
    """Состояния системы обучения"""
    STABLE = "stable"
    MONITORING = "monitoring"
    UNSTABLE = "unstable"
    CRITICAL = "critical"
    ROLLBACK_INITIATED = "rollback_initiated"
    ROLLBACK_IN_PROGRESS = "rollback_in_progress"
    ROLLBACK_COMPLETED = "rollback_completed"
    RECOVERY = "recovery"

class RollbackTrigger(Enum):
    """Триггеры для отката"""
    HIGH_RISK_ASSESSMENT = "high_risk_assessment"
    CRITICAL_RISK_ASSESSMENT = "critical_risk_assessment"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    TRAINING_INSTABILITY = "training_instability"
    MANUAL_TRIGGER = "manual_trigger"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class RollbackConfig:
    """Конфигурация для Rollback Controller"""
    # Пороги для триггеров
    high_risk_threshold: float = 0.8
    critical_risk_threshold: float = 0.95
    performance_degradation_threshold: float = 0.2
    
    # Настройки checkpointing
    max_checkpoints: int = 10
    checkpoint_interval_minutes: int = 30
    emergency_checkpoint_interval_minutes: int = 5
    
    # Пути для сохранений
    rollback_base_path: str = "rollback_data"
    checkpoint_base_path: str = "rollback_checkpoints"
    state_db_path: str = "rollback_states.db"
    
    # Настройки мониторинга
    monitoring_window_size: int = 50
    stability_check_interval_seconds: int = 60
    
    # Автоматический режим
    auto_rollback_enabled: bool = True
    require_confirmation: bool = False

@dataclass
class SystemCheckpoint:
    """Checkpoint системы"""
    checkpoint_id: str
    timestamp: str
    system_state: SystemState
    model_state_path: str
    optimizer_state_path: str
    scheduler_state_path: str
    training_metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    risk_assessment: Optional[Dict[str, Any]]
    description: str

@dataclass
class RollbackEvent:
    """Событие отката"""
    event_id: str
    timestamp: str
    trigger: RollbackTrigger
    from_state: SystemState
    to_state: SystemState
    target_checkpoint_id: str
    reason: str
    risk_score: Optional[float]
    success: bool
    recovery_time_seconds: float

class StateTransitionManager:
    """Менеджер переходов между состояниями"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.StateTransitionManager")
        
        # Определение допустимых переходов
        self.valid_transitions = {
            SystemState.STABLE: [
                SystemState.MONITORING,
                SystemState.UNSTABLE,
                SystemState.CRITICAL,
                SystemState.ROLLBACK_INITIATED
            ],
            SystemState.MONITORING: [
                SystemState.STABLE,
                SystemState.UNSTABLE,
                SystemState.CRITICAL,
                SystemState.ROLLBACK_INITIATED
            ],
            SystemState.UNSTABLE: [
                SystemState.STABLE,
                SystemState.MONITORING,
                SystemState.CRITICAL,
                SystemState.ROLLBACK_INITIATED
            ],
            SystemState.CRITICAL: [
                SystemState.ROLLBACK_INITIATED,
                SystemState.RECOVERY
            ],
            SystemState.ROLLBACK_INITIATED: [
                SystemState.ROLLBACK_IN_PROGRESS
            ],
            SystemState.ROLLBACK_IN_PROGRESS: [
                SystemState.ROLLBACK_COMPLETED,
                SystemState.CRITICAL  # В случае неудачи
            ],
            SystemState.ROLLBACK_COMPLETED: [
                SystemState.RECOVERY,
                SystemState.STABLE
            ],
            SystemState.RECOVERY: [
                SystemState.STABLE,
                SystemState.MONITORING,
                SystemState.UNSTABLE,
                SystemState.CRITICAL
            ]
        }
    
    def is_valid_transition(self, from_state: SystemState, to_state: SystemState) -> bool:
        """Проверка допустимости перехода"""
        return to_state in self.valid_transitions.get(from_state, [])
    
    def get_allowed_transitions(self, current_state: SystemState) -> List[SystemState]:
        """Получение списка допустимых переходов"""
        return self.valid_transitions.get(current_state, [])
    
    def suggest_next_state(self, current_state: SystemState, 
                          risk_score: float, performance_change: float) -> SystemState:
        """Предложение следующего состояния на основе метрик"""
        
        # Критическая ситуация
        if risk_score >= 0.95 or performance_change <= -0.5:
            if current_state != SystemState.CRITICAL:
                return SystemState.CRITICAL
        
        # Высокий риск
        elif risk_score >= 0.8 or performance_change <= -0.2:
            if current_state == SystemState.STABLE:
                return SystemState.UNSTABLE
            elif current_state == SystemState.MONITORING:
                return SystemState.UNSTABLE
        
        # Средний риск - мониторинг
        elif risk_score >= 0.5 or abs(performance_change) > 0.1:
            if current_state == SystemState.STABLE:
                return SystemState.MONITORING
        
        # Низкий риск - стабильное состояние
        else:
            if current_state in [SystemState.MONITORING, SystemState.RECOVERY]:
                return SystemState.STABLE
        
        return current_state

class CheckpointManager:
    """Менеджер checkpoint'ов для отката"""
    
    def __init__(self, config: RollbackConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.CheckpointManager")
        
        # Создание директорий
        self.checkpoint_dir = Path(config.checkpoint_base_path)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # История checkpoint'ов
        self.checkpoints: deque[SystemCheckpoint] = deque(maxlen=config.max_checkpoints)
        
        # Блокировка для thread safety
        self._lock = threading.Lock()
    
    def create_checkpoint(self, 
                         model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                         training_metadata: Dict[str, Any],
                         performance_metrics: Dict[str, float],
                         system_state: SystemState,
                         description: str = "",
                         risk_assessment: Optional[Dict[str, Any]] = None) -> SystemCheckpoint:
        """Создание checkpoint'а системы"""
        
        try:
            checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            checkpoint_path = self.checkpoint_dir / checkpoint_id
            checkpoint_path.mkdir(exist_ok=True)
            
            # Сохранение состояний
            model_path = checkpoint_path / "model_state.pth"
            optimizer_path = checkpoint_path / "optimizer_state.pth"
            scheduler_path = checkpoint_path / "scheduler_state.pth"
            
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            
            if scheduler is not None:
                torch.save(scheduler.state_dict(), scheduler_path)
            
            # Создание объекта checkpoint'а
            checkpoint = SystemCheckpoint(
                checkpoint_id=checkpoint_id,
                timestamp=datetime.now().isoformat(),
                system_state=system_state,
                model_state_path=str(model_path),
                optimizer_state_path=str(optimizer_path),
                scheduler_state_path=str(scheduler_path) if scheduler else "",
                training_metadata=training_metadata.copy(),
                performance_metrics=performance_metrics.copy(),
                risk_assessment=risk_assessment.copy() if risk_assessment else None,
                description=description
            )
            
            # Сохранение метаданных
            metadata_path = checkpoint_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(checkpoint), f, indent=2, cls=SystemStateEncoder)
            
            # Добавление в историю
            with self._lock:
                self.checkpoints.append(checkpoint)
            
            self.logger.info(f"Checkpoint created: {checkpoint_id}")
            
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            raise
    
    def restore_checkpoint(self, 
                          checkpoint_id: str,
                          model: torch.nn.Module,
                          optimizer: torch.optim.Optimizer,
                          scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> SystemCheckpoint:
        """Восстановление из checkpoint'а"""
        
        try:
            # Поиск checkpoint'а
            checkpoint = None
            with self._lock:
                for cp in self.checkpoints:
                    if cp.checkpoint_id == checkpoint_id:
                        checkpoint = cp
                        break
            
            if checkpoint is None:
                raise ValueError(f"Checkpoint {checkpoint_id} not found")
            
            # Восстановление состояний
            model.load_state_dict(torch.load(checkpoint.model_state_path))
            optimizer.load_state_dict(torch.load(checkpoint.optimizer_state_path))
            
            if scheduler is not None and checkpoint.scheduler_state_path:
                scheduler.load_state_dict(torch.load(checkpoint.scheduler_state_path))
            
            self.logger.info(f"Checkpoint restored: {checkpoint_id}")
            
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
            raise
    
    def get_latest_stable_checkpoint(self) -> Optional[SystemCheckpoint]:
        """Получение последнего стабильного checkpoint'а"""
        with self._lock:
            for checkpoint in reversed(self.checkpoints):
                if checkpoint.system_state == SystemState.STABLE:
                    return checkpoint
        return None
    
    def get_checkpoint_by_id(self, checkpoint_id: str) -> Optional[SystemCheckpoint]:
        """Получение checkpoint'а по ID"""
        with self._lock:
            for checkpoint in self.checkpoints:
                if checkpoint.checkpoint_id == checkpoint_id:
                    return checkpoint
        return None
    
    def list_checkpoints(self) -> List[SystemCheckpoint]:
        """Получение списка всех checkpoint'ов"""
        with self._lock:
            return list(self.checkpoints)
    
    def cleanup_old_checkpoints(self, max_age_days: int = 7):
        """Очистка старых checkpoint'ов"""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        with self._lock:
            to_remove = []
            for checkpoint in self.checkpoints:
                checkpoint_time = datetime.fromisoformat(checkpoint.timestamp)
                if checkpoint_time < cutoff_time:
                    to_remove.append(checkpoint)
            
            for checkpoint in to_remove:
                try:
                    # Удаление файлов
                    checkpoint_path = Path(checkpoint.model_state_path).parent
                    if checkpoint_path.exists():
                        shutil.rmtree(checkpoint_path)
                    
                    # Удаление из списка
                    self.checkpoints.remove(checkpoint)
                    
                    self.logger.info(f"Cleaned up old checkpoint: {checkpoint.checkpoint_id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to cleanup checkpoint {checkpoint.checkpoint_id}: {e}")

class RollbackDatabase:
    """База данных для истории rollback событий"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.RollbackDatabase")
        self._init_database()
    
    def _init_database(self):
        """Инициализация базы данных"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Таблица событий отката
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS rollback_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT UNIQUE NOT NULL,
                        timestamp TEXT NOT NULL,
                        trigger TEXT NOT NULL,
                        from_state TEXT NOT NULL,
                        to_state TEXT NOT NULL,
                        target_checkpoint_id TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        risk_score REAL,
                        success BOOLEAN NOT NULL,
                        recovery_time_seconds REAL NOT NULL
                    )
                ''')
                
                # Таблица истории состояний
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS state_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        state TEXT NOT NULL,
                        risk_score REAL,
                        performance_metrics TEXT,
                        trigger_reason TEXT
                    )
                ''')
                
                conn.commit()
                self.logger.info("Rollback database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def save_rollback_event(self, event: RollbackEvent):
        """Сохранение события отката"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Попытка вставки с обработкой дубликатов
                cursor.execute('''
                    INSERT OR REPLACE INTO rollback_events 
                    (event_id, timestamp, trigger, from_state, to_state,
                     target_checkpoint_id, reason, risk_score, success, recovery_time_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id,
                    event.timestamp,
                    event.trigger.value,
                    event.from_state.value,
                    event.to_state.value,
                    event.target_checkpoint_id,
                    event.reason,
                    event.risk_score,
                    event.success,
                    event.recovery_time_seconds
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save rollback event: {e}")
    
    def save_state_change(self, timestamp: str, state: SystemState, 
                         risk_score: Optional[float] = None,
                         performance_metrics: Optional[Dict] = None,
                         trigger_reason: str = ""):
        """Сохранение изменения состояния"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                metrics_json = json.dumps(performance_metrics) if performance_metrics else None
                
                cursor.execute('''
                    INSERT INTO state_history 
                    (timestamp, state, risk_score, performance_metrics, trigger_reason)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    timestamp,
                    state.value,
                    risk_score,
                    metrics_json,
                    trigger_reason
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save state change: {e}")

class RollbackController:
    """
    Главный контроллер автоматического отката для Enhanced Tacotron2 AI System
    
    Управляет состояниями системы, создает checkpoint'ы и выполняет
    автоматический откат при критических ситуациях.
    """
    
    def __init__(self, config: Optional[RollbackConfig] = None):
        self.config = config or RollbackConfig()
        self.logger = logging.getLogger(__name__)
        
        # Инициализация компонентов
        self.state_manager = StateTransitionManager()
        self.checkpoint_manager = CheckpointManager(self.config)
        self.database = RollbackDatabase(self.config.state_db_path)
        
        # Текущее состояние системы
        self.current_state = SystemState.STABLE
        
        # История метрик производительности
        self.performance_history: deque = deque(maxlen=self.config.monitoring_window_size)
        
        # Блокировка для thread safety
        self._lock = threading.Lock()
        
        # Флаги состояния
        self.rollback_in_progress = False
        self.monitoring_active = False
        
        # Callback функции
        self.state_change_callbacks: List[Callable] = []
        self.rollback_callbacks: List[Callable] = []
        
        self.logger.info("Rollback Controller initialized successfully")
    
    def add_state_change_callback(self, callback: Callable):
        """Добавление callback для изменений состояния"""
        self.state_change_callbacks.append(callback)
    
    def add_rollback_callback(self, callback: Callable):
        """Добавление callback для событий отката"""
        self.rollback_callbacks.append(callback)
    
    def transition_to_state(self, new_state: SystemState, 
                           reason: str = "", risk_score: Optional[float] = None,
                           performance_metrics: Optional[Dict] = None) -> bool:
        """Переход в новое состояние"""
        
        with self._lock:
            old_state = self.current_state
            
            # Проверка допустимости перехода
            if not self.state_manager.is_valid_transition(old_state, new_state):
                self.logger.warning(f"Invalid transition: {old_state} -> {new_state}")
                return False
            
            # Выполнение перехода
            self.current_state = new_state
            timestamp = datetime.now().isoformat()
            
            # Сохранение в базу данных
            self.database.save_state_change(
                timestamp, new_state, risk_score, performance_metrics, reason
            )
            
            self.logger.info(f"State transition: {old_state} -> {new_state} (reason: {reason})")
            
            # Вызов callback'ов
            for callback in self.state_change_callbacks:
                try:
                    callback(old_state, new_state, reason, risk_score)
                except Exception as e:
                    self.logger.error(f"State change callback failed: {e}")
            
            return True
    
    def assess_system_risk(self, risk_assessment_module=None, **kwargs) -> float:
        """Оценка текущего риска системы"""
        
        if risk_assessment_module is None:
            # Простая эвристическая оценка если модуль недоступен
            return self._calculate_heuristic_risk(**kwargs)
        
        try:
            # Использование Risk Assessment Module
            parameter_changes = kwargs.get('parameter_changes', {})
            if parameter_changes:
                assessment = risk_assessment_module.assess_system_risk(parameter_changes)
                return assessment.overall_risk_score
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            return 0.5  # Средний риск по умолчанию
    
    def _calculate_heuristic_risk(self, **kwargs) -> float:
        """Простая эвристическая оценка риска"""
        
        risk_score = 0.0
        
        # Оценка на основе изменения производительности
        performance_change = kwargs.get('performance_change', 0.0)
        if performance_change < -0.3:
            risk_score += 0.5
        elif performance_change < -0.1:
            risk_score += 0.3
        
        # Оценка на основе нестабильности
        instability_metrics = kwargs.get('instability_metrics', {})
        if instability_metrics:
            gradient_norm = instability_metrics.get('gradient_norm', 0)
            loss_variance = instability_metrics.get('loss_variance', 0)
            
            if gradient_norm > 10.0:
                risk_score += 0.3
            if loss_variance > 1.0:
                risk_score += 0.2
        
        return min(risk_score, 1.0)
    
    def update_performance_metrics(self, metrics: Dict[str, float]):
        """Обновление метрик производительности"""
        
        timestamp = datetime.now().isoformat()
        metrics_with_time = {'timestamp': timestamp, **metrics}
        
        with self._lock:
            self.performance_history.append(metrics_with_time)
        
        # Анализ тенденций
        if len(self.performance_history) >= 2:
            self._analyze_performance_trends()
    
    def _analyze_performance_trends(self):
        """Анализ трендов производительности"""
        
        if len(self.performance_history) < 3:
            return
        
        # Расчет изменения производительности
        recent_metrics = list(self.performance_history)[-3:]
        
        # Извлечение ключевых метрик
        loss_values = [m.get('loss', 0) for m in recent_metrics if 'loss' in m]
        
        if len(loss_values) >= 2:
            performance_change = (loss_values[0] - loss_values[-1]) / max(abs(loss_values[0]), 1e-6)
            
            # Определение состояния на основе изменений
            if performance_change <= -0.5:
                self._trigger_rollback(
                    RollbackTrigger.PERFORMANCE_DEGRADATION,
                    f"Severe performance degradation: {performance_change:.3f}"
                )
            elif performance_change <= -0.2 and self.current_state == SystemState.STABLE:
                self.transition_to_state(
                    SystemState.UNSTABLE,
                    f"Performance degradation detected: {performance_change:.3f}"
                )
    
    def create_checkpoint(self, model: torch.nn.Module, 
                         optimizer: torch.optim.Optimizer,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                         training_metadata: Optional[Dict] = None,
                         performance_metrics: Optional[Dict] = None,
                         description: str = "") -> SystemCheckpoint:
        """Создание checkpoint'а"""
        
        training_metadata = training_metadata or {}
        performance_metrics = performance_metrics or {}
        
        return self.checkpoint_manager.create_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            system_state=self.current_state,
            description=description
        )
    
    def _trigger_rollback(self, trigger: RollbackTrigger, reason: str, 
                         risk_score: Optional[float] = None):
        """Запуск процесса отката"""
        
        if self.rollback_in_progress:
            self.logger.warning("Rollback already in progress, ignoring trigger")
            return
        
        # Проверка автоматического режима
        if not self.config.auto_rollback_enabled and trigger != RollbackTrigger.MANUAL_TRIGGER:
            self.logger.warning(f"Auto rollback disabled, trigger ignored: {trigger}")
            return
        
        self.logger.warning(f"Rollback triggered: {trigger.value} - {reason}")
        
        # Переход в состояние инициации отката
        self.transition_to_state(
            SystemState.ROLLBACK_INITIATED,
            f"Rollback triggered: {reason}",
            risk_score
        )
        
        # Выполнение отката
        self._execute_rollback(trigger, reason, risk_score)
    
    def _execute_rollback(self, trigger: RollbackTrigger, reason: str, 
                         risk_score: Optional[float] = None):
        """Выполнение отката"""
        
        start_time = datetime.now()
        import uuid
        event_id = f"rollback_{start_time.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        try:
            self.rollback_in_progress = True
            
            # Переход в состояние выполнения отката
            self.transition_to_state(
                SystemState.ROLLBACK_IN_PROGRESS,
                "Executing rollback"
            )
            
            # Поиск подходящего checkpoint'а
            target_checkpoint = self._select_rollback_target()
            
            if target_checkpoint is None:
                raise RuntimeError("No suitable checkpoint found for rollback")
            
            self.logger.info(f"Rolling back to checkpoint: {target_checkpoint.checkpoint_id}")
            
            # Выполнение отката через callback'ы
            rollback_success = True
            for callback in self.rollback_callbacks:
                try:
                    callback(target_checkpoint)
                except Exception as e:
                    self.logger.error(f"Rollback callback failed: {e}")
                    rollback_success = False
            
            if rollback_success:
                # Успешный откат
                self.transition_to_state(
                    SystemState.ROLLBACK_COMPLETED,
                    f"Rollback completed successfully to {target_checkpoint.checkpoint_id}"
                )
                
                # Переход в режим восстановления
                self.transition_to_state(
                    SystemState.RECOVERY,
                    "Starting recovery process"
                )
                
                success = True
            else:
                # Неудачный откат
                self.transition_to_state(
                    SystemState.CRITICAL,
                    "Rollback failed"
                )
                success = False
            
        except Exception as e:
            self.logger.error(f"Rollback execution failed: {e}")
            self.transition_to_state(
                SystemState.CRITICAL,
                f"Rollback execution failed: {e}"
            )
            success = False
            target_checkpoint = None
        
        finally:
            self.rollback_in_progress = False
            
            # Сохранение события отката
            end_time = datetime.now()
            recovery_time = (end_time - start_time).total_seconds()
            
            rollback_event = RollbackEvent(
                event_id=event_id,
                timestamp=start_time.isoformat(),
                trigger=trigger,
                from_state=SystemState.STABLE,  # Предполагаем что откатываемся от стабильного
                to_state=self.current_state,
                target_checkpoint_id=target_checkpoint.checkpoint_id if target_checkpoint else "",
                reason=reason,
                risk_score=risk_score,
                success=success,
                recovery_time_seconds=recovery_time
            )
            
            self.database.save_rollback_event(rollback_event)
    
    def _select_rollback_target(self) -> Optional[SystemCheckpoint]:
        """Выбор checkpoint'а для отката"""
        
        # Сначала пытаемся найти последний стабильный checkpoint
        stable_checkpoint = self.checkpoint_manager.get_latest_stable_checkpoint()
        if stable_checkpoint:
            return stable_checkpoint
        
        # Если нет стабильных, берем последний доступный
        checkpoints = self.checkpoint_manager.list_checkpoints()
        if checkpoints:
            return checkpoints[-1]
        
        return None
    
    def manual_rollback(self, checkpoint_id: Optional[str] = None, reason: str = "Manual rollback"):
        """Ручной запуск отката"""
        
        if checkpoint_id:
            checkpoint = self.checkpoint_manager.get_checkpoint_by_id(checkpoint_id)
            if checkpoint is None:
                raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        self._trigger_rollback(RollbackTrigger.MANUAL_TRIGGER, reason)
    
    def emergency_stop(self, reason: str = "Emergency stop triggered"):
        """Экстренная остановка с откатом"""
        self._trigger_rollback(RollbackTrigger.EMERGENCY_STOP, reason, risk_score=1.0)
    
    def process_risk_assessment(self, assessment_result):
        """Обработка результата оценки рисков"""
        
        if hasattr(assessment_result, 'overall_risk_score'):
            risk_score = assessment_result.overall_risk_score
            is_safe = assessment_result.is_safe_to_proceed
            
            # Определение необходимости отката
            if risk_score >= self.config.critical_risk_threshold:
                self._trigger_rollback(
                    RollbackTrigger.CRITICAL_RISK_ASSESSMENT,
                    f"Critical risk detected: {risk_score:.3f}",
                    risk_score
                )
            elif risk_score >= self.config.high_risk_threshold:
                if not is_safe:
                    self._trigger_rollback(
                        RollbackTrigger.HIGH_RISK_ASSESSMENT,
                        f"High risk detected: {risk_score:.3f}",
                        risk_score
                    )
                else:
                    # Переход в состояние мониторинга
                    self.transition_to_state(
                        SystemState.MONITORING,
                        f"High risk detected, monitoring: {risk_score:.3f}",
                        risk_score
                    )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса системы отката"""
        
        with self._lock:
            checkpoints = self.checkpoint_manager.list_checkpoints()
            
            status = {
                'current_state': self.current_state.value,
                'rollback_in_progress': self.rollback_in_progress,
                'auto_rollback_enabled': self.config.auto_rollback_enabled,
                'total_checkpoints': len(checkpoints),
                'latest_checkpoint': checkpoints[-1].checkpoint_id if checkpoints else None,
                'performance_history_size': len(self.performance_history),
                'valid_transitions': [s.value for s in self.state_manager.get_allowed_transitions(self.current_state)]
            }
        
        return status
    
    def cleanup_old_data(self, max_age_days: int = 7):
        """Очистка старых данных"""
        self.checkpoint_manager.cleanup_old_checkpoints(max_age_days)

# Удобные функции для интеграции
def create_rollback_controller(config: Optional[RollbackConfig] = None) -> RollbackController:
    """Создание и настройка Rollback Controller"""
    return RollbackController(config)

def integrate_with_risk_assessment(rollback_controller: RollbackController, 
                                 risk_assessment_module) -> Callable:
    """Интеграция с Risk Assessment Module"""
    
    def risk_callback(assessment_result):
        rollback_controller.process_risk_assessment(assessment_result)
    
    return risk_callback

if __name__ == "__main__":
    # Демонстрация использования
    logging.basicConfig(level=logging.INFO)
    
    # Создание контроллера
    controller = create_rollback_controller()
    
    # Демонстрация переходов состояний
    print(f"Current state: {controller.current_state}")
    print(f"System status: {controller.get_system_status()}")
    
    # Имитация высокого риска
    controller.process_risk_assessment(type('Assessment', (), {
        'overall_risk_score': 0.85,
        'is_safe_to_proceed': False
    })())
    
    print(f"State after high risk: {controller.current_state}")
    print("Rollback Controller demonstration completed!") 