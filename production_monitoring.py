"""
Production Monitoring System для Enhanced Tacotron2 AI System

Система мониторинга и dashboard для визуального отслеживания всех
компонентов интеллектуальной системы с real-time метриками.
"""

import logging
import threading
import time
import json
import sqlite3
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import pandas as pd
import psutil
import torch

# Настройка логирования
logger = logging.getLogger(__name__)

class ComponentStatus(Enum):
    """Статусы компонентов системы"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"

class AlertSeverity(Enum):
    """Уровни серьезности алертов"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class ComponentMetrics:
    """Метрики компонента системы"""
    component_name: str
    timestamp: str
    status: ComponentStatus
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    gpu_memory: Optional[float]
    custom_metrics: Dict[str, Any]
    error_count: int
    uptime_seconds: float

@dataclass
class SystemAlert:
    """Системный алерт"""
    alert_id: str
    timestamp: str
    component: str
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    resolved: bool
    resolution_time: Optional[str]

@dataclass
class PerformanceSnapshot:
    """Снапшот производительности системы"""
    timestamp: str
    training_loss: Optional[float]
    validation_loss: Optional[float]
    attention_score: Optional[float]
    model_quality: Optional[float]
    throughput: Optional[float]
    memory_efficiency: Optional[float]

@dataclass
class MonitoringConfig:
    """Конфигурация мониторинга"""
    # Интервалы мониторинга
    metrics_collection_interval: int = 30  # секунды
    alert_check_interval: int = 10  # секунды
    dashboard_update_interval: int = 5  # секунды
    
    # Пороги для алертов
    cpu_warning_threshold: float = 80.0
    cpu_critical_threshold: float = 95.0
    memory_warning_threshold: float = 85.0
    memory_critical_threshold: float = 95.0
    gpu_warning_threshold: float = 90.0
    gpu_critical_threshold: float = 98.0
    
    # Настройки хранения
    database_path: str = "monitoring.db"
    metrics_retention_days: int = 30
    alerts_retention_days: int = 90
    
    # Dashboard настройки
    dashboard_host: str = "localhost"
    dashboard_port: int = 8050
    auto_refresh: bool = True
    
    # Компоненты для мониторинга
    monitored_components: List[str] = None
    
    def __post_init__(self):
        if self.monitored_components is None:
            self.monitored_components = [
                "training_stabilization",
                "attention_enhancement", 
                "checkpointing_system",
                "meta_learning_engine",
                "feedback_loop_manager",
                "risk_assessment_module",
                "rollback_controller"
            ]

class MonitoringDatabase:
    """База данных для мониторинга"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.MonitoringDatabase")
        self._init_database()
    
    def _init_database(self):
        """Инициализация базы данных мониторинга"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Таблица метрик компонентов
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS component_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        component_name TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        status TEXT NOT NULL,
                        cpu_usage REAL NOT NULL,
                        memory_usage REAL NOT NULL,
                        gpu_usage REAL,
                        gpu_memory REAL,
                        custom_metrics TEXT,
                        error_count INTEGER NOT NULL,
                        uptime_seconds REAL NOT NULL,
                        INDEX(component_name, timestamp)
                    )
                ''')
                
                # Таблица алертов
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT UNIQUE NOT NULL,
                        timestamp TEXT NOT NULL,
                        component TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details TEXT,
                        resolved BOOLEAN NOT NULL DEFAULT FALSE,
                        resolution_time TEXT,
                        INDEX(component, timestamp),
                        INDEX(severity, resolved)
                    )
                ''')
                
                # Таблица снапшотов производительности
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        training_loss REAL,
                        validation_loss REAL,
                        attention_score REAL,
                        model_quality REAL,
                        throughput REAL,
                        memory_efficiency REAL,
                        INDEX(timestamp)
                    )
                ''')
                
                conn.commit()
                self.logger.info("Monitoring database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def save_component_metrics(self, metrics: ComponentMetrics):
        """Сохранение метрик компонента"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO component_metrics 
                    (component_name, timestamp, status, cpu_usage, memory_usage,
                     gpu_usage, gpu_memory, custom_metrics, error_count, uptime_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.component_name,
                    metrics.timestamp,
                    metrics.status.value,
                    metrics.cpu_usage,
                    metrics.memory_usage,
                    metrics.gpu_usage,
                    metrics.gpu_memory,
                    json.dumps(metrics.custom_metrics),
                    metrics.error_count,
                    metrics.uptime_seconds
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save component metrics: {e}")
    
    def save_alert(self, alert: SystemAlert):
        """Сохранение алерта"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO system_alerts 
                    (alert_id, timestamp, component, severity, message, details, resolved, resolution_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.alert_id,
                    alert.timestamp,
                    alert.component,
                    alert.severity.value,
                    alert.message,
                    json.dumps(alert.details),
                    alert.resolved,
                    alert.resolution_time
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save alert: {e}")
    
    def save_performance_snapshot(self, snapshot: PerformanceSnapshot):
        """Сохранение снапшота производительности"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO performance_snapshots
                    (timestamp, training_loss, validation_loss, attention_score,
                     model_quality, throughput, memory_efficiency)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot.timestamp,
                    snapshot.training_loss,
                    snapshot.validation_loss,
                    snapshot.attention_score,
                    snapshot.model_quality,
                    snapshot.throughput,
                    snapshot.memory_efficiency
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save performance snapshot: {e}")
    
    def get_component_metrics(self, component_name: str, hours: int = 24) -> List[ComponentMetrics]:
        """Получение метрик компонента за последние часы"""
        try:
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM component_metrics 
                    WHERE component_name = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (component_name, cutoff_time))
                
                metrics = []
                for row in cursor.fetchall():
                    metrics.append(ComponentMetrics(
                        component_name=row[1],
                        timestamp=row[2],
                        status=ComponentStatus(row[3]),
                        cpu_usage=row[4],
                        memory_usage=row[5],
                        gpu_usage=row[6],
                        gpu_memory=row[7],
                        custom_metrics=json.loads(row[8]) if row[8] else {},
                        error_count=row[9],
                        uptime_seconds=row[10]
                    ))
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Failed to get component metrics: {e}")
            return []
    
    def get_active_alerts(self) -> List[SystemAlert]:
        """Получение активных алертов"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM system_alerts 
                    WHERE resolved = FALSE
                    ORDER BY timestamp DESC
                ''')
                
                alerts = []
                for row in cursor.fetchall():
                    alerts.append(SystemAlert(
                        alert_id=row[1],
                        timestamp=row[2],
                        component=row[3],
                        severity=AlertSeverity(row[4]),
                        message=row[5],
                        details=json.loads(row[6]) if row[6] else {},
                        resolved=row[7],
                        resolution_time=row[8]
                    ))
                
                return alerts
                
        except Exception as e:
            self.logger.error(f"Failed to get active alerts: {e}")
            return []

class MetricsCollector:
    """Сборщик метрик системы"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.MetricsCollector")
        
        # Реестр компонентов
        self.components: Dict[str, Any] = {}
        self.component_start_times: Dict[str, datetime] = {}
        self.component_error_counts: Dict[str, int] = defaultdict(int)
        
        # Кэш метрик
        self.metrics_cache: Dict[str, ComponentMetrics] = {}
        
        # Блокировка для thread safety
        self._lock = threading.Lock()
    
    def register_component(self, name: str, component: Any):
        """Регистрация компонента для мониторинга"""
        with self._lock:
            self.components[name] = component
            self.component_start_times[name] = datetime.now()
            self.logger.info(f"Registered component for monitoring: {name}")
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Сбор системных метрик"""
        try:
            # CPU и память
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            metrics = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage': psutil.disk_usage('/').percent
            }
            
            # GPU метрики (если доступно)
            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.memory_stats()
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    
                    metrics.update({
                        'gpu_usage': (gpu_memory['allocated_bytes.all.current'] / total_memory) * 100,
                        'gpu_memory_allocated_gb': gpu_memory['allocated_bytes.all.current'] / (1024**3),
                        'gpu_memory_cached_gb': gpu_memory['reserved_bytes.all.current'] / (1024**3)
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to collect GPU metrics: {e}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return {}
    
    def collect_component_metrics(self, component_name: str) -> Optional[ComponentMetrics]:
        """Сбор метрик конкретного компонента"""
        try:
            if component_name not in self.components:
                return None
            
            component = self.components[component_name]
            system_metrics = self.collect_system_metrics()
            
            # Базовые метрики
            uptime = (datetime.now() - self.component_start_times[component_name]).total_seconds()
            error_count = self.component_error_counts[component_name]
            
            # Статус компонента
            status = self._determine_component_status(component, system_metrics)
            
            # Кастомные метрики компонента
            custom_metrics = {}
            if hasattr(component, 'get_monitoring_metrics'):
                try:
                    custom_metrics = component.get_monitoring_metrics()
                except Exception as e:
                    self.logger.warning(f"Failed to get custom metrics for {component_name}: {e}")
            
            metrics = ComponentMetrics(
                component_name=component_name,
                timestamp=datetime.now().isoformat(),
                status=status,
                cpu_usage=system_metrics.get('cpu_usage', 0.0),
                memory_usage=system_metrics.get('memory_usage', 0.0),
                gpu_usage=system_metrics.get('gpu_usage'),
                gpu_memory=system_metrics.get('gpu_memory_allocated_gb'),
                custom_metrics=custom_metrics,
                error_count=error_count,
                uptime_seconds=uptime
            )
            
            # Кэширование
            with self._lock:
                self.metrics_cache[component_name] = metrics
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics for {component_name}: {e}")
            return None
    
    def _determine_component_status(self, component: Any, system_metrics: Dict[str, float]) -> ComponentStatus:
        """Определение статуса компонента"""
        
        # Проверка системных ресурсов
        cpu_usage = system_metrics.get('cpu_usage', 0)
        memory_usage = system_metrics.get('memory_usage', 0)
        
        if cpu_usage > self.config.cpu_critical_threshold or memory_usage > self.config.memory_critical_threshold:
            return ComponentStatus.CRITICAL
        
        if cpu_usage > self.config.cpu_warning_threshold or memory_usage > self.config.memory_warning_threshold:
            return ComponentStatus.WARNING
        
        # Проверка статуса компонента
        if hasattr(component, 'is_healthy'):
            try:
                if not component.is_healthy():
                    return ComponentStatus.CRITICAL
            except Exception:
                return ComponentStatus.OFFLINE
        
        if hasattr(component, 'get_status'):
            try:
                status = component.get_status()
                if status in ['critical', 'error']:
                    return ComponentStatus.CRITICAL
                elif status in ['warning', 'degraded']:
                    return ComponentStatus.WARNING
            except Exception:
                pass
        
        return ComponentStatus.HEALTHY
    
    def increment_error_count(self, component_name: str):
        """Увеличение счетчика ошибок компонента"""
        with self._lock:
            self.component_error_counts[component_name] += 1

class AlertManager:
    """Менеджер алертов"""
    
    def __init__(self, config: MonitoringConfig, database: MonitoringDatabase):
        self.config = config
        self.database = database
        self.logger = logging.getLogger(f"{__name__}.AlertManager")
        
        # Активные алерты
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.alert_callbacks: List[Callable] = []
        
        # Блокировка для thread safety
        self._lock = threading.Lock()
    
    def add_alert_callback(self, callback: Callable):
        """Добавление callback для алертов"""
        self.alert_callbacks.append(callback)
    
    def check_metrics_for_alerts(self, metrics: ComponentMetrics):
        """Проверка метрик на наличие алертов"""
        
        alerts_to_trigger = []
        
        # Проверка CPU
        if metrics.cpu_usage > self.config.cpu_critical_threshold:
            alerts_to_trigger.append(self._create_alert(
                component=metrics.component_name,
                severity=AlertSeverity.CRITICAL,
                message=f"Critical CPU usage: {metrics.cpu_usage:.1f}%",
                details={"cpu_usage": metrics.cpu_usage, "threshold": self.config.cpu_critical_threshold}
            ))
        elif metrics.cpu_usage > self.config.cpu_warning_threshold:
            alerts_to_trigger.append(self._create_alert(
                component=metrics.component_name,
                severity=AlertSeverity.WARNING,
                message=f"High CPU usage: {metrics.cpu_usage:.1f}%",
                details={"cpu_usage": metrics.cpu_usage, "threshold": self.config.cpu_warning_threshold}
            ))
        
        # Проверка памяти
        if metrics.memory_usage > self.config.memory_critical_threshold:
            alerts_to_trigger.append(self._create_alert(
                component=metrics.component_name,
                severity=AlertSeverity.CRITICAL,
                message=f"Critical memory usage: {metrics.memory_usage:.1f}%",
                details={"memory_usage": metrics.memory_usage, "threshold": self.config.memory_critical_threshold}
            ))
        elif metrics.memory_usage > self.config.memory_warning_threshold:
            alerts_to_trigger.append(self._create_alert(
                component=metrics.component_name,
                severity=AlertSeverity.WARNING,
                message=f"High memory usage: {metrics.memory_usage:.1f}%",
                details={"memory_usage": metrics.memory_usage, "threshold": self.config.memory_warning_threshold}
            ))
        
        # Проверка GPU
        if metrics.gpu_usage and metrics.gpu_usage > self.config.gpu_critical_threshold:
            alerts_to_trigger.append(self._create_alert(
                component=metrics.component_name,
                severity=AlertSeverity.CRITICAL,
                message=f"Critical GPU usage: {metrics.gpu_usage:.1f}%",
                details={"gpu_usage": metrics.gpu_usage, "threshold": self.config.gpu_critical_threshold}
            ))
        
        # Проверка статуса компонента
        if metrics.status == ComponentStatus.CRITICAL:
            alerts_to_trigger.append(self._create_alert(
                component=metrics.component_name,
                severity=AlertSeverity.CRITICAL,
                message=f"Component is in critical state",
                details={"status": metrics.status.value}
            ))
        elif metrics.status == ComponentStatus.OFFLINE:
            alerts_to_trigger.append(self._create_alert(
                component=metrics.component_name,
                severity=AlertSeverity.EMERGENCY,
                message=f"Component is offline",
                details={"status": metrics.status.value}
            ))
        
        # Проверка ошибок
        if metrics.error_count > 10:  # Порог ошибок
            alerts_to_trigger.append(self._create_alert(
                component=metrics.component_name,
                severity=AlertSeverity.WARNING,
                message=f"High error count: {metrics.error_count}",
                details={"error_count": metrics.error_count}
            ))
        
        # Триггер алертов
        for alert in alerts_to_trigger:
            self.trigger_alert(alert)
    
    def _create_alert(self, component: str, severity: AlertSeverity, 
                     message: str, details: Dict[str, Any]) -> SystemAlert:
        """Создание алерта"""
        alert_id = f"{component}_{severity.value}_{int(time.time())}"
        
        return SystemAlert(
            alert_id=alert_id,
            timestamp=datetime.now().isoformat(),
            component=component,
            severity=severity,
            message=message,
            details=details,
            resolved=False,
            resolution_time=None
        )
    
    def trigger_alert(self, alert: SystemAlert):
        """Триггер алерта"""
        with self._lock:
            # Проверка дубликатов
            existing_key = f"{alert.component}_{alert.severity.value}"
            if existing_key in self.active_alerts:
                # Обновляем существующий алерт
                self.active_alerts[existing_key].timestamp = alert.timestamp
                self.active_alerts[existing_key].details.update(alert.details)
            else:
                # Новый алерт
                self.active_alerts[existing_key] = alert
                self.logger.warning(f"Alert triggered: {alert.message}")
        
        # Сохранение в базу данных
        self.database.save_alert(alert)
        
        # Вызов callback'ов
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def resolve_alert(self, alert_id: str):
        """Разрешение алерта"""
        with self._lock:
            # Поиск и разрешение
            for key, alert in self.active_alerts.items():
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolution_time = datetime.now().isoformat()
                    
                    # Сохранение в базу данных
                    self.database.save_alert(alert)
                    
                    # Удаление из активных
                    del self.active_alerts[key]
                    
                    self.logger.info(f"Alert resolved: {alert_id}")
                    break
    
    def get_active_alerts(self) -> List[SystemAlert]:
        """Получение активных алертов"""
        with self._lock:
            return list(self.active_alerts.values())

class ProductionMonitor:
    """Главный класс мониторинга производственной системы"""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(f"{__name__}.ProductionMonitor")
        
        # Инициализация компонентов
        self.database = MonitoringDatabase(self.config.database_path)
        self.metrics_collector = MetricsCollector(self.config)
        self.alert_manager = AlertManager(self.config, self.database)
        
        # Состояние мониторинга
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.alert_thread: Optional[threading.Thread] = None
        
        # Кэш данных для dashboard
        self.dashboard_data_cache: Dict[str, Any] = {}
        self.cache_lock = threading.Lock()
        
        self.logger.info("Production Monitor initialized")
    
    def register_component(self, name: str, component: Any):
        """Регистрация компонента для мониторинга"""
        self.metrics_collector.register_component(name, component)
    
    def start_monitoring(self):
        """Запуск мониторинга"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        # Поток сбора метрик
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="MonitoringThread"
        )
        self.monitoring_thread.start()
        
        # Поток проверки алертов
        self.alert_thread = threading.Thread(
            target=self._alert_loop,
            daemon=True,
            name="AlertThread"
        )
        self.alert_thread.start()
        
        self.logger.info("Monitoring started")
    
    def stop_monitoring(self):
        """Остановка мониторинга"""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        if self.alert_thread:
            self.alert_thread.join(timeout=5.0)
        
        self.logger.info("Monitoring stopped")
    
    def _monitoring_loop(self):
        """Основной цикл мониторинга"""
        while self.monitoring_active:
            try:
                # Сбор метрик для всех компонентов
                for component_name in self.config.monitored_components:
                    metrics = self.metrics_collector.collect_component_metrics(component_name)
                    
                    if metrics:
                        # Сохранение в базу данных
                        self.database.save_component_metrics(metrics)
                        
                        # Обновление кэша для dashboard
                        self._update_dashboard_cache(component_name, metrics)
                
                # Создание снапшота производительности
                self._create_performance_snapshot()
                
                time.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)  # Короткая задержка при ошибке
    
    def _alert_loop(self):
        """Цикл проверки алертов"""
        while self.monitoring_active:
            try:
                # Проверка алертов для всех компонентов
                for component_name in self.config.monitored_components:
                    metrics = self.metrics_collector.metrics_cache.get(component_name)
                    
                    if metrics:
                        self.alert_manager.check_metrics_for_alerts(metrics)
                
                time.sleep(self.config.alert_check_interval)
                
            except Exception as e:
                self.logger.error(f"Alert loop error: {e}")
                time.sleep(5)
    
    def _update_dashboard_cache(self, component_name: str, metrics: ComponentMetrics):
        """Обновление кэша данных для dashboard"""
        with self.cache_lock:
            if 'components' not in self.dashboard_data_cache:
                self.dashboard_data_cache['components'] = {}
            
            self.dashboard_data_cache['components'][component_name] = {
                'timestamp': metrics.timestamp,
                'status': metrics.status.value,
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'gpu_usage': metrics.gpu_usage,
                'error_count': metrics.error_count,
                'uptime_hours': metrics.uptime_seconds / 3600,
                'custom_metrics': metrics.custom_metrics
            }
            
            # Обновление общих метрик
            self.dashboard_data_cache['last_update'] = datetime.now().isoformat()
            self.dashboard_data_cache['active_alerts'] = len(self.alert_manager.get_active_alerts())
    
    def _create_performance_snapshot(self):
        """Создание снапшота производительности"""
        try:
            # Сбор метрик производительности из компонентов
            training_loss = None
            validation_loss = None
            attention_score = None
            model_quality = None
            throughput = None
            memory_efficiency = None
            
            # Получение метрик от зарегистрированных компонентов
            for component_name in self.config.monitored_components:
                component = self.metrics_collector.components.get(component_name)
                if component and hasattr(component, 'get_performance_metrics'):
                    try:
                        perf_metrics = component.get_performance_metrics()
                        
                        # Агрегация метрик
                        training_loss = perf_metrics.get('training_loss', training_loss)
                        validation_loss = perf_metrics.get('validation_loss', validation_loss)
                        attention_score = perf_metrics.get('attention_score', attention_score)
                        model_quality = perf_metrics.get('model_quality', model_quality)
                        throughput = perf_metrics.get('throughput', throughput)
                        memory_efficiency = perf_metrics.get('memory_efficiency', memory_efficiency)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to get performance metrics from {component_name}: {e}")
            
            # Создание снапшота
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now().isoformat(),
                training_loss=training_loss,
                validation_loss=validation_loss,
                attention_score=attention_score,
                model_quality=model_quality,
                throughput=throughput,
                memory_efficiency=memory_efficiency
            )
            
            # Сохранение в базу данных
            self.database.save_performance_snapshot(snapshot)
            
        except Exception as e:
            self.logger.error(f"Failed to create performance snapshot: {e}")
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Получение обзора системы"""
        with self.cache_lock:
            overview = {
                'monitoring_active': self.monitoring_active,
                'total_components': len(self.config.monitored_components),
                'healthy_components': 0,
                'warning_components': 0,
                'critical_components': 0,
                'offline_components': 0,
                'active_alerts': len(self.alert_manager.get_active_alerts()),
                'last_update': self.dashboard_data_cache.get('last_update'),
                'components': self.dashboard_data_cache.get('components', {})
            }
            
            # Подсчет статусов компонентов
            for component_data in overview['components'].values():
                status = component_data.get('status', 'offline')
                if status == 'healthy':
                    overview['healthy_components'] += 1
                elif status == 'warning':
                    overview['warning_components'] += 1
                elif status == 'critical':
                    overview['critical_components'] += 1
                else:
                    overview['offline_components'] += 1
            
            return overview
    
    def get_component_history(self, component_name: str, hours: int = 24) -> List[Dict]:
        """Получение истории метрик компонента"""
        metrics = self.database.get_component_metrics(component_name, hours)
        
        return [
            {
                'timestamp': m.timestamp,
                'status': m.status.value,
                'cpu_usage': m.cpu_usage,
                'memory_usage': m.memory_usage,
                'gpu_usage': m.gpu_usage,
                'error_count': m.error_count,
                'custom_metrics': m.custom_metrics
            }
            for m in metrics
        ]
    
    def add_alert_callback(self, callback: Callable):
        """Добавление callback для алертов"""
        self.alert_manager.add_alert_callback(callback)
    
    def force_alert_check(self):
        """Принудительная проверка алертов"""
        for component_name in self.config.monitored_components:
            metrics = self.metrics_collector.collect_component_metrics(component_name)
            if metrics:
                self.alert_manager.check_metrics_for_alerts(metrics)
    
    def cleanup_old_data(self):
        """Очистка старых данных"""
        try:
            cutoff_metrics = (datetime.now() - timedelta(days=self.config.metrics_retention_days)).isoformat()
            cutoff_alerts = (datetime.now() - timedelta(days=self.config.alerts_retention_days)).isoformat()
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                
                # Очистка старых метрик
                cursor.execute('DELETE FROM component_metrics WHERE timestamp < ?', (cutoff_metrics,))
                
                # Очистка старых алертов
                cursor.execute('DELETE FROM system_alerts WHERE timestamp < ? AND resolved = TRUE', (cutoff_alerts,))
                
                # Очистка старых снапшотов
                cursor.execute('DELETE FROM performance_snapshots WHERE timestamp < ?', (cutoff_metrics,))
                
                conn.commit()
                
                self.logger.info("Old monitoring data cleaned up")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")

# Удобные функции для интеграции
def create_production_monitor(config: Optional[MonitoringConfig] = None) -> ProductionMonitor:
    """Создание и настройка Production Monitor"""
    return ProductionMonitor(config)

def setup_monitoring_for_component(monitor: ProductionMonitor, component_name: str, component: Any):
    """Настройка мониторинга для компонента"""
    monitor.register_component(component_name, component)
    
    # Добавление методов мониторинга в компонент (если их нет)
    if not hasattr(component, 'get_monitoring_metrics'):
        def get_monitoring_metrics():
            return {}
        component.get_monitoring_metrics = get_monitoring_metrics
    
    if not hasattr(component, 'is_healthy'):
        def is_healthy():
            return True
        component.is_healthy = is_healthy

if __name__ == "__main__":
    # Демонстрация использования
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Создание монитора
    monitor = create_production_monitor()
    
    # Симуляция компонентов
    class MockComponent:
        def __init__(self, name):
            self.name = name
            self.healthy = True
        
        def get_monitoring_metrics(self):
            return {
                'requests_per_second': np.random.uniform(10, 100),
                'average_response_time': np.random.uniform(0.1, 1.0),
                'queue_size': np.random.randint(0, 50)
            }
        
        def is_healthy(self):
            return self.healthy
        
        def get_performance_metrics(self):
            return {
                'training_loss': np.random.uniform(0.5, 2.0),
                'model_quality': np.random.uniform(0.7, 0.95)
            }
    
    # Регистрация mock компонентов
    for component_name in ["training_stabilization", "attention_enhancement", "risk_assessment_module"]:
        component = MockComponent(component_name)
        setup_monitoring_for_component(monitor, component_name, component)
    
    # Запуск мониторинга
    monitor.start_monitoring()
    
    try:
        # Демонстрация работы
        time.sleep(5)
        
        print("=== System Overview ===")
        overview = monitor.get_system_overview()
        for key, value in overview.items():
            print(f"{key}: {value}")
        
        print("\n=== Active Alerts ===")
        alerts = monitor.alert_manager.get_active_alerts()
        for alert in alerts:
            print(f"[{alert.severity.value.upper()}] {alert.component}: {alert.message}")
        
        print("\n=== Component History ===")
        history = monitor.get_component_history("training_stabilization", hours=1)
        print(f"Found {len(history)} metrics entries for training_stabilization")
        
        time.sleep(2)
        
    finally:
        monitor.stop_monitoring()
    
    print("Production Monitoring demonstration completed!") 