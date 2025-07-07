"""
Тесты для Production Monitoring System

Комплексное тестирование системы мониторинга для Enhanced Tacotron2 AI System.
"""

import unittest
import tempfile
import shutil
import os
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, call
import sqlite3
import json
import numpy as np

from production_monitoring import (
    ProductionMonitor,
    MonitoringConfig,
    MonitoringDatabase,
    MetricsCollector,
    AlertManager,
    ComponentStatus,
    AlertSeverity,
    ComponentMetrics,
    SystemAlert,
    PerformanceSnapshot,
    create_production_monitor,
    setup_monitoring_for_component
)

class TestMonitoringConfig(unittest.TestCase):
    """Тесты конфигурации мониторинга"""
    
    def test_default_config(self):
        """Тест дефолтной конфигурации"""
        config = MonitoringConfig()
        
        self.assertEqual(config.metrics_collection_interval, 30)
        self.assertEqual(config.alert_check_interval, 10)
        self.assertEqual(config.cpu_warning_threshold, 80.0)
        self.assertEqual(config.cpu_critical_threshold, 95.0)
        self.assertTrue(config.auto_refresh)
        self.assertIsNotNone(config.monitored_components)
        self.assertIn("training_stabilization", config.monitored_components)
    
    def test_custom_config(self):
        """Тест кастомной конфигурации"""
        config = MonitoringConfig(
            metrics_collection_interval=60,
            cpu_warning_threshold=70.0,
            monitored_components=["test_component"]
        )
        
        self.assertEqual(config.metrics_collection_interval, 60)
        self.assertEqual(config.cpu_warning_threshold, 70.0)
        self.assertEqual(config.monitored_components, ["test_component"])

class TestMonitoringDatabase(unittest.TestCase):
    """Тесты базы данных мониторинга"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_monitoring.db")
        self.database = MonitoringDatabase(self.db_path)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_database_initialization(self):
        """Тест инициализации базы данных"""
        self.assertTrue(os.path.exists(self.db_path))
        
        # Проверяем создание таблиц
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            self.assertIn('component_metrics', tables)
            self.assertIn('system_alerts', tables)
            self.assertIn('performance_snapshots', tables)
    
    def test_save_component_metrics(self):
        """Тест сохранения метрик компонента"""
        metrics = ComponentMetrics(
            component_name="test_component",
            timestamp=datetime.now().isoformat(),
            status=ComponentStatus.HEALTHY,
            cpu_usage=50.0,
            memory_usage=60.0,
            gpu_usage=70.0,
            gpu_memory=4.0,
            custom_metrics={"test_metric": 123.45},
            error_count=0,
            uptime_seconds=3600.0
        )
        
        self.database.save_component_metrics(metrics)
        
        # Проверяем сохранение
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM component_metrics")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
            
            cursor.execute("SELECT * FROM component_metrics")
            row = cursor.fetchone()
            self.assertEqual(row[1], "test_component")
            self.assertEqual(row[3], "healthy")
            self.assertEqual(row[4], 50.0)
    
    def test_save_alert(self):
        """Тест сохранения алерта"""
        alert = SystemAlert(
            alert_id="test_alert",
            timestamp=datetime.now().isoformat(),
            component="test_component",
            severity=AlertSeverity.WARNING,
            message="Test alert message",
            details={"cpu_usage": 85.0},
            resolved=False,
            resolution_time=None
        )
        
        self.database.save_alert(alert)
        
        # Проверяем сохранение
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM system_alerts")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
            
            cursor.execute("SELECT * FROM system_alerts")
            row = cursor.fetchone()
            self.assertEqual(row[1], "test_alert")
            self.assertEqual(row[3], "test_component")
            self.assertEqual(row[4], "warning")
    
    def test_save_performance_snapshot(self):
        """Тест сохранения снапшота производительности"""
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now().isoformat(),
            training_loss=1.5,
            validation_loss=1.8,
            attention_score=0.85,
            model_quality=0.9,
            throughput=100.0,
            memory_efficiency=0.75
        )
        
        self.database.save_performance_snapshot(snapshot)
        
        # Проверяем сохранение
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM performance_snapshots")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
            
            cursor.execute("SELECT * FROM performance_snapshots")
            row = cursor.fetchone()
            self.assertEqual(row[2], 1.5)  # training_loss
            self.assertEqual(row[3], 1.8)  # validation_loss
    
    def test_get_component_metrics(self):
        """Тест получения метрик компонента"""
        # Создаем несколько метрик
        for i in range(3):
            metrics = ComponentMetrics(
                component_name="test_component",
                timestamp=(datetime.now() - timedelta(hours=i)).isoformat(),
                status=ComponentStatus.HEALTHY,
                cpu_usage=50.0 + i,
                memory_usage=60.0,
                gpu_usage=None,
                gpu_memory=None,
                custom_metrics={},
                error_count=i,
                uptime_seconds=3600.0
            )
            self.database.save_component_metrics(metrics)
        
        # Получаем метрики
        retrieved_metrics = self.database.get_component_metrics("test_component", hours=24)
        
        self.assertEqual(len(retrieved_metrics), 3)
        self.assertEqual(retrieved_metrics[0].component_name, "test_component")
        
        # Проверяем сортировку по времени (DESC)
        self.assertGreater(retrieved_metrics[0].cpu_usage, retrieved_metrics[-1].cpu_usage)
    
    def test_get_active_alerts(self):
        """Тест получения активных алертов"""
        # Создаем активный и разрешенный алерты
        active_alert = SystemAlert(
            alert_id="active_alert",
            timestamp=datetime.now().isoformat(),
            component="test_component",
            severity=AlertSeverity.CRITICAL,
            message="Active alert",
            details={},
            resolved=False,
            resolution_time=None
        )
        
        resolved_alert = SystemAlert(
            alert_id="resolved_alert",
            timestamp=datetime.now().isoformat(),
            component="test_component",
            severity=AlertSeverity.WARNING,
            message="Resolved alert",
            details={},
            resolved=True,
            resolution_time=datetime.now().isoformat()
        )
        
        self.database.save_alert(active_alert)
        self.database.save_alert(resolved_alert)
        
        # Получаем только активные алерты
        active_alerts = self.database.get_active_alerts()
        
        self.assertEqual(len(active_alerts), 1)
        self.assertEqual(active_alerts[0].alert_id, "active_alert")
        self.assertFalse(active_alerts[0].resolved)

class MockComponent:
    """Mock компонент для тестирования"""
    
    def __init__(self, name, healthy=True):
        self.name = name
        self.healthy = healthy
        self.custom_metrics = {"mock_metric": 42.0}
        self.performance_metrics = {"model_quality": 0.85}
    
    def get_monitoring_metrics(self):
        return self.custom_metrics
    
    def is_healthy(self):
        return self.healthy
    
    def get_performance_metrics(self):
        return self.performance_metrics

class TestMetricsCollector(unittest.TestCase):
    """Тесты сборщика метрик"""
    
    def setUp(self):
        self.config = MonitoringConfig(
            monitored_components=["test_component"],
            cpu_warning_threshold=80.0
        )
        self.collector = MetricsCollector(self.config)
    
    def test_register_component(self):
        """Тест регистрации компонента"""
        component = MockComponent("test_component")
        
        self.collector.register_component("test_component", component)
        
        self.assertIn("test_component", self.collector.components)
        self.assertIn("test_component", self.collector.component_start_times)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_collect_system_metrics(self, mock_disk, mock_memory, mock_cpu):
        """Тест сбора системных метрик"""
        # Настройка моков
        mock_cpu.return_value = 75.0
        mock_memory.return_value = MagicMock(percent=65.0, available=8*1024**3)
        mock_disk.return_value = MagicMock(percent=45.0)
        
        metrics = self.collector.collect_system_metrics()
        
        self.assertEqual(metrics['cpu_usage'], 75.0)
        self.assertEqual(metrics['memory_usage'], 65.0)
        self.assertEqual(metrics['disk_usage'], 45.0)
        self.assertAlmostEqual(metrics['memory_available_gb'], 8.0, places=1)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_collect_component_metrics(self, mock_disk, mock_memory, mock_cpu):
        """Тест сбора метрик компонента"""
        # Настройка моков
        mock_cpu.return_value = 75.0
        mock_memory.return_value = MagicMock(percent=65.0, available=8*1024**3)
        mock_disk.return_value = MagicMock(percent=45.0)
        
        # Регистрация компонента
        component = MockComponent("test_component")
        self.collector.register_component("test_component", component)
        
        # Сбор метрик
        metrics = self.collector.collect_component_metrics("test_component")
        
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.component_name, "test_component")
        self.assertEqual(metrics.status, ComponentStatus.HEALTHY)
        self.assertEqual(metrics.cpu_usage, 75.0)
        self.assertEqual(metrics.memory_usage, 65.0)
        self.assertEqual(metrics.custom_metrics, {"mock_metric": 42.0})
        self.assertGreater(metrics.uptime_seconds, 0)
    
    def test_determine_component_status_healthy(self):
        """Тест определения здорового статуса компонента"""
        component = MockComponent("test", healthy=True)
        system_metrics = {"cpu_usage": 50.0, "memory_usage": 60.0}
        
        status = self.collector._determine_component_status(component, system_metrics)
        
        self.assertEqual(status, ComponentStatus.HEALTHY)
    
    def test_determine_component_status_critical_resources(self):
        """Тест определения критического статуса из-за ресурсов"""
        component = MockComponent("test", healthy=True)
        system_metrics = {"cpu_usage": 98.0, "memory_usage": 60.0}  # Превышение CPU
        
        status = self.collector._determine_component_status(component, system_metrics)
        
        self.assertEqual(status, ComponentStatus.CRITICAL)
    
    def test_determine_component_status_warning_resources(self):
        """Тест определения статуса предупреждения из-за ресурсов"""
        component = MockComponent("test", healthy=True)
        system_metrics = {"cpu_usage": 85.0, "memory_usage": 60.0}  # Превышение порога предупреждения
        
        status = self.collector._determine_component_status(component, system_metrics)
        
        self.assertEqual(status, ComponentStatus.WARNING)
    
    def test_determine_component_status_unhealthy_component(self):
        """Тест определения критического статуса нездорового компонента"""
        component = MockComponent("test", healthy=False)
        system_metrics = {"cpu_usage": 50.0, "memory_usage": 60.0}
        
        status = self.collector._determine_component_status(component, system_metrics)
        
        self.assertEqual(status, ComponentStatus.CRITICAL)
    
    def test_increment_error_count(self):
        """Тест увеличения счетчика ошибок"""
        self.assertEqual(self.collector.component_error_counts["test_component"], 0)
        
        self.collector.increment_error_count("test_component")
        self.collector.increment_error_count("test_component")
        
        self.assertEqual(self.collector.component_error_counts["test_component"], 2)

class TestAlertManager(unittest.TestCase):
    """Тесты менеджера алертов"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_monitoring.db")
        self.config = MonitoringConfig(
            cpu_warning_threshold=80.0,
            cpu_critical_threshold=95.0,
            memory_warning_threshold=85.0,
            memory_critical_threshold=95.0
        )
        self.database = MonitoringDatabase(self.db_path)
        self.alert_manager = AlertManager(self.config, self.database)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_check_metrics_cpu_critical(self):
        """Тест проверки критического использования CPU"""
        metrics = ComponentMetrics(
            component_name="test_component",
            timestamp=datetime.now().isoformat(),
            status=ComponentStatus.HEALTHY,
            cpu_usage=98.0,  # Критический уровень
            memory_usage=50.0,
            gpu_usage=None,
            gpu_memory=None,
            custom_metrics={},
            error_count=0,
            uptime_seconds=3600.0
        )
        
        self.alert_manager.check_metrics_for_alerts(metrics)
        
        # Проверяем что алерт был создан
        active_alerts = self.alert_manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 1)
        self.assertEqual(active_alerts[0].severity, AlertSeverity.CRITICAL)
        self.assertIn("CPU usage", active_alerts[0].message)
    
    def test_check_metrics_memory_warning(self):
        """Тест проверки предупреждения об использовании памяти"""
        metrics = ComponentMetrics(
            component_name="test_component",
            timestamp=datetime.now().isoformat(),
            status=ComponentStatus.HEALTHY,
            cpu_usage=50.0,
            memory_usage=88.0,  # Уровень предупреждения
            gpu_usage=None,
            gpu_memory=None,
            custom_metrics={},
            error_count=0,
            uptime_seconds=3600.0
        )
        
        self.alert_manager.check_metrics_for_alerts(metrics)
        
        # Проверяем что алерт был создан
        active_alerts = self.alert_manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 1)
        self.assertEqual(active_alerts[0].severity, AlertSeverity.WARNING)
        self.assertIn("memory usage", active_alerts[0].message)
    
    def test_check_metrics_component_critical(self):
        """Тест проверки критического состояния компонента"""
        metrics = ComponentMetrics(
            component_name="test_component",
            timestamp=datetime.now().isoformat(),
            status=ComponentStatus.CRITICAL,  # Критическое состояние
            cpu_usage=50.0,
            memory_usage=60.0,
            gpu_usage=None,
            gpu_memory=None,
            custom_metrics={},
            error_count=0,
            uptime_seconds=3600.0
        )
        
        self.alert_manager.check_metrics_for_alerts(metrics)
        
        # Проверяем что алерт был создан
        active_alerts = self.alert_manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 1)
        self.assertEqual(active_alerts[0].severity, AlertSeverity.CRITICAL)
        self.assertIn("critical state", active_alerts[0].message)
    
    def test_check_metrics_high_error_count(self):
        """Тест проверки высокого количества ошибок"""
        metrics = ComponentMetrics(
            component_name="test_component",
            timestamp=datetime.now().isoformat(),
            status=ComponentStatus.HEALTHY,
            cpu_usage=50.0,
            memory_usage=60.0,
            gpu_usage=None,
            gpu_memory=None,
            custom_metrics={},
            error_count=15,  # Высокое количество ошибок
            uptime_seconds=3600.0
        )
        
        self.alert_manager.check_metrics_for_alerts(metrics)
        
        # Проверяем что алерт был создан
        active_alerts = self.alert_manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 1)
        self.assertEqual(active_alerts[0].severity, AlertSeverity.WARNING)
        self.assertIn("error count", active_alerts[0].message)
    
    def test_trigger_alert(self):
        """Тест триггера алерта"""
        alert = SystemAlert(
            alert_id="test_alert",
            timestamp=datetime.now().isoformat(),
            component="test_component",
            severity=AlertSeverity.WARNING,
            message="Test alert",
            details={"test": "data"},
            resolved=False,
            resolution_time=None
        )
        
        self.alert_manager.trigger_alert(alert)
        
        # Проверяем что алерт добавлен в активные
        active_alerts = self.alert_manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 1)
        self.assertEqual(active_alerts[0].alert_id, "test_alert")
    
    def test_resolve_alert(self):
        """Тест разрешения алерта"""
        # Создаем алерт
        alert = SystemAlert(
            alert_id="test_alert",
            timestamp=datetime.now().isoformat(),
            component="test_component",
            severity=AlertSeverity.WARNING,
            message="Test alert",
            details={},
            resolved=False,
            resolution_time=None
        )
        
        self.alert_manager.trigger_alert(alert)
        
        # Разрешаем алерт
        self.alert_manager.resolve_alert("test_alert")
        
        # Проверяем что алерт больше не активен
        active_alerts = self.alert_manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 0)
    
    def test_alert_callbacks(self):
        """Тест callback'ов алертов"""
        callback_called = False
        callback_alert = None
        
        def test_callback(alert):
            nonlocal callback_called, callback_alert
            callback_called = True
            callback_alert = alert
        
        self.alert_manager.add_alert_callback(test_callback)
        
        # Триггер алерта
        alert = SystemAlert(
            alert_id="test_alert",
            timestamp=datetime.now().isoformat(),
            component="test_component",
            severity=AlertSeverity.CRITICAL,
            message="Test alert",
            details={},
            resolved=False,
            resolution_time=None
        )
        
        self.alert_manager.trigger_alert(alert)
        
        # Проверяем что callback был вызван
        self.assertTrue(callback_called)
        self.assertEqual(callback_alert.alert_id, "test_alert")

class TestProductionMonitor(unittest.TestCase):
    """Тесты главного класса мониторинга"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = MonitoringConfig(
            database_path=os.path.join(self.temp_dir, "test_monitoring.db"),
            monitored_components=["test_component"],
            metrics_collection_interval=1,  # Быстрый интервал для тестов
            alert_check_interval=1
        )
        self.monitor = ProductionMonitor(self.config)
    
    def tearDown(self):
        if self.monitor.monitoring_active:
            self.monitor.stop_monitoring()
        shutil.rmtree(self.temp_dir)
    
    def test_register_component(self):
        """Тест регистрации компонента"""
        component = MockComponent("test_component")
        
        self.monitor.register_component("test_component", component)
        
        self.assertIn("test_component", self.monitor.metrics_collector.components)
    
    def test_start_stop_monitoring(self):
        """Тест запуска и остановки мониторинга"""
        self.assertFalse(self.monitor.monitoring_active)
        
        # Запуск мониторинга
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.monitoring_active)
        self.assertIsNotNone(self.monitor.monitoring_thread)
        self.assertIsNotNone(self.monitor.alert_thread)
        
        # Остановка мониторинга
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.monitoring_active)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_monitoring_loop_integration(self, mock_disk, mock_memory, mock_cpu):
        """Интеграционный тест цикла мониторинга"""
        # Настройка моков
        mock_cpu.return_value = 75.0
        mock_memory.return_value = MagicMock(percent=65.0, available=8*1024**3)
        mock_disk.return_value = MagicMock(percent=45.0)
        
        # Регистрация компонента
        component = MockComponent("test_component")
        self.monitor.register_component("test_component", component)
        
        # Запуск мониторинга на короткое время
        self.monitor.start_monitoring()
        time.sleep(2)  # Даем время для сбора метрик
        self.monitor.stop_monitoring()
        
        # Проверяем что метрики были собраны
        self.assertIn("test_component", self.monitor.metrics_collector.metrics_cache)
        
        # Проверяем обновление cache для dashboard
        self.assertIn("components", self.monitor.dashboard_data_cache)
        self.assertIn("test_component", self.monitor.dashboard_data_cache["components"])
    
    def test_get_system_overview(self):
        """Тест получения обзора системы"""
        # Регистрация и создание метрик
        component = MockComponent("test_component")
        self.monitor.register_component("test_component", component)
        
        # Создание fake метрик в cache
        fake_metrics = {
            "test_component": {
                "timestamp": datetime.now().isoformat(),
                "status": "healthy",
                "cpu_usage": 50.0,
                "memory_usage": 60.0,
                "gpu_usage": None,
                "error_count": 0,
                "uptime_hours": 1.0,
                "custom_metrics": {}
            }
        }
        
        with self.monitor.cache_lock:
            self.monitor.dashboard_data_cache["components"] = fake_metrics
            self.monitor.dashboard_data_cache["last_update"] = datetime.now().isoformat()
        
        overview = self.monitor.get_system_overview()
        
        self.assertEqual(overview["total_components"], 1)
        self.assertEqual(overview["healthy_components"], 1)
        self.assertEqual(overview["warning_components"], 0)
        self.assertEqual(overview["critical_components"], 0)
        self.assertIn("test_component", overview["components"])
    
    def test_get_component_history(self):
        """Тест получения истории метрик компонента"""
        # Создаем несколько метрик в базе данных
        for i in range(3):
            metrics = ComponentMetrics(
                component_name="test_component",
                timestamp=(datetime.now() - timedelta(hours=i)).isoformat(),
                status=ComponentStatus.HEALTHY,
                cpu_usage=50.0 + i,
                memory_usage=60.0,
                gpu_usage=None,
                gpu_memory=None,
                custom_metrics={"test": i},
                error_count=0,
                uptime_seconds=3600.0
            )
            self.monitor.database.save_component_metrics(metrics)
        
        history = self.monitor.get_component_history("test_component", hours=24)
        
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["custom_metrics"]["test"], 0)  # Самая новая запись
    
    def test_force_alert_check(self):
        """Тест принудительной проверки алертов"""
        # Регистрация компонента с критическими метриками
        component = MockComponent("test_component", healthy=False)  # Нездоровый компонент
        self.monitor.register_component("test_component", component)
        
        # Принудительная проверка алертов
        self.monitor.force_alert_check()
        
        # Проверяем что алерт был создан
        active_alerts = self.monitor.alert_manager.get_active_alerts()
        self.assertGreater(len(active_alerts), 0)
    
    def test_cleanup_old_data(self):
        """Тест очистки старых данных"""
        # Создаем старые метрики
        old_timestamp = (datetime.now() - timedelta(days=35)).isoformat()
        
        old_metrics = ComponentMetrics(
            component_name="test_component",
            timestamp=old_timestamp,
            status=ComponentStatus.HEALTHY,
            cpu_usage=50.0,
            memory_usage=60.0,
            gpu_usage=None,
            gpu_memory=None,
            custom_metrics={},
            error_count=0,
            uptime_seconds=3600.0
        )
        
        self.monitor.database.save_component_metrics(old_metrics)
        
        # Проверяем что данные есть
        with sqlite3.connect(self.monitor.database.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM component_metrics")
            count_before = cursor.fetchone()[0]
            self.assertEqual(count_before, 1)
        
        # Очистка старых данных
        self.monitor.cleanup_old_data()
        
        # Проверяем что старые данные удалены
        with sqlite3.connect(self.monitor.database.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM component_metrics")
            count_after = cursor.fetchone()[0]
            self.assertEqual(count_after, 0)

class TestUtilityFunctions(unittest.TestCase):
    """Тесты вспомогательных функций"""
    
    def test_create_production_monitor(self):
        """Тест создания Production Monitor"""
        config = MonitoringConfig(metrics_collection_interval=60)
        
        monitor = create_production_monitor(config)
        
        self.assertIsInstance(monitor, ProductionMonitor)
        self.assertEqual(monitor.config.metrics_collection_interval, 60)
    
    def test_setup_monitoring_for_component(self):
        """Тест настройки мониторинга для компонента"""
        monitor = create_production_monitor()
        
        # Создаем компонент без методов мониторинга
        class SimpleComponent:
            pass
        
        component = SimpleComponent()
        
        # Настройка мониторинга
        setup_monitoring_for_component(monitor, "simple_component", component)
        
        # Проверяем что компонент зарегистрирован
        self.assertIn("simple_component", monitor.metrics_collector.components)
        
        # Проверяем что методы мониторинга добавлены
        self.assertTrue(hasattr(component, 'get_monitoring_metrics'))
        self.assertTrue(hasattr(component, 'is_healthy'))
        
        # Проверяем что методы работают
        self.assertEqual(component.get_monitoring_metrics(), {})
        self.assertTrue(component.is_healthy())

class TestIntegrationScenarios(unittest.TestCase):
    """Интеграционные тесты реальных сценариев"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = MonitoringConfig(
            database_path=os.path.join(self.temp_dir, "test_monitoring.db"),
            monitored_components=["training_system", "risk_module"],
            metrics_collection_interval=1,
            alert_check_interval=1,
            cpu_critical_threshold=90.0
        )
        self.monitor = ProductionMonitor(self.config)
    
    def tearDown(self):
        if self.monitor.monitoring_active:
            self.monitor.stop_monitoring()
        shutil.rmtree(self.temp_dir)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_full_monitoring_scenario(self, mock_disk, mock_memory, mock_cpu):
        """Тест полного сценария мониторинга"""
        
        # Настройка моков для нормальных условий
        mock_cpu.return_value = 75.0
        mock_memory.return_value = MagicMock(percent=65.0, available=8*1024**3)
        mock_disk.return_value = MagicMock(percent=45.0)
        
        # Создаем mock компоненты
        training_system = MockComponent("training_system")
        risk_module = MockComponent("risk_module")
        
        # Регистрация компонентов
        self.monitor.register_component("training_system", training_system)
        self.monitor.register_component("risk_module", risk_module)
        
        # Запуск мониторинга
        self.monitor.start_monitoring()
        
        # Даем время для сбора нескольких метрик
        time.sleep(3)
        
        # Проверяем что система работает
        overview = self.monitor.get_system_overview()
        self.assertEqual(overview["total_components"], 2)
        self.assertTrue(overview["monitoring_active"])
        
        # Симулируем проблему с компонентом
        training_system.healthy = False
        
        # Принудительная проверка алертов
        self.monitor.force_alert_check()
        
        # Проверяем что алерт создан
        active_alerts = self.monitor.alert_manager.get_active_alerts()
        self.assertGreater(len(active_alerts), 0)
        
        # Исправляем проблему
        training_system.healthy = True
        
        # Разрешаем алерт
        if active_alerts:
            self.monitor.alert_manager.resolve_alert(active_alerts[0].alert_id)
        
        # Проверяем что алерт разрешен
        remaining_alerts = self.monitor.alert_manager.get_active_alerts()
        self.assertEqual(len(remaining_alerts), 0)
        
        # Остановка мониторинга
        self.monitor.stop_monitoring()
        
        # Проверяем что данные сохранены в базе
        history = self.monitor.get_component_history("training_system", hours=1)
        self.assertGreater(len(history), 0)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_resource_stress_scenario(self, mock_disk, mock_memory, mock_cpu):
        """Тест сценария нагрузки на ресурсы"""
        
        # Регистрация компонента
        component = MockComponent("test_component")
        self.monitor.register_component("test_component", component)
        
        alert_triggered = False
        alert_data = None
        
        def alert_callback(alert):
            nonlocal alert_triggered, alert_data
            alert_triggered = True
            alert_data = alert
        
        self.monitor.add_alert_callback(alert_callback)
        
        # Симулируем нормальную нагрузку
        mock_cpu.return_value = 70.0
        mock_memory.return_value = MagicMock(percent=60.0, available=8*1024**3)
        mock_disk.return_value = MagicMock(percent=40.0)
        
        self.monitor.force_alert_check()
        
        # Алерт не должен быть создан
        self.assertFalse(alert_triggered)
        
        # Симулируем критическую нагрузку
        mock_cpu.return_value = 95.0  # Критический уровень CPU
        
        self.monitor.force_alert_check()
        
        # Алерт должен быть создан
        self.assertTrue(alert_triggered)
        self.assertEqual(alert_data.severity, AlertSeverity.CRITICAL)
        self.assertIn("CPU", alert_data.message)
    
    def test_performance_snapshot_creation(self):
        """Тест создания снапшотов производительности"""
        
        # Создаем компонент с метриками производительности
        class PerformanceComponent:
            def get_monitoring_metrics(self):
                return {"requests_per_second": 100}
            
            def is_healthy(self):
                return True
            
            def get_performance_metrics(self):
                return {
                    "training_loss": 1.5,
                    "model_quality": 0.85,
                    "throughput": 150.0
                }
        
        component = PerformanceComponent()
        self.monitor.register_component("perf_component", component)
        
        # Создание снапшота
        self.monitor._create_performance_snapshot()
        
        # Проверяем что снапшот сохранен в базе данных
        with sqlite3.connect(self.monitor.database.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM performance_snapshots")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
            
            cursor.execute("SELECT * FROM performance_snapshots")
            row = cursor.fetchone()
            self.assertEqual(row[2], 1.5)  # training_loss

if __name__ == '__main__':
    # Настройка логирования для тестов
    logging.basicConfig(level=logging.WARNING)
    
    # Запуск тестов
    unittest.main(verbosity=2) 