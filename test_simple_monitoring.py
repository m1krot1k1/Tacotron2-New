"""
Тесты для Simple Production Monitoring System

Упрощенные тесты системы мониторинга без внешних зависимостей.
"""

import unittest
import tempfile
import shutil
import os
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sqlite3
import json

from simple_monitoring import (
    SimpleProductionMonitor,
    MonitoringConfig,
    MonitoringDatabase,
    MetricsCollector,
    AlertManager,
    ComponentStatus,
    AlertSeverity,
    ComponentMetrics,
    SystemAlert,
    PerformanceSnapshot,
    MockSystemMonitor,
    create_simple_production_monitor,
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

class TestMockSystemMonitor(unittest.TestCase):
    """Тесты mock системного монитора"""
    
    def test_get_cpu_percent(self):
        """Тест получения процента CPU"""
        cpu_percent = MockSystemMonitor.get_cpu_percent()
        
        self.assertIsInstance(cpu_percent, float)
        self.assertGreaterEqual(cpu_percent, 0)
        self.assertLessEqual(cpu_percent, 100)
    
    def test_get_memory_info(self):
        """Тест получения информации о памяти"""
        memory_info = MockSystemMonitor.get_memory_info()
        
        self.assertIn('percent', memory_info)
        self.assertIn('available_gb', memory_info)
        self.assertIsInstance(memory_info['percent'], float)
        self.assertIsInstance(memory_info['available_gb'], float)
    
    def test_get_disk_usage(self):
        """Тест получения использования диска"""
        disk_usage = MockSystemMonitor.get_disk_usage()
        
        self.assertIsInstance(disk_usage, float)
        self.assertGreaterEqual(disk_usage, 0)
        self.assertLessEqual(disk_usage, 100)

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
    
    def test_collect_system_metrics(self):
        """Тест сбора системных метрик"""
        metrics = self.collector.collect_system_metrics()
        
        self.assertIn('cpu_usage', metrics)
        self.assertIn('memory_usage', metrics)
        self.assertIn('disk_usage', metrics)
        self.assertIsInstance(metrics['cpu_usage'], float)
        self.assertIsInstance(metrics['memory_usage'], float)
    
    def test_collect_component_metrics(self):
        """Тест сбора метрик компонента"""
        # Регистрация компонента
        component = MockComponent("test_component")
        self.collector.register_component("test_component", component)
        
        # Сбор метрик
        metrics = self.collector.collect_component_metrics("test_component")
        
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.component_name, "test_component")
        self.assertEqual(metrics.status, ComponentStatus.HEALTHY)
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

class TestSimpleProductionMonitor(unittest.TestCase):
    """Тесты главного класса мониторинга"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = MonitoringConfig(
            database_path=os.path.join(self.temp_dir, "test_monitoring.db"),
            monitored_components=["test_component"],
            metrics_collection_interval=1,  # Быстрый интервал для тестов
            alert_check_interval=1
        )
        self.monitor = SimpleProductionMonitor(self.config)
    
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
    
    def test_monitoring_loop_integration(self):
        """Интеграционный тест цикла мониторинга"""
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
    
    def test_create_simple_production_monitor(self):
        """Тест создания Simple Production Monitor"""
        config = MonitoringConfig(metrics_collection_interval=60)
        
        monitor = create_simple_production_monitor(config)
        
        self.assertIsInstance(monitor, SimpleProductionMonitor)
        self.assertEqual(monitor.config.metrics_collection_interval, 60)
    
    def test_setup_monitoring_for_component(self):
        """Тест настройки мониторинга для компонента"""
        monitor = create_simple_production_monitor()
        
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
        self.monitor = SimpleProductionMonitor(self.config)
    
    def tearDown(self):
        if self.monitor.monitoring_active:
            self.monitor.stop_monitoring()
        shutil.rmtree(self.temp_dir)
    
    def test_full_monitoring_scenario(self):
        """Тест полного сценария мониторинга"""
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
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Запуск тестов
    unittest.main(verbosity=2) 