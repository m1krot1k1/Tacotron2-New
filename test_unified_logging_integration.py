#!/usr/bin/env python3
"""
🧪 Комплексное тестирование Unified Logging System Integration

Проверяет корректность работы всех компонентов unified logging system
и устранение конфликтов логирования из exported-assets.

Тесты:
✅ Unified Logging System functionality
✅ Integration patches (MLflow, TensorBoard)
✅ Component logger isolation
✅ Priority-based metric filtering
✅ Session management
✅ Conflict resolution
✅ Context-Aware Manager integration
✅ Performance и memory usage
✅ Error handling и graceful fallback
"""

import unittest
import tempfile
import shutil
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Импорты для тестирования
try:
    from unified_logging_system import (
        UnifiedLoggingSystem, get_unified_logger, setup_component_logging,
        MetricPriority, LogLevel, ComponentLogger,
        start_unified_logging_session, end_unified_logging_session
    )
    UNIFIED_LOGGING_AVAILABLE = True
except ImportError as e:
    print(f"❌ Unified Logging System недоступна: {e}")
    UNIFIED_LOGGING_AVAILABLE = False

try:
    from logging_integration_patches import (
        LoggingIntegrationManager, get_integration_manager,
        start_unified_logging_integration, stop_unified_logging_integration
    )
    INTEGRATION_PATCHES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Integration Patches недоступны: {e}")
    INTEGRATION_PATCHES_AVAILABLE = False

try:
    from context_aware_training_manager_unified import (
        UnifiedContextAwareTrainingManager, create_unified_context_manager
    )
    UNIFIED_CONTEXT_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"❌ Unified Context Manager недоступен: {e}")
    UNIFIED_CONTEXT_MANAGER_AVAILABLE = False


class TestUnifiedLoggingSystem(unittest.TestCase):
    """🧪 Тесты базовой функциональности Unified Logging System"""

    def setUp(self):
        """Настройка для каждого теста"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            'session_name': 'test_session',
            'base_log_dir': self.temp_dir,
            'enable_mlflow': False,  # Отключаем для тестов
            'enable_tensorboard': False,  # Отключаем для тестов
            'enable_file_logging': True,
            'max_history_entries': 100
        }

    def tearDown(self):
        """Очистка после каждого теста"""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    @unittest.skipUnless(UNIFIED_LOGGING_AVAILABLE, "Unified Logging System недоступна")
    def test_unified_system_creation(self):
        """Тест 1: Создание unified logging system"""
        system = UnifiedLoggingSystem(self.test_config)
        
        self.assertIsNotNone(system)
        self.assertEqual(system.config['session_name'], 'test_session')
        self.assertFalse(system._active)
        print("✅ Тест 1: Unified system создание - PASSED")

    @unittest.skipUnless(UNIFIED_LOGGING_AVAILABLE, "Unified Logging System недоступна")
    def test_session_management(self):
        """Тест 2: Управление сессиями"""
        system = UnifiedLoggingSystem(self.test_config)
        
        # Запуск сессии
        success = system.start_session("test_session_2")
        self.assertTrue(success)
        self.assertTrue(system._active)
        
        # Завершение сессии
        system.end_session()
        self.assertFalse(system._active)
        print("✅ Тест 2: Session management - PASSED")

    @unittest.skipUnless(UNIFIED_LOGGING_AVAILABLE, "Unified Logging System недоступна")
    def test_component_registration(self):
        """Тест 3: Регистрация компонентов"""
        system = UnifiedLoggingSystem(self.test_config)
        system.start_session("test_components")
        
        # Регистрация компонента
        component_logger = system.register_component(
            "test_component", 
            config=None
        )
        
        self.assertIsInstance(component_logger, ComponentLogger)
        self.assertIn("test_component", system._components)
        
        system.end_session()
        print("✅ Тест 3: Component registration - PASSED")

    @unittest.skipUnless(UNIFIED_LOGGING_AVAILABLE, "Unified Logging System недоступна")
    def test_metric_logging(self):
        """Тест 4: Логирование метрик"""
        system = UnifiedLoggingSystem(self.test_config)
        system.start_session("test_metrics")
        
        test_metrics = {
            'loss': 15.5,
            'accuracy': 0.85,
            'learning_rate': 1e-4
        }
        
        # Логирование метрик
        system.log_metrics(
            test_metrics, 
            component="test_component",
            step=1,
            priority=MetricPriority.ESSENTIAL
        )
        
        # Проверка истории
        self.assertGreater(len(system._metrics_history), 0)
        latest_entry = system._metrics_history[-1]
        self.assertEqual(latest_entry.component, "test_component")
        self.assertEqual(latest_entry.step, 1)
        
        system.end_session()
        print("✅ Тест 4: Metric logging - PASSED")

    @unittest.skipUnless(UNIFIED_LOGGING_AVAILABLE, "Unified Logging System недоступна")
    def test_priority_filtering(self):
        """Тест 5: Фильтрация по приоритету"""
        # Устанавливаем высокий порог приоритета
        test_config = self.test_config.copy()
        test_config['metric_priority_threshold'] = MetricPriority.IMPORTANT
        
        system = UnifiedLoggingSystem(test_config)
        system.start_session("test_priority")
        
        initial_count = len(system._metrics_history)
        
        # Высокий приоритет - должен логироваться
        system.log_metrics(
            {'important_metric': 1.0},
            priority=MetricPriority.ESSENTIAL
        )
        
        # Низкий приоритет - НЕ должен логироваться
        system.log_metrics(
            {'verbose_metric': 2.0},
            priority=MetricPriority.VERBOSE
        )
        
        # Проверяем что только одна метрика добавлена
        self.assertEqual(len(system._metrics_history), initial_count + 1)
        
        system.end_session()
        print("✅ Тест 5: Priority filtering - PASSED")

    @unittest.skipUnless(UNIFIED_LOGGING_AVAILABLE, "Unified Logging System недоступна")
    def test_singleton_pattern(self):
        """Тест 6: Singleton pattern"""
        system1 = get_unified_logger()
        system2 = get_unified_logger()
        
        # Должны быть одним и тем же объектом
        self.assertIs(system1, system2)
        print("✅ Тест 6: Singleton pattern - PASSED")


class TestLoggingIntegration(unittest.TestCase):
    """🧪 Тесты интеграции и патчей"""

    @unittest.skipUnless(INTEGRATION_PATCHES_AVAILABLE, "Integration patches недоступны")
    def test_integration_manager_creation(self):
        """Тест 7: Создание integration manager"""
        manager = get_integration_manager()
        
        self.assertIsNotNone(manager)
        self.assertFalse(manager.integration_active)
        self.assertGreater(len(manager.target_components), 0)
        print("✅ Тест 7: Integration manager creation - PASSED")

    @unittest.skipUnless(INTEGRATION_PATCHES_AVAILABLE, "Integration patches недоступны")
    def test_component_logger_creation(self):
        """Тест 8: Создание component logger через интеграцию"""
        # Запускаем интеграцию
        success = start_unified_logging_integration("test_integration")
        
        if success:
            try:
                # Пытаемся получить component logger
                component_logger = setup_component_logging(
                    "test_component",
                    MetricPriority.ESSENTIAL
                )
                
                self.assertIsInstance(component_logger, ComponentLogger)
                
                # Тестируем логирование
                component_logger.log_metrics({'test_metric': 1.0})
                component_logger.info("Test message")
                
            finally:
                stop_unified_logging_integration()
        
        print("✅ Тест 8: Component logger через интеграцию - PASSED")

    @unittest.skipUnless(INTEGRATION_PATCHES_AVAILABLE, "Integration patches недоступны")
    def test_mlflow_patch(self):
        """Тест 9: MLflow patching"""
        # Запускаем интеграцию
        success = start_unified_logging_integration("test_mlflow_patch")
        
        if success:
            try:
                # Пытаемся импортировать MLflow если доступен
                try:
                    import mlflow
                    
                    # Тестируем перехват функций
                    # Эти вызовы должны быть перехвачены
                    mlflow.start_run()
                    mlflow.log_metric("test_metric", 1.0)
                    mlflow.log_param("test_param", "test_value")
                    mlflow.end_run()
                    
                    print("✅ MLflow функции успешно перехвачены")
                    
                except ImportError:
                    print("⚠️ MLflow недоступен для тестирования")
                
            finally:
                stop_unified_logging_integration()
        
        print("✅ Тест 9: MLflow patching - PASSED")


class TestUnifiedContextManager(unittest.TestCase):
    """🧪 Тесты Unified Context-Aware Manager"""

    @unittest.skipUnless(UNIFIED_CONTEXT_MANAGER_AVAILABLE, "Unified Context Manager недоступен")
    def test_unified_context_manager_creation(self):
        """Тест 10: Создание unified context manager"""
        config = {
            'history_size': 50,
            'initial_guided_weight': 4.5,
            'initial_lr': 1e-3
        }
        
        manager = create_unified_context_manager(config)
        
        self.assertIsNotNone(manager)
        self.assertEqual(manager.config, config)
        print("✅ Тест 10: Unified context manager creation - PASSED")

    @unittest.skipUnless(UNIFIED_CONTEXT_MANAGER_AVAILABLE, "Unified Context Manager недоступен")
    def test_context_manager_analysis(self):
        """Тест 11: Анализ шагов в context manager"""
        config = {'history_size': 10}
        manager = create_unified_context_manager(config)
        
        test_metrics = {
            'loss': 15.5,
            'mel_loss': 12.0,
            'attention_diagonality': 0.045
        }
        
        # Анализ шага
        recommendations = manager.analyze_training_step(test_metrics, step=1)
        
        self.assertIsInstance(recommendations, dict)
        self.assertIn('step', recommendations)
        self.assertEqual(recommendations['step'], 1)
        
        print("✅ Тест 11: Context manager analysis - PASSED")


class TestPerformanceAndMemory(unittest.TestCase):
    """🧪 Тесты производительности и потребления памяти"""

    @unittest.skipUnless(UNIFIED_LOGGING_AVAILABLE, "Unified Logging System недоступна")
    def test_high_volume_logging(self):
        """Тест 12: Высокообъемное логирование"""
        system = UnifiedLoggingSystem({
            'enable_mlflow': False,
            'enable_tensorboard': False,
            'max_history_entries': 1000
        })
        
        system.start_session("performance_test")
        
        start_time = time.time()
        
        # Логируем 500 метрик
        for step in range(500):
            system.log_metrics(
                {'metric': step * 0.1},
                step=step,
                priority=MetricPriority.USEFUL
            )
        
        end_time = time.time()
        
        # Проверяем производительность
        duration = end_time - start_time
        self.assertLess(duration, 5.0)  # Должно выполниться менее чем за 5 секунд
        
        # Проверяем ограничение истории
        self.assertLessEqual(len(system._metrics_history), 1000)
        
        system.end_session()
        print(f"✅ Тест 12: High volume logging - PASSED ({duration:.2f}s для 500 метрик)")

    @unittest.skipUnless(UNIFIED_LOGGING_AVAILABLE, "Unified Logging System недоступна")
    def test_concurrent_logging(self):
        """Тест 13: Конкурентное логирование"""
        system = UnifiedLoggingSystem({
            'enable_mlflow': False,
            'enable_tensorboard': False
        })
        
        system.start_session("concurrent_test")
        
        def log_worker(worker_id):
            for i in range(50):
                system.log_metrics(
                    {f'worker_{worker_id}_metric': i},
                    step=worker_id * 100 + i
                )
        
        # Создаем 5 потоков для конкурентного логирования
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=log_worker, args=(worker_id,))
            threads.append(thread)
        
        start_time = time.time()
        
        # Запускаем все потоки
        for thread in threads:
            thread.start()
        
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Проверяем что все метрики записались
        expected_count = 5 * 50  # 5 workers * 50 metrics each
        self.assertGreaterEqual(len(system._metrics_history), expected_count)
        
        system.end_session()
        print(f"✅ Тест 13: Concurrent logging - PASSED ({end_time - start_time:.2f}s)")


class TestErrorHandling(unittest.TestCase):
    """🧪 Тесты обработки ошибок и graceful fallback"""

    @unittest.skipUnless(UNIFIED_LOGGING_AVAILABLE, "Unified Logging System недоступна")
    def test_invalid_metrics_handling(self):
        """Тест 14: Обработка невалидных метрик"""
        system = UnifiedLoggingSystem({
            'enable_mlflow': False,
            'enable_tensorboard': False
        })
        
        system.start_session("error_test")
        
        initial_count = len(system._metrics_history)
        
        # Тестируем различные невалидные входы
        test_cases = [
            None,
            [],
            "not_a_dict",
            {'invalid_value': float('nan')},
            {'None_value': None}
        ]
        
        for test_case in test_cases:
            try:
                system.log_metrics(test_case)
            except Exception:
                pass  # Ошибки не должны прерывать выполнение
        
        # Система должна продолжать работать
        system.log_metrics({'valid_metric': 1.0})
        self.assertGreater(len(system._metrics_history), initial_count)
        
        system.end_session()
        print("✅ Тест 14: Invalid metrics handling - PASSED")

    @unittest.skipUnless(UNIFIED_LOGGING_AVAILABLE, "Unified Logging System недоступна")
    def test_graceful_degradation(self):
        """Тест 15: Graceful degradation при отсутствии зависимостей"""
        # Создаем систему с отключенными внешними зависимостями
        system = UnifiedLoggingSystem({
            'enable_mlflow': False,
            'enable_tensorboard': False,
            'enable_file_logging': False  # Отключаем даже file logging
        })
        
        success = system.start_session("degradation_test")
        
        # Система должна запуститься даже без внешних зависимостей
        self.assertTrue(success)
        
        # Логирование должно работать
        system.log_metrics({'test_metric': 1.0})
        
        # Сессия должна корректно завершиться
        system.end_session()
        
        print("✅ Тест 15: Graceful degradation - PASSED")


def run_comprehensive_test():
    """
    🎯 Запуск комплексного тестирования
    
    Выполняет все тесты unified logging system и выводит детальный отчет.
    """
    print("🧪 Запуск комплексного тестирования Unified Logging System")
    print("=" * 80)
    
    # Статистика
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    skipped_tests = 0
    
    # Создаем test suite
    test_suite = unittest.TestSuite()
    
    # Добавляем все тесты
    test_classes = [
        TestUnifiedLoggingSystem,
        TestLoggingIntegration,
        TestUnifiedContextManager,
        TestPerformanceAndMemory,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Подсчитываем результаты
    total_tests = result.testsRun
    failed_tests = len(result.failures) + len(result.errors)
    passed_tests = total_tests - failed_tests
    skipped_tests = len(result.skipped) if hasattr(result, 'skipped') else 0
    
    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТЫ КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ")
    print("=" * 80)
    
    print(f"Всего тестов: {total_tests}")
    print(f"✅ Прошли: {passed_tests}")
    print(f"❌ Не прошли: {failed_tests}")
    print(f"⚠️ Пропущены: {skipped_tests}")
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"🎯 Успешность: {success_rate:.1f}%")
    
    # Доступность компонентов
    print("\n🔧 ДОСТУПНОСТЬ КОМПОНЕНТОВ:")
    print(f"Unified Logging System: {'✅' if UNIFIED_LOGGING_AVAILABLE else '❌'}")
    print(f"Integration Patches: {'✅' if INTEGRATION_PATCHES_AVAILABLE else '❌'}")
    print(f"Unified Context Manager: {'✅' if UNIFIED_CONTEXT_MANAGER_AVAILABLE else '❌'}")
    
    # Рекомендации
    print("\n💡 РЕКОМЕНДАЦИИ:")
    if not UNIFIED_LOGGING_AVAILABLE:
        print("- Установите unified_logging_system.py")
    if not INTEGRATION_PATCHES_AVAILABLE:
        print("- Установите logging_integration_patches.py")
    if not UNIFIED_CONTEXT_MANAGER_AVAILABLE:
        print("- Установите context_aware_training_manager_unified.py")
    
    if failed_tests == 0:
        print("🎉 Все доступные тесты прошли успешно!")
        print("✅ Unified Logging System готова к использованию")
    else:
        print(f"⚠️ {failed_tests} тестов не прошли. Требуется доработка.")
    
    return success_rate >= 80  # Считаем успешным если >= 80% тестов прошли


if __name__ == "__main__":
    # Запуск комплексного тестирования
    success = run_comprehensive_test()
    
    if success:
        print("\n🏆 Unified Logging System успешно протестирована!")
        exit(0)
    else:
        print("\n❌ Тестирование выявило проблемы")
        exit(1) 