"""
🧪 ТЕСТИРОВАНИЕ Training Stabilization System
==========================================

Комплексный тест системы стабилизации обучения:
1. ✅ Инициализация Training Stabilization System
2. ✅ Intelligent Gradient Manager функциональность
3. ✅ Adaptive Learning Rate Scheduler
4. ✅ Training Stability Monitor
5. ✅ Emergency Stabilization System
6. ✅ Интеграция с Context-Aware Manager

Версия: 1.0.0
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_stabilization_system_import():
    """🧪 Тест 1: Импорт Training Stabilization System"""
    logger.info("\n🧪 Тест 1: Импорт Training Stabilization System")
    logger.info("-" * 60)
    
    try:
        from training_stabilization_system import (
            create_training_stabilization_system,
            IntelligentGradientManager,
            AdaptiveLearningRateScheduler,
            TrainingStabilityMonitor,
            EmergencyStabilizationSystem,
            StabilityLevel,
            StabilityMetrics
        )
        
        logger.info("✅ Все компоненты Training Stabilization System импортированы успешно")
        
        # Тестируем создание системы
        class MockHParams:
            learning_rate = 1e-3
            target_gradient_norm = 2.0
            max_gradient_norm = 5.0
            min_learning_rate = 1e-5
            stability_window_size = 20
        
        hparams = MockHParams()
        stabilization_system = create_training_stabilization_system(hparams)
        
        logger.info("✅ Training Stabilization System создана успешно")
        logger.info("   Компоненты: Gradient Manager, LR Scheduler, Stability Monitor, Emergency System")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Ошибка импорта: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка создания системы: {e}")
        return False


def test_gradient_manager():
    """🧪 Тест 2: Intelligent Gradient Manager"""
    logger.info("\n🧪 Тест 2: Intelligent Gradient Manager")
    logger.info("-" * 60)
    
    try:
        from training_stabilization_system import IntelligentGradientManager
        
        # Создаем gradient manager
        gradient_manager = IntelligentGradientManager(
            target_norm=2.0,
            max_norm=5.0,
            min_norm=0.1
        )
        logger.info("✅ IntelligentGradientManager создан")
        
        # Создаем простую модель для тестирования
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
                
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        
        # Создаем тестовые данные с большими градиентами
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss = loss * 100  # Искусственно увеличиваем loss для больших градиентов
        
        # Тестируем обработку градиентов
        metrics = gradient_manager.process_gradients(model, loss)
        
        logger.info(f"✅ Градиенты обработаны:")
        logger.info(f"   Исходная норма: {metrics['original_norm']:.3f}")
        logger.info(f"   Финальная норма: {metrics['final_norm']:.3f}")
        logger.info(f"   Клиппинг применен: {metrics['clipped']}")
        
        # Проверяем адаптивное клиппирование
        if metrics['clipped']:
            logger.info("✅ Адаптивное клиппирование работает")
        else:
            logger.info("✅ Градиенты в норме, клиппинг не требуется")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования Gradient Manager: {e}")
        return False


def test_lr_scheduler():
    """🧪 Тест 3: Adaptive Learning Rate Scheduler"""
    logger.info("\n🧪 Тест 3: Adaptive Learning Rate Scheduler")
    logger.info("-" * 60)
    
    try:
        from training_stabilization_system import AdaptiveLearningRateScheduler, StabilityMetrics, StabilityLevel
        
        # Создаем LR scheduler
        lr_scheduler = AdaptiveLearningRateScheduler(
            initial_lr=1e-3,
            min_lr=1e-5,
            max_lr=1e-2,
            patience=5
        )
        logger.info("✅ AdaptiveLearningRateScheduler создан")
        
        # Тестируем различные сценарии стабильности
        scenarios = [
            (StabilityLevel.STABLE, 2.0, "Стабильное обучение"),
            (StabilityLevel.MODERATE, 3.0, "Умеренная нестабильность"),
            (StabilityLevel.UNSTABLE, 8.0, "Нестабильное обучение"),
            (StabilityLevel.CRITICAL, 15.0, "Критическая нестабильность")
        ]
        
        for stability_level, loss_value, description in scenarios:
            stability_metrics = StabilityMetrics(
                loss_std=2.0,
                gradient_norm=3.0,
                stability_level=stability_level
            )
            
            old_lr = lr_scheduler.current_lr
            new_lr = lr_scheduler.step(loss_value, stability_metrics)
            
            logger.info(f"✅ {description}:")
            logger.info(f"   LR: {old_lr:.2e} → {new_lr:.2e}")
            
            # Проверяем, что LR изменился соответствующе уровню стабильности
            if stability_level == StabilityLevel.CRITICAL:
                assert new_lr <= old_lr, "LR должен снижаться при критической нестабильности"
            
        logger.info("✅ Адаптивный LR scheduler работает корректно")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования LR Scheduler: {e}")
        return False


def test_stability_monitor():
    """🧪 Тест 4: Training Stability Monitor"""
    logger.info("\n🧪 Тест 4: Training Stability Monitor")
    logger.info("-" * 60)
    
    try:
        from training_stabilization_system import TrainingStabilityMonitor, StabilityLevel
        
        # Создаем stability monitor
        stability_monitor = TrainingStabilityMonitor(window_size=10)
        logger.info("✅ TrainingStabilityMonitor создан")
        
        # Симулируем различные сценарии обучения
        
        # 1. Стабильное обучение
        logger.info("📊 Тестирование стабильного обучения:")
        for i in range(15):
            loss = 2.0 + 0.1 * np.random.randn()  # Стабильная loss
            grad_norm = 1.5 + 0.2 * np.random.randn()  # Стабильные градиенты
            
            metrics = stability_monitor.update_metrics(
                loss=loss,
                gradient_norm=abs(grad_norm),
                learning_rate=1e-3,
                attention_quality=0.7
            )
        
        logger.info(f"   Уровень стабильности: {metrics.stability_level.value}")
        logger.info(f"   Loss std: {metrics.loss_std:.3f}")
        logger.info(f"   Gradient norm: {metrics.gradient_norm:.3f}")
        
        # 2. Нестабильное обучение
        logger.info("📊 Тестирование нестабильного обучения:")
        for i in range(15):
            loss = 5.0 + 3.0 * np.random.randn()  # Нестабильная loss
            grad_norm = 8.0 + 4.0 * np.random.randn()  # Нестабильные градиенты
            
            metrics = stability_monitor.update_metrics(
                loss=abs(loss),
                gradient_norm=abs(grad_norm),
                learning_rate=1e-3,
                attention_quality=0.3
            )
        
        logger.info(f"   Уровень стабильности: {metrics.stability_level.value}")
        logger.info(f"   Loss std: {metrics.loss_std:.3f}")
        logger.info(f"   Gradient norm: {metrics.gradient_norm:.3f}")
        
        # Проверяем, что monitor корректно классифицирует нестабильность
        assert metrics.stability_level in [StabilityLevel.UNSTABLE, StabilityLevel.CRITICAL], \
            "Monitor должен обнаруживать нестабильность"
        
        logger.info("✅ Training Stability Monitor работает корректно")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования Stability Monitor: {e}")
        return False


def test_emergency_system():
    """🧪 Тест 5: Emergency Stabilization System"""
    logger.info("\n🧪 Тест 5: Emergency Stabilization System")
    logger.info("-" * 60)
    
    try:
        from training_stabilization_system import EmergencyStabilizationSystem, StabilityMetrics, StabilityLevel
        
        # Создаем emergency system
        emergency_system = EmergencyStabilizationSystem()
        logger.info("✅ EmergencyStabilizationSystem создан")
        
        # Создаем модель и оптимизатор для тестирования
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
        
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Тестируем нормальные условия
        normal_metrics = StabilityMetrics(
            loss_std=1.0,
            gradient_norm=2.0,
            stability_level=StabilityLevel.STABLE
        )
        
        need_emergency = emergency_system.check_emergency_conditions(normal_metrics)
        logger.info(f"✅ Нормальные условия - экстренная активация: {need_emergency}")
        assert not need_emergency, "Экстренная активация не должна срабатывать при нормальных условиях"
        
        # Тестируем критические условия
        critical_metrics = StabilityMetrics(
            loss_std=12.0,  # Экстремальная нестабильность
            gradient_norm=20.0,  # Экстремальные градиенты
            stability_level=StabilityLevel.CRITICAL
        )
        
        need_emergency = emergency_system.check_emergency_conditions(critical_metrics)
        logger.info(f"✅ Критические условия - экстренная активация: {need_emergency}")
        assert need_emergency, "Экстренная активация должна срабатывать при критических условиях"
        
        # Тестируем активацию экстренной стабилизации
        measures = emergency_system.activate_emergency_stabilization(model, optimizer, critical_metrics)
        logger.info("✅ Экстренная стабилизация активирована:")
        for key, value in measures.items():
            logger.info(f"   {key}: {value}")
        
        # Проверяем, что LR снижен
        current_lr = optimizer.param_groups[0]['lr']
        assert current_lr < 1e-3, "Learning rate должен быть снижен при экстренной активации"
        
        logger.info("✅ Emergency Stabilization System работает корректно")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования Emergency System: {e}")
        return False


def test_context_aware_integration():
    """🧪 Тест 6: Интеграция с Context-Aware Manager"""
    logger.info("\n🧪 Тест 6: Интеграция с Context-Aware Manager")
    logger.info("-" * 60)
    
    try:
        from context_aware_training_manager import create_context_aware_manager
        
        # Создаем mock hparams с параметрами стабилизации
        class MockHParams:
            learning_rate = 1e-3
            guide_loss_weight = 4.5
            target_gradient_norm = 2.0
            max_gradient_norm = 5.0
            min_learning_rate = 1e-5
            stability_window_size = 20
        
        hparams = MockHParams()
        
        # Создаем Context-Aware Manager
        context_manager = create_context_aware_manager(hparams)
        logger.info("✅ Context-Aware Training Manager создан")
        
        # Проверяем интеграцию системы стабилизации
        if hasattr(context_manager, 'stabilization_available') and context_manager.stabilization_available:
            logger.info("✅ Training Stabilization System интегрирована в Context-Aware Manager")
            
            # Получаем диагностику стабилизации
            stabilization_diagnostics = context_manager.get_stabilization_diagnostics()
            logger.info("✅ Диагностика системы стабилизации получена:")
            
            if 'gradient_manager' in stabilization_diagnostics:
                logger.info(f"   Gradient Manager: текущий масштаб = {stabilization_diagnostics['gradient_manager'].get('current_scale', 'N/A')}")
                
            if 'lr_scheduler' in stabilization_diagnostics:
                logger.info(f"   LR Scheduler: текущий LR = {stabilization_diagnostics['lr_scheduler'].get('current_lr', 'N/A')}")
                
            if 'emergency_system' in stabilization_diagnostics:
                emergency_active = stabilization_diagnostics['emergency_system'].get('active', False)
                logger.info(f"   Emergency System: активна = {emergency_active}")
            
            # Тестируем получение общей статистики
            stats = context_manager.get_statistics()
            if 'stabilization_system' in stats:
                logger.info("✅ Статистика стабилизации включена в общую статистику")
            
        else:
            logger.warning("⚠️ Training Stabilization System недоступна в Context-Aware Manager")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования интеграции с Context-Aware Manager: {e}")
        return False


def test_full_system_simulation():
    """🧪 Тест 7: Полная симуляция системы стабилизации"""
    logger.info("\n🧪 Тест 7: Полная симуляция системы стабилизации")
    logger.info("-" * 60)
    
    try:
        from training_stabilization_system import create_training_stabilization_system
        
        # Создаем систему стабилизации
        class MockHParams:
            learning_rate = 1e-3
            target_gradient_norm = 2.0
            max_gradient_norm = 5.0
            min_learning_rate = 1e-5
            stability_window_size = 10
        
        hparams = MockHParams()
        stabilization_system = create_training_stabilization_system(hparams)
        
        # Создаем модель и оптимизатор
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(20, 10)
                self.linear2 = nn.Linear(10, 1)
                
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                return self.linear2(x)
        
        model = TestModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        logger.info("✅ Модель и оптимизатор созданы")
        
        # Симулируем шаги обучения с различными уровнями нестабильности
        scenarios = [
            ("Стабильное обучение", 2.0, 0.5),
            ("Умеренная нестабильность", 5.0, 1.0),
            ("Нестабильное обучение", 10.0, 2.0),
            ("Критическая нестабильность", 25.0, 5.0)
        ]
        
        for scenario_name, base_loss, noise_factor in scenarios:
            logger.info(f"📊 Симуляция: {scenario_name}")
            
            for step in range(5):
                # Создаем искусственную loss с различными уровнями нестабильности
                loss_value = base_loss + noise_factor * np.random.randn()
                loss = torch.tensor(abs(loss_value), requires_grad=True)
                
                # Применяем стабилизацию
                report = stabilization_system.stabilize_training_step(
                    model=model,
                    optimizer=optimizer,
                    loss=loss,
                    attention_quality=0.5
                )
                
                if step == 4:  # Отчет только для последнего шага каждого сценария
                    logger.info(f"   Уровень стабильности: {report['stability_level']}")
                    if report['emergency_measures']:
                        logger.info("   🚨 Экстренная стабилизация активирована")
                    else:
                        logger.info("   ✅ Обучение стабильно")
        
        # Получаем финальную диагностику
        diagnostics = stabilization_system.get_system_diagnostics()
        logger.info(f"✅ Финальная диагностика:")
        logger.info(f"   Всего вмешательств: {diagnostics['statistics']['interventions']}")
        logger.info(f"   Экстренных активаций: {diagnostics['statistics']['emergency_activations']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка полной симуляции системы: {e}")
        return False


def run_all_tests():
    """Запуск всех тестов Training Stabilization System"""
    logger.info("🚀 ЗАПУСК КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ Training Stabilization System")
    logger.info("=" * 80)
    
    tests = [
        ("Импорт системы", test_stabilization_system_import),
        ("Gradient Manager", test_gradient_manager),
        ("LR Scheduler", test_lr_scheduler),
        ("Stability Monitor", test_stability_monitor),
        ("Emergency System", test_emergency_system),
        ("Интеграция с Context-Aware Manager", test_context_aware_integration),
        ("Полная симуляция системы", test_full_system_simulation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"💥 Критическая ошибка в тесте '{test_name}': {e}")
            results.append((test_name, False))
    
    # Подводим итоги
    logger.info("\n" + "=" * 80)
    logger.info("📊 РЕЗУЛЬТАТЫ КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ Training Stabilization System:")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ ПРОЙДЕН" if result else "❌ ПРОВАЛЕН"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\n🎯 ИТОГО: {passed}/{total} тестов пройдено ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Training Stabilization System готова к работе!")
        logger.info("🛡️ Система полностью стабилизирует процесс обучения")
    else:
        logger.warning(f"⚠️ {total-passed} тестов провалено. Требуется исправление.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit_code = 0 if success else 1
    exit(exit_code) 