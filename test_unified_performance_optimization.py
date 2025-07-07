#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 COMPREHENSIVE TESTS: Unified Performance Optimization System
Комплексное тестирование единой системы оптимизации производительности

Тестирует:
1. SystemProfiler - профилирование системы и обнаружение bottleneck'ов
2. PerformanceOptimizer - генерация рекомендаций по оптимизации
3. AdaptiveParameterController - адаптивное управление параметрами
4. UnifiedPerformanceOptimizationSystem - интеграцию всех компонентов
5. Интеграцию с существующими системами
6. Экстренную оптимизацию при критических условиях
"""

import os
import sys
import time
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Добавляем current directory в path для импортов
sys.path.insert(0, os.getcwd())

try:
    from unified_performance_optimization_system import (
        UnifiedPerformanceOptimizationSystem,
        SystemProfiler,
        PerformanceOptimizer,
        AdaptiveParameterController,
        PerformanceMetrics,
        OptimizationRecommendation,
        OptimizationPriority,
        PerformanceMetricType,
        create_performance_optimization_system
    )
    OPTIMIZATION_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"❌ Unified Performance Optimization System недоступна: {e}")
    OPTIMIZATION_SYSTEM_AVAILABLE = False

try:
    from hparams import create_hparams
    HPARAMS_AVAILABLE = True
except ImportError:
    HPARAMS_AVAILABLE = False


class MockHparams:
    """Mock объект для hparams"""
    def __init__(self):
        self.learning_rate = 1e-3
        self.batch_size = 16
        self.gradient_accumulation_steps = 1
        self.attention_dropout = 0.1
        self.decoder_dropout = 0.1
        self.gradient_clip_thresh = 1.0


def test_system_profiler():
    """Тестирование SystemProfiler"""
    print("\n🧪 ТЕСТ 1: SystemProfiler")
    
    try:
        profiler = SystemProfiler()
        
        # Тестирование профилирования системы
        metrics = profiler.profile_system_performance()
        
        # Проверка базовых метрик
        assert isinstance(metrics, PerformanceMetrics), "Метрики должны быть типа PerformanceMetrics"
        assert metrics.timestamp > 0, "Timestamp должен быть установлен"
        assert 0 <= metrics.cpu_usage <= 100, f"CPU usage должен быть в диапазоне 0-100, получен: {metrics.cpu_usage}"
        assert 0 <= metrics.memory_usage <= 100, f"Memory usage должен быть в диапазоне 0-100, получен: {metrics.memory_usage}"
        assert metrics.memory_available_gb >= 0, "Доступная память должна быть >= 0"
        
        # Проверка эффективности
        assert 0 <= metrics.memory_efficiency <= 100, "Memory efficiency должна быть 0-100%"
        assert 0 <= metrics.compute_efficiency <= 100, "Compute efficiency должна быть 0-100%"
        
        print(f"✅ Системные метрики: CPU={metrics.cpu_usage:.1f}%, RAM={metrics.memory_usage:.1f}%")
        print(f"✅ Эффективность: Memory={metrics.memory_efficiency:.1f}%, Compute={metrics.compute_efficiency:.1f}%")
        
        # Тестирование обнаружения bottleneck'ов
        if metrics.bottleneck_detected:
            print(f"⚠️ Обнаружен bottleneck: {metrics.bottleneck_type}")
        else:
            print("✅ Bottleneck'и не обнаружены")
        
        # Проверка истории профилирования
        assert len(profiler.profiling_history) == 1, "История должна содержать 1 запись"
        
        print("✅ SystemProfiler: Корректно профилирует систему")
        return True
        
    except Exception as e:
        print(f"❌ SystemProfiler: {e}")
        return False


def test_performance_optimizer():
    """Тестирование PerformanceOptimizer"""
    print("\n🧪 ТЕСТ 2: PerformanceOptimizer")
    
    try:
        optimizer = PerformanceOptimizer()
        
        # Создание тестовых метрик с различными сценариями
        test_scenarios = [
            {
                'name': 'Низкое использование GPU',
                'metrics': PerformanceMetrics(
                    timestamp=time.time(),
                    gpu_usage=30.0,
                    gpu_memory_usage=40.0,
                    memory_usage=50.0
                ),
                'training_context': {'loss': 5.0, 'gradient_norm': 2.0}
            },
            {
                'name': 'Высокое использование памяти',
                'metrics': PerformanceMetrics(
                    timestamp=time.time(),
                    memory_usage=95.0,
                    gpu_memory_usage=85.0
                ),
                'training_context': {'loss': 3.0, 'gradient_norm': 1.5}
            },
            {
                'name': 'Нестабильные градиенты',
                'metrics': PerformanceMetrics(
                    timestamp=time.time(),
                    gpu_usage=80.0,
                    memory_usage=60.0
                ),
                'training_context': {'loss': 10.0, 'gradient_norm': 15.0, 'learning_rate': 1e-3}
            }
        ]
        
        total_recommendations = 0
        
        for scenario in test_scenarios:
            recommendations = optimizer.generate_optimization_recommendations(
                scenario['metrics'], 
                scenario['training_context']
            )
            
            print(f"   📊 {scenario['name']}: {len(recommendations)} рекомендаций")
            
            # Проверка структуры рекомендаций
            for rec in recommendations:
                assert isinstance(rec, OptimizationRecommendation), "Должна быть OptimizationRecommendation"
                assert isinstance(rec.priority, OptimizationPriority), "Должен быть установлен приоритет"
                assert isinstance(rec.metric_type, PerformanceMetricType), "Должен быть установлен тип метрики"
                assert rec.description, "Описание не должно быть пустым"
                assert rec.suggested_action, "Предлагаемое действие не должно быть пустым"
                assert 0 <= rec.expected_improvement <= 100, "Ожидаемое улучшение должно быть 0-100%"
                assert 0 <= rec.estimated_risk <= 1, "Риск должен быть 0.0-1.0"
                assert 0 <= rec.confidence <= 1, "Уверенность должна быть 0.0-1.0"
                assert isinstance(rec.parameters_to_change, dict), "Параметры должны быть словарем"
                
                print(f"     • {rec.priority.value}: {rec.description}")
                
            total_recommendations += len(recommendations)
        
        print(f"✅ Всего сгенерировано {total_recommendations} рекомендаций")
        print("✅ PerformanceOptimizer: Корректно генерирует рекомендации")
        return True
        
    except Exception as e:
        print(f"❌ PerformanceOptimizer: {e}")
        return False


def test_adaptive_parameter_controller():
    """Тестирование AdaptiveParameterController"""
    print("\n🧪 ТЕСТ 3: AdaptiveParameterController")
    
    try:
        hparams = MockHparams()
        controller = AdaptiveParameterController(hparams)
        
        # Создание тестовых рекомендаций
        recommendations = [
            OptimizationRecommendation(
                priority=OptimizationPriority.HIGH,
                metric_type=PerformanceMetricType.TRAINING,
                description="Тест изменения learning rate",
                suggested_action="Уменьшить learning rate",
                expected_improvement=20.0,
                estimated_risk=0.2,
                parameters_to_change={'learning_rate': 5e-4},
                confidence=0.8
            ),
            OptimizationRecommendation(
                priority=OptimizationPriority.MEDIUM,
                metric_type=PerformanceMetricType.EFFICIENCY,
                description="Тест изменения batch size",
                suggested_action="Увеличить batch size",
                expected_improvement=15.0,
                estimated_risk=0.3,
                parameters_to_change={'batch_size': 24},
                confidence=0.7
            ),
            OptimizationRecommendation(
                priority=OptimizationPriority.LOW,
                metric_type=PerformanceMetricType.SYSTEM,
                description="Высокий риск - не должен применяться",
                suggested_action="Рискованное изменение",
                expected_improvement=50.0,
                estimated_risk=0.9,  # Высокий риск
                parameters_to_change={'learning_rate': 1e-1},
                confidence=0.3  # Низкая уверенность
            )
        ]
        
        # Сохранение исходных значений
        original_lr = hparams.learning_rate
        original_batch_size = hparams.batch_size
        
        # Применение рекомендаций
        applied_changes = controller.apply_optimization_recommendations(recommendations)
        
        # Проверка применения изменений
        assert len(applied_changes) >= 1, "Должны быть применены изменения"
        
        # Проверка что безопасные изменения применились
        if 'learning_rate' in applied_changes:
            assert hparams.learning_rate != original_lr, "Learning rate должен измениться"
            assert applied_changes['learning_rate']['old_value'] == original_lr
            assert applied_changes['learning_rate']['new_value'] == hparams.learning_rate
            print(f"✅ Learning rate изменен: {original_lr} → {hparams.learning_rate}")
        
        if 'batch_size' in applied_changes:
            assert hparams.batch_size != original_batch_size, "Batch size должен измениться"
            print(f"✅ Batch size изменен: {original_batch_size} → {hparams.batch_size}")
        
        # Проверка что рискованные изменения НЕ применились
        # (learning_rate не должен стать 0.1)
        assert hparams.learning_rate < 1e-2, "Рискованное изменение LR не должно применяться"
        
        print(f"✅ Применено {len(applied_changes)} безопасных изменений")
        print("✅ AdaptiveParameterController: Корректно управляет параметрами")
        return True
        
    except Exception as e:
        print(f"❌ AdaptiveParameterController: {e}")
        return False


def test_unified_optimization_system():
    """Тестирование UnifiedPerformanceOptimizationSystem"""
    print("\n🧪 ТЕСТ 4: UnifiedPerformanceOptimizationSystem")
    
    try:
        hparams = MockHparams()
        optimization_system = UnifiedPerformanceOptimizationSystem(
            hparams=hparams,
            enable_auto_optimization=True
        )
        
        # Проверка инициализации
        assert optimization_system.profiler is not None, "Profiler должен быть инициализирован"
        assert optimization_system.optimizer is not None, "Optimizer должен быть инициализирован"
        assert optimization_system.parameter_controller is not None, "Parameter controller должен быть инициализирован"
        
        # Тестирование шага оптимизации
        training_metrics = {
            'loss': 4.5,
            'learning_rate': 1e-3,
            'gradient_norm': 3.2,
            'attention_quality': 0.6
        }
        
        # Принудительная оптимизация
        result = optimization_system.optimize_performance_step(
            training_metrics=training_metrics,
            force_optimization=True
        )
        
        # Проверка результата
        assert result['status'] == 'completed', f"Статус должен быть 'completed', получен: {result['status']}"
        assert 'performance_metrics' in result, "Должны быть метрики производительности"
        assert 'recommendations_count' in result, "Должно быть количество рекомендаций"
        assert 'applied_changes' in result, "Должны быть примененные изменения"
        
        print(f"✅ Рекомендаций сгенерировано: {result['recommendations_count']}")
        print(f"✅ Изменений применено: {len(result['applied_changes'])}")
        
        # Проверка отчета о производительности
        report = optimization_system.get_performance_report()
        
        assert 'current_performance' in report, "Должна быть текущая производительность"
        assert 'optimization_stats' in report, "Должна быть статистика оптимизации"
        assert report['status'] in ['healthy', 'needs_attention'], "Должен быть корректный статус"
        
        print(f"✅ Статус системы: {report['status']}")
        print(f"✅ Эффективность: {report['current_performance']['compute_efficiency']:.1f}%")
        
        print("✅ UnifiedPerformanceOptimizationSystem: Работает корректно")
        return True
        
    except Exception as e:
        print(f"❌ UnifiedPerformanceOptimizationSystem: {e}")
        return False


def test_emergency_optimization():
    """Тестирование экстренной оптимизации"""
    print("\n🧪 ТЕСТ 5: Emergency Optimization")
    
    try:
        hparams = MockHparams()
        optimization_system = UnifiedPerformanceOptimizationSystem(
            hparams=hparams,
            enable_auto_optimization=True
        )
        
        # Критические метрики
        critical_metrics = {
            'gpu_memory_usage': 97.0,  # Критическое использование GPU памяти
            'memory_usage': 96.0,      # Критическое использование RAM
            'loss': 15.0,
            'gradient_norm': 25.0
        }
        
        # Сохранение исходного batch size
        original_batch_size = hparams.batch_size
        
        # Активация экстренной оптимизации
        result = optimization_system.activate_emergency_optimization(critical_metrics)
        
        # Проверка экстренных мер
        assert result.get('emergency_activation') == True, "Должна быть активирована экстренная оптимизация"
        assert 'emergency_changes' in result, "Должны быть экстренные изменения"
        
        emergency_changes = result['emergency_changes']
        
        # Проверка экстренного снижения batch size
        if emergency_changes.get('emergency_batch_size_reduction'):
            assert hparams.batch_size < original_batch_size, "Batch size должен быть уменьшен"
            print(f"✅ Экстренное снижение batch size: {original_batch_size} → {hparams.batch_size}")
        
        # Проверка очистки памяти
        if emergency_changes.get('emergency_memory_cleanup'):
            print("✅ Выполнена экстренная очистка памяти")
        
        print("✅ Emergency Optimization: Корректно реагирует на критические условия")
        return True
        
    except Exception as e:
        print(f"❌ Emergency Optimization: {e}")
        return False


def test_integration_with_hparams():
    """Тестирование интеграции с реальными hparams"""
    print("\n🧪 ТЕСТ 6: Integration with Real Hparams")
    
    if not HPARAMS_AVAILABLE:
        print("⚠️ Hparams недоступны, пропускаем тест")
        return True
    
    try:
        from hparams import create_hparams
        
        hparams = create_hparams()
        
        # Создание системы оптимизации
        optimization_system = create_performance_optimization_system(
            hparams=hparams,
            enable_auto_optimization=True
        )
        
        # Проверка успешной инициализации
        assert optimization_system is not None, "Система должна быть создана"
        
        # Тестирование с реальными параметрами
        training_metrics = {
            'loss': float(getattr(hparams, 'target_loss', 5.0)),
            'learning_rate': float(hparams.learning_rate),
            'gradient_norm': 2.0,
            'attention_quality': 0.5
        }
        
        result = optimization_system.optimize_performance_step(
            training_metrics=training_metrics,
            force_optimization=True
        )
        
        assert result['status'] == 'completed', "Должна успешно работать с реальными hparams"
        
        print("✅ Интеграция с реальными hparams работает корректно")
        return True
        
    except Exception as e:
        print(f"❌ Integration with Real Hparams: {e}")
        return False


def test_performance_monitoring_cycle():
    """Тестирование полного цикла мониторинга производительности"""
    print("\n🧪 ТЕСТ 7: Performance Monitoring Cycle")
    
    try:
        hparams = MockHparams()
        optimization_system = UnifiedPerformanceOptimizationSystem(
            hparams=hparams,
            enable_auto_optimization=True
        )
        
        # Включение непрерывной оптимизации
        optimization_system.enable_continuous_optimization()
        assert optimization_system.optimization_active == True, "Непрерывная оптимизация должна быть включена"
        
        # Симуляция нескольких циклов мониторинга
        simulation_results = []
        
        for cycle in range(3):
            training_metrics = {
                'loss': 5.0 - cycle * 0.5,  # Постепенное улучшение
                'learning_rate': 1e-3,
                'gradient_norm': 2.0 + cycle * 0.5,
                'attention_quality': 0.5 + cycle * 0.1
            }
            
            result = optimization_system.optimize_performance_step(
                training_metrics=training_metrics,
                force_optimization=True
            )
            
            simulation_results.append(result)
            time.sleep(0.1)  # Небольшая пауза между циклами
            
            print(f"   Цикл {cycle + 1}: {result['recommendations_count']} рекомендаций, "
                  f"{len(result['applied_changes'])} изменений")
        
        # Проверка накопления истории
        assert len(optimization_system.optimization_results) >= 3, "Должна накапливаться история оптимизации"
        
        # Финальный отчет
        final_report = optimization_system.get_performance_report()
        print(f"✅ Финальный статус: {final_report['status']}")
        print(f"✅ Всего оптимизаций: {final_report['optimization_stats']['total_optimizations']}")
        
        # Отключение непрерывной оптимизации
        optimization_system.disable_continuous_optimization()
        assert optimization_system.optimization_active == False, "Непрерывная оптимизация должна быть отключена"
        
        print("✅ Performance Monitoring Cycle: Полный цикл работает корректно")
        return True
        
    except Exception as e:
        print(f"❌ Performance Monitoring Cycle: {e}")
        return False


def run_all_tests():
    """Запуск всех тестов Unified Performance Optimization System"""
    print("🚀 НАЧАЛО ТЕСТИРОВАНИЯ: Unified Performance Optimization System")
    print("=" * 80)
    
    if not OPTIMIZATION_SYSTEM_AVAILABLE:
        print("❌ Unified Performance Optimization System недоступна для тестирования")
        return False
    
    tests = [
        test_system_profiler,
        test_performance_optimizer,
        test_adaptive_parameter_controller,
        test_unified_optimization_system,
        test_emergency_optimization,
        test_integration_with_hparams,
        test_performance_monitoring_cycle
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ Критическая ошибка в тесте {test_func.__name__}: {e}")
    
    # Финальный отчет
    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"✅ Пройдено тестов: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("\n🚀 Unified Performance Optimization System готова к production использованию:")
        print("   • SystemProfiler - профилирование и обнаружение bottleneck'ов")
        print("   • PerformanceOptimizer - интеллектуальные рекомендации")
        print("   • AdaptiveParameterController - безопасное управление параметрами")
        print("   • Emergency Optimization - реакция на критические условия")
        print("   • Real-time Monitoring - непрерывный мониторинг производительности")
        print("   • Integration - интеграция с существующими системами")
        return True
    else:
        print(f"⚠️ {total_tests - passed_tests} тестов не прошли")
        print("   Требуется доработка перед production использованием")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 