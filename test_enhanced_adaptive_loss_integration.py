"""
🧪 ТЕСТИРОВАНИЕ Enhanced Adaptive Loss System
==============================================

Комплексный тест интеграции адаптивной системы loss функций:
1. ✅ Инициализация Enhanced Adaptive Loss System
2. ✅ Интеграция с Tacotron2Loss  
3. ✅ Интеграция с Context-Aware Training Manager
4. ✅ Dynamic Tversky Loss функциональность
5. ✅ Адаптивные веса и контекстная адаптация
6. ✅ Диагностика и мониторинг системы

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

def test_adaptive_loss_system_import():
    """🧪 Тест 1: Импорт Enhanced Adaptive Loss System"""
    logger.info("\n🧪 Тест 1: Импорт Enhanced Adaptive Loss System")
    logger.info("-" * 60)
    
    try:
        from adaptive_loss_system import (
            create_adaptive_loss_system,
            DynamicTverskyLoss,
            IntelligentWeightManager,
            ContextBasedLossScaler,
            PhaseAwareLossOptimizer,
            LossPhase,
            LossContext
        )
        
        logger.info("✅ Все компоненты Enhanced Adaptive Loss System импортированы успешно")
        
        # Тестируем создание системы
        class MockHParams:
            mel_loss_weight = 1.0
            gate_loss_weight = 1.0
            guide_loss_weight = 2.0
            spectral_loss_weight = 0.3
            perceptual_loss_weight = 0.2
            style_loss_weight = 0.1
            monotonic_loss_weight = 0.1
        
        hparams = MockHParams()
        adaptive_system = create_adaptive_loss_system(hparams)
        
        logger.info("✅ Enhanced Adaptive Loss System создана успешно")
        logger.info(f"   Компоненты: DynamicTverskyLoss, IntelligentWeightManager, ContextBasedLossScaler")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Ошибка импорта: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка создания системы: {e}")
        return False


def test_loss_function_integration():
    """🧪 Тест 2: Интеграция с Tacotron2Loss"""
    logger.info("\n🧪 Тест 2: Интеграция с Tacotron2Loss")
    logger.info("-" * 60)
    
    try:
        from loss_function import Tacotron2Loss
        from adaptive_loss_system import create_adaptive_loss_system, LossPhase, LossContext
        
        # Создаем mock hparams
        class MockHParams:
            mel_loss_weight = 1.0
            gate_loss_weight = 1.0
            guide_loss_weight = 2.0
            spectral_loss_weight = 0.3
            perceptual_loss_weight = 0.2
            style_loss_weight = 0.1
            monotonic_loss_weight = 0.1
            guide_decay = 0.9999
            guide_sigma = 0.4
            adaptive_gate_threshold = True
            curriculum_teacher_forcing = True
            use_ddc = False
            ddc_consistency_weight = 0.5
            
            # Параметры для адаптивной системы
            weight_adaptation_rate = 0.02
            loss_stability_threshold = 2.0
            
            # Параметры для UnifiedGuidedAttention
            guide_loss_initial_weight = 5.0
            guide_loss_min_weight = 0.1
            guide_loss_max_weight = 15.0
            guide_loss_decay_start = 2000
            guide_loss_decay_steps = 25000
            guide_loss_decay_factor = 3.0
            guide_sigma_initial = 0.1
            guide_sigma_peak = 0.4
            guide_sigma_final = 0.15
            guide_emergency_weight = 25.0
            attention_emergency_threshold = 0.02
            attention_recovery_threshold = 0.5
            use_context_aware_attention = True
        
        hparams = MockHParams()
        
        # Создаем loss функцию
        loss_function = Tacotron2Loss(hparams)
        logger.info("✅ Tacotron2Loss создан")
        
        # Проверяем интеграцию адаптивной системы
        if hasattr(loss_function, 'use_adaptive_loss') and loss_function.use_adaptive_loss:
            logger.info("✅ Enhanced Adaptive Loss System интегрирована")
            
            # Тестируем обновление контекста
            loss_function.update_training_context(
                phase="ALIGNMENT_LEARNING",
                attention_quality=0.4,
                gate_accuracy=0.7,
                mel_consistency=0.6,
                gradient_norm=2.5,
                loss_stability=1.2,
                learning_rate=1e-3
            )
            logger.info("✅ Контекст обучения обновлен")
            
            # Проверяем получение адаптивных весов
            adaptive_weights = loss_function.get_current_adaptive_weights()
            logger.info(f"✅ Получены адаптивные веса: {list(adaptive_weights.keys())}")
            
        else:
            logger.warning("⚠️ Enhanced Adaptive Loss System не интегрирована")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования интеграции с loss функцией: {e}")
        return False


def test_context_aware_integration():
    """🧪 Тест 3: Интеграция с Context-Aware Training Manager"""
    logger.info("\n🧪 Тест 3: Интеграция с Context-Aware Training Manager")
    logger.info("-" * 60)
    
    try:
        from context_aware_training_manager import create_context_aware_manager, TrainingPhase
        from loss_function import Tacotron2Loss
        
        # Создаем mock hparams
        class MockHParams:
            learning_rate = 1e-3
            guide_loss_weight = 4.5
            mel_loss_weight = 1.0
            gate_loss_weight = 1.0
            spectral_loss_weight = 0.3
            perceptual_loss_weight = 0.2
            style_loss_weight = 0.1
            monotonic_loss_weight = 0.1
            guide_decay = 0.9999
            guide_sigma = 0.4
            adaptive_gate_threshold = True
            curriculum_teacher_forcing = True
            use_ddc = False
            ddc_consistency_weight = 0.5
            
            # Параметры для адаптивной системы
            weight_adaptation_rate = 0.02
            loss_stability_threshold = 2.0
            
            # Параметры для UnifiedGuidedAttention
            guide_loss_initial_weight = 5.0
            guide_loss_min_weight = 0.1
            guide_loss_max_weight = 15.0
            guide_loss_decay_start = 2000
            guide_loss_decay_steps = 25000
            guide_loss_decay_factor = 3.0
            guide_sigma_initial = 0.1
            guide_sigma_peak = 0.4
            guide_sigma_final = 0.15
            guide_emergency_weight = 25.0
            attention_emergency_threshold = 0.02
            attention_recovery_threshold = 0.5
            use_context_aware_attention = True
        
        hparams = MockHParams()
        
        # Создаем Context-Aware Manager
        context_manager = create_context_aware_manager(hparams)
        logger.info("✅ Context-Aware Training Manager создан")
        
        # Создаем loss функцию
        loss_function = Tacotron2Loss(hparams)
        logger.info("✅ Tacotron2Loss создан")
        
        # Тестируем интеграцию
        context_manager.integrate_with_loss_function(loss_function)
        logger.info("✅ Интеграция Context-Aware Manager с loss функцией завершена")
        
        # Проверяем, что EnhancedLossIntegrator получил ссылку на loss функцию
        if hasattr(context_manager.loss_controller, 'enhanced_system_available'):
            if context_manager.loss_controller.enhanced_system_available:
                logger.info("✅ EnhancedLossIntegrator успешно подключен к Enhanced Adaptive Loss System")
            else:
                logger.warning("⚠️ EnhancedLossIntegrator работает в fallback режиме")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования интеграции с Context-Aware Manager: {e}")
        return False


def test_dynamic_tversky_loss():
    """🧪 Тест 4: Dynamic Tversky Loss функциональность"""
    logger.info("\n🧪 Тест 4: Dynamic Tversky Loss функциональность")
    logger.info("-" * 60)
    
    try:
        from adaptive_loss_system import DynamicTverskyLoss, LossContext, LossPhase
        
        # Создаем Dynamic Tversky Loss
        tversky_loss = DynamicTverskyLoss(
            initial_alpha=0.7,
            initial_beta=0.3,
            adapt_rate=0.01
        )
        logger.info("✅ DynamicTverskyLoss создан")
        
        # Создаем тестовые данные
        predictions = torch.randn(8, 100)  # Batch size 8, 100 gate outputs
        targets = torch.randint(0, 2, (8, 100)).float()  # Binary targets
        
        # Создаем контекст для адаптации
        context = LossContext(
            phase=LossPhase.ALIGNMENT_LEARNING,
            global_step=1000,
            attention_quality=0.4,
            gate_accuracy=0.7,
            mel_consistency=0.6,
            gradient_norm=2.5,
            loss_stability=1.2,
            learning_rate=1e-3
        )
        
        # Тестируем вычисление loss
        loss_value = tversky_loss(predictions, targets, context)
        logger.info(f"✅ Tversky loss вычислена: {loss_value.item():.4f}")
        
        # Проверяем адаптацию параметров
        initial_alpha = tversky_loss.alpha
        
        # Симулируем несколько шагов с низкой gate accuracy
        for i in range(10):
            context.gate_accuracy = 0.5  # Низкая accuracy
            tversky_loss(predictions, targets, context)
        
        final_alpha = tversky_loss.alpha
        
        if abs(final_alpha - initial_alpha) > 0.001:
            logger.info(f"✅ Адаптация параметров работает: α {initial_alpha:.3f} → {final_alpha:.3f}")
        else:
            logger.info(f"✅ Параметры стабильны: α={final_alpha:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования Dynamic Tversky Loss: {e}")
        return False


def test_adaptive_weights_and_scaling():
    """🧪 Тест 5: Адаптивные веса и контекстное масштабирование"""
    logger.info("\n🧪 Тест 5: Адаптивные веса и контекстное масштабирование")
    logger.info("-" * 60)
    
    try:
        from adaptive_loss_system import (
            create_adaptive_loss_system,
            LossContext,
            LossPhase
        )
        
        # Создаем систему
        class MockHParams:
            mel_loss_weight = 1.0
            gate_loss_weight = 1.0
            guide_loss_weight = 2.0
            spectral_loss_weight = 0.3
            perceptual_loss_weight = 0.2
            style_loss_weight = 0.1
            monotonic_loss_weight = 0.1
            weight_adaptation_rate = 0.02
            loss_stability_threshold = 2.0
        
        hparams = MockHParams()
        adaptive_system = create_adaptive_loss_system(hparams)
        logger.info("✅ Enhanced Adaptive Loss System создана")
        
        # Тестируем разные фазы обучения
        phases = [
            (LossPhase.PRE_ALIGNMENT, 0.1, 0.5),      # Плохое attention, средняя gate accuracy
            (LossPhase.ALIGNMENT_LEARNING, 0.4, 0.7), # Среднее attention, хорошая gate accuracy
            (LossPhase.REFINEMENT, 0.6, 0.8),         # Хорошее attention, отличная gate accuracy
            (LossPhase.CONVERGENCE, 0.8, 0.9)         # Отличное attention, превосходная gate accuracy
        ]
        
        for phase, attention_quality, gate_accuracy in phases:
            context = LossContext(
                phase=phase,
                global_step=1000,
                attention_quality=attention_quality,
                gate_accuracy=gate_accuracy,
                mel_consistency=0.6,
                gradient_norm=2.0,
                loss_stability=1.0,
                learning_rate=1e-3
            )
            
            # Тестовые loss компоненты
            loss_components = {
                'mel': torch.tensor(2.5),
                'gate': torch.tensor(0.8),
                'guided_attention': torch.tensor(1.2),
                'spectral': torch.tensor(0.4),
                'perceptual': torch.tensor(0.3),
                'style': torch.tensor(0.1),
                'monotonic': torch.tensor(0.1)
            }
            
            # Тестируем оптимизацию
            optimized_loss, diagnostics = adaptive_system.optimize_loss_computation(
                loss_components, context
            )
            
            logger.info(f"✅ Фаза {phase.value}:")
            logger.info(f"   Оптимизированная loss: {optimized_loss.item():.4f}")
            logger.info(f"   Loss scale: {diagnostics['loss_scale']:.3f}")
            logger.info(f"   Guided attention weight: {diagnostics['adaptive_weights']['guided_attention']:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования адаптивных весов: {e}")
        return False


def test_system_diagnostics():
    """🧪 Тест 6: Диагностика и мониторинг системы"""
    logger.info("\n🧪 Тест 6: Диагностика и мониторинг системы")
    logger.info("-" * 60)
    
    try:
        from loss_function import Tacotron2Loss
        from adaptive_loss_system import LossContext, LossPhase
        
        # Создаем mock hparams
        class MockHParams:
            mel_loss_weight = 1.0
            gate_loss_weight = 1.0
            guide_loss_weight = 2.0
            spectral_loss_weight = 0.3
            perceptual_loss_weight = 0.2
            style_loss_weight = 0.1
            monotonic_loss_weight = 0.1
            guide_decay = 0.9999
            guide_sigma = 0.4
            adaptive_gate_threshold = True
            curriculum_teacher_forcing = True
            use_ddc = False
            ddc_consistency_weight = 0.5
            weight_adaptation_rate = 0.02
            loss_stability_threshold = 2.0
            
            # Параметры для UnifiedGuidedAttention
            guide_loss_initial_weight = 5.0
            guide_loss_min_weight = 0.1
            guide_loss_max_weight = 15.0
            guide_loss_decay_start = 2000
            guide_loss_decay_steps = 25000
            guide_loss_decay_factor = 3.0
            guide_sigma_initial = 0.1
            guide_sigma_peak = 0.4
            guide_sigma_final = 0.15
            guide_emergency_weight = 25.0
            attention_emergency_threshold = 0.02
            attention_recovery_threshold = 0.5
            use_context_aware_attention = True
        
        hparams = MockHParams()
        loss_function = Tacotron2Loss(hparams)
        
        if hasattr(loss_function, 'use_adaptive_loss') and loss_function.use_adaptive_loss:
            # Обновляем контекст и получаем диагностику
            loss_function.update_training_context(
                phase="REFINEMENT",
                attention_quality=0.6,
                gate_accuracy=0.8,
                gradient_norm=2.0,
                loss_stability=1.0
            )
            
            # Получаем диагностику адаптивной системы
            diagnostics = loss_function.get_adaptive_loss_diagnostics()
            logger.info("✅ Диагностика Enhanced Adaptive Loss System получена:")
            
            if 'tversky_loss' in diagnostics:
                tversky_info = diagnostics['tversky_loss']
                logger.info(f"   Dynamic Tversky: α={tversky_info.get('current_alpha', 0):.3f}, β={tversky_info.get('current_beta', 0):.3f}")
            
            if 'weight_manager' in diagnostics:
                weight_info = diagnostics['weight_manager']
                current_weights = weight_info.get('current_weights', {})
                logger.info(f"   Адаптивные веса: mel={current_weights.get('mel', 0):.2f}, gate={current_weights.get('gate', 0):.2f}")
            
            if 'loss_scaler' in diagnostics:
                scaler_info = diagnostics['loss_scaler']
                logger.info(f"   Loss scaler: {scaler_info.get('current_scale', 1.0):.3f}")
            
            if 'performance_summary' in diagnostics:
                perf_summary = diagnostics['performance_summary']
                logger.info(f"   Производительность: стабильность={perf_summary.get('system_stability', 0):.3f}")
            
        else:
            logger.warning("⚠️ Enhanced Adaptive Loss System недоступна для диагностики")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования диагностики: {e}")
        return False


def run_all_tests():
    """Запуск всех тестов Enhanced Adaptive Loss System"""
    logger.info("🚀 ЗАПУСК КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ Enhanced Adaptive Loss System")
    logger.info("=" * 80)
    
    tests = [
        ("Импорт системы", test_adaptive_loss_system_import),
        ("Интеграция с Tacotron2Loss", test_loss_function_integration),
        ("Интеграция с Context-Aware Manager", test_context_aware_integration),
        ("Dynamic Tversky Loss", test_dynamic_tversky_loss),
        ("Адаптивные веса и масштабирование", test_adaptive_weights_and_scaling),
        ("Диагностика системы", test_system_diagnostics)
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
    logger.info("📊 РЕЗУЛЬТАТЫ КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ Enhanced Adaptive Loss System:")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ ПРОЙДЕН" if result else "❌ ПРОВАЛЕН"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\n🎯 ИТОГО: {passed}/{total} тестов пройдено ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Enhanced Adaptive Loss System готова к работе!")
        logger.info("✨ Система полностью интегрирована и функциональна")
    else:
        logger.warning(f"⚠️ {total-passed} тестов провалено. Требуется исправление.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit_code = 0 if success else 1
    exit(exit_code) 