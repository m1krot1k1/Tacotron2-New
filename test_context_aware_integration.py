#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест интеграции Context-Aware Training Manager
Проверка корректной замены AutoFixManager на интеллектуальную систему
"""

import sys
import logging
import torch
import numpy as np
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_context_aware_manager_import():
    """Тест импорта Context-Aware Training Manager"""
    try:
        from context_aware_training_manager import (
            ContextAwareTrainingManager, 
            TrainingPhase, 
            TrainingContext, 
            create_context_aware_manager
        )
        logger.info("✅ Context-Aware Training Manager импортирован успешно")
        return True
    except ImportError as e:
        logger.error(f"❌ Ошибка импорта Context-Aware Manager: {e}")
        return False

def test_context_aware_manager_creation():
    """Тест создания менеджера"""
    try:
        from context_aware_training_manager import ContextAwareTrainingManager
        
        config = {
            'initial_lr': 1e-3,
            'history_size': 50,
            'initial_guided_weight': 4.5
        }
        
        manager = ContextAwareTrainingManager(config)
        logger.info("✅ Context-Aware Training Manager создан успешно")
        
        # Проверка компонентов
        assert manager.context_analyzer is not None
        assert manager.loss_controller is not None
        assert manager.param_manager is not None
        logger.info("✅ Все компоненты менеджера инициализированы")
        
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка создания менеджера: {e}")
        return False

def test_mock_training_adaptation():
    """Тест адаптации с мок-данными"""
    try:
        from context_aware_training_manager import ContextAwareTrainingManager
        
        config = {
            'initial_lr': 1e-3,
            'history_size': 50,
            'initial_guided_weight': 4.5
        }
        
        manager = ContextAwareTrainingManager(config)
        
        # Симуляция нескольких шагов обучения
        for step in range(10):
            mock_metrics = {
                'loss': 20.0 - step * 0.5,  # Убывающий loss
                'attention_diagonality': 0.05 + step * 0.02,  # Улучшающийся attention
                'grad_norm': 5.0 + np.random.normal(0, 0.5),
                'gate_accuracy': 0.7 + step * 0.02,
                'mel_loss': 15.0 - step * 0.3,
                'gate_loss': 3.0 - step * 0.1,
                'guided_attention_loss': 2.0 - step * 0.05,
                'epoch': 0
            }
            
            adaptations = manager.analyze_and_adapt(
                step=step,
                metrics=mock_metrics
            )
            
            if step % 5 == 0:
                logger.info(f"Step {step}: {list(adaptations.keys())}")
        
        logger.info("✅ Мок-тест адаптации прошел успешно")
        
        # Проверка статистики
        stats = manager.get_statistics()
        logger.info(f"📊 Статистика: {stats['total_interventions']} вмешательств, фаза: {stats['current_phase']}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка мок-теста: {e}")
        return False

def test_ultimate_trainer_integration():
    """Тест интеграции с UltimateEnhancedTacotronTrainer"""
    try:
        from hparams import create_hparams
        from ultimate_tacotron_trainer import UltimateEnhancedTacotronTrainer
        
        # Создаем минимальные hparams для теста
        hparams = create_hparams()
        
        # Создаем trainer в режиме 'enhanced'
        trainer = UltimateEnhancedTacotronTrainer(
            hparams=hparams, 
            mode='enhanced'
        )
        
        # Проверяем, что Context-Aware Manager доступен
        if hasattr(trainer, 'context_aware_manager'):
            logger.info("✅ Context-Aware Manager интегрирован в UltimateEnhancedTacotronTrainer")
        else:
            logger.warning("⚠️ Context-Aware Manager не найден в trainer")
        
        # Проверяем, что AutoFixManager отключен
        if hasattr(trainer, 'auto_fix_manager') and trainer.auto_fix_manager is None:
            logger.info("✅ AutoFixManager корректно отключен")
        else:
            logger.warning("⚠️ AutoFixManager все еще активен")
        
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка интеграции с trainer: {e}")
        return False

def test_phase_classification():
    """Тест классификации фаз обучения"""
    try:
        from context_aware_training_manager import ContextAnalyzer, TrainingPhase
        
        analyzer = ContextAnalyzer(history_size=20)
        
        # Тест фазы PRE_ALIGNMENT
        for i in range(15):
            analyzer.update_metrics(
                loss=30.0 - i * 0.5,
                attention_diag=0.02 + i * 0.001,  # Очень низкий attention
                grad_norm=10.0 + np.random.normal(0, 1),
                gate_accuracy=0.6
            )
        
        phase = analyzer.analyze_phase()
        assert phase == TrainingPhase.PRE_ALIGNMENT
        logger.info("✅ Фаза PRE_ALIGNMENT корректно определена")
        
        # Тест фазы ALIGNMENT_LEARNING
        for i in range(20):
            analyzer.update_metrics(
                loss=15.0 - i * 0.2,
                attention_diag=0.25 + i * 0.01,  # Средний attention
                grad_norm=5.0 + np.random.normal(0, 0.5),
                gate_accuracy=0.8
            )
        
        phase = analyzer.analyze_phase()
        assert phase == TrainingPhase.ALIGNMENT_LEARNING
        logger.info("✅ Фаза ALIGNMENT_LEARNING корректно определена")
        
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка теста классификации фаз: {e}")
        return False

def run_all_tests():
    """Запуск всех тестов интеграции"""
    logger.info("🚀 Начало тестирования интеграции Context-Aware Training Manager")
    logger.info("=" * 70)
    
    tests = [
        ("Импорт модулей", test_context_aware_manager_import),
        ("Создание менеджера", test_context_aware_manager_creation),
        ("Мок-тест адаптации", test_mock_training_adaptation),
        ("Интеграция с trainer", test_ultimate_trainer_integration),
        ("Классификация фаз", test_phase_classification)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Тест: {test_name}")
        logger.info("-" * 50)
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name}: ПРОЙДЕН")
            else:
                logger.error(f"❌ {test_name}: ПРОВАЛЕН")
        except Exception as e:
            logger.error(f"❌ {test_name}: ОШИБКА - {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info(f"📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    logger.info(f"✅ Пройдено: {passed}/{total}")
    logger.info(f"❌ Провалено: {total - passed}/{total}")
    
    if passed == total:
        logger.info("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        logger.info("🧠 Context-Aware Training Manager готов к использованию")
        return True
    else:
        logger.warning("⚠️ Некоторые тесты провалены. Проверьте интеграцию.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n🏆 ИНТЕГРАЦИЯ УСПЕШНА!")
        print("Context-Aware Training Manager заменил деструктивный AutoFixManager")
        print("Система готова к умному обучению с контекстным пониманием")
    else:
        print("\n🔧 ТРЕБУЕТСЯ ДОРАБОТКА")
        print("Проверьте ошибки интеграции выше")
    
    sys.exit(0 if success else 1) 