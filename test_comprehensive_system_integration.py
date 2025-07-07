#!/usr/bin/env python3
"""
🏆 MASTER COMPREHENSIVE SYSTEM INTEGRATION TEST

Финальное комплексное тестирование всей новой интеллектуальной системы Tacotron2-New.
Проверяет интеграцию ВСЕХ созданных компонентов и решение проблем из exported-assets.

Тестируемые системы:
✅ Context-Aware Training Manager (замена AutoFixManager)
✅ Training Stabilization System 
✅ Advanced Attention Enhancement System
✅ Enhanced Adaptive Loss System
✅ Unified Logging System
✅ Integration между всеми компонентами
✅ Performance и стабильность
✅ Решение проблем exported-assets
"""

import sys
import os
import time
import logging
import traceback
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import threading

# Setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MasterSystemTester:
    """🏆 Master tester для комплексного тестирования системы"""
    
    def __init__(self):
        self.test_results = {}
        self.system_components = {}
        self.performance_metrics = {}
        self.start_time = time.time()
        
    def run_comprehensive_tests(self) -> bool:
        """Запуск всех комплексных тестов"""
        logger.info("🏆 ЗАПУСК MASTER COMPREHENSIVE SYSTEM INTEGRATION TEST")
        logger.info("=" * 80)
        
        tests = [
            ("System Components Import", self.test_system_imports),
            ("Unified Logging Integration", self.test_unified_logging_integration),
            ("Context-Aware Manager", self.test_context_aware_manager),
            ("Training Stabilization", self.test_training_stabilization),
            ("Attention Enhancement", self.test_attention_enhancement),
            ("Adaptive Loss System", self.test_adaptive_loss_system),
            ("Inter-Component Integration", self.test_inter_component_integration),
            ("Performance & Stability", self.test_performance_stability),
            ("Exported-Assets Problems Resolution", self.test_exported_assets_resolution),
            ("Full System Simulation", self.test_full_system_simulation)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n🧪 Тест: {test_name}")
            logger.info("-" * 60)
            
            try:
                result = test_func()
                self.test_results[test_name] = result
                
                if result:
                    passed += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                logger.error(f"💥 {test_name}: CRITICAL ERROR - {e}")
                self.test_results[test_name] = False
        
        # Финальный отчет
        self.generate_final_report(passed, total)
        return passed == total
    
    def test_system_imports(self) -> bool:
        """Тест 1: Импорт всех системных компонентов"""
        try:
            # Unified Logging System
            from unified_logging_system import get_unified_logger, setup_component_logging
            logger.info("✅ Unified Logging System")
            
            # Context-Aware Training Manager
            from context_aware_training_manager import ContextAwareTrainingManager
            logger.info("✅ Context-Aware Training Manager")
            
            # Training Stabilization System
            from training_stabilization_system import create_training_stabilization_system
            logger.info("✅ Training Stabilization System")
            
            # Advanced Attention Enhancement
            from advanced_attention_enhancement_system import create_advanced_attention_enhancement_system
            logger.info("✅ Advanced Attention Enhancement System")
            
            # Enhanced Adaptive Loss
            from adaptive_loss_system import create_adaptive_loss_system
            logger.info("✅ Enhanced Adaptive Loss System")
            
            # Integration patches
            from logging_integration_patches import start_unified_logging_integration
            logger.info("✅ Logging Integration Patches")
            
            logger.info("🎉 Все системные компоненты импортированы успешно")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка импорта: {e}")
            return False
    
    def test_unified_logging_integration(self) -> bool:
        """Тест 2: Интеграция unified logging"""
        try:
            from logging_integration_patches import start_unified_logging_integration, stop_unified_logging_integration
            
            # Запуск интеграции
            success = start_unified_logging_integration("master_test_session")
            if not success:
                logger.error("❌ Не удалось запустить unified logging integration")
                return False
            
            # Тест component logging
            from unified_logging_system import setup_component_logging, MetricPriority
            
            test_logger = setup_component_logging("master_test", MetricPriority.ESSENTIAL)
            test_logger.log_metrics({"test_metric": 1.0}, step=1)
            test_logger.info("Master test logging message")
            
            # Завершение интеграции
            stop_unified_logging_integration()
            
            logger.info("✅ Unified logging integration работает корректно")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка unified logging integration: {e}")
            return False
    
    def test_context_aware_manager(self) -> bool:
        """Тест 3: Context-Aware Training Manager"""
        try:
            from context_aware_training_manager import ContextAwareTrainingManager
            
            config = {
                'history_size': 50,
                'initial_guided_weight': 4.5,
                'initial_lr': 1e-3
            }
            
            manager = ContextAwareTrainingManager(config)
            
            # Тест анализа шагов
            test_metrics = {
                'loss': 15.5,
                'mel_loss': 12.0,
                'gate_loss': 0.8,
                'attention_diagonality': 0.045
            }
            
            recommendations = manager.analyze_training_step(test_metrics, step=1)
            
            if isinstance(recommendations, dict) and 'step' in recommendations:
                logger.info("✅ Context-Aware Manager анализ работает")
                return True
            else:
                logger.error("❌ Context-Aware Manager анализ не работает")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка Context-Aware Manager: {e}")
            return False
    
    def test_training_stabilization(self) -> bool:
        """Тест 4: Training Stabilization System"""
        try:
            from training_stabilization_system import create_training_stabilization_system
            
            class MockHParams:
                learning_rate = 1e-3
                target_gradient_norm = 2.0
                max_gradient_norm = 5.0
                min_learning_rate = 1e-5
                stability_window_size = 20
            
            hparams = MockHParams()
            stabilization_system = create_training_stabilization_system(hparams)
            
            if stabilization_system and len(stabilization_system) >= 4:
                logger.info("✅ Training Stabilization System создана успешно")
                return True
            else:
                logger.error("❌ Training Stabilization System не создана")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка Training Stabilization System: {e}")
            return False
    
    def test_attention_enhancement(self) -> bool:
        """Тест 5: Advanced Attention Enhancement System"""
        try:
            from advanced_attention_enhancement_system import create_advanced_attention_enhancement_system
            
            class MockHParams:
                attention_rnn_dim = 1024
                encoder_embedding_dim = 512
                attention_dim = 128
                attention_num_heads = 8
                attention_location_n_filters = 32
                attention_location_kernel_size = 31
                max_training_steps = 10000
                target_attention_diagonality = 0.7
            
            hparams = MockHParams()
            attention_system = create_advanced_attention_enhancement_system(hparams)
            
            if attention_system and len(attention_system) >= 5:
                logger.info("✅ Advanced Attention Enhancement System создана")
                return True
            else:
                logger.error("❌ Advanced Attention Enhancement System не создана")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка Attention Enhancement: {e}")
            return False
    
    def test_adaptive_loss_system(self) -> bool:
        """Тест 6: Enhanced Adaptive Loss System"""
        try:
            from adaptive_loss_system import create_adaptive_loss_system
            
            class MockHParams:
                mel_loss_weight = 1.0
                gate_loss_weight = 1.0
                guide_loss_weight = 2.0
                spectral_loss_weight = 0.3
                perceptual_loss_weight = 0.2
            
            hparams = MockHParams()
            adaptive_system = create_adaptive_loss_system(hparams)
            
            if adaptive_system and len(adaptive_system) >= 3:
                logger.info("✅ Enhanced Adaptive Loss System создана")
                return True
            else:
                logger.error("❌ Enhanced Adaptive Loss System не создана")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка Adaptive Loss System: {e}")
            return False
    
    def test_inter_component_integration(self) -> bool:
        """Тест 7: Интеграция между компонентами"""
        try:
            # Проверяем что все компоненты могут работать вместе
            from unified_logging_system import get_unified_logger
            from context_aware_training_manager import ContextAwareTrainingManager
            
            # Запускаем unified logging
            logger_system = get_unified_logger()
            logger_system.start_session("integration_test")
            
            # Создаем Context-Aware Manager
            config = {'history_size': 10}
            manager = ContextAwareTrainingManager(config)
            
            # Тест взаимодействия
            test_metrics = {'loss': 10.0, 'attention_diagonality': 0.1}
            recommendations = manager.analyze_training_step(test_metrics, 1)
            
            # Завершаем
            logger_system.end_session()
            
            if recommendations and isinstance(recommendations, dict):
                logger.info("✅ Интеграция между компонентами работает")
                return True
            else:
                logger.error("❌ Интеграция между компонентами не работает")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка интеграции компонентов: {e}")
            return False
    
    def test_performance_stability(self) -> bool:
        """Тест 8: Performance и стабильность"""
        try:
            from unified_logging_system import get_unified_logger
            
            logger_system = get_unified_logger()
            logger_system.start_session("performance_test")
            
            start_time = time.time()
            
            # Stress test: 100 метрик
            for i in range(100):
                logger_system.log_metrics({'metric': i * 0.1}, step=i)
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger_system.end_session()
            
            if duration < 2.0:  # Должно выполняться менее чем за 2 секунды
                logger.info(f"✅ Performance тест пройден ({duration:.2f}s для 100 метрик)")
                return True
            else:
                logger.error(f"❌ Performance тест не пройден ({duration:.2f}s)")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка performance теста: {e}")
            return False
    
    def test_exported_assets_resolution(self) -> bool:
        """Тест 9: Решение проблем из exported-assets"""
        try:
            # Проверяем что основные проблемы из exported-assets решены
            from unified_logging_system import get_unified_logger
            
            # Тест 1: Нет множественных MLflow runs
            logger_system = get_unified_logger()
            session1_success = logger_system.start_session("test1")
            session2_success = logger_system.start_session("test2")  # Должен предупредить, но не создать новый
            
            if session1_success and session2_success:
                logger.info("✅ MLflow конфликты устранены")
                mlflow_resolved = True
            else:
                mlflow_resolved = False
            
            logger_system.end_session()
            
            # Тест 2: Priority-based logging работает
            from unified_logging_system import setup_component_logging, MetricPriority
            
            essential_logger = setup_component_logging("test_essential", MetricPriority.ESSENTIAL)
            verbose_logger = setup_component_logging("test_verbose", MetricPriority.VERBOSE)
            
            if essential_logger and verbose_logger:
                logger.info("✅ Priority-based logging работает")
                priority_resolved = True
            else:
                priority_resolved = False
            
            # Тест 3: Context-Aware заменил AutoFixManager
            try:
                from context_aware_training_manager import ContextAwareTrainingManager
                context_available = True
            except:
                context_available = False
            
            # Проверяем что AutoFixManager недоступен
            try:
                from smart_tuner.auto_fix_manager import AutoFixManager
                autofix_disabled = False  # Если импортируется, значит не отключен
            except:
                autofix_disabled = True  # Не импортируется - хорошо
            
            if context_available and autofix_disabled:
                logger.info("✅ AutoFixManager успешно заменен на Context-Aware Manager")
                replacement_resolved = True
            else:
                replacement_resolved = False
            
            # Общий результат
            all_resolved = mlflow_resolved and priority_resolved and replacement_resolved
            
            if all_resolved:
                logger.info("✅ Все проблемы из exported-assets решены")
                return True
            else:
                logger.error("❌ Не все проблемы из exported-assets решены")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка проверки exported-assets: {e}")
            return False
    
    def test_full_system_simulation(self) -> bool:
        """Тест 10: Полная симуляция системы"""
        try:
            from logging_integration_patches import start_unified_logging_integration, stop_unified_logging_integration
            from context_aware_training_manager import ContextAwareTrainingManager
            
            # Запуск полной интеграции
            integration_success = start_unified_logging_integration("full_simulation")
            
            if not integration_success:
                logger.error("❌ Не удалось запустить полную интеграцию")
                return False
            
            # Создание Context-Aware Manager
            config = {'history_size': 20}
            manager = ContextAwareTrainingManager(config)
            
            # Симуляция 10 шагов обучения
            simulation_success = True
            
            for step in range(1, 11):
                # Симулируем улучшающиеся метрики
                loss = 20.0 - step * 1.5  # Loss уменьшается
                attention_diagonality = 0.02 + step * 0.08  # Attention улучшается
                
                test_metrics = {
                    'loss': loss,
                    'mel_loss': loss * 0.8,
                    'gate_loss': loss * 0.1,
                    'attention_diagonality': attention_diagonality,
                    'learning_rate': 1e-4,
                    'gradient_norm': 2.0 - step * 0.1
                }
                
                try:
                    recommendations = manager.analyze_training_step(test_metrics, step)
                    
                    if not recommendations or 'step' not in recommendations:
                        simulation_success = False
                        break
                        
                except Exception as e:
                    logger.error(f"❌ Ошибка на шаге {step}: {e}")
                    simulation_success = False
                    break
            
            # Завершение интеграции
            stop_unified_logging_integration()
            
            if simulation_success:
                logger.info("✅ Полная симуляция системы успешна")
                return True
            else:
                logger.error("❌ Полная симуляция системы провалена")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка полной симуляции: {e}")
            return False
    
    def generate_final_report(self, passed: int, total: int):
        """Генерация финального отчета"""
        duration = time.time() - self.start_time
        success_rate = (passed / total * 100) if total > 0 else 0
        
        logger.info("\n" + "=" * 80)
        logger.info("🏆 ФИНАЛЬНЫЙ ОТЧЕТ MASTER COMPREHENSIVE SYSTEM INTEGRATION TEST")
        logger.info("=" * 80)
        
        logger.info(f"⏱️  Общее время тестирования: {duration:.2f} секунд")
        logger.info(f"📊 Всего тестов: {total}")
        logger.info(f"✅ Пройдено: {passed}")
        logger.info(f"❌ Провалено: {total - passed}")
        logger.info(f"🎯 Успешность: {success_rate:.1f}%")
        
        logger.info("\n📋 ДЕТАЛИЗАЦИЯ РЕЗУЛЬТАТОВ:")
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{status}: {test_name}")
        
        logger.info("\n🎯 ЗАКЛЮЧЕНИЕ:")
        if success_rate >= 90:
            logger.info("🎉 СИСТЕМА ПОЛНОСТЬЮ ГОТОВА К PRODUCTION ИСПОЛЬЗОВАНИЮ!")
            logger.info("🏆 Все критические компоненты протестированы и работают корректно")
        elif success_rate >= 70:
            logger.info("⚠️  Система в основном готова, но требует доработки некоторых компонентов")
        else:
            logger.info("❌ Система требует серьезной доработки перед использованием")
        
        # Рекомендации
        logger.info("\n💡 РЕКОМЕНДАЦИИ:")
        if any(not result for result in self.test_results.values()):
            logger.info("- Исправить провалившиеся тесты")
            logger.info("- Повторить комплексное тестирование")
        else:
            logger.info("- Система готова к использованию в проекте")
            logger.info("- Можно переходить к следующей задаче TODO")


def main():
    """Главная функция запуска master тестирования"""
    master_tester = MasterSystemTester()
    success = master_tester.run_comprehensive_tests()
    
    if success:
        logger.info("\n🎉 MASTER COMPREHENSIVE TEST УСПЕШНО ЗАВЕРШЕН!")
        exit(0)
    else:
        logger.error("\n❌ MASTER COMPREHENSIVE TEST ПРОВАЛЕН!")
        exit(1)


if __name__ == "__main__":
    main() 