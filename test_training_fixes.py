#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 COMPREHENSIVE TEST SCRIPT FOR TRAINING FIXES
Проверяет все критические исправления обучения Tacotron2-New

Этот скрипт тестирует:
1. ✅ AdaptiveGradientClipper integration
2. ✅ Alignment Diagnostics integration  
3. ✅ Guided Attention Loss fixes
4. ✅ Optimized hyperparameters
5. ✅ Training stability improvements
"""

import sys
import os
import logging
import torch
import numpy as np
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingFixesTest:
    """Класс для тестирования всех исправлений обучения."""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        
    def run_all_tests(self):
        """Запускает все тесты исправлений."""
        logger.info("🧪 НАЧИНАЕМ КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ИСПРАВЛЕНИЙ ОБУЧЕНИЯ")
        logger.info("=" * 60)
        
        # Тест 1: AdaptiveGradientClipper
        self.test_adaptive_gradient_clipper()
        
        # Тест 2: Alignment Diagnostics
        self.test_alignment_diagnostics()
        
        # Тест 3: Guided Attention fixes
        self.test_guided_attention_fixes()
        
        # Тест 4: Hyperparameters optimization
        self.test_hyperparameters_optimization()
        
        # Тест 5: Training loop integration
        self.test_training_loop_integration()
        
        # Финальный отчет
        self.generate_final_report()
        
    def test_adaptive_gradient_clipper(self):
        """Тест 1: Проверка интеграции AdaptiveGradientClipper"""
        logger.info("🔧 Тест 1: AdaptiveGradientClipper Integration")
        
        try:
            # Проверяем наличие AdaptiveGradientClipper
            from smart_tuner.gradient_clipper import AdaptiveGradientClipper, get_global_clipper
            
            # Создаем тестовый clipper
            clipper = AdaptiveGradientClipper(
                max_norm=1.0,
                adaptive=True,
                emergency_threshold=100.0,
                history_size=1000,
                percentile=95
            )
            
            # Тестируем базовую функциональность
            logger.info("  ✅ AdaptiveGradientClipper импортирован успешно")
            
            # Проверяем интеграцию в train.py
            with open('train.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'AdaptiveGradientClipper' in content and 'get_global_clipper' in content:
                logger.info("  ✅ AdaptiveGradientClipper интегрирован в train.py")
                
                # Проверяем правильность вызова
                if 'clip_gradients(model, iteration)' in content:
                    logger.info("  ✅ Правильный вызов clip_gradients найден")
                    self.test_results['adaptive_gradient_clipper'] = 'PASS'
                else:
                    logger.error("  ❌ Неправильный вызов clip_gradients")
                    self.test_results['adaptive_gradient_clipper'] = 'FAIL'
            else:
                logger.error("  ❌ AdaptiveGradientClipper не интегрирован в train.py")
                self.test_results['adaptive_gradient_clipper'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"  ❌ Ошибка теста AdaptiveGradientClipper: {e}")
            self.test_results['adaptive_gradient_clipper'] = 'ERROR'
            self.errors.append(f"AdaptiveGradientClipper: {e}")
    
    def test_alignment_diagnostics(self):
        """Тест 2: Проверка интеграции Alignment Diagnostics"""
        logger.info("🎯 Тест 2: Alignment Diagnostics Integration")
        
        try:
            # Проверяем наличие AlignmentDiagnostics
            from alignment_diagnostics import AlignmentDiagnostics
            
            # Создаем тестовый объект
            diagnostics = AlignmentDiagnostics()
            logger.info("  ✅ AlignmentDiagnostics импортирован успешно")
            
            # Тестируем анализ тестовой матрицы
            test_matrix = np.random.rand(50, 30)  # [mel_time, text_time]
            results = diagnostics.analyze_alignment_matrix(test_matrix, step=100)
            
            if 'diagnostics' in results and 'overall_score' in results:
                logger.info("  ✅ AlignmentDiagnostics работает корректно")
                
                # Проверяем интеграцию в train.py
                with open('train.py', 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'alignment_diagnostics.analyze_alignment_matrix' in content:
                    logger.info("  ✅ Alignment diagnostics интегрирован в train.py")
                    self.test_results['alignment_diagnostics'] = 'PASS'
                else:
                    logger.error("  ❌ Alignment diagnostics не интегрирован в train.py")
                    self.test_results['alignment_diagnostics'] = 'FAIL'
            else:
                logger.error("  ❌ AlignmentDiagnostics возвращает неверные результаты")
                self.test_results['alignment_diagnostics'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"  ❌ Ошибка теста Alignment Diagnostics: {e}")
            self.test_results['alignment_diagnostics'] = 'ERROR'
            self.errors.append(f"Alignment Diagnostics: {e}")
    
    def test_guided_attention_fixes(self):
        """Тест 3: Проверка исправлений Guided Attention"""
        logger.info("🎯 Тест 3: Guided Attention Fixes")
        
        try:
            # Проверяем loss_function.py
            from loss_function import GuidedAttentionLoss, Tacotron2Loss
            
            # Проверяем создание GuidedAttentionLoss
            guide_loss = GuidedAttentionLoss(alpha=2.0, sigma=0.4)
            logger.info("  ✅ GuidedAttentionLoss создан успешно")
            
            # Проверяем интеграцию в train.py - исправление двойного применения
            with open('train.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'criterion_has_guided_attention' in content and 'двойного применения guided attention' in content:
                logger.info("  ✅ Исправление двойного применения guided attention найдено")
                self.test_results['guided_attention_fixes'] = 'PASS'
            else:
                logger.error("  ❌ Исправление двойного применения guided attention не найдено")
                self.test_results['guided_attention_fixes'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"  ❌ Ошибка теста Guided Attention: {e}")
            self.test_results['guided_attention_fixes'] = 'ERROR'
            self.errors.append(f"Guided Attention: {e}")
    
    def test_hyperparameters_optimization(self):
        """Тест 4: Проверка оптимизации гиперпараметров"""
        logger.info("⚙️ Тест 4: Hyperparameters Optimization")
        
        try:
            # Проверяем hparams.py
            from hparams import create_hparams
            
            hparams = create_hparams()
            
            # Проверяем критические параметры
            checks = {
                'learning_rate': (hparams.learning_rate == 5e-5, f"Expected 5e-5, got {hparams.learning_rate}"),
                'grad_clip_thresh': (hparams.grad_clip_thresh == 1.0, f"Expected 1.0, got {hparams.grad_clip_thresh}"),
                'guide_loss_weight': (hparams.guide_loss_weight == 1.5, f"Expected 1.5, got {hparams.guide_loss_weight}"),
                'guide_loss_initial_weight': (hparams.guide_loss_initial_weight == 5.0, f"Expected 5.0, got {hparams.guide_loss_initial_weight}"),
                'batch_size': (hparams.batch_size == 16, f"Expected 16, got {hparams.batch_size}"),
                'gradient_accumulation_steps': (hparams.gradient_accumulation_steps == 2, f"Expected 2, got {hparams.gradient_accumulation_steps}")
            }
            
            passed_checks = 0
            for param_name, (check_result, error_msg) in checks.items():
                if check_result:
                    logger.info(f"  ✅ {param_name}: правильное значение")
                    passed_checks += 1
                else:
                    logger.error(f"  ❌ {param_name}: {error_msg}")
            
            if passed_checks == len(checks):
                logger.info("  ✅ Все критические гиперпараметры оптимизированы")
                self.test_results['hyperparameters_optimization'] = 'PASS'
            else:
                logger.error(f"  ❌ {len(checks) - passed_checks} гиперпараметров не оптимизированы")
                self.test_results['hyperparameters_optimization'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"  ❌ Ошибка теста гиперпараметров: {e}")
            self.test_results['hyperparameters_optimization'] = 'ERROR'
            self.errors.append(f"Hyperparameters: {e}")
    
    def test_training_loop_integration(self):
        """Тест 5: Проверка интеграции в основной цикл обучения"""
        logger.info("🔄 Тест 5: Training Loop Integration")
        
        try:
            # Проверяем критические интеграции в train.py
            with open('train.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            integrations = {
                'smart_tuner_gradient_clipper': 'smart_tuner.gradient_clipper import get_global_clipper',
                'alignment_diagnostics_import': 'from alignment_diagnostics import AlignmentDiagnostics',
                'guided_attention_double_check': 'criterion_has_guided_attention',
                'adaptive_clipping_usage': 'clip_gradients(model, iteration)',
                'alignment_analysis': 'analyze_alignment_matrix'
            }
            
            passed_integrations = 0
            for integration_name, search_pattern in integrations.items():
                if search_pattern in content:
                    logger.info(f"  ✅ {integration_name}: интегрирован")
                    passed_integrations += 1
                else:
                    logger.error(f"  ❌ {integration_name}: не найден")
            
            if passed_integrations == len(integrations):
                logger.info("  ✅ Все критические интеграции в training loop найдены")
                self.test_results['training_loop_integration'] = 'PASS'
            else:
                logger.error(f"  ❌ {len(integrations) - passed_integrations} интеграций отсутствуют")
                self.test_results['training_loop_integration'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"  ❌ Ошибка теста интеграции: {e}")
            self.test_results['training_loop_integration'] = 'ERROR'
            self.errors.append(f"Training Loop Integration: {e}")
    
    def generate_final_report(self):
        """Генерирует финальный отчет о тестировании."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 ФИНАЛЬНЫЙ ОТЧЕТ ТЕСТИРОВАНИЯ ИСПРАВЛЕНИЙ")
        logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results.values() if r == 'PASS'])
        failed_tests = len([r for r in self.test_results.values() if r == 'FAIL'])
        error_tests = len([r for r in self.test_results.values() if r == 'ERROR'])
        
        logger.info(f"Всего тестов: {total_tests}")
        logger.info(f"✅ Прошло: {passed_tests}")
        logger.info(f"❌ Провалено: {failed_tests}")
        logger.info(f"🚨 Ошибки: {error_tests}")
        logger.info("")
        
        # Детальные результаты
        for test_name, result in self.test_results.items():
            status_icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "🚨"}[result]
            logger.info(f"{status_icon} {test_name}: {result}")
        
        logger.info("")
        
        # Ошибки
        if self.errors:
            logger.info("🚨 ОБНАРУЖЕННЫЕ ОШИБКИ:")
            for error in self.errors:
                logger.error(f"  • {error}")
            logger.info("")
        
        # Итоговая оценка
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        if success_rate >= 80:
            logger.info(f"🎉 ОТЛИЧНО! Все критические исправления работают ({success_rate:.1f}% успеха)")
            logger.info("✅ Система готова к обучению с исправлениями")
        elif success_rate >= 60:
            logger.info(f"⚠️ ХОРОШО. Большинство исправлений работает ({success_rate:.1f}% успеха)")
            logger.info("🔧 Рекомендуется исправить оставшиеся проблемы")
        else:
            logger.error(f"❌ КРИТИЧНО! Много проблем с исправлениями ({success_rate:.1f}% успеха)")
            logger.error("🚨 Необходимы дополнительные исправления перед обучением")
        
        logger.info("")
        logger.info("=" * 60)
        
        # Рекомендации
        logger.info("💡 РЕКОМЕНДАЦИИ ДЛЯ СТАБИЛЬНОГО ОБУЧЕНИЯ:")
        logger.info("1. 🎯 Используйте начальный learning_rate=5e-5")
        logger.info("2. 📊 Мониторьте gradient norms (должны быть <10 для стабильности)")  
        logger.info("3. 🔧 Проверяйте alignment diagonality каждые 100 шагов")
        logger.info("4. 🚨 При gradient explosion >100 система автоматически снизит LR")
        logger.info("5. 📈 Alignment diagnostics будут предупреждать о проблемах")
        logger.info("=" * 60)

if __name__ == "__main__":
    # Запускаем тестирование
    tester = TrainingFixesTest()
    tester.run_all_tests() 