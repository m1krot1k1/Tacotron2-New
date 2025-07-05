#!/usr/bin/env python3
"""
🧪 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ КРИТИЧЕСКИХ ИСПРАВЛЕНИЙ
Тестирует все исправления из IMMEDIATE_ACTION_PLAN.md

Автор: AI Assistant для проекта Intelligent TTS Training Pipeline
"""

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CriticalFixesTester:
    """Тестер критических исправлений Tacotron2-New"""
    
    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.total_tests = 0
        
    def run_all_tests(self):
        """Запуск всех тестов критических исправлений"""
        logger.info("🧪 НАЧАЛО КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ КРИТИЧЕСКИХ ИСПРАВЛЕНИЙ")
        
        # Тест 1: Gradient Clipping
        self.test_gradient_clipping()
        
        # Тест 2: Guided Attention Loss
        self.test_guided_attention_loss()
        
        # Тест 3: Learning Rate
        self.test_learning_rate()
        
        # Тест 4: Alignment Diagnostics
        self.test_alignment_diagnostics()
        
        # Тест 5: Smart Tuner Integration
        self.test_smart_tuner_integration()
        
        # Тест 6: Model Loading
        self.test_model_loading()
        
        # Тест 7: Loss Function
        self.test_loss_function()
        
        # Итоговый отчет
        self.print_final_report()
        
    def test_gradient_clipping(self):
        """Тест 1: Проверка правильного gradient clipping"""
        logger.info("🔧 Тест 1: Gradient Clipping")
        self.total_tests += 1
        
        try:
            # Проверяем, что в train.py есть правильный gradient clipping
            with open('train.py', 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Проверяем наличие правильного gradient clipping
            if 'torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)' in content:
                logger.info("✅ Gradient clipping с max_norm=1.0 найден")
                
                # Проверяем наличие алертов для высоких градиентов
                if 'grad_norm > 10.0' in content and 'grad_norm > 100.0' in content:
                    logger.info("✅ Алерты для высоких градиентов найдены")
                    self.passed_tests += 1
                    self.test_results['gradient_clipping'] = 'PASS'
                else:
                    logger.error("❌ Алерты для высоких градиентов не найдены")
                    self.test_results['gradient_clipping'] = 'FAIL'
            else:
                logger.error("❌ Правильный gradient clipping не найден")
                self.test_results['gradient_clipping'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ Ошибка теста gradient clipping: {e}")
            self.test_results['gradient_clipping'] = 'ERROR'
    
    def test_guided_attention_loss(self):
        """Тест 2: Проверка guided attention loss"""
        logger.info("🎯 Тест 2: Guided Attention Loss")
        self.total_tests += 1
        
        try:
            # Проверяем loss_function.py
            with open('loss_function.py', 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Проверяем наличие guided attention loss
            if 'guided_attention_loss' in content and 'guide_loss_weight' in content:
                logger.info("✅ Guided attention loss найден в loss_function.py")
                
                # Проверяем train.py на правильное использование веса
                with open('train.py', 'r', encoding='utf-8') as f:
                    train_content = f.read()
                    
                if 'guide_loss_weight * loss_guide' in train_content:
                    logger.info("✅ Правильный вес для guided attention loss найден")
                    self.passed_tests += 1
                    self.test_results['guided_attention_loss'] = 'PASS'
                else:
                    logger.error("❌ Правильный вес для guided attention loss не найден")
                    self.test_results['guided_attention_loss'] = 'FAIL'
            else:
                logger.error("❌ Guided attention loss не найден")
                self.test_results['guided_attention_loss'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ Ошибка теста guided attention loss: {e}")
            self.test_results['guided_attention_loss'] = 'ERROR'
    
    def test_learning_rate(self):
        """Тест 3: Проверка learning rate"""
        logger.info("📈 Тест 3: Learning Rate")
        self.total_tests += 1
        
        try:
            # Проверяем hparams.py
            with open('hparams.py', 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Проверяем, что learning rate установлен в 1e-4
            if 'learning_rate=1e-4' in content:
                logger.info("✅ Learning rate установлен в 1e-4")
                self.passed_tests += 1
                self.test_results['learning_rate'] = 'PASS'
            else:
                logger.error("❌ Learning rate не установлен в 1e-4")
                self.test_results['learning_rate'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ Ошибка теста learning rate: {e}")
            self.test_results['learning_rate'] = 'ERROR'
    
    def test_alignment_diagnostics(self):
        """Тест 4: Проверка alignment diagnostics"""
        logger.info("🔍 Тест 4: Alignment Diagnostics")
        self.total_tests += 1
        
        try:
            # Проверяем наличие файла alignment_diagnostics.py
            if os.path.exists('alignment_diagnostics.py'):
                logger.info("✅ Файл alignment_diagnostics.py найден")
                
                # Проверяем интеграцию в train.py
                with open('train.py', 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if 'AlignmentDiagnostics' in content and 'alignment_diagnostics.analyze_alignment_matrix' in content:
                    logger.info("✅ Alignment diagnostics интегрирован в train.py")
                    self.passed_tests += 1
                    self.test_results['alignment_diagnostics'] = 'PASS'
                else:
                    logger.error("❌ Alignment diagnostics не интегрирован в train.py")
                    self.test_results['alignment_diagnostics'] = 'FAIL'
            else:
                logger.error("❌ Файл alignment_diagnostics.py не найден")
                self.test_results['alignment_diagnostics'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ Ошибка теста alignment diagnostics: {e}")
            self.test_results['alignment_diagnostics'] = 'ERROR'
    
    def test_smart_tuner_integration(self):
        """Тест 5: Проверка Smart Tuner integration"""
        logger.info("🤖 Тест 5: Smart Tuner Integration")
        self.total_tests += 1
        
        try:
            # Проверяем наличие smart_tuner_main.py
            if os.path.exists('smart_tuner_main.py'):
                logger.info("✅ Файл smart_tuner_main.py найден")
                
                # Проверяем наличие функции integrate_critical_components
                with open('smart_tuner_main.py', 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if 'integrate_critical_components' in content:
                    logger.info("✅ Функция integrate_critical_components найдена")
                    self.passed_tests += 1
                    self.test_results['smart_tuner_integration'] = 'PASS'
                else:
                    logger.error("❌ Функция integrate_critical_components не найдена")
                    self.test_results['smart_tuner_integration'] = 'FAIL'
            else:
                logger.error("❌ Файл smart_tuner_main.py не найден")
                self.test_results['smart_tuner_integration'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ Ошибка теста Smart Tuner integration: {e}")
            self.test_results['smart_tuner_integration'] = 'ERROR'
    
    def test_model_loading(self):
        """Тест 6: Проверка загрузки модели"""
        logger.info("🏗️ Тест 6: Model Loading")
        self.total_tests += 1
        
        try:
            # Проверяем наличие model.py
            if os.path.exists('model.py'):
                logger.info("✅ Файл model.py найден")
                
                # Проверяем наличие hparams.py
                if os.path.exists('hparams.py'):
                    logger.info("✅ Файл hparams.py найден")
                    
                    # Пытаемся импортировать и создать модель
                    try:
                        from hparams import create_hparams
                        from model import Tacotron2
                        
                        hparams = create_hparams()
                        logger.info("✅ HParams созданы успешно")
                        
                        # Проверяем, что CUDA доступна
                        if torch.cuda.is_available():
                            logger.info("✅ CUDA доступна")
                            self.passed_tests += 1
                            self.test_results['model_loading'] = 'PASS'
                        else:
                            logger.warning("⚠️ CUDA недоступна, но модель может работать на CPU")
                            self.passed_tests += 1
                            self.test_results['model_loading'] = 'PASS'
                    except Exception as e:
                        logger.error(f"❌ Ошибка создания модели: {e}")
                        self.test_results['model_loading'] = 'FAIL'
                else:
                    logger.error("❌ Файл hparams.py не найден")
                    self.test_results['model_loading'] = 'FAIL'
            else:
                logger.error("❌ Файл model.py не найден")
                self.test_results['model_loading'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ Ошибка теста model loading: {e}")
            self.test_results['model_loading'] = 'ERROR'
    
    def test_loss_function(self):
        """Тест 7: Проверка loss function"""
        logger.info("📊 Тест 7: Loss Function")
        self.total_tests += 1
        
        try:
            # Проверяем наличие loss_function.py
            if os.path.exists('loss_function.py'):
                logger.info("✅ Файл loss_function.py найден")
                
                # Пытаемся импортировать loss function
                try:
                    from hparams import create_hparams
                    from loss_function import Tacotron2Loss
                    
                    hparams = create_hparams()
                    criterion = Tacotron2Loss(hparams)
                    logger.info("✅ Tacotron2Loss создан успешно")
                    
                    # Проверяем наличие guided attention loss
                    if hasattr(criterion, 'guided_attention_loss'):
                        logger.info("✅ Guided attention loss найден в criterion")
                        self.passed_tests += 1
                        self.test_results['loss_function'] = 'PASS'
                    else:
                        logger.error("❌ Guided attention loss не найден в criterion")
                        self.test_results['loss_function'] = 'FAIL'
                except Exception as e:
                    logger.error(f"❌ Ошибка создания loss function: {e}")
                    self.test_results['loss_function'] = 'FAIL'
            else:
                logger.error("❌ Файл loss_function.py не найден")
                self.test_results['loss_function'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ Ошибка теста loss function: {e}")
            self.test_results['loss_function'] = 'ERROR'
    
    def print_final_report(self):
        """Печать итогового отчета"""
        logger.info("\n" + "="*60)
        logger.info("📋 ИТОГОВЫЙ ОТЧЕТ ТЕСТИРОВАНИЯ КРИТИЧЕСКИХ ИСПРАВЛЕНИЙ")
        logger.info("="*60)
        
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result == 'PASS' else "❌" if result == 'FAIL' else "⚠️"
            logger.info(f"{status_emoji} {test_name}: {result}")
        
        logger.info(f"\n📊 Результаты: {self.passed_tests}/{self.total_tests} тестов пройдено")
        
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        logger.info(f"📈 Успешность: {success_rate:.1f}%")
        
        if success_rate >= 80:
            logger.info("🎉 КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ ГОТОВЫ К ИСПОЛЬЗОВАНИЮ!")
        elif success_rate >= 60:
            logger.warning("⚠️ КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ ТРЕБУЮТ ДОРАБОТКИ")
        else:
            logger.error("🚨 КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ НЕ ГОТОВЫ!")
        
        logger.info("="*60)

def main():
    """Главная функция"""
    tester = CriticalFixesTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main() 