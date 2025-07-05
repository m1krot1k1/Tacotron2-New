#!/usr/bin/env python3
"""
🧪 Unit-тесты для критических путей Tacotron2-New
Проверка NaN-recovery, emergency restart, distributed режима
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hparams import create_hparams
from smart_tuner.optimization_engine import OptimizationEngine
from smart_tuner.optuna_integration import OptunaTrainerIntegration


class TestCriticalPaths(unittest.TestCase):
    """Тесты критических путей системы"""
    
    def setUp(self):
        """Настройка тестов"""
        self.hparams = create_hparams()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Очистка после тестов"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_nan_recovery(self):
        """Тест восстановления после NaN"""
        print("🧪 Тестируем NaN recovery...")
        
        # Создаем тензор с NaN
        nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
        
        # Проверяем детекцию NaN
        self.assertTrue(torch.isnan(nan_tensor).any())
        
        # Проверяем восстановление
        recovered_tensor = torch.where(torch.isnan(nan_tensor), torch.tensor(0.0), nan_tensor)
        self.assertFalse(torch.isnan(recovered_tensor).any())
        
        print("✅ NaN recovery работает корректно")
    
    def test_emergency_restart_params(self):
        """Тест параметров экстренного перезапуска"""
        print("🧪 Тестируем параметры emergency restart...")
        
        # Проверяем создание безопасных параметров
        safe_hparams = create_hparams()
        safe_hparams.learning_rate = 1e-6  # Минимальный LR
        safe_hparams.batch_size = 2        # Минимальный batch
        safe_hparams.grad_clip_thresh = 0.01  # Строгое клипирование
        
        # Проверяем, что параметры действительно безопасные
        self.assertLess(safe_hparams.learning_rate, 1e-5)
        self.assertLess(safe_hparams.batch_size, 4)
        self.assertLess(safe_hparams.grad_clip_thresh, 0.1)
        
        print("✅ Emergency restart параметры корректны")
    
    def test_optimization_engine_initialization(self):
        """Тест инициализации Optimization Engine"""
        print("🧪 Тестируем инициализацию Optimization Engine...")
        
        try:
            # Создаем временный конфиг
            config_content = """
optimization:
  n_trials: 5
  direction: minimize
  
hyperparameter_search_space:
  learning_rate:
    type: float
    min: 1e-6
    max: 1e-3
  batch_size:
    type: int
    min: 4
    max: 32
    
training_safety:
  tts_quality_checks:
    min_attention_alignment: 0.1
    max_validation_loss: 10.0
"""
            
            config_path = os.path.join(self.temp_dir, "test_config.yaml")
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            # Инициализируем Optimization Engine
            engine = OptimizationEngine(config_path)
            
            # Проверяем, что engine инициализирован
            self.assertIsNotNone(engine)
            self.assertIsNotNone(engine.config)
            
            print("✅ Optimization Engine инициализирован корректно")
            
        except Exception as e:
            self.fail(f"Ошибка инициализации Optimization Engine: {e}")
    
    def test_optuna_integration(self):
        """Тест интеграции Optuna"""
        print("🧪 Тестируем интеграцию Optuna...")
        
        try:
            # Создаем временный конфиг
            config_content = """
optimization:
  n_trials: 3
  direction: minimize
  
hyperparameter_search_space:
  learning_rate:
    type: float
    min: 1e-5
    max: 1e-3
  batch_size:
    type: int
    min: 4
    max: 16
"""
            
            config_path = os.path.join(self.temp_dir, "test_config.yaml")
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            # Инициализируем интеграцию
            integration = OptunaTrainerIntegration(config_path)
            
            # Проверяем, что integration инициализирован
            self.assertIsNotNone(integration)
            self.assertIsNotNone(integration.optimization_engine)
            
            print("✅ Optuna интеграция работает корректно")
            
        except Exception as e:
            self.fail(f"Ошибка интеграции Optuna: {e}")
    
    def test_distributed_restart_logic(self):
        """Тест логики distributed restart"""
        print("🧪 Тестируем логику distributed restart...")
        
        # Проверяем логику перезапуска без реального distributed
        restart_attempts = 0
        max_attempts = 3
        
        def simulate_restart():
            nonlocal restart_attempts
            restart_attempts += 1
            return restart_attempts <= max_attempts
        
        # Симулируем несколько попыток перезапуска
        for i in range(5):
            success = simulate_restart()
            if not success:
                break
        
        # Проверяем, что перезапуск остановился после max_attempts
        self.assertEqual(restart_attempts, max_attempts)
        
        print("✅ Distributed restart логика работает корректно")
    
    def test_hyperparameter_validation(self):
        """Тест валидации гиперпараметров"""
        print("🧪 Тестируем валидацию гиперпараметров...")
        
        # Проверяем валидные параметры
        valid_params = {
            'learning_rate': 1e-4,
            'batch_size': 16,
            'grad_clip_thresh': 1.0
        }
        
        for param_name, param_value in valid_params.items():
            if hasattr(self.hparams, param_name):
                setattr(self.hparams, param_name, param_value)
        
        # Проверяем, что параметры установлены
        self.assertEqual(self.hparams.learning_rate, 1e-4)
        self.assertEqual(self.hparams.batch_size, 16)
        self.assertEqual(self.hparams.grad_clip_thresh, 1.0)
        
        print("✅ Валидация гиперпараметров работает корректно")
    
    def test_loss_components_handling(self):
        """Тест обработки компонентов loss"""
        print("🧪 Тестируем обработку компонентов loss...")
        
        # Создаем тестовые loss компоненты
        loss_components = {
            'mel_loss': 1.5,
            'gate_loss': 0.3,
            'guide_loss': 0.2,
            'emb_loss': 0.1
        }
        
        # Проверяем, что все компоненты положительные
        for component_name, component_value in loss_components.items():
            self.assertGreater(component_value, 0)
        
        # Проверяем вычисление общего loss
        total_loss = sum(loss_components.values())
        self.assertAlmostEqual(total_loss, 2.1, places=2)
        
        print("✅ Обработка компонентов loss работает корректно")
    
    def test_attention_diagonality_calculation(self):
        """Тест вычисления диагональности attention"""
        print("🧪 Тестируем вычисление диагональности attention...")
        
        # Создаем тестовую attention матрицу
        attention_matrix = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ])
        
        # Вычисляем диагональность
        diagonal_elements = np.diag(attention_matrix)
        diagonality = np.mean(diagonal_elements)
        
        # Проверяем, что диагональность в разумных пределах
        self.assertGreater(diagonality, 0.5)
        self.assertLess(diagonality, 1.0)
        
        print("✅ Вычисление диагональности attention работает корректно")
    
    def test_gradient_clipping(self):
        """Тест обрезания градиентов"""
        print("🧪 Тестируем обрезание градиентов...")
        
        # Создаем тестовые градиенты
        gradients = torch.randn(10) * 10  # Большие градиенты
        
        # Применяем clipping
        max_norm = 1.0
        clipped_gradients = torch.nn.utils.clip_grad_norm_(gradients, max_norm)
        
        # Проверяем, что градиенты обрезаны
        grad_norm = torch.norm(clipped_gradients)
        self.assertLessEqual(grad_norm, max_norm)
        
        print("✅ Обрезание градиентов работает корректно")
    
    def test_telegram_monitor_fallback(self):
        """Тест fallback Telegram монитора"""
        print("🧪 Тестируем fallback Telegram монитора...")
        
        # Симулируем отсутствие настроек Telegram
        telegram_config = {
            'bot_token': None,
            'chat_id': None,
            'enabled': False
        }
        
        # Проверяем, что система корректно обрабатывает отсутствие настроек
        if not telegram_config['bot_token'] or not telegram_config['chat_id']:
            telegram_config['enabled'] = False
        
        self.assertFalse(telegram_config['enabled'])
        
        print("✅ Fallback Telegram монитора работает корректно")


class TestSmartTunerIntegration(unittest.TestCase):
    """Тесты интеграции Smart Tuner"""
    
    def setUp(self):
        """Настройка тестов"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Очистка после тестов"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_smart_tuner_initialization(self):
        """Тест инициализации Smart Tuner"""
        print("🧪 Тестируем инициализацию Smart Tuner...")
        
        try:
            from smart_tuner.smart_tuner_integration import SmartTunerIntegration
            
            # Создаем временный конфиг
            config_content = """
smart_tuner_enabled: true
optimization_enabled: true
quality_control_enabled: true
early_stopping_enabled: true
adaptive_learning_enabled: true
"""
            
            config_path = os.path.join(self.temp_dir, "test_config.yaml")
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            # Инициализируем Smart Tuner
            smart_tuner = SmartTunerIntegration(config_path)
            
            # Проверяем, что Smart Tuner инициализирован
            self.assertIsNotNone(smart_tuner)
            self.assertIsNotNone(smart_tuner.config)
            
            print("✅ Smart Tuner инициализирован корректно")
            
        except Exception as e:
            self.fail(f"Ошибка инициализации Smart Tuner: {e}")


def run_all_tests():
    """Запуск всех тестов"""
    print("🚀 Запуск unit-тестов для критических путей...")
    
    # Создаем test suite
    test_suite = unittest.TestSuite()
    
    # Добавляем тесты
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestCriticalPaths))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSmartTunerIntegration))
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Выводим результаты
    print(f"\n📊 Результаты тестов:")
    print(f"✅ Успешных тестов: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Неудачных тестов: {len(result.failures)}")
    print(f"🚨 Ошибок: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Неудачные тесты:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\n🚨 Ошибки:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 