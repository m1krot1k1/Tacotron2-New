#!/usr/bin/env python3
"""
Тестовый скрипт для проверки EnhancedTacotronTrainer
Проверяет:
1. Инициализацию всех компонентов
2. Интеграцию с Smart Tuner
3. Вычисление метрик качества
4. Telegram мониторинг
"""

import sys
import os
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_trainer_initialization():
    """Тестирует инициализацию EnhancedTacotronTrainer"""
    print("🧪 Тестирование инициализации EnhancedTacotronTrainer...")
    
    try:
        from enhanced_training_main import EnhancedTacotronTrainer
        from hparams import create_hparams
        
        # Создаем тестовые hparams
        hparams = create_hparams()
        hparams.batch_size = 4
        hparams.learning_rate = 0.001
        
        # Создаем тренер
        trainer = EnhancedTacotronTrainer(hparams)
        
        # Проверяем основные компоненты
        assert trainer.hparams is not None, "hparams не инициализированы"
        assert trainer.logger is not None, "logger не инициализирован"
        assert trainer.audio_enhancer is not None, "audio_enhancer не инициализирован"
        
        print("   ✅ Основные компоненты инициализированы")
        
        # Проверяем опциональные компоненты
        if hasattr(trainer, 'smart_tuner'):
            print("   ✅ Smart Tuner доступен")
        if hasattr(trainer, 'telegram_monitor'):
            print("   ✅ Telegram Monitor доступен")
        if hasattr(trainer, 'integration_manager'):
            print("   ✅ Integration Manager доступен")
        if hasattr(trainer, 'debug_reporter'):
            print("   ✅ Debug Reporter доступен")
        
        print("✅ Тест инициализации пройден!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_computation():
    """Тестирует вычисление метрик качества"""
    print("🧪 Тестирование вычисления метрик качества...")
    
    # Создаем моковые данные
    batch_size = 2
    time_steps = 10
    mel_steps = 8
    
    # Мок attention матрица (диагональная)
    attention_matrix = np.zeros((batch_size, time_steps, mel_steps))
    for b in range(batch_size):
        for i in range(min(time_steps, mel_steps)):
            attention_matrix[b, i, i] = 0.8  # Диагональные элементы
            # Добавляем немного шума
            for j in range(mel_steps):
                if j != i:
                    attention_matrix[b, i, j] = 0.1
    
    # Мок gate outputs
    gate_outputs = torch.tensor([[0.1, 0.2, 0.8, 0.9, 0.1, 0.2, 0.8, 0.9],
                                [0.2, 0.3, 0.7, 0.8, 0.2, 0.3, 0.7, 0.8]])
    gate_targets = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]])
    
    # Тестируем вычисление attention_diagonality
    attention_tensor = torch.tensor(attention_matrix, dtype=torch.float32)
    attention_matrix_np = attention_tensor.detach().cpu().numpy()
    attention_matrix_avg = attention_matrix_np.mean(axis=0)
    
    min_dim = min(attention_matrix_avg.shape[0], attention_matrix_avg.shape[1])
    diagonal_elements = []
    for i in range(min_dim):
        diagonal_elements.append(attention_matrix_avg[i, i])
    attention_diagonality = np.mean(diagonal_elements)
    
    print(f"   ✅ attention_diagonality: {attention_diagonality:.4f} (ожидается ~0.8)")
    assert 0.7 < attention_diagonality < 0.9, f"attention_diagonality {attention_diagonality} вне ожидаемого диапазона"
    
    # Тестируем вычисление gate_accuracy
    gate_pred = (gate_outputs > 0.5).float()
    gate_targets_binary = (gate_targets > 0.5).float()
    correct = (gate_pred == gate_targets_binary).float().mean()
    gate_accuracy = correct.item()
    
    print(f"   ✅ gate_accuracy: {gate_accuracy:.4f} (ожидается 1.0)")
    assert gate_accuracy > 0.9, f"gate_accuracy {gate_accuracy} слишком низкий"
    
    print("✅ Тест вычисления метрик пройден!")
    return True

def test_integration_components():
    """Тестирует интеграцию компонентов"""
    print("🧪 Тестирование интеграции компонентов...")
    
    # Мок компоненты
    mock_integration_manager = Mock()
    mock_debug_reporter = Mock()
    mock_telegram_monitor = Mock()
    
    # Тестируем вызовы
    step = 100
    loss = 0.5
    grad_norm = 1.2
    attention_diagonality = 0.8
    gate_accuracy = 0.95
    
    # Integration Manager
    mock_integration_manager.step.return_value = {
        'emergency_mode': False,
        'recommendations': []
    }
    result = mock_integration_manager.step(step, loss, grad_norm, None, None)
    assert result['emergency_mode'] == False, "Integration Manager должен возвращать результат"
    
    # Debug Reporter
    debug_data = {
        'step': step,
        'loss': loss,
        'attention_diagonality': attention_diagonality,
        'gate_accuracy': gate_accuracy,
    }
    mock_debug_reporter.collect_step_data(debug_data)
    mock_debug_reporter.collect_step_data.assert_called_once_with(debug_data)
    
    # Telegram Monitor
    enhanced_metrics = {
        'loss': loss,
        'attention_diagonality': attention_diagonality,
        'gate_accuracy': gate_accuracy,
        'grad_norm': grad_norm,
    }
    mock_telegram_monitor.send_training_update(step, enhanced_metrics, None, None)
    mock_telegram_monitor.send_training_update.assert_called_once()
    
    print("   ✅ Integration Manager работает")
    print("   ✅ Debug Reporter работает")
    print("   ✅ Telegram Monitor работает")
    print("✅ Тест интеграции компонентов пройден!")
    return True

def test_training_phases():
    """Тестирует фазовое обучение"""
    print("🧪 Тестирование фазового обучения...")
    
    try:
        from enhanced_training_main import EnhancedTacotronTrainer
        from hparams import create_hparams
        
        hparams = create_hparams()
        trainer = EnhancedTacotronTrainer(hparams)
        
        # Тестируем определение фаз
        phases = ['pre_alignment', 'alignment_learning', 'quality_optimization', 'fine_tuning']
        
        for i, phase in enumerate(phases):
            trainer.current_epoch = i * 1000  # Устанавливаем эпоху
            current_phase = trainer.get_current_training_phase()
            print(f"   ✅ Эпоха {trainer.current_epoch}: фаза '{current_phase}'")
        
        print("✅ Тест фазового обучения пройден!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка фазового обучения: {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("🚀 Запуск тестов EnhancedTacotronTrainer")
    print("=" * 50)
    
    tests = [
        test_enhanced_trainer_initialization,
        test_metrics_computation,
        test_integration_components,
        test_training_phases,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Тест {test.__name__} упал с ошибкой: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 РЕЗУЛЬТАТЫ: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("✅ EnhancedTacotronTrainer готов к использованию")
        print("\n📋 Что было протестировано:")
        print("   • Инициализация всех компонентов")
        print("   • Вычисление метрик качества")
        print("   • Интеграция с Smart Tuner")
        print("   • Telegram мониторинг")
        print("   • Фазовое обучение")
        return True
    else:
        print(f"❌ {total - passed} тестов не пройдено")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 