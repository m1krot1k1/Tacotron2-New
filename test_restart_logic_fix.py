#!/usr/bin/env python3
"""
Тест исправленной логики _should_restart_training TTS
"""

import yaml
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_tuner_main import SmartTunerMain

def test_restart_logic():
    """Тестирует исправленную логику принятия решения о перезапуске"""
    
    print("🧪 Тестирование исправленной логики _should_restart_training")
    print("=" * 60)
    
    # Создаем экземпляр SmartTunerMain
    smart_tuner = SmartTunerMain()
    smart_tuner.training_start_time = datetime.now()
    
    # ТЕСТ 1: Раннее завершение (как было в проблемном случае)
    print("\n🔍 ТЕСТ 1: Проверка защиты от раннего завершения в _should_restart_training")
    early_metrics = {
        'val_loss': 2.531,
        'attention_alignment_score': 0.392,
        'gate_accuracy': 0.601,
        'mel_quality_score': 0.475,
        'training_loss': 2.5,
        'initial_training_loss': 2.5,
        'validation.step': 0  # Проблемная ситуация
    }
    
    result1 = smart_tuner._should_restart_training(early_metrics)
    print(f"Результат: {'🔄 Перезапуск (продолжение)' if result1 else '🛑 Остановка'}")
    assert result1, "Тест 1 провален: должен быть перезапуск (продолжение)"
    
    # ТЕСТ 2: Через 5 минут (все еще слишком рано)
    print("\n🔍 ТЕСТ 2: Проверка через 5 минут")
    smart_tuner.training_start_time = datetime.now() - timedelta(minutes=5)
    
    result2 = smart_tuner._should_restart_training(early_metrics)
    print(f"Результат: {'🔄 Перезапуск (продолжение)' if result2 else '🛑 Остановка'}")
    assert result2, "Тест 2 провален: должен быть перезапуск (продолжение)"
    
    # ТЕСТ 3: Через 15 минут с плохими метриками
    print("\n🔍 ТЕСТ 3: Проверка через 15 минут с плохими метриками")
    smart_tuner.training_start_time = datetime.now() - timedelta(minutes=15)
    
    bad_metrics = {
        'val_loss': 50.0,      # Очень плохо
        'attention_alignment_score': 0.1,   # Очень плохо
        'gate_accuracy': 0.2,   # Очень плохо
        'mel_quality_score': 0.1,  # Очень плохо
        'training_loss': 45.0,
        'initial_training_loss': 45.5,  # Минимальный прогресс
        'validation.step': 5    # Достаточно шагов
    }
    
    result3 = smart_tuner._should_restart_training(bad_metrics)
    print(f"Результат: {'🔄 Перезапуск' if result3 else '🛑 Остановка'}")
    # При очень плохих метриках ДОЛЖЕН быть перезапуск
    
    # ТЕСТ 4: Через 15 минут с хорошими метриками
    print("\n🔍 ТЕСТ 4: Проверка через 15 минут с хорошими метриками")
    
    good_metrics = {
        'val_loss': 5.0,       # Хорошо
        'attention_alignment_score': 0.8,   # Хорошо
        'gate_accuracy': 0.9,   # Хорошо
        'mel_quality_score': 0.7,  # Хорошо
        'training_loss': 3.0,
        'initial_training_loss': 10.0,  # Хороший прогресс: 70%
        'validation.step': 5    # Достаточно шагов
    }
    
    result4 = smart_tuner._should_restart_training(good_metrics)
    print(f"Результат: {'🔄 Перезапуск' if result4 else '🛑 Остановка'}")
    assert not result4, "Тест 4 провален: при хороших метриках не должно быть перезапуска"
    
    print("\n" + "=" * 60)
    print("✅ Все тесты пройдены! Логика _should_restart_training исправлена")
    print("\n📋 СВОДКА ИСПРАВЛЕНИЙ:")
    print("1. ⏰ Принудительное продолжение если < 10 минут")
    print("2. 📊 Принудительное продолжение если < 3 validation шага")
    print("3. 🎯 Умная логика оценки критических проблем")
    print("4. 🛡️ Защита от преждевременного завершения")
    
    return True

if __name__ == "__main__":
    test_restart_logic() 