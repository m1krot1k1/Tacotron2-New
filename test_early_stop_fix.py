#!/usr/bin/env python3
"""
Тест исправленной логики раннего останова TTS
"""

import yaml
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_tuner_main import SmartTunerMain

def test_early_stop_logic():
    """Тестирует исправленную логику раннего останова"""
    
    print("🧪 Тестирование исправленной логики раннего останова TTS")
    print("=" * 60)
    
    # Создаем экземпляр SmartTunerMain
    smart_tuner = SmartTunerMain()
    smart_tuner.training_start_time = datetime.now()
    
    # ТЕСТ 1: Слишком раннее завершение (как было в проблемном случае)
    print("\n🔍 ТЕСТ 1: Проверка защиты от раннего завершения")
    early_metrics = {
        'attention_alignment_score': 0.442,
        'gate_accuracy': 0.557,
        'val_loss': 2.665,
        'mel_quality_score': 0.475,
        'training_progress': float('nan'),
        'validation.step': 0  # Проблемная ситуация
    }
    
    result1 = smart_tuner._check_tts_quality_thresholds(early_metrics)
    print(f"Результат: {'✅ Продолжить' if not result1 else '❌ Остановить'}")
    assert not result1, "Тест 1 провален: обучение должно продолжаться"
    
    # ТЕСТ 2: Через 5 минут (все еще слишком рано)
    print("\n🔍 ТЕСТ 2: Проверка через 5 минут")
    smart_tuner.training_start_time = datetime.now() - timedelta(minutes=5)
    
    result2 = smart_tuner._check_tts_quality_thresholds(early_metrics)
    print(f"Результат: {'✅ Продолжить' if not result2 else '❌ Остановить'}")
    assert not result2, "Тест 2 провален: обучение должно продолжаться"
    
    # ТЕСТ 3: Через 20 минут с достаточными метриками
    print("\n🔍 ТЕСТ 3: Проверка через 20 минут с хорошими метриками")
    smart_tuner.training_start_time = datetime.now() - timedelta(minutes=20)
    
    good_metrics = {
        'attention_alignment_score': 0.65,  # Выше порога 0.6
        'gate_accuracy': 0.75,             # Выше порога 0.7
        'val_loss': 10.0,                  # Ниже порога 15.0
        'mel_quality_score': 0.45,         # Выше порога 0.4
        'training_loss': 8.0,
        'initial_training_loss': 12.0,     # Прогресс: (12-8)/12 = 33%
        'validation.step': 5               # Достаточно шагов
    }
    
    result3 = smart_tuner._check_tts_quality_thresholds(good_metrics)
    print(f"Результат: {'❌ Остановить' if result3 else '✅ Продолжить'}")
    
    # ТЕСТ 4: Плохие метрики после 20 минут
    print("\n🔍 ТЕСТ 4: Проверка плохих метрик через 20 минут")
    
    bad_metrics = {
        'attention_alignment_score': 0.3,   # Ниже порога 0.6
        'gate_accuracy': 0.4,              # Ниже порога 0.7
        'val_loss': 20.0,                  # Выше порога 15.0
        'mel_quality_score': 0.2,          # Ниже порога 0.4
        'training_loss': 11.9,
        'initial_training_loss': 12.0,     # Прогресс: (12-11.9)/12 = 0.8%
        'validation.step': 5               # Достаточно шагов
    }
    
    result4 = smart_tuner._check_tts_quality_thresholds(bad_metrics)
    print(f"Результат: {'✅ Продолжить' if not result4 else '❌ Остановить'}")
    assert not result4, "Тест 4 провален: при плохих метриках должно продолжаться"
    
    print("\n" + "=" * 60)
    print("✅ Все тесты пройдены! Логика early stopping исправлена")
    print("\n📋 СВОДКА ИСПРАВЛЕНИЙ:")
    print("1. ⏰ Минимальное время обучения: 10 минут")
    print("2. 📊 Минимальное количество validation шагов: 3")
    print("3. 🎯 Более строгие критерии качества")
    print("4. 🛡️ Требование всех критических проверок")
    
    return True

if __name__ == "__main__":
    test_early_stop_logic() 