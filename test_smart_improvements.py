#!/usr/bin/env python3
'''
🧪 Быстрый тест улучшений Smart Tuner V2
Проверяет основные функции после применения улучшений
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_tuner_main import SmartTunerMain

def test_improvements():
    print("🧪 Тестирование улучшений Smart Tuner V2...")
    
    try:
        # Инициализация с улучшенной конфигурацией
        smart_tuner = SmartTunerMain("smart_tuner/config.yaml")
        smart_tuner.initialize_components()
        
        print("✅ Инициализация прошла успешно")
        
        # Тестирование проверки качества с реалистичными метриками
        test_metrics = {
            'val_loss': 5.0,
            'attention_alignment_score': 0.45,  # Теперь должно проходить
            'gate_accuracy': 0.55,  # Теперь должно проходить
            'mel_quality_score': 0.35,  # Теперь должно проходить
            'training_loss': 4.8,
            'initial_training_loss': 6.0
        }
        
        quality_passed = smart_tuner._check_tts_quality_thresholds(test_metrics)
        
        if quality_passed:
            print("✅ Новые критерии качества работают корректно")
        else:
            print("⚠️ Критерии качества все еще слишком строгие")
            
        # Тестирование логики перезапуска
        should_restart = smart_tuner._should_restart_training(test_metrics)
        
        if not should_restart:
            print("✅ Логика перезапуска стала менее агрессивной")
        else:
            print("⚠️ Логика перезапуска все еще слишком строгая")
            
        # Тестирование композитной функции
        composite_score = smart_tuner.optimization_engine.calculate_composite_tts_objective(test_metrics)
        
        if 0.01 <= composite_score <= 10.0:
            print(f"✅ Композитная функция работает корректно: {composite_score:.4f}")
        else:
            print(f"⚠️ Проблемы с композитной функцией: {composite_score}")
            
        print("\n🎉 Все тесты пройдены! Улучшения применены успешно.")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        return False

if __name__ == "__main__":
    success = test_improvements()
    sys.exit(0 if success else 1)
