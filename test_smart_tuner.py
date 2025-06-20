#!/usr/bin/env python3
"""Тестирование Smart Tuner V2"""

import subprocess
import time
import os

def test_basic_training():
    """Тест базового обучения"""
    print("🧪 Тестирование базового обучения...")
    
    cmd = [
        'python', 'train.py',
        '-o', 'output/test_basic',
        '-l', 'output/test_basic/logs',
        '--hparams=epochs=1,iters_per_checkpoint=5,batch_size=2'
    ]
    
    try:
        result = subprocess.run(cmd, timeout=60, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Базовое обучение работает")
            return True
        else:
            print(f"❌ Ошибка обучения: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Обучение превысило таймаут (60 сек)")
        return False
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")
        return False

def test_smart_tuner():
    """Тест Smart Tuner"""
    print("\n🚀 Тестирование Smart Tuner...")
    
    cmd = ['python', 'smart_tuner_main.py', '--mode', 'train']
    
    try:
        result = subprocess.run(cmd, timeout=30, capture_output=True, text=True)
        if "Процесс обучения запущен" in result.stderr:
            print("✅ Smart Tuner запускается")
            return True
        else:
            print(f"❌ Smart Tuner не работает: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Ошибка Smart Tuner: {e}")
        return False

def main():
    print("🔬 Тестирование системы обучения Tacotron2 + Smart Tuner V2")
    print("=" * 60)
    
    # Создаем необходимые директории
    os.makedirs("output/test_basic", exist_ok=True)
    os.makedirs("output/test_basic/logs", exist_ok=True)
    
    # Тестируем компоненты
    basic_ok = test_basic_training()
    smart_ok = test_smart_tuner()
    
    print("\n📊 Результаты тестирования:")
    print(f"  Базовое обучение: {'✅' if basic_ok else '❌'}")
    print(f"  Smart Tuner: {'✅' if smart_ok else '❌'}")
    
    if basic_ok and smart_ok:
        print("\n🎉 Все системы готовы к работе!")
        print("\nДля запуска полной оптимизации используйте:")
        print("  python smart_tuner_main.py --mode optimize --trials 10")
    else:
        print("\n⚠️ Есть проблемы, требующие решения")

if __name__ == '__main__':
    main()
