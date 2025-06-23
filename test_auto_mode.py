#!/usr/bin/env python3
"""
🧪 Тест автоматического режима Smart Tuner V2
Проверяет работу нового режима "оптимизация + обучение"
"""

import sys
import os
import subprocess
import time
from datetime import datetime

def test_auto_mode():
    """Тестирует автоматический режим"""
    print("🧪 Тестирование автоматического режима Smart Tuner V2")
    print("=" * 60)
    
    # Проверяем доступность файлов
    required_files = [
        "smart_tuner_main.py",
        "smart_tuner/config.yaml"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Не найден файл: {file_path}")
            return False
    
    print("✅ Все необходимые файлы найдены")
    
    # Проверяем аргументы командной строки
    print("🔍 Проверка аргументов командной строки...")
    
    try:
        # Проверяем help
        python_path = "venv/bin/python" if os.path.exists("venv/bin/python") else sys.executable
        result = subprocess.run([
            python_path, "smart_tuner_main.py", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if "auto" in result.stdout:
            print("✅ Автоматический режим доступен в аргументах")
        else:
            print("❌ Автоматический режим не найден в help")
            print(f"Stderr: {result.stderr[:200]}")
            if result.stdout:
                print(f"Stdout: {result.stdout[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при проверке аргументов: {e}")
        return False
    
    # Тестируем импорт и инициализацию
    print("🔧 Проверка импорта и инициализации...")
    
    try:
        # Попытка импорта
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from smart_tuner_main import SmartTunerMain
        
        # Попытка инициализации
        smart_tuner = SmartTunerMain("smart_tuner/config.yaml")
        
        # Проверяем наличие метода автоматического режима
        if hasattr(smart_tuner, 'run_automatic_mode'):
            print("✅ Метод run_automatic_mode найден")
        else:
            print("❌ Метод run_automatic_mode не найден")
            return False
            
        print("✅ Инициализация прошла успешно")
        
    except Exception as e:
        print(f"❌ Ошибка при инициализации: {e}")
        return False
    
    print("=" * 60)
    print("🎉 ВСЕ ТЕСТЫ АВТОМАТИЧЕСКОГО РЕЖИМА ПРОЙДЕНЫ!")
    print("=" * 60)
    print("📋 Что протестировано:")
    print("  ✅ Наличие необходимых файлов")
    print("  ✅ Аргументы командной строки")
    print("  ✅ Импорт модулей")
    print("  ✅ Инициализация Smart Tuner")
    print("  ✅ Метод автоматического режима")
    
    print("\n🚀 Автоматический режим готов к использованию!")
    print("Запустите: ./install.sh → выберите 3 → выберите 1")
    
    return True

def demo_command_line():
    """Демонстрирует использование автоматического режима"""
    print("\n" + "=" * 60)
    print("📚 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ АВТОМАТИЧЕСКОГО РЕЖИМА")
    print("=" * 60)
    
    examples = [
        ("🤖 Автоматический режим (по умолчанию - 15 trials):", 
         "python smart_tuner_main.py --mode auto"),
        ("🎯 Автоматический режим с 20 trials:", 
         "python smart_tuner_main.py --mode auto --trials 20"),
        ("⚡ Быстрый автоматический режим (10 trials):", 
         "python smart_tuner_main.py --mode auto --trials 10"),
        ("🔧 Автоматический режим с кастомной конфигурацией:", 
         "python smart_tuner_main.py --mode auto --config smart_tuner/config_improved.yaml --trials 25")
    ]
    
    for description, command in examples:
        print(f"\n{description}")
        print(f"  $ {command}")
    
    print("\n💡 СОВЕТ: Используйте меню в install.sh для удобства!")

if __name__ == "__main__":
    print(f"🕐 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = test_auto_mode()
    
    if success:
        demo_command_line()
        
    sys.exit(0 if success else 1) 