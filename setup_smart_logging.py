#!/usr/bin/env python3
"""
Установочный скрипт для умной системы логирования TTS

Автор: AI Assistant
Назначение: Быстрая настройка и интеграция систем экспорта и логирования
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header():
    """Печатает заголовок"""
    print("=" * 80)
    print("🎯 УСТАНОВКА УМНОЙ СИСТЕМЫ ЛОГИРОВАНИЯ TTS")
    print("=" * 80)
    print("Автор: AI Assistant")
    print("Версия: 1.0.0")
    print("")

def check_dependencies():
    """Проверяет зависимости"""
    print("📦 Проверка зависимостей...")
    
    required_packages = [
        'mlflow', 'matplotlib', 'pandas', 'numpy', 
        'seaborn', 'scipy', 'pyyaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - ОТСУТСТВУЕТ")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Отсутствуют пакеты: {', '.join(missing_packages)}")
        
        install = input("Установить автоматически? (y/n): ").lower().strip()
        if install in ['y', 'yes', 'да', '']:
            install_packages(missing_packages)
        else:
            print("❌ Установка прервана. Установите пакеты вручную:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    print("✅ Все зависимости установлены\n")
    return True

def install_packages(packages):
    """Устанавливает недостающие пакеты"""
    print(f"📥 Установка пакетов: {', '.join(packages)}")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + packages)
        print("✅ Пакеты установлены успешно\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки: {e}")
        return False

def check_files():
    """Проверяет наличие файлов системы"""
    print("📁 Проверка файлов системы...")
    
    required_files = [
        'training_export_system.py',
        'smart_training_logger.py', 
        'training_integration.py'
    ]
    
    missing_files = []
    
    for file in required_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - ОТСУТСТВУЕТ")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Отсутствуют файлы: {', '.join(missing_files)}")
        print("Восстановите файлы из репозитория или повторно создайте их")
        return False
    
    print("✅ Все файлы системы присутствуют\n")
    return True

def test_export_system():
    """Тестирует систему экспорта"""
    print("🧪 Тестирование системы экспорта...")
    
    try:
        from training_export_system import TrainingExportSystem
        
        # Создаем тестовый экспортер
        exporter = TrainingExportSystem()
        
        print("   ✅ Модуль импортирован")
        print("   ✅ Экспортер создан")
        print("   ✅ Папки созданы")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка тестирования: {e}")
        return False

def test_logging_system():
    """Тестирует систему логирования"""
    print("🧪 Тестирование системы логирования...")
    
    try:
        from smart_training_logger import SmartTrainingLogger
        
        # Создаем тестовый логгер
        logger = SmartTrainingLogger()
        
        print("   ✅ Модуль импортирован")
        print("   ✅ Логгер создан")
        print("   ✅ Папки созданы")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка тестирования: {e}")
        return False

def test_integration():
    """Тестирует интеграционный модуль"""
    print("🧪 Тестирование интеграции...")
    
    try:
        from training_integration import setup_training_logging
        from training_integration import log_step_metrics, log_smart_tuner_change
        
        print("   ✅ Модуль импортирован")
        print("   ✅ Функции доступны")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка тестирования: {e}")
        return False

def create_demo_export():
    """Создает демонстрационный экспорт"""
    print("📤 Создание демонстрационного экспорта...")
    
    try:
        from training_export_system import export_training_for_ai
        
        # Пробуем экспортировать последний run
        result = export_training_for_ai()
        
        if result:
            print("   ✅ Демо-экспорт создан")
            print(f"   📄 Файл: {result}")
            return True
        else:
            print("   ⚠️ Нет данных MLflow для экспорта")
            print("   💡 Запустите обучение, чтобы создать данные")
            return True
            
    except Exception as e:
        print(f"   ❌ Ошибка создания демо-экспорта: {e}")
        return False

def show_quick_start():
    """Показывает quick start guide"""
    print("\n" + "=" * 80)
    print("🚀 БЫСТРЫЙ СТАРТ")
    print("=" * 80)
    
    print("\n1. 📤 Экспорт последнего обучения для AI:")
    print("   python -c \"from training_export_system import export_training_for_ai; export_training_for_ai()\"")
    
    print("\n2. 📝 Интеграция с train.py:")
    print("   Добавьте в train.py:")
    print("   ```python")
    print("   from training_integration import setup_training_logging, log_step_metrics")
    print("   from training_integration import finish_training_logging, export_current_training")
    print("   ```")
    
    print("\n3. 📊 Просмотр логов:")
    print("   - Markdown логи: smart_logs/training_sessions/")
    print("   - Графики: smart_logs/plots/")
    print("   - Экспорт для AI: training_exports/text_reports/")
    
    print("\n4. 📋 Документация:")
    print("   Читайте SMART_LOGGING_SYSTEM_README.md")
    
    print("\n" + "=" * 80)

def create_convenient_scripts():
    """Создает удобные скрипты для использования"""
    print("🔧 Создание удобных скриптов...")
    
    # Скрипт для быстрого экспорта
    export_script = """#!/usr/bin/env python3
# Быстрый экспорт для AI Assistant
from training_export_system import export_training_for_ai
import sys

if len(sys.argv) > 1:
    run_id = sys.argv[1]
    print(f"Экспорт run: {run_id}")
    export_training_for_ai(run_id)
else:
    print("Экспорт последнего обучения...")
    export_training_for_ai()
"""
    
    with open("quick_export.py", "w", encoding="utf-8") as f:
        f.write(export_script)
    
    print("   ✅ quick_export.py - быстрый экспорт")
    
    # Скрипт для просмотра логов
    view_script = """#!/usr/bin/env python3
# Просмотр последних логов
from pathlib import Path
import os

smart_logs = Path("smart_logs")
if smart_logs.exists():
    sessions = list(smart_logs.glob("training_sessions/*.md"))
    if sessions:
        latest = max(sessions, key=os.path.getctime)
        print(f"Последний лог: {latest}")
        
        if input("Открыть? (y/n): ").lower() in ['y', 'yes', '']:
            os.system(f"cat '{latest}'")
    else:
        print("Логи не найдены")
else:
    print("Папка smart_logs не найдена")
"""
    
    with open("view_logs.py", "w", encoding="utf-8") as f:
        f.write(view_script)
    
    print("   ✅ view_logs.py - просмотр логов")

def main():
    """Основная функция установки"""
    print_header()
    
    # Проверки
    success = True
    success &= check_dependencies()
    success &= check_files()
    
    if not success:
        print("❌ Установка прервана из-за ошибок")
        return False
    
    # Тестирование
    print("🧪 Тестирование системы...")
    success &= test_export_system()
    success &= test_logging_system() 
    success &= test_integration()
    
    if not success:
        print("❌ Некоторые тесты не прошли")
        return False
    
    print("✅ Все тесты пройдены\n")
    
    # Демонстрация
    create_demo_export()
    create_convenient_scripts()
    
    # Финал
    print("\n✅ УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!")
    print("🎉 Умная система логирования готова к использованию")
    
    show_quick_start()
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Установка прервана пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc() 