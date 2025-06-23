#!/usr/bin/env python3
"""
Интеграция систем экспорта и логирования в процесс обучения TTS

Автор: AI Assistant  
Назначение: Подключение умных систем к train.py
"""

import os
import sys
from pathlib import Path

# Добавляем текущую директорию в путь
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from training_export_system import TrainingExportSystem, export_training_for_ai
    from smart_training_logger import SmartTrainingLogger, get_training_logger
    from smart_training_logger import log_training_start, log_training_metrics, log_param_change
    from smart_training_logger import log_training_warning, log_training_end
except ImportError as e:
    print(f"⚠️ Ошибка импорта модулей логирования: {e}")
    print("Создайте файлы training_export_system.py и smart_training_logger.py")

def setup_training_logging(run_id: str, hparams):
    """
    Настройка систем логирования для обучения
    
    Args:
        run_id: MLflow run ID
        hparams: параметры обучения
    
    Returns:
        Tuple[TrainingExportSystem, SmartTrainingLogger]
    """
    print("🔧 Настройка систем логирования...")
    
    # Создаем системы
    export_system = TrainingExportSystem()
    
    # Начинаем сессию логирования
    training_params = {
        "learning_rate": hparams.learning_rate,
        "batch_size": hparams.batch_size,
        "warmup_steps": hparams.warmup_steps,
        "model": "Tacotron2",
        "optimizer": "Adam",
        "scheduler": hparams.lr_scheduler_type if hasattr(hparams, 'lr_scheduler_type') else "StepLR",
        "early_stopping": hparams.early_stopping if hasattr(hparams, 'early_stopping') else True,
        "mixed_precision": hparams.fp16_run if hasattr(hparams, 'fp16_run') else False
    }
    
    session_id = log_training_start(run_id, training_params)
    
    print(f"✅ Логирование настроено. Session ID: {session_id}")
    
    return export_system, get_training_logger()

def log_step_metrics(step: int, metrics: dict):
    """
    Логирование метрик шага обучения
    
    Args:
        step: номер шага
        metrics: словарь с метриками
    """
    try:
        # Приводим метрики к нужному формату
        clean_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                clean_metrics[key] = float(value)
            elif hasattr(value, 'item'):  # torch.Tensor
                clean_metrics[key] = float(value.item())
            else:
                clean_metrics[key] = str(value)
        
        log_training_metrics(step, clean_metrics)
        
    except Exception as e:
        print(f"⚠️ Ошибка логирования метрик на шаге {step}: {e}")

def log_smart_tuner_change(param_name: str, old_value, new_value, reason: str):
    """
    Логирование изменения параметра умной системой
    
    Args:
        param_name: имя параметра
        old_value: старое значение
        new_value: новое значение
        reason: причина изменения
    """
    try:
        log_param_change(param_name, old_value, new_value, reason)
        print(f"📝 Логировано изменение {param_name}: {old_value} → {new_value}")
        
    except Exception as e:
        print(f"⚠️ Ошибка логирования изменения параметра: {e}")

def log_training_warning_event(warning_type: str, message: str, data: dict = None):
    """
    Логирование предупреждения
    
    Args:
        warning_type: тип предупреждения
        message: сообщение
        data: дополнительные данные
    """
    try:
        log_training_warning(warning_type, message, data)
        print(f"⚠️ Логировано предупреждение: {warning_type}")
        
    except Exception as e:
        print(f"❌ Ошибка логирования предупреждения: {e}")

def finish_training_logging(final_metrics: dict = None, status: str = "completed"):
    """
    Завершение логирования обучения
    
    Args:
        final_metrics: финальные метрики
        status: статус завершения
    """
    try:
        # Приводим метрики к нужному формату
        clean_final_metrics = {}
        if final_metrics:
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    clean_final_metrics[key] = float(value)
                elif hasattr(value, 'item'):  # torch.Tensor
                    clean_final_metrics[key] = float(value.item())
                else:
                    clean_final_metrics[key] = str(value)
        
        log_training_end(clean_final_metrics, status)
        print(f"🏁 Логирование завершено со статусом: {status}")
        
    except Exception as e:
        print(f"❌ Ошибка завершения логирования: {e}")

def export_current_training(run_id: str = None):
    """
    Экспорт текущего обучения для отправки AI
    
    Args:
        run_id: MLflow run ID
    
    Returns:
        Путь к файлу для отправки AI
    """
    try:
        text_file = export_training_for_ai(run_id)
        
        if text_file:
            print(f"\n" + "="*60)
            print("📤 ЭКСПОРТ ДЛЯ AI ГОТОВ!")
            print("="*60)
            print(f"Файл: {text_file}")
            print("📋 Скопируйте содержимое файла и отправьте AI Assistant")
            print("="*60)
            
            # Показываем первые строки файла
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:10]
                    print("Предпросмотр:")
                    for line in lines:
                        print(f"  {line.rstrip()}")
                    if len(lines) >= 10:
                        print("  ...")
            except Exception:
                pass
        
        return text_file
        
    except Exception as e:
        print(f"❌ Ошибка экспорта обучения: {e}")
        return None

# Патчи для интеграции с train.py
def patch_train_logging():
    """
    Автоматическое патчинг train.py для добавления логирования
    """
    train_file = Path("train.py")
    
    if not train_file.exists():
        print("❌ Файл train.py не найден")
        return False
    
    try:
        # Читаем train.py
        with open(train_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Проверяем, уже ли добавлено логирование
        if "from training_integration import" in content:
            print("✅ Логирование уже интегрировано в train.py")
            return True
        
        # Добавляем импорты
        import_line = """
# === ИНТЕГРАЦИЯ УМНОГО ЛОГИРОВАНИЯ ===
try:
    from training_integration import setup_training_logging, log_step_metrics
    from training_integration import log_smart_tuner_change, log_training_warning_event
    from training_integration import finish_training_logging, export_current_training
    SMART_LOGGING_ENABLED = True
    print("✅ Умное логирование активировано")
except ImportError as e:
    print(f"⚠️ Умное логирование недоступно: {e}")
    SMART_LOGGING_ENABLED = False
# === КОНЕЦ ИНТЕГРАЦИИ ===
"""
        
        # Вставляем после импортов MLflow
        if "import mlflow" in content:
            content = content.replace(
                "import mlflow",
                "import mlflow" + import_line
            )
        else:
            # Если нет MLflow, добавляем в начало после основных импортов
            lines = content.split('\n')
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    insert_pos = i + 1
            
            lines.insert(insert_pos, import_line)
            content = '\n'.join(lines)
        
        # Создаем резервную копию
        backup_file = train_file.with_suffix('.py.backup')
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"📄 Создана резервная копия: {backup_file}")
        
        # Инструкции по интеграции
        print("\n" + "="*60)
        print("📋 ИНСТРУКЦИИ ПО ИНТЕГРАЦИИ")
        print("="*60)
        print("Добавьте следующие вызовы в train.py:")
        print("")
        print("1. В начале обучения (после mlflow.start_run):")
        print("   if SMART_LOGGING_ENABLED:")
        print("       export_system, logger = setup_training_logging(run.info.run_id, hparams)")
        print("")
        print("2. В цикле обучения (каждые N шагов):")
        print("   if SMART_LOGGING_ENABLED and iteration % 10 == 0:")
        print("       log_step_metrics(iteration, {")
        print("           'training.loss': train_loss,")
        print("           'validation.loss': val_loss,")
        print("           'grad_norm': grad_norm,")
        print("           'learning_rate': optimizer.param_groups[0]['lr']")
        print("       })")
        print("")
        print("3. При изменении параметров умной системой:")
        print("   if SMART_LOGGING_ENABLED:")
        print("       log_smart_tuner_change('learning_rate', old_lr, new_lr, 'Градиенты нестабильны')")
        print("")
        print("4. В конце обучения:")
        print("   if SMART_LOGGING_ENABLED:")
        print("       finish_training_logging({'final_loss': final_loss}, 'completed')")
        print("       export_current_training(run.info.run_id)")
        print("")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка патчинга train.py: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Тестирование интеграции умного логирования")
    
    # Тест настройки логирования
    class MockHparams:
        learning_rate = 0.001
        batch_size = 32
        warmup_steps = 1000
        
    mock_hparams = MockHparams()
    
    try:
        export_system, logger = setup_training_logging("test_run_12345", mock_hparams)
        
        # Тест логирования метрик
        log_step_metrics(100, {
            "training.loss": 2.5,
            "validation.loss": 2.8,
            "grad_norm": 5.2,
            "learning_rate": 0.001
        })
        
        # Тест логирования изменения параметра
        log_smart_tuner_change(
            "learning_rate", 
            0.001, 
            0.0008, 
            "Высокая норма градиентов"
        )
        
        # Тест предупреждения
        log_training_warning_event(
            "GradientWarning",
            "Градиенты превышают порог",
            {"grad_norm": 50.0}
        )
        
        # Завершение
        finish_training_logging(
            {"final_loss": 1.2}, 
            "completed"
        )
        
        print("✅ Тест интеграции завершен успешно")
        
    except Exception as e:
        print(f"❌ Ошибка теста: {e}")
    
    # Тест патчинга
    print("\n🔧 Тестирование патчинга train.py...")
    patch_train_logging() 