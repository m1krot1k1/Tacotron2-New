
# ФАЙЛ: smart_tuner_main.py
# ПРОБЛЕМА: Утечки памяти в долгих trial

import gc
import torch
import psutil
import time

class MemoryManager:
    def __init__(self, memory_threshold=85):
        self.memory_threshold = memory_threshold
        self.last_cleanup = time.time()

    def cleanup_trial_memory(self, force=False):
        '''
        Принудительная очистка памяти после trial
        '''
        current_time = time.time()
        memory_percent = psutil.virtual_memory().percent

        # Очищаем каждые 30 секунд или при превышении порога
        if force or (current_time - self.last_cleanup > 30) or memory_percent > self.memory_threshold:

            # Python garbage collection
            collected = gc.collect()

            # PyTorch cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Дополнительная очистка для del объектов
            for obj in gc.get_objects():
                if isinstance(obj, torch.Tensor) and obj.device.type == 'cuda':
                    del obj

            gc.collect()  # Еще один проход

            self.last_cleanup = current_time
            new_memory_percent = psutil.virtual_memory().percent

            print(f"🧹 Очистка памяти: {memory_percent:.1f}% -> {new_memory_percent:.1f}% "
                  f"(освобождено {collected} объектов)")

    def monitor_memory_usage(self):
        '''
        Мониторинг использования памяти
        '''
        memory = psutil.virtual_memory()
        gpu_memory = None

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_summary()

        return {
            "ram_percent": memory.percent,
            "ram_available": memory.available // (1024**3),  # GB
            "gpu_memory": gpu_memory
        }

    def check_memory_health(self):
        '''
        Проверка состояния памяти
        '''
        stats = self.monitor_memory_usage()

        if stats["ram_percent"] > 90:
            print(f"⚠️ Критическое использование RAM: {stats['ram_percent']:.1f}%")
            self.cleanup_trial_memory(force=True)
            return False

        return True

# ИНТЕГРАЦИЯ В ОБУЧЕНИЕ:
def tts_objective_function_with_memory_management(trial):
    memory_manager = MemoryManager()

    try:
        # Получаем параметры trial
        suggested_params = get_trial_params(trial)

        # Проверяем память перед началом
        if not memory_manager.check_memory_health():
            print("⚠️ Недостаточно памяти для trial")
            return float('inf')

        # Обучение модели
        metrics = train_model_with_params(suggested_params)

        return metrics.get('validation_loss', float('inf'))

    except Exception as e:
        print(f"❌ Ошибка в trial: {e}")
        return float('inf')

    finally:
        # ОБЯЗАТЕЛЬНАЯ очистка памяти
        memory_manager.cleanup_trial_memory(force=True)
