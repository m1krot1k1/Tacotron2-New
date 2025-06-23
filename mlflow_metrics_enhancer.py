#!/usr/bin/env python3
"""
Патч для улучшения логирования метрик в MLflow
Добавляет системные метрики и улучшенную визуализацию
"""

import mlflow
import psutil
import torch
import time
import os

# Проверяем доступность GPU мониторинга
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    gpu_count = pynvml.nvmlDeviceGetCount()
    print(f"✅ GPU мониторинг активен для {gpu_count} устройств")
except:
    GPU_AVAILABLE = False
    print("⚠️ GPU мониторинг недоступен")

def log_system_metrics(step):
    """Логирование системных метрик"""
    try:
        system_metrics = {}
        
        # CPU метрики
        system_metrics['system/cpu_usage_percent'] = psutil.cpu_percent(interval=0.1)
        system_metrics['system/cpu_count'] = psutil.cpu_count()
        
        # Память
        memory = psutil.virtual_memory()
        system_metrics['system/memory_usage_percent'] = memory.percent
        system_metrics['system/memory_available_gb'] = memory.available / (1024**3)
        system_metrics['system/memory_used_gb'] = memory.used / (1024**3)
        
        # Диск
        disk = psutil.disk_usage('.')
        system_metrics['system/disk_usage_percent'] = (disk.used / disk.total) * 100
        system_metrics['system/disk_free_gb'] = disk.free / (1024**3)
        
        # Процесс
        process = psutil.Process()
        system_metrics['system/process_memory_mb'] = process.memory_info().rss / (1024**2)
        
        # GPU метрики
        if GPU_AVAILABLE:
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Использование GPU
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                system_metrics[f'gpu_{i}/utilization_percent'] = util.gpu
                system_metrics[f'gpu_{i}/memory_utilization_percent'] = util.memory
                
                # Память GPU
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                system_metrics[f'gpu_{i}/memory_used_mb'] = mem_info.used / (1024**2)
                system_metrics[f'gpu_{i}/memory_free_mb'] = mem_info.free / (1024**2)
                system_metrics[f'gpu_{i}/memory_usage_percent'] = (mem_info.used / mem_info.total) * 100
                
                # Температура
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    system_metrics[f'gpu_{i}/temperature_celsius'] = temp
                except:
                    pass
        
        # Логируем все системные метрики
        mlflow.log_metrics(system_metrics, step=step)
        
    except Exception as e:
        print(f"❌ Ошибка логирования системных метрик: {e}")

def log_enhanced_training_metrics(metrics_dict, step):
    """Улучшенное логирование метрик обучения"""
    try:
        # Реорганизуем метрики по группам для лучшей визуализации
        enhanced_metrics = {}
        
        for key, value in metrics_dict.items():
            if key == 'training.loss':
                enhanced_metrics['Loss/Total_Training'] = value
            elif key == 'training.taco_loss':
                enhanced_metrics['Loss/Tacotron'] = value
            elif key == 'training.mi_loss':
                enhanced_metrics['Loss/MI'] = value
            elif key == 'training.guide_loss':
                enhanced_metrics['Loss/Guide'] = value
            elif key == 'training.gate_loss':
                enhanced_metrics['Loss/Gate'] = value
            elif key == 'training.emb_loss':
                enhanced_metrics['Loss/Embedding'] = value
            elif key == 'validation.loss':
                enhanced_metrics['Loss/Validation'] = value
            elif key == 'grad_norm':
                enhanced_metrics['Gradients/Norm'] = value
            elif key == 'learning_rate':
                enhanced_metrics['Optimizer/Learning_Rate'] = value
            elif key == 'duration':
                enhanced_metrics['Performance/Step_Duration'] = value
            elif key == 'guide_loss_weight':
                enhanced_metrics['Weights/Guide_Loss'] = value
            else:
                enhanced_metrics[key] = value
        
        # Логируем улучшенные метрики
        mlflow.log_metrics(enhanced_metrics, step=step)
        
        # Логируем системные метрики каждые 10 шагов
        if step % 10 == 0:
            log_system_metrics(step)
            
    except Exception as e:
        print(f"❌ Ошибка улучшенного логирования: {e}")

# Функция для патчинга train.py
def patch_training_logging():
    """Инструкции по патчингу"""
    print("""
🔧 Для применения улучшенного логирования:

1. В train.py найдите строку:
   mlflow.log_metrics({...}, step=iteration)

2. Замените на:
   from mlflow_metrics_enhancer import log_enhanced_training_metrics
   log_enhanced_training_metrics({...}, iteration)

3. Добавьте в начало train.py:
   import mlflow_metrics_enhancer

Это добавит:
✅ Системные метрики (CPU, GPU, RAM)
✅ Улучшенную группировку метрик
✅ Дополнительные графики в MLflow UI
""")

if __name__ == "__main__":
    patch_training_logging()
