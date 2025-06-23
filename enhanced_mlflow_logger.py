#!/usr/bin/env python3
"""
Улучшенный MLflow логгер для Tacotron2

Особенности:
- Полное логирование всех метрик обучения
- Системные метрики (CPU, GPU, RAM)
- Визуализация спектрограмм и выравниваний
- Автоматические графики для MLflow UI
"""

import mlflow
import psutil
import torch
import time
import os
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path

# Проверяем доступность GPU мониторинга
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMLflowLogger:
    """
    Улучшенный логгер для MLflow с полными метриками и системными показателями
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.step_times = []
        self.memory_usage_history = []
        
        # Инициализация GPU мониторинга
        if GPU_AVAILABLE:
            try:
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_handles = []
                for i in range(self.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.gpu_handles.append(handle)
                logger.info(f"GPU мониторинг активен для {self.gpu_count} устройств")
            except Exception as e:
                logger.warning(f"Ошибка инициализации GPU мониторинга: {e}")
                GPU_AVAILABLE = False
        
        logger.info("Enhanced MLflow Logger инициализирован")
    
    def log_training_metrics(self, metrics: Dict[str, float], step: int):
        """
        Логирование основных метрик обучения
        
        Args:
            metrics: Словарь с метриками
            step: Номер шага
        """
        try:
            # Основные метрики обучения
            training_metrics = {}
            
            # Группируем метрики по категориям
            for key, value in metrics.items():
                if key.startswith('training.'):
                    training_metrics[key] = value
                elif key == 'grad_norm':
                    training_metrics['gradients/norm'] = value
                elif key == 'learning_rate':
                    training_metrics['optimizer/learning_rate'] = value
                elif key == 'duration':
                    training_metrics['performance/step_duration'] = value
                else:
                    training_metrics[key] = value
            
            # Логируем основные метрики
            if training_metrics:
                mlflow.log_metrics(training_metrics, step=step)
            
            # Добавляем производные метрики
            self._log_derived_metrics(metrics, step)
            
        except Exception as e:
            logger.error(f"Ошибка логирования метрик обучения: {e}")
    
    def log_validation_metrics(self, val_loss: float, step: int, model=None, 
                              spectrograms: Optional[Dict] = None):
        """
        Логирование метрик валидации
        
        Args:
            val_loss: Validation loss
            step: Номер шага
            model: Модель (для логирования весов)
            spectrograms: Спектрограммы для визуализации
        """
        try:
            validation_metrics = {
                'validation/loss': val_loss,
                'validation/step': step
            }
            
            mlflow.log_metrics(validation_metrics, step=step)
            
            # Логируем веса модели (каждые N шагов)
            if model is not None and step % 1000 == 0:
                self._log_model_weights(model, step)
            
            # Логируем спектрограммы как артефакты
            if spectrograms is not None:
                self._log_spectrograms(spectrograms, step)
                
        except Exception as e:
            logger.error(f"Ошибка логирования метрик валидации: {e}")
    
    def log_system_metrics(self, step: int):
        """
        Логирование системных метрик
        
        Args:
            step: Номер шага
        """
        try:
            system_metrics = {}
            
            # CPU метрики
            cpu_percent = psutil.cpu_percent(interval=1)
            system_metrics['system/cpu_usage_percent'] = cpu_percent
            system_metrics['system/cpu_count'] = psutil.cpu_count()
            
            # Память
            memory = psutil.virtual_memory()
            system_metrics['system/memory_usage_percent'] = memory.percent
            system_metrics['system/memory_available_gb'] = memory.available / (1024**3)
            system_metrics['system/memory_used_gb'] = memory.used / (1024**3)
            system_metrics['system/memory_total_gb'] = memory.total / (1024**3)
            
            # Диск
            disk = psutil.disk_usage('.')
            system_metrics['system/disk_usage_percent'] = (disk.used / disk.total) * 100
            system_metrics['system/disk_free_gb'] = disk.free / (1024**3)
            
            # Процесс
            process = psutil.Process()
            system_metrics['system/process_memory_mb'] = process.memory_info().rss / (1024**2)
            system_metrics['system/process_cpu_percent'] = process.cpu_percent()
            
            # GPU метрики (если доступны)
            if GPU_AVAILABLE and self.gpu_handles:
                self._log_gpu_metrics(system_metrics)
            
            # Логируем все системные метрики
            mlflow.log_metrics(system_metrics, step=step)
            
            # Сохраняем историю для анализа
            self.memory_usage_history.append(memory.percent)
            if len(self.memory_usage_history) > 1000:
                self.memory_usage_history.pop(0)
                
        except Exception as e:
            logger.error(f"Ошибка логирования системных метрик: {e}")
    
    def _log_gpu_metrics(self, system_metrics: Dict[str, float]):
        """Логирование GPU метрик"""
        try:
            for i, handle in enumerate(self.gpu_handles):
                # Использование GPU
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                system_metrics[f'gpu_{i}/utilization_percent'] = util.gpu
                system_metrics[f'gpu_{i}/memory_utilization_percent'] = util.memory
                
                # Память GPU
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                system_metrics[f'gpu_{i}/memory_used_mb'] = mem_info.used / (1024**2)
                system_metrics[f'gpu_{i}/memory_free_mb'] = mem_info.free / (1024**2)
                system_metrics[f'gpu_{i}/memory_total_mb'] = mem_info.total / (1024**2)
                system_metrics[f'gpu_{i}/memory_usage_percent'] = (mem_info.used / mem_info.total) * 100
                
                # Температура GPU
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    system_metrics[f'gpu_{i}/temperature_celsius'] = temp
                except:
                    pass
                
                # Потребление энергии
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # мВт -> Вт
                    system_metrics[f'gpu_{i}/power_usage_watts'] = power
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Ошибка логирования GPU метрик: {e}")
    
    def _log_derived_metrics(self, metrics: Dict[str, float], step: int):
        """Логирование производных метрик"""
        try:
            derived_metrics = {}
            
            # Скорость обучения
            current_time = time.time()
            step_time = current_time - self.last_log_time
            self.step_times.append(step_time)
            
            if len(self.step_times) > 100:
                self.step_times.pop(0)
            
            # Средняя скорость за последние шаги
            if len(self.step_times) > 1:
                avg_step_time = np.mean(self.step_times[-10:])  # Последние 10 шагов
                derived_metrics['performance/avg_step_time'] = avg_step_time
                derived_metrics['performance/steps_per_second'] = 1.0 / avg_step_time
                
                # Оценка времени до завершения (если есть цель)
                if 'target_steps' in metrics:
                    remaining_steps = metrics['target_steps'] - step
                    eta_seconds = remaining_steps * avg_step_time
                    derived_metrics['performance/eta_hours'] = eta_seconds / 3600
            
            # Общее время обучения
            total_time = current_time - self.start_time
            derived_metrics['performance/total_training_hours'] = total_time / 3600
            
            # Эффективность градиентов
            if 'grad_norm' in metrics:
                if not hasattr(self, 'grad_history'):
                    self.grad_history = []
                self.grad_history.append(metrics['grad_norm'])
                if len(self.grad_history) > 100:
                    self.grad_history.pop(0)
                
                if len(self.grad_history) > 1:
                    derived_metrics['gradients/std'] = np.std(self.grad_history)
                    derived_metrics['gradients/mean'] = np.mean(self.grad_history)
            
            # Стабильность loss
            if 'training.loss' in metrics:
                if not hasattr(self, 'loss_history'):
                    self.loss_history = []
                self.loss_history.append(metrics['training.loss'])
                if len(self.loss_history) > 100:
                    self.loss_history.pop(0)
                
                if len(self.loss_history) > 10:
                    # Тренд loss (улучшается/ухудшается)
                    recent_loss = np.mean(self.loss_history[-10:])
                    older_loss = np.mean(self.loss_history[-20:-10]) if len(self.loss_history) >= 20 else recent_loss
                    derived_metrics['training/loss_trend'] = (older_loss - recent_loss) / older_loss
                    derived_metrics['training/loss_stability'] = 1.0 / (1.0 + np.std(self.loss_history[-20:]))
            
            # Логируем производные метрики
            if derived_metrics:
                mlflow.log_metrics(derived_metrics, step=step)
            
            self.last_log_time = current_time
            
        except Exception as e:
            logger.error(f"Ошибка логирования производных метрик: {e}")
    
    def _log_model_weights(self, model, step: int):
        """Логирование статистики весов модели"""
        try:
            weight_metrics = {}
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.data.numel() > 0:
                    # Базовые статистики
                    weight_data = param.data.cpu().numpy().flatten()
                    
                    # Преобразуем имя параметра для MLflow
                    clean_name = name.replace('.', '_').replace('/', '_')
                    
                    weight_metrics[f'weights/{clean_name}/mean'] = np.mean(weight_data)
                    weight_metrics[f'weights/{clean_name}/std'] = np.std(weight_data)
                    weight_metrics[f'weights/{clean_name}/min'] = np.min(weight_data)
                    weight_metrics[f'weights/{clean_name}/max'] = np.max(weight_data)
                    
                    # Процент нулевых весов
                    zero_percent = np.sum(np.abs(weight_data) < 1e-8) / len(weight_data) * 100
                    weight_metrics[f'weights/{clean_name}/zero_percent'] = zero_percent
            
            # Логируем метрики весов
            if weight_metrics:
                mlflow.log_metrics(weight_metrics, step=step)
                
        except Exception as e:
            logger.error(f"Ошибка логирования весов модели: {e}")
    
    def _log_spectrograms(self, spectrograms: Dict, step: int):
        """Логирование спектрограмм как артефакты"""
        try:
            # Создаем временную директорию для артефактов
            temp_dir = Path(f"temp_artifacts_{step}")
            temp_dir.mkdir(exist_ok=True)
            
            for name, spectrogram in spectrograms.items():
                if isinstance(spectrogram, torch.Tensor):
                    spectrogram = spectrogram.cpu().numpy()
                
                # Сохраняем как numpy файл
                file_path = temp_dir / f"{name}_step_{step}.npy"
                np.save(file_path, spectrogram)
                
                # Логируем как артефакт
                mlflow.log_artifact(str(file_path), f"spectrograms/step_{step}")
            
            # Очищаем временные файлы
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            logger.error(f"Ошибка логирования спектрограмм: {e}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Логирование гиперпараметров"""
        try:
            # Фильтруем и конвертируем гиперпараметры
            clean_hparams = {}
            for key, value in hparams.items():
                # Конвертируем в подходящий тип для MLflow
                if isinstance(value, (int, float, str, bool)):
                    clean_hparams[key] = value
                elif isinstance(value, (list, tuple)):
                    clean_hparams[key] = str(value)
                elif value is None:
                    clean_hparams[key] = "None"
                else:
                    clean_hparams[key] = str(value)
            
            mlflow.log_params(clean_hparams)
            logger.info(f"Залогировано {len(clean_hparams)} гиперпараметров")
            
        except Exception as e:
            logger.error(f"Ошибка логирования гиперпараметров: {e}")
    
    def log_training_summary(self, final_metrics: Dict[str, float]):
        """Логирование итоговой сводки обучения"""
        try:
            summary_metrics = {}
            
            # Основные результаты
            for key, value in final_metrics.items():
                summary_metrics[f'final/{key}'] = value
            
            # Сводка по производительности
            if self.step_times:
                summary_metrics['summary/avg_step_time'] = np.mean(self.step_times)
                summary_metrics['summary/total_steps'] = len(self.step_times)
            
            # Сводка по памяти
            if self.memory_usage_history:
                summary_metrics['summary/max_memory_usage'] = max(self.memory_usage_history)
                summary_metrics['summary/avg_memory_usage'] = np.mean(self.memory_usage_history)
            
            # Общее время обучения
            total_time = time.time() - self.start_time
            summary_metrics['summary/total_training_time_hours'] = total_time / 3600
            
            mlflow.log_metrics(summary_metrics)
            logger.info("Залогирована сводка обучения")
            
        except Exception as e:
            logger.error(f"Ошибка логирования сводки: {e}")

# Глобальный экземпляр логгера
enhanced_logger = None

def get_enhanced_logger():
    """Получить глобальный экземпляр улучшенного логгера"""
    global enhanced_logger
    if enhanced_logger is None:
        enhanced_logger = EnhancedMLflowLogger()
    return enhanced_logger

def log_all_metrics(training_metrics: Dict[str, float], step: int, 
                   validation_loss: Optional[float] = None,
                   system_metrics: bool = True):
    """
    Удобная функция для логирования всех типов метрик
    
    Args:
        training_metrics: Метрики обучения
        step: Номер шага
        validation_loss: Validation loss (если есть)
        system_metrics: Логировать ли системные метрики
    """
    logger_instance = get_enhanced_logger()
    
    # Логируем метрики обучения
    logger_instance.log_training_metrics(training_metrics, step)
    
    # Логируем validation metrics
    if validation_loss is not None:
        logger_instance.log_validation_metrics(validation_loss, step)
    
    # Логируем системные метрики (каждые 10 шагов)
    if system_metrics and step % 10 == 0:
        logger_instance.log_system_metrics(step) 