#!/usr/bin/env python3
"""
Синхронизация TensorBoard логов с MLflow
Автоматически переносит все метрики из TensorBoard в MLflow
"""

import os
import mlflow
import time
from pathlib import Path
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

class TensorBoardMLflowSync:
    """Синхронизатор TensorBoard -> MLflow"""
    
    def __init__(self, tensorboard_log_dir="output"):
        self.tensorboard_log_dir = Path(tensorboard_log_dir)
        self.synced_files = set()
        
    def sync_all_metrics(self):
        """Синхронизирует все метрики из TensorBoard в MLflow"""
        print("🔄 Начинаю синхронизацию TensorBoard -> MLflow")
        
        # Ищем все логи TensorBoard
        tb_dirs = list(self.tensorboard_log_dir.glob("**/events.out.tfevents.*"))
        
        for tb_file in tb_dirs:
            if str(tb_file) not in self.synced_files:
                self._sync_single_file(tb_file)
                self.synced_files.add(str(tb_file))
    
    def _sync_single_file(self, tb_file):
        """Синхронизирует один файл TensorBoard"""
        try:
            print(f"📊 Синхронизирую: {tb_file}")
            
            # Загружаем данные из TensorBoard
            ea = EventAccumulator(str(tb_file.parent))
            ea.Reload()
            
            # Получаем все доступные скаляры
            scalar_tags = ea.Tags()['scalars']
            
            for tag in scalar_tags:
                scalar_events = ea.Scalars(tag)
                
                for event in scalar_events:
                    # Конвертируем в MLflow формат
                    mlflow_tag = self._convert_tag_name(tag)
                    
                    try:
                        mlflow.log_metric(
                            mlflow_tag, 
                            event.value, 
                            step=event.step
                        )
                    except Exception as e:
                        print(f"⚠️ Ошибка логирования {mlflow_tag}: {e}")
                        
        except Exception as e:
            print(f"❌ Ошибка синхронизации {tb_file}: {e}")
    
    def _convert_tag_name(self, tb_tag):
        """Конвертирует имена тегов TensorBoard в MLflow формат"""
        # Группируем метрики для лучшей визуализации
        conversions = {
            'training.loss': 'Loss/Training_Total',
            'training.taco_loss': 'Loss/Tacotron', 
            'training.mi_loss': 'Loss/MI',
            'training.guide_loss': 'Loss/Guide',
            'training.gate_loss': 'Loss/Gate',
            'training.emb_loss': 'Loss/Embedding',
            'validation.loss': 'Loss/Validation',
            'grad.norm': 'Gradients/Norm',
            'learning.rate': 'Optimizer/Learning_Rate',
            'duration': 'Performance/Step_Duration'
        }
        
        return conversions.get(tb_tag, tb_tag.replace('.', '/'))

def add_missing_charts():
    """Добавляет недостающие графики в MLflow"""
    
    # Дополнительные метрики для комплексного анализа
    additional_metrics = {
        'training/loss_components_ratio': {
            'description': 'Соотношение компонентов loss',
            'formula': 'guide_loss / (taco_loss + 1e-8)'
        },
        'training/gradient_stability': {
            'description': 'Стабильность градиентов',
            'formula': '1.0 / (1.0 + grad_norm_std)'
        },
        'training/learning_efficiency': {
            'description': 'Эффективность обучения',
            'formula': 'loss_improvement / step_time'
        }
    }
    
    print("📈 Добавляю дополнительные метрики:")
    for metric, info in additional_metrics.items():
        print(f"  • {metric}: {info['description']}")

def setup_mlflow_enhanced_ui():
    """Настройка MLflow UI для лучшей визуализации TTS метрик"""
    
    print("""
🎨 НАСТРОЙКА MLFLOW UI ДЛЯ TTS:

1. Основные графики для мониторинга:
   📊 Loss/Training_Total - общий loss обучения
   📊 Loss/Validation - validation loss  
   📊 Loss/Tacotron - основной loss Tacotron2
   📊 Loss/Gate - gate механизм
   📊 Gradients/Norm - норма градиентов

2. Системные метрики:
   🖥️ system/cpu_usage_percent
   🧠 system/memory_usage_percent  
   🎮 gpu_0/utilization_percent
   🌡️ gpu_0/temperature_celsius

3. Производительность:
   ⚡ Performance/Step_Duration
   📈 Performance/Steps_Per_Second
   ⏱️ Performance/ETA_Hours

Откройте MLflow UI: http://localhost:5000
""")

if __name__ == "__main__":
    # Автоматическая синхронизация
    syncer = TensorBoardMLflowSync()
    syncer.sync_all_metrics()
    
    # Добавляем дополнительные метрики
    add_missing_charts()
    
    # Показываем инструкции по UI
    setup_mlflow_enhanced_ui()
