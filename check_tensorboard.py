#!/usr/bin/env python3
"""
Скрипт для проверки метрик в TensorBoard
"""

import os
import time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def check_tensorboard_logs(log_dir="logs"):
    """Проверяет метрики в TensorBoard логах"""
    
    if not os.path.exists(log_dir):
        print(f"❌ Директория {log_dir} не найдена")
        return
    
    # Находим все event файлы
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    
    if not event_files:
        print(f"❌ Event файлы не найдены в {log_dir}")
        return
    
    print(f"📊 Найдено {len(event_files)} event файлов:")
    
    for event_file in event_files:
        event_path = os.path.join(log_dir, event_file)
        file_size = os.path.getsize(event_path)
        mod_time = time.ctime(os.path.getmtime(event_path))
        
        print(f"\n📁 {event_file}")
        print(f"   Размер: {file_size} байт")
        print(f"   Время изменения: {mod_time}")
        
        try:
            # Загружаем event файл
            ea = EventAccumulator(event_path)
            ea.Reload()
            
            # Получаем все скалярные метрики
            scalar_tags = ea.Tags()['scalars']
            
            if scalar_tags:
                print(f"   ✅ Найдено {len(scalar_tags)} метрик:")
                for tag in sorted(scalar_tags):
                    events = ea.Scalars(tag)
                    if events:
                        latest_value = events[-1].value
                        latest_step = events[-1].step
                        print(f"      • {tag}: {latest_value:.4f} (шаг {latest_step})")
            else:
                print(f"   ⚠️ Скалярные метрики не найдены")
                
        except Exception as e:
            print(f"   ❌ Ошибка чтения файла: {e}")

if __name__ == "__main__":
    print("🔍 Проверка TensorBoard логов...")
    check_tensorboard_logs() 