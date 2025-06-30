#!/usr/bin/env python3
"""
Простой тест функции train для отладки
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hparams import create_hparams
from train import train
import torch

def test_train():
    print("🧪 Тестируем функцию train...")
    
    # Создаем базовые hparams
    hparams = create_hparams()
    
    # Устанавливаем минимальные параметры для тестирования
    hparams.epochs = 2  # Всего 2 эпохи для теста
    hparams.batch_size = 4  # Маленький batch
    hparams.validation_freq = 10  # Частая валидация
    
    print(f"📊 Параметры теста:")
    print(f"   - epochs: {hparams.epochs}")
    print(f"   - batch_size: {hparams.batch_size}")
    print(f"   - validation_freq: {hparams.validation_freq}")
    
    # Создаем директории
    output_dir = "test_output"
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        print("🚀 Запускаем train...")
        result = train(
            output_directory=output_dir,
            log_directory=log_dir,
            checkpoint_path=None,
            warm_start=False,
            ignore_mmi_layers=False,
            ignore_gst_layers=False,
            ignore_tsgst_layers=False,
            n_gpus=1,
            rank=0,
            group_name="test",
            hparams=hparams
        )
        
        print(f"✅ Результат: {result}")
        return result
        
    except Exception as e:
        import traceback
        print(f"❌ Ошибка: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    test_train() 