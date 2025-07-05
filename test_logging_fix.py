#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест исправления логирования TensorBoard и MLflow
Проверяет, что все компоненты работают корректно после исправления импорта utils
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_utils_import():
    """Тестирует импорт utils модулей"""
    logger.info("🔍 Тестируем импорт utils модулей...")
    
    try:
        from utils.dynamic_padding import DynamicPaddingCollator
        logger.info("✅ DynamicPaddingCollator импортирован успешно")
    except Exception as e:
        logger.error(f"❌ Ошибка импорта DynamicPaddingCollator: {e}")
        return False
    
    try:
        from utils.bucket_batching import BucketBatchSampler
        logger.info("✅ BucketBatchSampler импортирован успешно")
    except Exception as e:
        logger.error(f"❌ Ошибка импорта BucketBatchSampler: {e}")
        return False
    
    return True

def test_tensorboard_setup():
    """Тестирует настройку TensorBoard"""
    logger.info("🔍 Тестируем настройку TensorBoard...")
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        logger.info("✅ SummaryWriter импортирован успешно")
        
        # Создаем тестовый writer
        log_dir = Path("test_tensorboard_logs")
        log_dir.mkdir(exist_ok=True)
        
        writer = SummaryWriter(log_dir=str(log_dir))
        logger.info(f"✅ TensorBoard writer создан: {log_dir}")
        
        # Тестируем запись метрик
        writer.add_scalar("test/loss", 1.0, 0)
        writer.add_scalar("test/accuracy", 0.8, 0)
        writer.flush()
        logger.info("✅ Метрики записаны в TensorBoard")
        
        writer.close()
        logger.info("✅ TensorBoard writer закрыт")
        
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка настройки TensorBoard: {e}")
        return False

def test_mlflow_setup():
    """Тестирует настройку MLflow"""
    logger.info("🔍 Тестируем настройку MLflow...")
    
    try:
        import mlflow
        logger.info("✅ MLflow импортирован успешно")
        
        # Настраиваем MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("test_experiment")
        
        # Создаем тестовый run
        with mlflow.start_run(run_name="test_run"):
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 1.0, step=0)
            logger.info("✅ MLflow run создан и метрики записаны")
        
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка настройки MLflow: {e}")
        return False

def test_training_imports():
    """Тестирует импорт основных компонентов обучения"""
    logger.info("🔍 Тестируем импорт компонентов обучения...")
    
    try:
        from model import Tacotron2
        logger.info("✅ Tacotron2 импортирован успешно")
    except Exception as e:
        logger.error(f"❌ Ошибка импорта Tacotron2: {e}")
        return False
    
    try:
        from loss_function import Tacotron2Loss
        logger.info("✅ Tacotron2Loss импортирован успешно")
    except Exception as e:
        logger.error(f"❌ Ошибка импорта Tacotron2Loss: {e}")
        return False
    
    try:
        from hparams import create_hparams
        logger.info("✅ create_hparams импортирован успешно")
    except Exception as e:
        logger.error(f"❌ Ошибка импорта create_hparams: {e}")
        return False
    
    return True

def main():
    """Основная функция тестирования"""
    logger.info("🚀 Начинаем тестирование исправлений логирования...")
    
    # Тест 1: Импорт utils
    if not test_utils_import():
        logger.error("❌ Тест импорта utils провален")
        return False
    
    # Тест 2: TensorBoard
    if not test_tensorboard_setup():
        logger.error("❌ Тест TensorBoard провален")
        return False
    
    # Тест 3: MLflow
    if not test_mlflow_setup():
        logger.error("❌ Тест MLflow провален")
        return False
    
    # Тест 4: Компоненты обучения
    if not test_training_imports():
        logger.error("❌ Тест компонентов обучения провален")
        return False
    
    logger.info("✅ Все тесты пройдены успешно!")
    logger.info("🎯 Теперь можно запускать обучение")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 