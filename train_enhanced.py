#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Основной entrypoint для обучения Tacotron2 с полной интеграцией EnhancedTacotronTrainer
- Использует только современную архитектуру
- Поддержка Smart Tuner, Telegram, MLflow
- Совместимость с legacy train.py (только для отладки)
"""

import argparse
import logging
import os
import sys
import yaml
import torch
import numpy as np
from typing import Dict, Any, Optional

from enhanced_training_main import EnhancedTacotronTrainer, prepare_dataloaders
from hparams import create_hparams

# === MLflow: безопасная инициализация ===
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# === Telegram: безопасная инициализация ===
try:
    from smart_tuner.telegram_monitor_enhanced import TelegramMonitorEnhanced
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description='Enhanced Tacotron2 Training Entrypoint')
    parser.add_argument('--config', type=str, default='smart_tuner/config.yaml', help='Путь к конфигу Smart Tuner')
    parser.add_argument('--epochs', type=int, default=None, help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=None, help='Размер batch')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--mode', type=str, choices=['train', 'validate', 'test'], default='train', help='Режим работы')
    parser.add_argument('--output', type=str, default='output', help='Директория для чекпоинтов и логов')
    args = parser.parse_args()

    # === Загрузка гиперпараметров ===
    hparams = create_hparams()
    if args.epochs:
        hparams.epochs = args.epochs
    if args.batch_size:
        hparams.batch_size = args.batch_size
    if args.learning_rate:
        hparams.learning_rate = args.learning_rate

    # === Загрузка конфигурации Smart Tuner ===
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    # === Telegram Monitor ===
    telegram_monitor = None
    if TELEGRAM_AVAILABLE and config.get('telegram', {}).get('enabled', False):
        tg = config['telegram']
        telegram_monitor = TelegramMonitorEnhanced(
            bot_token=tg.get('bot_token'),
            chat_id=tg.get('chat_id'),
            enabled=True
        )

    # === Подготовка данных ===
    train_loader, val_loader = prepare_dataloaders(hparams)

    # === Информация о датасете (можно расширить) ===
    dataset_info = {
        'total_duration_minutes': 120,
        'num_speakers': 1,
        'voice_complexity': 'moderate',
        'audio_quality': 'good',
        'language': 'ru'
    }

    # === Инициализация тренера ===
    trainer = EnhancedTacotronTrainer(hparams, dataset_info)
    if telegram_monitor:
        trainer.telegram_monitor = telegram_monitor
    
    # 🔧 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Инициализация модели
    print("🔧 Инициализация компонентов обучения...")
    trainer.initialize_training()

    # === Запуск обучения ===
    if args.mode == 'train':
        print('🚀 Запуск обучения через EnhancedTacotronTrainer...')
        trainer.train(train_loader, val_loader, max_epochs=hparams.epochs)
        print('🎉 Обучение завершено!')
    elif args.mode == 'validate':
        print('🔍 Валидация модели...')
        val_result = trainer.validate_step(val_loader)
        print(f'Валидация завершена: {val_result}')
    elif args.mode == 'test':
        print('🧪 Тестовый запуск...')
        batch = next(iter(train_loader))
        result = trainer.train_step(batch)
        print(f'Тестовый шаг: {result}')

if __name__ == "__main__":
    main() 