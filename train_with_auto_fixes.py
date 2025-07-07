#!/usr/bin/env python3
"""
🚀 Улучшенный запуск обучения с автоматическими исправлениями
Интегрирует все компоненты: EnhancedTacotronTrainer + AutoFixManager + Smart Tuner + Telegram

Использование:
python train_with_auto_fixes.py --epochs 10 --batch_size 16
"""

import argparse
import sys
import os
import torch
import logging
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_training_main import EnhancedTacotronTrainer, prepare_dataloaders
from hparams import create_hparams

def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training_with_auto_fixes.log')
        ]
    )
    return logging.getLogger('TrainingWithAutoFixes')

def create_optimized_hparams(args):
    """Создает оптимизированные гиперпараметры для стабильного обучения"""
    hparams = create_hparams()
    
    # 🔧 Безопасные начальные параметры
    hparams.learning_rate = 1e-4  # Консервативный learning rate
    hparams.batch_size = args.batch_size if args.batch_size else 8
    hparams.grad_clip_thresh = 1.0  # Строгое клипирование градиентов
    
    # 🎯 Принудительная активация guided attention
    hparams.use_guided_attn = True
    hparams.guide_loss_weight = 10.0  # Высокий начальный вес
    hparams.guide_loss_initial_weight = 10.0
    
    # 🛡️ Консервативные dropout параметры
    hparams.p_attention_dropout = 0.01  # Минимальный dropout
    hparams.p_decoder_dropout = 0.01
    
    # 🚪 Сбалансированный gate threshold
    hparams.gate_threshold = 0.5
    
    # 📊 Увеличенные интервалы проверки для стабильности
    hparams.iters_per_checkpoint = 500
    hparams.epochs_per_checkpoint = 1
    
    # 🚫 Отключение нестабильных функций
    hparams.use_mmi = False
    hparams.fp16_run = False  # Отключаем fp16 для максимальной стабильности
    
    return hparams

def main():
    """Основная функция запуска обучения"""
    parser = argparse.ArgumentParser(description='Обучение Tacotron2 с автоматическими исправлениями')
    parser.add_argument('--epochs', type=int, default=10, help='Количество эпох')
    parser.add_argument('--batch_size', type=int, default=8, help='Размер батча')
    parser.add_argument('--checkpoint', type=str, help='Путь к checkpoint для продолжения')
    parser.add_argument('--output_dir', type=str, default='output_auto_fixes', help='Директория для сохранения')
    parser.add_argument('--log_dir', type=str, default='logs_auto_fixes', help='Директория для логов')
    parser.add_argument('--disable_telegram', action='store_true', help='Отключить Telegram уведомления')
    parser.add_argument('--disable_mlflow', action='store_true', help='Отключить MLflow логирование')
    
    args = parser.parse_args()
    
    # Настройка логирования
    logger = setup_logging()
    logger.info("🚀 Запуск обучения с автоматическими исправлениями")
    
    try:
        # Создаем директории
        Path(args.output_dir).mkdir(exist_ok=True)
        Path(args.log_dir).mkdir(exist_ok=True)
        
        # Создаем оптимизированные гиперпараметры
        hparams = create_optimized_hparams(args)
        logger.info(f"📋 Гиперпараметры: LR={hparams.learning_rate:.2e}, BS={hparams.batch_size}")
        
        # Подготавливаем данные
        logger.info("📦 Подготовка данных...")
        train_loader, val_loader = prepare_dataloaders(hparams)
        logger.info(f"✅ Данные загружены: {len(train_loader)} батчей для обучения")
        
        # Создаем EnhancedTacotronTrainer с автоматическими исправлениями
        logger.info("🤖 Инициализация EnhancedTacotronTrainer с AutoFixManager...")
        trainer = EnhancedTacotronTrainer(
            hparams=hparams,
            dataset_info={'train_size': len(train_loader), 'val_size': len(val_loader) if val_loader else 0}
        )
        
        # Инициализируем обучение
        trainer.initialize_training()
        
        # Проверяем интеграцию AutoFixManager
        if hasattr(trainer, 'auto_fix_manager') and trainer.auto_fix_manager:
            logger.info("✅ AutoFixManager успешно интегрирован")
        else:
            logger.warning("⚠️ AutoFixManager не найден - автоматические исправления отключены")
        
        # Проверяем интеграцию Smart Tuner
        if hasattr(trainer, 'smart_tuner') and trainer.smart_tuner:
            logger.info("✅ Smart Tuner успешно интегрирован")
        else:
            logger.warning("⚠️ Smart Tuner не найден - оптимизация гиперпараметров отключена")
        
        # Проверяем интеграцию Telegram мониторинга
        if hasattr(trainer, 'telegram_monitor') and trainer.telegram_monitor:
            logger.info("✅ Telegram мониторинг успешно интегрирован")
        else:
            logger.warning("⚠️ Telegram мониторинг не найден - уведомления отключены")
        
        # Загружаем checkpoint если указан
        if args.checkpoint and os.path.exists(args.checkpoint):
            logger.info(f"📂 Загрузка checkpoint: {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)
        
        # Запускаем обучение
        logger.info(f"🎯 Начинаем обучение на {args.epochs} эпох...")
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=args.epochs
        )
        
        # Сохраняем финальную модель
        final_checkpoint = os.path.join(args.output_dir, 'final_model_auto_fixes.pth')
        trainer.save_checkpoint(final_checkpoint, {'epoch': args.epochs, 'final': True})
        logger.info(f"💾 Финальная модель сохранена: {final_checkpoint}")
        
        # Выводим статистику автоматических исправлений
        if hasattr(trainer, 'auto_fix_manager') and trainer.auto_fix_manager:
            stats = trainer.auto_fix_manager.get_fix_statistics()
            logger.info("📊 Статистика автоматических исправлений:")
            logger.info(f"  Всего исправлений: {stats['total_fixes']}")
            logger.info(f"  Успешных исправлений: {stats['successful_fixes']}")
            logger.info(f"  Счетчики проблем: {stats['problem_counters']}")
        
        logger.info("🎉 Обучение завершено успешно!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Обучение прервано пользователем")
        if 'trainer' in locals():
            trainer.save_checkpoint(
                os.path.join(args.output_dir, 'interrupted_auto_fixes.pth'),
                {'interrupted': True}
            )
            logger.info("💾 Checkpoint сохранен при прерывании")
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Пытаемся сохранить checkpoint при ошибке
        if 'trainer' in locals():
            try:
                trainer.save_checkpoint(
                    os.path.join(args.output_dir, 'error_auto_fixes.pth'),
                    {'error': str(e)}
                )
                logger.info("💾 Checkpoint сохранен при ошибке")
            except:
                pass
        
        sys.exit(1)

if __name__ == "__main__":
    main() 