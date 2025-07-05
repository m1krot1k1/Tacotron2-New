#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Интеграционный модуль для перехода на EnhancedTacotronTrainer
Переход с train.py на enhanced_training_main.py как основной движок

Этот модуль обеспечивает:
1. Плавный переход на EnhancedTacotronTrainer
2. Сохранение совместимости с существующими компонентами
3. Интеграцию всех улучшений Smart Tuner
4. Автоматическую миграцию конфигураций
"""

import os
import sys
import logging
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import argparse

# Импорт основных компонентов
from enhanced_training_main import EnhancedTacotronTrainer, prepare_dataloaders
from hparams import create_hparams
from model import Tacotron2

# Импорт Smart Tuner компонентов
try:
    from smart_tuner.smart_tuner_integration import SmartTunerIntegration
    from smart_tuner.telegram_monitor import TelegramMonitor
    from smart_tuner.optimization_engine import OptimizationEngine
    SMART_TUNER_AVAILABLE = True
except ImportError:
    SMART_TUNER_AVAILABLE = False
    logging.warning("Smart Tuner не найден, используется стандартное обучение")

class EnhancedTrainingIntegration:
    """
    Интеграционный класс для перехода на EnhancedTacotronTrainer
    """
    
    def __init__(self, config_path: str = "smart_tuner/config.yaml"):
        """
        Инициализация интеграционного модуля
        
        Args:
            config_path: Путь к конфигурации Smart Tuner
        """
        self.config_path = config_path
        self.logger = self._setup_logger()
        
        # Загрузка конфигурации
        self.config = self._load_config()
        
        # Состояние интеграции
        self.integration_status = {
            'enhanced_trainer_ready': False,
            'smart_tuner_integrated': False,
            'optimization_engine_ready': False,
            'telegram_monitor_ready': False
        }
        
        self.logger.info("🎯 Enhanced Training Integration инициализирован")
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger('EnhancedTrainingIntegration')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Конфигурация {self.config_path} не найдена, используем значения по умолчанию")
            return {}
        except yaml.YAMLError as e:
            self.logger.error(f"Ошибка парсинга конфигурации: {e}")
            return {}
    
    def validate_enhanced_trainer(self) -> bool:
        """
        Валидация EnhancedTacotronTrainer
        
        Returns:
            True если EnhancedTacotronTrainer готов к использованию
        """
        try:
            # Проверяем доступность всех необходимых компонентов
            from enhanced_training_main import EnhancedTacotronTrainer
            
            # Создаем тестовые гиперпараметры
            hparams = create_hparams()
            
            # Создаем тестовый тренер
            trainer = EnhancedTacotronTrainer(hparams)
            
            # Проверяем основные методы
            required_methods = [
                'initialize_training',
                'train_step', 
                'validate_step',
                'train_epoch',
                'train'
            ]
            
            for method_name in required_methods:
                if not hasattr(trainer, method_name):
                    self.logger.error(f"Метод {method_name} не найден в EnhancedTacotronTrainer")
                    return False
            
            self.integration_status['enhanced_trainer_ready'] = True
            self.logger.info("✅ EnhancedTacotronTrainer валидирован успешно")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка валидации EnhancedTacotronTrainer: {e}")
            return False
    
    def integrate_smart_tuner(self) -> bool:
        """
        Интеграция Smart Tuner с EnhancedTacotronTrainer
        
        Returns:
            True если интеграция успешна
        """
        if not SMART_TUNER_AVAILABLE:
            self.logger.warning("Smart Tuner недоступен, пропускаем интеграцию")
            return False
        
        try:
            # Проверяем доступность основных компонентов Smart Tuner
            from smart_tuner.smart_tuner_integration import SmartTunerIntegration
            from smart_tuner.optimization_engine import OptimizationEngine
            
            # Создаем тестовые компоненты
            smart_tuner = SmartTunerIntegration()
            optimization_engine = OptimizationEngine(self.config_path)
            
            # Проверяем основные методы
            if hasattr(smart_tuner, 'on_training_start') and hasattr(smart_tuner, 'on_batch_end'):
                self.integration_status['smart_tuner_integrated'] = True
                self.logger.info("✅ Smart Tuner интегрирован успешно")
                return True
            else:
                self.logger.error("Smart Tuner не имеет необходимых методов")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка интеграции Smart Tuner: {e}")
            return False
    
    def setup_optimization_engine(self) -> bool:
        """
        Настройка Optimization Engine
        
        Returns:
            True если настройка успешна
        """
        if not SMART_TUNER_AVAILABLE:
            self.logger.warning("Smart Tuner недоступен, пропускаем Optimization Engine")
            return False
        
        try:
            from smart_tuner.optimization_engine import OptimizationEngine
            
            # Создаем Optimization Engine
            optimization_engine = OptimizationEngine(self.config_path)
            
            # Проверяем основные методы
            if hasattr(optimization_engine, 'create_study_with_retry') and hasattr(optimization_engine, 'optimize'):
                self.integration_status['optimization_engine_ready'] = True
                self.logger.info("✅ Optimization Engine настроен успешно")
                return True
            else:
                self.logger.error("Optimization Engine не имеет необходимых методов")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка настройки Optimization Engine: {e}")
            return False
    
    def setup_telegram_monitor(self) -> bool:
        """
        Настройка Telegram Monitor
        
        Returns:
            True если настройка успешна
        """
        if not SMART_TUNER_AVAILABLE:
            self.logger.warning("Smart Tuner недоступен, пропускаем Telegram Monitor")
            return False
        
        try:
            from smart_tuner.telegram_monitor import TelegramMonitor
            
            # Проверяем конфигурацию Telegram
            telegram_config = self.config.get('telegram', {})
            bot_token = telegram_config.get('bot_token')
            chat_id = telegram_config.get('chat_id')
            enabled = telegram_config.get('enabled', False)
            
            if bot_token and chat_id and enabled:
                # Создаем тестовый монитор
                monitor = TelegramMonitor(bot_token, chat_id)
                
                if hasattr(monitor, 'send_training_update'):
                    self.integration_status['telegram_monitor_ready'] = True
                    self.logger.info("✅ Telegram Monitor настроен успешно")
                    return True
                else:
                    self.logger.error("Telegram Monitor не имеет необходимых методов")
                    return False
            else:
                self.logger.warning("Telegram Monitor отключен (неполные настройки)")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка настройки Telegram Monitor: {e}")
            return False
    
    def migrate_configuration(self) -> Dict[str, Any]:
        """
        Миграция конфигурации с train.py на EnhancedTacotronTrainer
        
        Returns:
            Обновленная конфигурация
        """
        try:
            # Загружаем текущие гиперпараметры
            hparams = create_hparams()
            
            # Создаем мигрированную конфигурацию
            migrated_config = {
                'hparams': vars(hparams),
                'smart_tuner': {
                    'enabled': SMART_TUNER_AVAILABLE,
                    'config_path': self.config_path
                },
                'training': {
                    'max_epochs': getattr(hparams, 'epochs', 500000),
                    'validation_interval': getattr(hparams, 'validate_interval', 200),
                    'checkpoint_interval': getattr(hparams, 'iters_per_checkpoint', 1000)
                },
                'monitoring': {
                    'telegram_enabled': self.integration_status['telegram_monitor_ready'],
                    'tensorboard_enabled': True,
                    'mlflow_enabled': True
                }
            }
            
            self.logger.info("✅ Конфигурация успешно мигрирована")
            return migrated_config
            
        except Exception as e:
            self.logger.error(f"Ошибка миграции конфигурации: {e}")
            return {}
    
    def create_enhanced_trainer(self, hparams=None, dataset_info=None) -> Optional[EnhancedTacotronTrainer]:
        """
        Создание EnhancedTacotronTrainer с полной интеграцией
        
        Args:
            hparams: Гиперпараметры (если None, создаются автоматически)
            dataset_info: Информация о датасете
            
        Returns:
            EnhancedTacotronTrainer или None в случае ошибки
        """
        try:
            # Проверяем готовность интеграции
            if not self.integration_status['enhanced_trainer_ready']:
                self.logger.error("EnhancedTacotronTrainer не готов")
                return None
            
            # Создаем гиперпараметры если не предоставлены
            if hparams is None:
                hparams = create_hparams()
            
            # Создаем информацию о датасете если не предоставлена
            if dataset_info is None:
                dataset_info = {
                    'total_duration_minutes': 120,
                    'num_speakers': 1,
                    'voice_complexity': 'moderate',
                    'audio_quality': 'good',
                    'language': 'ru'
                }
            
            # Создаем EnhancedTacotronTrainer
            trainer = EnhancedTacotronTrainer(hparams, dataset_info)
            
            self.logger.info("✅ EnhancedTacotronTrainer создан успешно")
            return trainer
            
        except Exception as e:
            self.logger.error(f"Ошибка создания EnhancedTacotronTrainer: {e}")
            return None
    
    def run_full_integration_test(self) -> bool:
        """
        Полный тест интеграции всех компонентов
        
        Returns:
            True если все тесты пройдены
        """
        self.logger.info("🧪 Начинаем полный тест интеграции...")
        
        # Тест 1: Валидация EnhancedTacotronTrainer
        if not self.validate_enhanced_trainer():
            self.logger.error("❌ Тест 1 провален: EnhancedTacotronTrainer")
            return False
        
        # Тест 2: Интеграция Smart Tuner
        if not self.integrate_smart_tuner():
            self.logger.warning("⚠️ Тест 2: Smart Tuner недоступен")
        
        # Тест 3: Настройка Optimization Engine
        if not self.setup_optimization_engine():
            self.logger.warning("⚠️ Тест 3: Optimization Engine недоступен")
        
        # Тест 4: Настройка Telegram Monitor
        if not self.setup_telegram_monitor():
            self.logger.warning("⚠️ Тест 4: Telegram Monitor недоступен")
        
        # Тест 5: Создание тренера
        trainer = self.create_enhanced_trainer()
        if trainer is None:
            self.logger.error("❌ Тест 5 провален: создание тренера")
            return False
        
        # Тест 6: Подготовка DataLoader'ов
        try:
            hparams = create_hparams()
            train_loader, val_loader = prepare_dataloaders(hparams)
            self.logger.info("✅ Тест 6 пройден: DataLoader'ы подготовлены")
        except Exception as e:
            self.logger.error(f"❌ Тест 6 провален: {e}")
            return False
        
        self.logger.info("🎉 Все тесты интеграции пройдены успешно!")
        return True
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Возвращает статус интеграции
        
        Returns:
            Словарь со статусом всех компонентов
        """
        return {
            'integration_status': self.integration_status,
            'smart_tuner_available': SMART_TUNER_AVAILABLE,
            'config_loaded': bool(self.config),
            'ready_for_training': self.integration_status['enhanced_trainer_ready']
        }


def main():
    """Главная функция для запуска интеграции"""
    parser = argparse.ArgumentParser(description='Enhanced Training Integration')
    parser.add_argument('--config', type=str, default='smart_tuner/config.yaml',
                       help='Путь к конфигурации Smart Tuner')
    parser.add_argument('--test', action='store_true',
                       help='Запустить полный тест интеграции')
    parser.add_argument('--migrate', action='store_true',
                       help='Мигрировать конфигурацию')
    
    args = parser.parse_args()
    
    # Создаем интеграционный модуль
    integration = EnhancedTrainingIntegration(args.config)
    
    if args.test:
        # Запускаем полный тест
        success = integration.run_full_integration_test()
        if success:
            print("🎉 Интеграция готова к использованию!")
        else:
            print("❌ Интеграция требует доработки")
            sys.exit(1)
    
    elif args.migrate:
        # Мигрируем конфигурацию
        config = integration.migrate_configuration()
        print("✅ Конфигурация мигрирована:")
        print(yaml.dump(config, default_flow_style=False))
    
    else:
        # Показываем статус
        status = integration.get_integration_status()
        print("📊 Статус интеграции:")
        print(yaml.dump(status, default_flow_style=False))


if __name__ == "__main__":
    main() 