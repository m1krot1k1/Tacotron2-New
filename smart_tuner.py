#!/usr/bin/env python3
"""
Smart Tuner V2 для Tacotron2
Главный файл системы автоматического тюнинга гиперпараметров
"""

import yaml
import logging
import argparse
from pathlib import Path

def setup_logging():
    """Настройка базового логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class SmartTunerV2:
    """
    Главный класс Smart Tuner V2
    Координирует работу всех компонентов системы
    """
    
    def __init__(self, config_path: str = "smart_tuner/config.yaml"):
        """Инициализация Smart Tuner V2"""
        self.config_path = config_path
        self.logger = logging.getLogger('SmartTunerV2')
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self.logger.info("Smart Tuner V2 инициализирован успешно")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки конфигурации: {e}")
            self.config = {}
            
    def run_demo(self):
        """Запуск демонстрации возможностей"""
        self.logger.info("=== Демонстрация Smart Tuner V2 ===")
        
        # Проверка конфигурации
        self.logger.info("1. Проверка конфигурации...")
        self.logger.info(f"   Эксперимент: {self.config.get('experiment_name', 'Unknown')}")
        self.logger.info(f"   Датасет: {self.config.get('dataset_path', 'Unknown')}")
        
        # Проверка пространства поиска
        self.logger.info("2. Проверка пространства поиска...")
        search_space = self.config.get('hyperparameter_search_space', {})
        for param_name, param_config in search_space.items():
            param_type = param_config.get('type', 'unknown')
            param_range = f"[{param_config.get('min', 'N/A')} - {param_config.get('max', 'N/A')}]"
            self.logger.info(f"   {param_name}: {param_type} {param_range}")
                
        # Проверка настроек оптимизации
        self.logger.info("3. Проверка настроек оптимизации...")
        opt_config = self.config.get('optimization', {})
        self.logger.info(f"   Направление: {opt_config.get('direction', 'Unknown')}")
        self.logger.info(f"   Целевая метрика: {opt_config.get('objective_metric', 'Unknown')}")
        self.logger.info(f"   Количество trials: {opt_config.get('n_trials', 'Unknown')}")
        
        # Проверка настроек раннего останова
        self.logger.info("4. Проверка настроек раннего останова...")
        early_stop_config = self.config.get('early_stopping', {})
        enabled_criteria = [name for name, config in early_stop_config.items() 
                          if config.get('enabled', False)]
        self.logger.info(f"   Активные критерии: {enabled_criteria}")
        
        # Проверка уведомлений
        self.logger.info("5. Проверка уведомлений...")
        telegram_config = self.config.get('telegram', {})
        self.logger.info(f"   Telegram включен: {telegram_config.get('enabled', False)}")
        
        self.logger.info("=== Демонстрация завершена ===")
        
    def test_components(self):
        """Тестирование компонентов"""
        self.logger.info("=== Тестирование компонентов Smart Tuner V2 ===")
        
        components = [
            'trainer_wrapper.py',
            'metrics_store.py', 
            'log_watcher.py',
            'optimization_engine.py',
            'param_scheduler.py',
            'early_stop_controller.py',
            'alert_manager.py',
            'model_registry.py'
        ]
        
        for component in components:
            component_path = Path(f"smart_tuner/{component}")
            if component_path.exists():
                self.logger.info(f"✅ {component} - найден")
            else:
                self.logger.warning(f"❌ {component} - не найден")
                
        self.logger.info("=== Тестирование завершено ===")
        
    def show_status(self):
        """Показать статус системы"""
        self.logger.info("=== Статус Smart Tuner V2 ===")
        self.logger.info(f"Конфигурация: {self.config_path}")
        self.logger.info(f"Эксперимент: {self.config.get('experiment_name', 'Unknown')}")
        
        # Проверка файлов
        smart_tuner_dir = Path("smart_tuner")
        if smart_tuner_dir.exists():
            py_files = list(smart_tuner_dir.glob("*.py"))
            self.logger.info(f"Python файлов: {len(py_files)}")
            
            config_file = smart_tuner_dir / "config.yaml"
            if config_file.exists():
                self.logger.info("✅ Конфигурация найдена")
            else:
                self.logger.warning("❌ Конфигурация не найдена")
        else:
            self.logger.error("❌ Папка smart_tuner не найдена")
            
        self.logger.info("=== Конец статуса ===")

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Smart Tuner V2 для Tacotron2')
    parser.add_argument('--config', default='smart_tuner/config.yaml', 
                       help='Путь к файлу конфигурации')
    parser.add_argument('--test', action='store_true', 
                       help='Запуск тестирования компонентов')
    parser.add_argument('--demo', action='store_true',
                       help='Запуск демонстрации возможностей')
    parser.add_argument('--status', action='store_true',
                       help='Показать статус системы')
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        smart_tuner = SmartTunerV2(args.config)
        
        if args.test:
            smart_tuner.test_components()
        elif args.demo:
            smart_tuner.run_demo()
        elif args.status:
            smart_tuner.show_status()
        else:
            # По умолчанию запускаем демонстрацию
            smart_tuner.run_demo()
            
        return 0
        
    except Exception as e:
        logging.error(f"Критическая ошибка: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
 