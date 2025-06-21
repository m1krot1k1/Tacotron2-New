#!/usr/bin/env python3
"""
Тест интеграции с Optuna Dashboard
Запускает небольшой эксперимент для проверки, что данные попадают в базу данных.
"""

import os
import sys
import logging
from smart_tuner_main import SmartTunerMain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_optuna_integration():
    """Тестирует интеграцию с Optuna Dashboard"""
    
    logger.info("🧪 Запуск теста интеграции с Optuna...")
    
    try:
        # Инициализируем Smart Tuner
        tuner = SmartTunerMain("smart_tuner/config.yaml")
        
        # Запускаем небольшую оптимизацию (2 trials для теста)
        logger.info("Запуск оптимизации с 2 trials для тестирования...")
        best_params = tuner.run_optimization(n_trials=2)
        
        logger.info(f"✅ Тест завершен. Лучшие параметры: {best_params}")
        
        # Проверяем, что база данных создана
        db_path = "smart_tuner/optuna_studies.db"
        if os.path.exists(db_path):
            size = os.path.getsize(db_path)
            logger.info(f"✅ База данных Optuna создана: {db_path} ({size} байт)")
        else:
            logger.error("❌ База данных Optuna не найдена!")
            
        logger.info("🌐 Проверьте Optuna Dashboard: http://localhost:5002")
        
    except Exception as e:
        logger.error(f"❌ Ошибка в тесте: {e}")
        raise

if __name__ == "__main__":
    test_optuna_integration() 