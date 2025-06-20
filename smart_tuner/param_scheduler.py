"""
Parameter Scheduler для Smart Tuner V2
Динамическое изменение гиперпараметров во время обучения
"""

import yaml
import logging
import math
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import numpy as np

class ParamScheduler:
    """
    Планировщик гиперпараметров для динамического изменения во время обучения
    Поддерживает различные стратегии: linear, exponential, cosine, step, plateau
    """
    
    def __init__(self, config_path: str = "smart_tuner/config.yaml"):
        """
        Инициализация планировщика параметров
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logger()
        self.current_epoch = 0
        self.current_step = 0
        self.history = []
        
        # Кэш для хранения функций планирования
        self.schedulers = {}
        self._initialize_schedulers()
