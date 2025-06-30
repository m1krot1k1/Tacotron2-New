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
from datetime import datetime

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
        
        # Отслеживание изменений параметров
        self.parameter_changes = []
        self.current_phase = "initialization"
    
    def _load_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию из файла."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Не удалось загрузить конфигурацию: {e}")
            return {}
    
    def _setup_logger(self) -> logging.Logger:
        """Настраивает логгер."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_schedulers(self):
        """Инициализирует планировщики для различных параметров."""
        scheduler_config = self.config.get('param_scheduler', {})
        
        # Планировщик learning rate
        lr_config = scheduler_config.get('learning_rate', {})
        self.schedulers['learning_rate'] = self._create_lr_scheduler(lr_config)
        
        # Планировщик dropout
        dropout_config = scheduler_config.get('dropout', {})
        self.schedulers['p_attention_dropout'] = self._create_dropout_scheduler(dropout_config)
        self.schedulers['p_decoder_dropout'] = self._create_dropout_scheduler(dropout_config)
        
        # Планировщик gate threshold
        gate_config = scheduler_config.get('gate_threshold', {})
        self.schedulers['gate_threshold'] = self._create_gate_scheduler(gate_config)
    
    def _create_lr_scheduler(self, config: Dict[str, Any]) -> Callable:
        """Создает планировщик learning rate."""
        strategy = config.get('strategy', 'cosine')
        initial_lr = config.get('initial', 1e-3)
        final_lr = config.get('final', 1e-5)
        warmup_steps = config.get('warmup_steps', 1000)
        total_steps = config.get('total_steps', 10000)
        
        def lr_scheduler(step: int) -> float:
            if step < warmup_steps:
                # Linear warmup
                return initial_lr * (step / warmup_steps)
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                progress = min(1.0, max(0.0, progress))
                return final_lr + (initial_lr - final_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        return lr_scheduler
    
    def _create_dropout_scheduler(self, config: Dict[str, Any]) -> Callable:
        """Создает планировщик dropout."""
        initial_dropout = config.get('initial', 0.1)
        final_dropout = config.get('final', 0.05)
        decay_steps = config.get('decay_steps', 5000)
        
        def dropout_scheduler(step: int) -> float:
            progress = min(1.0, step / decay_steps)
            return initial_dropout + (final_dropout - initial_dropout) * progress
        
        return dropout_scheduler
    
    def _create_gate_scheduler(self, config: Dict[str, Any]) -> Callable:
        """Создает планировщик gate threshold."""
        initial_threshold = config.get('initial', 0.5)
        final_threshold = config.get('final', 0.3)
        decay_steps = config.get('decay_steps', 3000)
        
        def gate_scheduler(step: int) -> float:
            progress = min(1.0, step / decay_steps)
            return initial_threshold + (final_threshold - initial_threshold) * progress
        
        return gate_scheduler
    
    def update(self, step: int) -> Dict[str, float]:
        """
        Обновляет параметры на основе текущего шага.
        
        Args:
            step: Текущий шаг обучения
            
        Returns:
            Словарь с новыми значениями параметров
        """
        self.current_step = step
        new_params = {}
        
        # Обновляем каждый параметр
        for param_name, scheduler in self.schedulers.items():
            new_value = scheduler(step)
            new_params[param_name] = new_value
        
        # Сохраняем в историю
        self.history.append({
            'step': step,
            'params': new_params.copy(),
            'timestamp': datetime.now().isoformat()
        })
        
        # Обновляем фазу
        self._update_phase(step)
        
        return new_params
    
    def _update_phase(self, step: int):
        """Обновляет текущую фазу планировщика."""
        if step < 1000:
            self.current_phase = "warmup"
        elif step < 5000:
            self.current_phase = "main_training"
        elif step < 10000:
            self.current_phase = "fine_tuning"
        else:
            self.current_phase = "final_adjustment"
    
    def get_status(self) -> Dict[str, Any]:
        """Возвращает текущий статус планировщика."""
        return {
            "active": True,
            "phase": self.current_phase,
            "current_step": self.current_step,
            "total_updates": len(self.history),
            "current_params": self.history[-1]['params'] if self.history else {}
        }
    
    def get_recommendations(self) -> List[str]:
        """Возвращает рекомендации планировщика."""
        recommendations = []
        
        if self.current_phase == "warmup":
            recommendations.append("Фаза разогрева - постепенное увеличение learning rate")
        elif self.current_phase == "main_training":
            recommendations.append("Основное обучение - стабильные параметры")
        elif self.current_phase == "fine_tuning":
            recommendations.append("Тонкая настройка - снижение learning rate и dropout")
        elif self.current_phase == "final_adjustment":
            recommendations.append("Финальная настройка - минимальные изменения")
        
        return recommendations
    
    def track_parameter_change(self, param_name: str, old_value: Any, new_value: Any, reason: str, step: int):
        """Отслеживает изменение параметра."""
        change_info = {
            'param_name': param_name,
            'old_value': old_value,
            'new_value': new_value,
            'reason': reason,
            'step': step,
            'timestamp': datetime.now().isoformat()
        }
        self.parameter_changes.append(change_info)
        
        # Ограничиваем историю изменений
        if len(self.parameter_changes) > 50:
            self.parameter_changes = self.parameter_changes[-50:]
    
    def get_parameter_changes(self) -> List[Dict[str, Any]]:
        """Возвращает последние изменения параметров."""
        return self.parameter_changes[-10:]  # Последние 10 изменений
