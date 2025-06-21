"""
Early Stop Controller для Smart Tuner V2
Интеллектуальный контроль раннего останова и проактивное вмешательство в обучение.
"""

import yaml
import logging
import numpy as np
from typing import Dict, Any, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - (EarlyStopController) - %(message)s')

class EarlyStopController:
    """
    Контроллер, который совмещает:
    1. Проактивные меры: пытается "вылечить" обучение, если оно идет не так.
    2. Ранний останов: останавливает безнадежное обучение для экономии ресурсов.
    """
    
    def __init__(self, config_path: str = "smart_tuner/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.proactive_config = self.config.get('proactive_measures', {})
        self.early_stop_config = self.config.get('early_stopping', {})
        
        self.metrics_history = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Состояние для проактивных мер
        self.stagnation_counter = 0
        self.overfitting_counter = 0

        # Состояние для раннего останова
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        
        self.logger.info("EarlyStopController с проактивными мерами инициализирован.")

    def add_metrics(self, metrics: Dict[str, float]):
        """Добавляет новый набор метрик в историю."""
        if 'train_loss' in metrics and 'val_loss' in metrics and 'grad_norm' in metrics:
            self.metrics_history.append(metrics)
    
    def decide_next_step(self, current_hparams: Dict) -> Dict[str, Any]:
        """
        Главный метод, принимающий решение о следующем шаге.
        Возвращает словарь с действием: 'continue', 'stop' или 'restart'.
        """
        if len(self.metrics_history) < self.proactive_config.get('min_history_points', 5):
            return {'action': 'continue', 'reason': 'Not enough data for a decision'}

        # 1. Сначала проверяем, нужны ли проактивные меры
        if self.proactive_config.get('enabled', False):
            proactive_decision = self._check_proactive_measures(current_hparams)
            if proactive_decision['action'] != 'continue':
                self.reset_counters() # Сбрасываем счетчики после вмешательства
                return proactive_decision

        # 2. Если вмешательство не требуется, проверяем условия для полной остановки
        stop_decision = self._check_hard_stop_conditions()
        return stop_decision

    def _check_proactive_measures(self, hparams: Dict) -> Dict[str, Any]:
        """Анализирует метрики и решает, нужно ли "лечить" обучение."""
        
        last_metrics = self.metrics_history[-1]
        
        # --- Проверка №1: Стагнация ---
        stagnation_conf = self.proactive_config.get('stagnation_detection', {})
        if stagnation_conf.get('enabled', False):
            if len(self.metrics_history) > 1:
                # Считаем среднее улучшение за последние N шагов
                recent_losses = [m['val_loss'] for m in self.metrics_history[-stagnation_conf.get('patience', 15):]]
                if len(recent_losses) == stagnation_conf.get('patience', 15):
                    improvement = recent_losses[0] - recent_losses[-1]
                    if improvement < stagnation_conf.get('min_delta', 0.001):
                        self.stagnation_counter += 1
                    else:
                        self.stagnation_counter = 0

                if self.stagnation_counter >= 3: # Если стагнация наблюдается 3 окна подряд
                    new_hparams = hparams.copy()
                    new_hparams['learning_rate'] *= stagnation_conf['action'].get('learning_rate_multiplier', 1.2)
                    self.logger.warning("Обнаружена стагнация! Увеличиваю learning_rate.")
                    return {
                        'action': 'restart',
                        'reason': 'Stagnation detected',
                        'new_params': new_hparams
                    }

        # --- Проверка №2: Переобучение ---
        overfitting_conf = self.proactive_config.get('overfitting_detection', {})
        if overfitting_conf.get('enabled', False):
            gap = last_metrics['val_loss'] - last_metrics['train_loss']
            if gap > overfitting_conf.get('threshold', 0.1):
                self.overfitting_counter += 1
            else:
                self.overfitting_counter = 0

            if self.overfitting_counter >= overfitting_conf.get('patience', 5):
                new_hparams = hparams.copy()
                new_hparams['learning_rate'] *= overfitting_conf['action'].get('learning_rate_multiplier', 0.7)
                # Предполагаем, что hparams.py имеет 'dropout_rate'
                new_hparams['dropout_rate'] = min(hparams.get('dropout_rate', 0.5) + overfitting_conf['action'].get('dropout_rate_increase', 0.1), 0.9)
                self.logger.warning("Обнаружено переобучение! Уменьшаю LR, увеличиваю dropout.")
                return {
                    'action': 'restart',
                    'reason': 'Overfitting detected',
                    'new_params': new_hparams
                }

        # --- Проверка №3: Нестабильность ---
        instability_conf = self.proactive_config.get('instability_detection', {})
        if instability_conf.get('enabled', False) and 'grad_norm' in last_metrics:
            if last_metrics['grad_norm'] > instability_conf.get('grad_norm_threshold', 50.0):
                new_hparams = hparams.copy()
                if instability_conf['action'].get('enable_gradient_clipping', True):
                    new_hparams['grad_clip_thresh'] = instability_conf['action'].get('gradient_clip_thresh', 1.0)
                new_hparams['batch_size'] = int(hparams.get('batch_size', 16) * instability_conf['action'].get('batch_size_multiplier', 1.2))
                self.logger.warning("Обнаружена нестабильность! Включаю clipping, увеличиваю batch_size.")
                return {
                    'action': 'restart',
                    'reason': 'Instability detected (gradient explosion)',
                    'new_params': new_hparams
                }

        return {'action': 'continue'}

    def _check_hard_stop_conditions(self) -> Dict[str, Any]:
        """Проверяет классические условия для полной остановки обучения."""
        patience = self.early_stop_config.get('patience', 25) # Увеличим терпение
        min_delta = self.early_stop_config.get('min_delta', 0.001)
        metric_to_check = self.early_stop_config.get('metric', 'val_loss')
        
        current_metric_val = self.metrics_history[-1].get(metric_to_check)
        if current_metric_val is None:
            return {'action': 'continue'} 

        if current_metric_val < self.best_val_loss - min_delta:
            self.best_val_loss = current_metric_val
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= patience:
            self.logger.info(f"Early stopping triggered after {patience} checks without improvement.")
            return {
                'action': 'stop',
                'reason': f'Metric {metric_to_check} did not improve for {patience} checks.'
            }
            
        return {'action': 'continue'}

    def reset_counters(self):
        """Сбрасывает счетчики после вмешательства, чтобы дать изменениям время подействовать."""
        self.stagnation_counter = 0
        self.overfitting_counter = 0
        self.logger.info("Счетчики проактивных мер сброшены.")

    def reset(self):
        """Полностью сбрасывает состояние контроллера для нового запуска."""
        self.metrics_history = []
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        self.reset_counters()
        self.logger.info("EarlyStopController был полностью сброшен в исходное состояние.")