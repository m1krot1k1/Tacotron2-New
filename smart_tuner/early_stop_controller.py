"""
Early Stop Controller для Smart Tuner V2
Интеллектуальный контроль раннего останова обучения
"""

import yaml
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import os
import sys

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - (EarlyStopController) - %(message)s')

# Добавляем обработчик исключений
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Критическая ошибка:", exc_info=(exc_type, exc_value, exc_traceback))

import sys
sys.excepthook = handle_exception

class ProactiveController:
    """Проактивные меры предотвращения проблем обучения"""
    
    def __init__(self, config):
        self.config = config
        self.metrics_history = []
        self.interventions_history = []
        
    def analyze_training_health(self, metrics: Dict) -> Dict[str, Any]:
        """Анализ здоровья обучения и предложение мер"""
        health_report = {
            "status": "healthy",
            "warnings": [],
            "interventions": [],
            "severity": "low"
        }
        
        if not self.metrics_history:
            return health_report
            
        recent_metrics = self.metrics_history[-10:]  # Последние 10 значений
        
        # 1. Анализ стагнации loss
        train_losses = [m.get('train_loss', 0) for m in recent_metrics if 'train_loss' in m]
        if len(train_losses) >= 5:
            recent_change = abs(train_losses[-1] - train_losses[-5])
            if recent_change < 0.001:  # Очень малое изменение
                health_report["warnings"].append("⚠️ Стагнация train_loss")
                health_report["interventions"].append({
                    "type": "learning_rate_boost",
                    "action": "Увеличить learning rate на 20%",
                    "reason": "Преодоление плато"
                })
                health_report["severity"] = "medium"
        
        # 2. Анализ расхождения train/val loss
        if 'train_loss' in metrics and 'val_loss' in metrics:
            gap = metrics['val_loss'] - metrics['train_loss']
            if gap > 0.5:  # Большой разрыв
                health_report["warnings"].append("🚨 Признаки переобучения")
                health_report["interventions"].append({
                    "type": "regularization_boost",
                    "action": "Увеличить dropout до 0.3",
                    "reason": "Предотвращение переобучения"
                })
                health_report["interventions"].append({
                    "type": "learning_rate_reduce",
                    "action": "Снизить learning rate на 30%",
                    "reason": "Замедление переобучения"
                })
                health_report["severity"] = "high"
        
        # 3. Анализ нестабильности
        if len(train_losses) >= 5:
            volatility = np.std(train_losses[-5:])
            if volatility > 0.1:  # Высокая волатильность
                health_report["warnings"].append("📈 Нестабильное обучение")
                health_report["interventions"].append({
                    "type": "gradient_clipping",
                    "action": "Включить gradient clipping (max_norm=1.0)",
                    "reason": "Стабилизация обучения"
                })
                health_report["interventions"].append({
                    "type": "batch_size_increase",
                    "action": "Увеличить batch size на 50%",
                    "reason": "Сглаживание градиентов"
                })
        
        # 4. Анализ скорости сходимости
        if len(train_losses) >= 10:
            early_avg = np.mean(train_losses[:5])
            recent_avg = np.mean(train_losses[-5:])
            improvement_rate = (early_avg - recent_avg) / early_avg
            
            if improvement_rate < 0.01:  # Очень медленное улучшение
                health_report["warnings"].append("🐌 Медленная сходимость")
                health_report["interventions"].append({
                    "type": "optimizer_change",
                    "action": "Переключиться на AdamW с weight_decay=0.01",
                    "reason": "Ускорение сходимости"
                })
        
        return health_report
    
    def apply_intervention(self, intervention: Dict) -> Dict[str, Any]:
        """Применение проактивной меры"""
        intervention_type = intervention["type"]
        timestamp = datetime.now().isoformat()
        
        # Запись в историю
        self.interventions_history.append({
            "timestamp": timestamp,
            "intervention": intervention,
            "applied": True
        })
        
        # Генерация новых гиперпараметров
        new_params = {}
        
        if intervention_type == "learning_rate_boost":
            current_lr = self.get_current_lr()
            new_params["learning_rate"] = current_lr * 1.2
            
        elif intervention_type == "learning_rate_reduce":
            current_lr = self.get_current_lr()
            new_params["learning_rate"] = current_lr * 0.7
            
        elif intervention_type == "regularization_boost":
            new_params["dropout"] = 0.3
            new_params["weight_decay"] = 0.01
            
        elif intervention_type == "gradient_clipping":
            new_params["gradient_clip_val"] = 1.0
            new_params["gradient_clip_algorithm"] = "norm"
            
        elif intervention_type == "batch_size_increase":
            current_batch = self.get_current_batch_size()
            new_params["batch_size"] = int(current_batch * 1.5)
            
        elif intervention_type == "optimizer_change":
            new_params["optimizer"] = "AdamW"
            new_params["weight_decay"] = 0.01
            new_params["betas"] = [0.9, 0.999]
        
        return {
            "status": "applied",
            "new_params": new_params,
            "intervention": intervention,
            "timestamp": timestamp
        }
    
    def get_current_lr(self) -> float:
        """Получение текущего learning rate"""
        if self.metrics_history:
            return self.metrics_history[-1].get('learning_rate', 0.001)
        return 0.001
    
    def get_current_batch_size(self) -> int:
        """Получение текущего batch size"""
        # Можно получить из конфигурации или метрик
        return self.config.get('hyperparameter_search_space', {}).get('batch_size', {}).get('default', 32)

class EarlyStopController:
    """
    Контроллер раннего останова с продвинутыми алгоритмами детекции переобучения
    Поддерживает множественные критерии и адаптивные пороги
    """
    
    def __init__(self, config_path: str = "smart_tuner/config.yaml"):
        """
        Инициализация контроллера раннего останова
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logger()
        
        # Состояние контроллера
        self.best_metrics = {}
        self.patience_counters = {}
        self.metric_history = {}
        self.should_stop = False
        self.stop_reasons = []
        
        # Инициализация критериев останова
        self._initialize_stop_criteria()
        
        # Инициализация проактивного контроллера
        self.proactive = ProactiveController(self.config)
        
        self.logger.info("EarlyStopController инициализирован с проактивными мерами")
        
    def _load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации из YAML файла"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Файл конфигурации {self.config_path} не найден")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Ошибка парсинга YAML: {e}")
            raise
            
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger('EarlyStopController')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _initialize_stop_criteria(self):
        """Инициализация критериев останова"""
        early_stop_config = self.config.get('early_stopping', {})
        
        for criterion_name, criterion_config in early_stop_config.items():
            if criterion_config.get('enabled', False):
                metric_name = criterion_config.get('metric', 'val_loss')
                
                # Инициализация истории метрик
                if metric_name not in self.metric_history:
                    self.metric_history[metric_name] = []
                    
                # Инициализация лучших значений
                if criterion_config.get('mode', 'min') == 'min':
                    self.best_metrics[metric_name] = float('inf')
                else:
                    self.best_metrics[metric_name] = float('-inf')
                    
                # Инициализация счетчиков терпения
                self.patience_counters[metric_name] = 0
                
                self.logger.info(f"Инициализирован критерий останова: {criterion_name}")
                
    def update_metrics(self, metrics: Dict[str, float], step: int) -> bool:
        """
        Обновление метрик и проверка условий останова
        
        Args:
            metrics: Словарь с текущими метриками
            step: Номер шага обучения
            
        Returns:
            True, если обучение нужно остановить
        """
        # Обновление истории метрик
        for metric_name, value in metrics.items():
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = []
            self.metric_history[metric_name].append({
                'step': step,
                'value': value,
                'timestamp': datetime.now().isoformat()
            })
            
        # Проверка всех критериев останова
        early_stop_config = self.config.get('early_stopping', {})
        
        for criterion_name, criterion_config in early_stop_config.items():
            if criterion_config.get('enabled', False):
                should_stop_by_criterion = self._check_criterion(
                    criterion_name, criterion_config, metrics, step
                )
                
                if should_stop_by_criterion:
                    self.should_stop = True
                    self.stop_reasons.append(f"{criterion_name} (шаг {step})")
                    
        return self.should_stop
        
    def _check_criterion(self, criterion_name: str, config: Dict[str, Any], 
                        metrics: Dict[str, float], step: int) -> bool:
        """
        Проверка конкретного критерия останова
        
        Args:
            criterion_name: Имя критерия
            config: Конфигурация критерия
            metrics: Текущие метрики
            step: Номер шага
            
        Returns:
            True, если критерий срабатывает
        """
        criterion_type = config.get('type', 'patience')
        
        if criterion_type == 'patience':
            return self._check_patience_criterion(config, metrics)
        elif criterion_type == 'threshold':
            return self._check_threshold_criterion(config, metrics)
        elif criterion_type == 'gradient':
            return self._check_gradient_criterion(config, metrics)
        elif criterion_type == 'plateau':
            return self._check_plateau_criterion(config, metrics)
        elif criterion_type == 'overfitting':
            return self._check_overfitting_criterion(config, metrics)
        elif criterion_type == 'loss_divergence':
            return self._check_divergence_criterion(config, metrics)
        else:
            self.logger.warning(f"Неизвестный тип критерия: {criterion_type}")
            return False
            
    def _check_patience_criterion(self, config: Dict[str, Any], 
                                metrics: Dict[str, float]) -> bool:
        """Проверка критерия терпения"""
        metric_name = config.get('metric', 'val_loss')
        patience = config.get('patience', 10)
        min_delta = config.get('min_delta', 0.0)
        mode = config.get('mode', 'min')
        
        if metric_name not in metrics:
            return False
            
        current_value = metrics[metric_name]
        best_value = self.best_metrics.get(metric_name)
        
        # Проверка улучшения
        improved = False
        if mode == 'min':
            if current_value < best_value - min_delta:
                improved = True
                self.best_metrics[metric_name] = current_value
        else:  # mode == 'max'
            if current_value > best_value + min_delta:
                improved = True
                self.best_metrics[metric_name] = current_value
                
        # Обновление счетчика терпения
        if improved:
            self.patience_counters[metric_name] = 0
        else:
            self.patience_counters[metric_name] += 1
            
        # Проверка превышения терпения
        if self.patience_counters[metric_name] >= patience:
            self.logger.info(
                f"Критерий терпения сработал для {metric_name}: "
                f"{self.patience_counters[metric_name]} >= {patience}"
            )
            return True
            
        return False
        
    def _check_threshold_criterion(self, config: Dict[str, Any], 
                                 metrics: Dict[str, float]) -> bool:
        """Проверка порогового критерия"""
        metric_name = config.get('metric', 'val_loss')
        threshold = config.get('threshold', 0.0)
        mode = config.get('mode', 'min')
        
        if metric_name not in metrics:
            return False
            
        current_value = metrics[metric_name]
        
        if mode == 'min' and current_value <= threshold:
            self.logger.info(f"Пороговый критерий сработал: {metric_name} <= {threshold}")
            return True
        elif mode == 'max' and current_value >= threshold:
            self.logger.info(f"Пороговый критерий сработал: {metric_name} >= {threshold}")
            return True
            
        return False
        
    def _check_gradient_criterion(self, config: Dict[str, Any], 
                                metrics: Dict[str, float]) -> bool:
        """Проверка критерия градиента (скорости изменения метрики)"""
        metric_name = config.get('metric', 'val_loss')
        window_size = config.get('window_size', 5)
        gradient_threshold = config.get('gradient_threshold', 1e-4)
        
        if metric_name not in self.metric_history:
            return False
            
        history = self.metric_history[metric_name]
        if len(history) < window_size:
            return False
            
        # Вычисление градиента за последнее окно
        recent_values = [h['value'] for h in history[-window_size:]]
        gradient = np.mean(np.diff(recent_values))
        
        if abs(gradient) < gradient_threshold:
            self.logger.info(
                f"Критерий градиента сработал: |{gradient:.6f}| < {gradient_threshold}"
            )
            return True
            
        return False
        
    def _check_plateau_criterion(self, config: Dict[str, Any], 
                               metrics: Dict[str, float]) -> bool:
        """Проверка критерия плато"""
        metric_name = config.get('metric', 'val_loss')
        window_size = config.get('window_size', 10)
        plateau_threshold = config.get('plateau_threshold', 0.01)
        
        if metric_name not in self.metric_history:
            return False
            
        history = self.metric_history[metric_name]
        if len(history) < window_size:
            return False
            
        # Проверка стабильности метрики
        recent_values = [h['value'] for h in history[-window_size:]]
        std_dev = np.std(recent_values)
        mean_value = np.mean(recent_values)
        
        # Коэффициент вариации
        cv = std_dev / abs(mean_value) if mean_value != 0 else float('inf')
        
        if cv < plateau_threshold:
            self.logger.info(
                f"Критерий плато сработал: CV = {cv:.6f} < {plateau_threshold}"
            )
            return True
            
        return False
        
    def _check_overfitting_criterion(self, config: Dict[str, Any], 
                                   metrics: Dict[str, float]) -> bool:
        """Проверка критерия переобучения"""
        train_metric = config.get('train_metric', 'train_loss')
        val_metric = config.get('val_metric', 'val_loss')
        overfitting_threshold = config.get('overfitting_threshold', 0.1)
        window_size = config.get('window_size', 5)
        
        if train_metric not in metrics or val_metric not in metrics:
            return False
            
        # Проверка разности между train и val метриками
        train_value = metrics[train_metric]
        val_value = metrics[val_metric]
        
        if train_value <= 0:
            return False
            
        overfitting_ratio = (val_value - train_value) / train_value
        
        # Проверка тренда переобучения
        if (train_metric in self.metric_history and 
            val_metric in self.metric_history):
            
            train_history = self.metric_history[train_metric]
            val_history = self.metric_history[val_metric]
            
            if len(train_history) >= window_size and len(val_history) >= window_size:
                recent_train = [h['value'] for h in train_history[-window_size:]]
                recent_val = [h['value'] for h in val_history[-window_size:]]
                
                # Тренд: train loss уменьшается, val loss увеличивается
                train_trend = np.mean(np.diff(recent_train))
                val_trend = np.mean(np.diff(recent_val))
                
                if train_trend < 0 and val_trend > 0 and overfitting_ratio > overfitting_threshold:
                    self.logger.info(
                        f"Критерий переобучения сработал: "
                        f"ratio = {overfitting_ratio:.4f} > {overfitting_threshold}"
                    )
                    return True
                    
        return False
        
    def _check_divergence_criterion(self, config: Dict[str, Any], 
                                  metrics: Dict[str, float]) -> bool:
        """Проверка критерия расхождения loss"""
        metric_name = config.get('metric', 'train_loss')
        divergence_threshold = config.get('divergence_threshold', 10.0)
        
        if metric_name not in metrics:
            return False
            
        current_value = metrics[metric_name]
        
        # Проверка на NaN или Inf
        if np.isnan(current_value) or np.isinf(current_value):
            self.logger.info(f"Критерий расхождения сработал: {metric_name} = {current_value}")
            return True
            
        # Проверка превышения порога
        if current_value > divergence_threshold:
            self.logger.info(
                f"Критерий расхождения сработал: "
                f"{current_value} > {divergence_threshold}"
            )
            return True
            
        return False
        
    def get_best_metrics(self) -> Dict[str, float]:
        """Получение лучших значений метрик"""
        return self.best_metrics.copy()
        
    def get_stop_reasons(self) -> List[str]:
        """Получение причин останова"""
        return self.stop_reasons.copy()
        
    def reset(self):
        """Сброс состояния контроллера"""
        self.should_stop = False
        self.stop_reasons = []
        self.best_metrics = {}
        self.patience_counters = {}
        self.metric_history = {}
        
        # Повторная инициализация
        self._initialize_stop_criteria()
        
        self.logger.info("Состояние контроллера раннего останова сброшено")
        
        # Сброс проактивного контроллера
        self.proactive.metrics_history = []
        self.proactive.interventions_history = []
        
    def save_state(self, output_path: str = "smart_tuner/early_stop_state.json"):
        """
        Сохранение состояния контроллера
        
        Args:
            output_path: Путь для сохранения состояния
        """
        state = {
            'should_stop': self.should_stop,
            'stop_reasons': self.stop_reasons,
            'best_metrics': self.best_metrics,
            'patience_counters': self.patience_counters,
            'metric_history': self.metric_history,
            'interventions_history': self.proactive.interventions_history
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Состояние сохранено в {output_path}")
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения состояния: {e}")
            
    def load_state(self, input_path: str = "smart_tuner/early_stop_state.json"):
        """
        Загрузка состояния контроллера
        
        Args:
            input_path: Путь к файлу состояния
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
                
            self.should_stop = state.get('should_stop', False)
            self.stop_reasons = state.get('stop_reasons', [])
            self.best_metrics = state.get('best_metrics', {})
            self.patience_counters = state.get('patience_counters', {})
            self.metric_history = state.get('metric_history', {})
            self.proactive.interventions_history = state.get('interventions_history', [])
            
            self.logger.info(f"Состояние загружено из {input_path}")
            
        except FileNotFoundError:
            self.logger.warning(f"Файл состояния {input_path} не найден")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки состояния: {e}")
            
    def get_status_report(self) -> Dict[str, Any]:
        """
        Получение отчета о состоянии контроллера
        
        Returns:
            Словарь с информацией о состоянии
        """
        report = {
            'should_stop': self.should_stop,
            'stop_reasons': self.stop_reasons,
            'best_metrics': self.best_metrics,
            'patience_counters': self.patience_counters,
            'active_criteria': [],
            'proactive_interventions': len(self.proactive.interventions_history)
        }
        
        # Добавление информации об активных критериях
        early_stop_config = self.config.get('early_stopping', {})
        for criterion_name, criterion_config in early_stop_config.items():
            if criterion_config.get('enabled', False):
                report['active_criteria'].append({
                    'name': criterion_name,
                    'type': criterion_config.get('type', 'patience'),
                    'metric': criterion_config.get('metric', 'val_loss'),
                    'config': criterion_config
                })
                
        return report 

    def should_stop_training(self, current_metrics: Dict[str, float]) -> Tuple[bool, str, Dict]:
        """
        Определяет, следует ли остановить обучение с проактивными мерами.
        
        Args:
            current_metrics: Текущие метрики обучения
            
        Returns:
            Tuple[bool, str, Dict]: (should_stop, reason, details)
        """
        # Обновляем историю метрик для проактивного контроллера
        timestamped_metrics = {
            **current_metrics,
            'timestamp': datetime.now().isoformat(),
            'epoch': len(self.proactive.metrics_history) + 1
        }
        self.proactive.metrics_history.append(timestamped_metrics)
        
        # Сначала проверяем проактивные меры
        health_report = self.proactive.analyze_training_health(current_metrics)
        
        # Если есть критические проблемы, применяем меры
        if health_report["severity"] == "high" and health_report["interventions"]:
            self.logger.warning(f"🚨 Обнаружены проблемы: {health_report['warnings']}")
            
            # Применяем первую интервенцию
            intervention_result = self.proactive.apply_intervention(health_report["interventions"][0])
            self.logger.info(f"🛡️ Применена проактивная мера: {intervention_result}")
            
            # Возвращаем рекомендацию о корректировке, но не останавливаем
            return False, "proactive_intervention", {
                "health_report": health_report,
                "intervention": intervention_result,
                "continue_training": True
            }
        
        # Проверяем стандартные критерии останова через update_metrics
        step = len(self.proactive.metrics_history)
        should_stop = self.update_metrics(current_metrics, step)
        
        if should_stop:
            reason = f"early_stopping: {', '.join(self.stop_reasons)}"
            details = {
                "stop_reasons": self.stop_reasons,
                "best_metrics": self.best_metrics,
                "patience_counters": self.patience_counters
            }
            self.logger.info(f"🛑 Решение об остановке: {reason}")
            return True, reason, details
        
        # Если есть предупреждения, но не критичные
        if health_report["warnings"]:
            self.logger.info(f"⚠️ Предупреждения: {health_report['warnings']}")
        
        return False, "continue", {"health_report": health_report} 