"""
Optimization Engine для Smart Tuner V2
Автоматический подбор гиперпараметров с использованием Optuna
"""

import optuna
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
from datetime import datetime

class OptimizationEngine:
    """
    Движок оптимизации гиперпараметров на основе Optuna
    Автоматически подбирает лучшие гиперпараметры на основе метрик обучения
    """
    
    def __init__(self, config_path: str = "smart_tuner/config.yaml"):
        """
        Инициализация движка оптимизации
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.study = None
        self.logger = self._setup_logger()
        
        # Настройка Optuna
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler())
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
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
        logger = logging.getLogger('OptimizationEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def create_study(self, study_name: str = None) -> optuna.Study:
        """
        Создание нового исследования Optuna
        
        Args:
            study_name: Имя исследования (по умолчанию генерируется автоматически)
            
        Returns:
            Объект исследования Optuna
        """
        if study_name is None:
            study_name = f"tacotron2_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Настройки из конфигурации
        optimization_config = self.config.get('optimization', {})
        direction = optimization_config.get('direction', 'minimize')
        
        # Создание базы данных для хранения результатов
        storage_path = Path("smart_tuner/optuna_studies.db")
        storage_url = f"sqlite:///{storage_path}"
        
        try:
            self.study = optuna.create_study(
                study_name=study_name,
                direction=direction,
                storage=storage_url,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            
            self.logger.info(f"Создано исследование: {study_name}")
            self.logger.info(f"База данных: {storage_url}")
            
        except Exception as e:
            self.logger.error(f"Ошибка создания исследования: {e}")
            raise
            
        return self.study
        
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Предложение гиперпараметров для текущего trial
        
        Args:
            trial: Объект trial от Optuna
            
        Returns:
            Словарь с предложенными гиперпараметрами
        """
        search_space = self.config.get('hyperparameter_search_space', {})
        suggested_params = {}
        
        for param_name, param_config in search_space.items():
            param_type = param_config.get('type')
            
            try:
                if param_type == 'float':
                    suggested_params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['min'],
                        param_config['max'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'int':
                    suggested_params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['min'],
                        param_config['max'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'categorical':
                    suggested_params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
                else:
                    self.logger.warning(f"Неизвестный тип параметра: {param_type} для {param_name}")
                    
            except Exception as e:
                self.logger.error(f"Ошибка при генерации параметра {param_name}: {e}")
                
        self.logger.info(f"Предложенные гиперпараметры: {suggested_params}")
        return suggested_params
        
    def calculate_objective_value(self, metrics: Dict[str, float]) -> float:
        """
        Вычисление целевой функции на основе метрик
        
        Args:
            metrics: Словарь с метриками обучения
            
        Returns:
            Значение целевой функции для оптимизации
        """
        optimization_config = self.config.get('optimization', {})
        objective_metric = optimization_config.get('objective_metric', 'val_loss')
        
        if objective_metric not in metrics:
            available_metrics = list(metrics.keys())
            self.logger.warning(
                f"Метрика {objective_metric} не найдена. "
                f"Доступные метрики: {available_metrics}"
            )
            # Используем первую доступную метрику
            if available_metrics:
                objective_metric = available_metrics[0]
                self.logger.info(f"Используется метрика: {objective_metric}")
            else:
                raise ValueError("Нет доступных метрик для оптимизации")
                
        objective_value = metrics[objective_metric]
        
        # Дополнительные штрафы за переобучение
        if 'train_loss' in metrics and 'val_loss' in metrics:
            overfitting_penalty = self._calculate_overfitting_penalty(
                metrics['train_loss'], 
                metrics['val_loss']
            )
            objective_value += overfitting_penalty
            
        self.logger.info(f"Целевая функция: {objective_value} (метрика: {objective_metric})")
        return objective_value
        
    def optimize(self, objective_function, n_trials: int = 10) -> Dict[str, Any]:
        """
        Запуск оптимизации гиперпараметров
        
        Args:
            objective_function: Функция для оптимизации
            n_trials: Количество попыток оптимизации
            
        Returns:
            Лучшие найденные гиперпараметры
        """
        if self.study is None:
            self.create_study()
            
        self.logger.info(f"Начало оптимизации с {n_trials} trials")
        
        try:
            self.study.optimize(objective_function, n_trials=n_trials)
            
            best_params = self.study.best_params
            best_value = self.study.best_value
            
            self.logger.info(f"Оптимизация завершена!")
            self.logger.info(f"Лучшие параметры: {best_params}")
            self.logger.info(f"Лучшее значение: {best_value}")
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"Ошибка во время оптимизации: {e}")
            raise
        
    def _calculate_overfitting_penalty(self, train_loss: float, val_loss: float) -> float:
        """
        Расчет штрафа за переобучение
        
        Args:
            train_loss: Лосс на обучающей выборке
            val_loss: Лосс на валидационной выборке
            
        Returns:
            Штраф за переобучение
        """
        if train_loss <= 0 or val_loss <= 0:
            return 0.0
            
        # Штраф пропорционален разности между validation и training loss
        overfitting_ratio = (val_loss - train_loss) / train_loss
        penalty_weight = self.config.get('optimization', {}).get('overfitting_penalty', 0.1)
        
        penalty = max(0, overfitting_ratio * penalty_weight)
        
        if penalty > 0:
            self.logger.info(f"Штраф за переобучение: {penalty:.4f}")
            
        return penalty
        
    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Получение лучших найденных гиперпараметров
        
        Returns:
            Словарь с лучшими гиперпараметрами или None
        """
        if self.study is None:
            self.logger.warning("Исследование не создано")
            return None
            
        if len(self.study.trials) == 0:
            self.logger.warning("Нет завершенных trials")
            return None
            
        best_trial = self.study.best_trial
        
        self.logger.info(f"Лучший результат: {best_trial.value}")
        self.logger.info(f"Лучшие параметры: {best_trial.params}")
        
        return best_trial.params
        
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Получение истории оптимизации
        
        Returns:
            Список с историей всех trials
        """
        if self.study is None:
            return []
            
        history = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'datetime_start': trial.datetime_start,
                    'datetime_complete': trial.datetime_complete,
                    'duration': trial.duration
                })
                
        return history
        
    def prune_trial(self, trial: optuna.Trial, step: int, value: float) -> bool:
        """
        Проверка, нужно ли прервать текущий trial
        
        Args:
            trial: Текущий trial
            step: Номер шага обучения
            value: Текущее значение метрики
            
        Returns:
            True, если trial нужно прервать
        """
        # Сообщаем промежуточное значение
        trial.report(value, step)
        
        # Проверяем, нужно ли прервать
        if trial.should_prune():
            self.logger.info(f"Trial {trial.number} прерван на шаге {step}")
            return True
            
        return False
        
    def save_study_summary(self, output_path: str = "smart_tuner/optimization_summary.yaml"):
        """
        Сохранение сводки по оптимизации
        
        Args:
            output_path: Путь для сохранения сводки
        """
        if self.study is None:
            self.logger.warning("Нет данных для сохранения")
            return
            
        summary = {
            'study_name': self.study.study_name,
            'direction': self.study.direction.name,
            'n_trials': len(self.study.trials),
            'best_trial': {
                'number': self.study.best_trial.number,
                'value': self.study.best_trial.value,
                'params': self.study.best_trial.params
            },
            'optimization_history': self.get_optimization_history()
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)
                
            self.logger.info(f"Сводка сохранена в {output_path}")
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения сводки: {e}")
            
    def cleanup_old_trials(self, keep_best_n: int = 10):
        """
        Очистка старых trials, оставляя только лучшие
        
        Args:
            keep_best_n: Количество лучших trials для сохранения
        """
        if self.study is None:
            return
            
        completed_trials = [
            trial for trial in self.study.trials 
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        
        if len(completed_trials) <= keep_best_n:
            return
            
        # Сортируем trials по значению (лучшие первыми)
        sorted_trials = sorted(
            completed_trials, 
            key=lambda t: t.value,
            reverse=(self.study.direction == optuna.study.StudyDirection.MAXIMIZE)
        )
        
        # Удаляем худшие trials
        trials_to_delete = sorted_trials[keep_best_n:]
        
        for trial in trials_to_delete:
            try:
                self.study.delete_trial(trial.number)
            except Exception as e:
                self.logger.warning(f"Не удалось удалить trial {trial.number}: {e}")
                
        self.logger.info(f"Удалено {len(trials_to_delete)} старых trials") 