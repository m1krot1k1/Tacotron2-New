#!/usr/bin/env python3
"""
🎯 Интеграция Optuna HPO с train.py
Автоматический поиск гиперпараметров для Tacotron2
"""

import optuna
import yaml
import logging
import os
import sys
from typing import Dict, Any, Optional, Callable
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_tuner.optimization_engine import OptimizationEngine
from hparams import create_hparams


class OptunaTrainerIntegration:
    """
    Интеграция Optuna HPO с процессом обучения Tacotron2
    """
    
    def __init__(self, config_path: str = "smart_tuner/config.yaml"):
        self.config_path = config_path
        self.optimization_engine = OptimizationEngine(config_path)
        self.logger = logging.getLogger(__name__)
        
        # История trials
        self.trial_history = []
        self.best_trial = None
        
    def create_objective_function(self, 
                                output_directory: str,
                                log_directory: str,
                                n_gpus: int = 1,
                                rank: int = 0,
                                group_name: str = "group_name") -> Callable:
        """
        Создает objective функцию для Optuna
        
        Args:
            output_directory: Директория для сохранения чекпоинтов
            log_directory: Директория для логов
            n_gpus: Количество GPU
            rank: Ранг текущего GPU
            group_name: Имя группы для distributed обучения
            
        Returns:
            Objective функция для Optuna
        """
        
        def objective(trial: optuna.Trial) -> float:
            """
            Objective функция для оптимизации гиперпараметров
            """
            try:
                # Получаем гиперпараметры от Optuna
                suggested_params = self.optimization_engine.suggest_hyperparameters(trial)
                
                # Создаем hparams с предложенными параметрами
                hparams = create_hparams()
                
                # Применяем предложенные параметры
                for param_name, param_value in suggested_params.items():
                    if hasattr(hparams, param_name):
                        setattr(hparams, param_name, param_value)
                
                # Логируем trial
                trial_info = {
                    'trial_number': trial.number,
                    'params': suggested_params,
                    'timestamp': optuna.trial.TrialState.RUNNING
                }
                self.trial_history.append(trial_info)
                
                self.logger.info(f"🎯 Trial {trial.number}: Запуск с параметрами {suggested_params}")
                
                # Импортируем train функцию
                from train import train
                
                # Создаем уникальные директории для этого trial
                trial_output_dir = os.path.join(output_directory, f"trial_{trial.number}")
                trial_log_dir = os.path.join(log_directory, f"trial_{trial.number}")
                
                os.makedirs(trial_output_dir, exist_ok=True)
                os.makedirs(trial_log_dir, exist_ok=True)
                
                # Запускаем обучение с текущими параметрами
                # Используем callback для передачи метрик в Optuna
                metrics_callback = self._create_metrics_callback(trial)
                
                # Запускаем обучение
                final_metrics = train(
                    output_directory=trial_output_dir,
                    log_directory=trial_log_dir,
                    checkpoint_path=None,
                    warm_start=False,
                    ignore_mmi_layers=False,
                    ignore_gst_layers=False,
                    ignore_tsgst_layers=False,
                    n_gpus=n_gpus,
                    rank=rank,
                    group_name=group_name,
                    hparams=hparams,
                    smart_tuner_trial=trial,
                    smart_tuner_logger=metrics_callback,
                    tensorboard_writer=None,
                    telegram_monitor=None
                )
                
                # Вычисляем финальное значение objective
                if final_metrics:
                    objective_value = self.optimization_engine.calculate_composite_tts_objective(final_metrics)
                    
                    # Сохраняем лучший trial
                    if self.best_trial is None or objective_value < self.best_trial['value']:
                        self.best_trial = {
                            'trial_number': trial.number,
                            'params': suggested_params,
                            'metrics': final_metrics,
                            'value': objective_value
                        }
                    
                    self.logger.info(f"✅ Trial {trial.number} завершен. Objective: {objective_value:.4f}")
                    return objective_value
                else:
                    # Если обучение не вернуло метрики, возвращаем плохое значение
                    self.logger.warning(f"⚠️ Trial {trial.number}: Нет метрик, возвращаем плохое значение")
                    return float('inf')
                    
            except Exception as e:
                self.logger.error(f"❌ Trial {trial.number} завершился с ошибкой: {e}")
                return float('inf')
        
        return objective
    
    def _create_metrics_callback(self, trial: optuna.Trial) -> Callable:
        """
        Создает callback для передачи промежуточных метрик в Optuna
        """
        def callback(step: int, metrics: Dict[str, float]):
            try:
                # Вычисляем промежуточное значение objective
                if metrics:
                    intermediate_value = self.optimization_engine.calculate_composite_tts_objective(metrics)
                    
                    # Отправляем в Optuna
                    trial.report(intermediate_value, step)
                    
                    # Проверяем, нужно ли остановить trial
                    if trial.should_prune():
                        self.logger.info(f"✂️ Trial {trial.number} остановлен на шаге {step}")
                        raise optuna.TrialPruned()
                        
            except Exception as e:
                self.logger.warning(f"⚠️ Ошибка в metrics callback: {e}")
        
        return callback
    
    def run_optimization(self, 
                        output_directory: str,
                        log_directory: str,
                        n_trials: int = 10,
                        n_gpus: int = 1,
                        timeout: int = None) -> Dict[str, Any]:
        """
        Запускает оптимизацию гиперпараметров
        
        Args:
            output_directory: Директория для сохранения результатов
            log_directory: Директория для логов
            n_trials: Количество trials
            n_gpus: Количество GPU
            timeout: Таймаут в секундах
            
        Returns:
            Результаты оптимизации
        """
        try:
            self.logger.info(f"🚀 Запуск оптимизации гиперпараметров: {n_trials} trials")
            
            # Создаем study
            study_name = f"tacotron2_optimization_{optuna.trial.TrialState.RUNNING}"
            study = self.optimization_engine.create_study_with_retry(
                study_name=study_name,
                direction='minimize'  # Минимизируем loss
            )
            
            # Создаем objective функцию
            objective = self.create_objective_function(
                output_directory=output_directory,
                log_directory=log_directory,
                n_gpus=n_gpus
            )
            
            # Запускаем оптимизацию
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=[
                    self.optimization_engine._tts_progress_callback,
                    self.optimization_engine._tts_early_stop_callback
                ]
            )
            
            # Анализируем результаты
            results = self._analyze_optimization_results(study)
            
            self.logger.info(f"🎉 Оптимизация завершена! Лучший trial: {results['best_trial_number']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка оптимизации: {e}")
            raise
    
    def _analyze_optimization_results(self, study: optuna.Study) -> Dict[str, Any]:
        """
        Анализирует результаты оптимизации
        """
        best_trial = study.best_trial
        
        results = {
            'best_trial_number': best_trial.number,
            'best_params': best_trial.params,
            'best_value': best_trial.value,
            'n_trials': len(study.trials),
            'n_completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'optimization_history': self.trial_history,
            'study_statistics': self.optimization_engine.get_study_statistics()
        }
        
        return results
    
    def save_optimization_results(self, results: Dict[str, Any], output_path: str):
        """
        Сохраняет результаты оптимизации
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"💾 Результаты оптимизации сохранены: {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения результатов: {e}")


def main():
    """
    Главная функция для запуска оптимизации гиперпараметров
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Optuna HPO для Tacotron2')
    parser.add_argument('--output-dir', type=str, default='output/optimization',
                       help='Директория для сохранения результатов')
    parser.add_argument('--log-dir', type=str, default='logs/optimization',
                       help='Директория для логов')
    parser.add_argument('--n-trials', type=int, default=10,
                       help='Количество trials')
    parser.add_argument('--n-gpus', type=int, default=1,
                       help='Количество GPU')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Таймаут в секундах')
    parser.add_argument('--config', type=str, default='smart_tuner/config.yaml',
                       help='Путь к конфигурации')
    
    args = parser.parse_args()
    
    # Создаем интеграцию
    integration = OptunaTrainerIntegration(args.config)
    
    # Запускаем оптимизацию
    results = integration.run_optimization(
        output_directory=args.output_dir,
        log_directory=args.log_dir,
        n_trials=args.n_trials,
        n_gpus=args.n_gpus,
        timeout=args.timeout
    )
    
    # Сохраняем результаты
    results_path = os.path.join(args.output_dir, 'optimization_results.yaml')
    integration.save_optimization_results(results, results_path)
    
    print(f"\n🎉 Оптимизация завершена!")
    print(f"📊 Лучший trial: {results['best_trial_number']}")
    print(f"🎯 Лучшее значение: {results['best_value']:.4f}")
    print(f"📁 Результаты сохранены: {results_path}")


if __name__ == "__main__":
    main() 