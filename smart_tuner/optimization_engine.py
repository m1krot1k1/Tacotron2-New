"""
Optimization Engine для Smart Tuner V2
Автоматический подбор гиперпараметров с использованием Optuna
Теперь с TTS-специфичными возможностями и композитной целевой функцией
"""

import optuna
import yaml
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

class OptimizationEngine:
    """
    TTS-оптимизированный движок оптимизации гиперпараметров на основе Optuna
    Автоматически подбирает лучшие гиперпараметры на основе композитных TTS метрик
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
        
        # TTS-специфичные настройки
        self.tts_config = self.config.get('optimization', {}).get('tts_specific', {})
        self.composite_config = self.config.get('optimization', {}).get('composite_objective', {})
        
        # Настройка Optuna с TTS-адаптированными параметрами
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler())
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.logger.info("TTS-оптимизированный OptimizationEngine инициализирован")
        
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
        Создание нового исследования Optuna с TTS-специфичными настройками
        
        Args:
            study_name: Имя исследования (по умолчанию генерируется автоматически)
            
        Returns:
            Объект исследования Optuna
        """
        if study_name is None:
            study_name = f"tacotron2_tts_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Настройки из конфигурации
        optimization_config = self.config.get('optimization', {})
        direction = optimization_config.get('direction', 'minimize')
        
        # Создание базы данных для хранения результатов
        storage_path = Path("smart_tuner/optuna_studies.db")
        storage_url = f"sqlite:///{storage_path}"
        
        # TTS-специфичный sampler для лучшей сходимости
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=20,  # Увеличено для TTS
            n_ei_candidates=48,   # Увеличено для лучшего поиска
            multivariate=True,    # Поддержка связанных параметров TTS
            seed=42
        )
        
        # TTS-адаптированный pruner
        early_pruning_disabled_epochs = self.tts_config.get('early_pruning_disabled_epochs', 100)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10,      # Увеличено для TTS
            n_warmup_steps=early_pruning_disabled_epochs,  # Не обрезать первые 100 эпох
            interval_steps=20         # Проверять каждые 20 эпох
        )
        
        try:
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction=direction,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True
            )
            
            self.logger.info(f"TTS исследование создано: {study_name}")
            self.logger.info(f"Storage: {storage_url}")
            self.logger.info(f"Direction: {direction}")
            self.logger.info(f"Early pruning отключен для первых {early_pruning_disabled_epochs} эпох")
            
            return self.study
            
        except Exception as e:
            self.logger.error(f"Ошибка создания TTS исследования: {e}")
            raise
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Предлагает TTS-специфичные гиперпараметры для trial
        
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
                    min_val = param_config.get('min', 0.0)
                    max_val = param_config.get('max', 1.0)
                    log_scale = param_config.get('log', False)
                    
                    if log_scale:
                        suggested_params[param_name] = trial.suggest_float(
                            param_name, min_val, max_val, log=True
                        )
                    else:
                        suggested_params[param_name] = trial.suggest_float(
                            param_name, min_val, max_val
                        )
                        
                elif param_type == 'int':
                    min_val = param_config.get('min', 1)
                    max_val = param_config.get('max', 100)
                    
                    suggested_params[param_name] = trial.suggest_int(
                        param_name, min_val, max_val
                    )
                    
                elif param_type == 'categorical':
                    choices = param_config.get('choices', [])
                    if choices:
                        suggested_params[param_name] = trial.suggest_categorical(
                            param_name, choices
                        )
                    else:
                        self.logger.warning(f"Пустой список choices для {param_name}")
                        
                else:
                    self.logger.warning(f"Неизвестный тип параметра: {param_type} для {param_name}")
                    
            except Exception as e:
                self.logger.error(f"Ошибка при предложении параметра {param_name}: {e}")
                # Используем значение по умолчанию
                default_value = param_config.get('default')
                if default_value is not None:
                    suggested_params[param_name] = default_value
        
        # Логируем предложенные параметры для TTS
        self.logger.debug(f"TTS параметры для trial {trial.number}: {suggested_params}")
        
        # Проверяем совместимость TTS параметров
        suggested_params = self._validate_tts_parameter_compatibility(suggested_params)
        
        return suggested_params
    
    def _validate_tts_parameter_compatibility(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Проверяет и корректирует совместимость TTS параметров
        """
        validated_params = params.copy()
        
        # Корректировка learning rate в зависимости от batch size
        if 'learning_rate' in params and 'batch_size' in params:
            batch_size = params['batch_size']
            learning_rate = params['learning_rate']
            
            # Для TTS меньшие batch sizes требуют меньшие learning rates
            if batch_size <= 16 and learning_rate > 0.003:
                validated_params['learning_rate'] = min(learning_rate, 0.003)
                self.logger.debug(f"Скорректирован learning_rate для batch_size {batch_size}: {validated_params['learning_rate']}")
        
        # Корректировка guided attention в зависимости от эпох
        if 'guided_attention_enabled' in params and 'epochs' in params:
            epochs = params['epochs']
            # Для коротких обучений принудительно включаем guided attention
            if epochs < 150:
                validated_params['guided_attention_enabled'] = True
                if 'guide_loss_weight' in validated_params:
                    validated_params['guide_loss_weight'] = max(validated_params.get('guide_loss_weight', 1.0), 1.5)
        
        # Корректировка dropout параметров для стабильности
        dropout_params = ['attention_dropout', 'prenet_dropout', 'postnet_dropout']
        for dropout_param in dropout_params:
            if dropout_param in validated_params:
                # Ограничиваем dropout для стабильности TTS
                validated_params[dropout_param] = min(validated_params[dropout_param], 0.6)
        
        return validated_params
    
    def calculate_composite_tts_objective(self, metrics: Dict[str, float]) -> float:
        """
        Вычисляет композитную TTS целевую функцию на основе множественных метрик
        
        Args:
            metrics: Словарь с метриками обучения
            
        Returns:
            Значение композитной целевой функции для минимизации
        """
        if not metrics:
            self.logger.warning("Получены пустые метрики для композитной целевой функции")
            return float('inf')
        
        weights = self.composite_config.get('weights', {
            'validation_loss': 0.4,
            'attention_alignment_score': 0.3,
            'gate_accuracy': 0.2,
            'mel_quality_score': 0.1
        })
        
        normalize_scores = self.composite_config.get('normalize_scores', True)
        
        total_score = 0.0
        total_weight = 0.0
        
        # Validation Loss (минимизируем)
        if 'validation_loss' in weights and 'val_loss' in metrics:
            val_loss = metrics['val_loss']
            if val_loss > 0:  # Проверка на валидность
                weight = weights['validation_loss']
                # Нормализуем loss для TTS (типичный диапазон 0.5-5.0)
                normalized_loss = min(val_loss / 5.0, 2.0) if normalize_scores else val_loss
                total_score += normalized_loss * weight
                total_weight += weight
        
        # Attention Alignment Score (максимизируем, конвертируем в минимизацию)
        if 'attention_alignment_score' in weights and 'attention_alignment_score' in metrics:
            att_score = metrics['attention_alignment_score']
            if 0.0 <= att_score <= 1.0:  # Проверка диапазона
                weight = weights['attention_alignment_score']
                # Конвертируем в минимизацию: чем выше alignment, тем меньше значение
                alignment_penalty = (1.0 - att_score) * 2.0 if normalize_scores else (1.0 - att_score)
                total_score += alignment_penalty * weight
                total_weight += weight
        
        # Gate Accuracy (максимизируем, конвертируем в минимизацию)
        if 'gate_accuracy' in weights and 'gate_accuracy' in metrics:
            gate_acc = metrics['gate_accuracy']
            if 0.0 <= gate_acc <= 1.0:  # Проверка диапазона
                weight = weights['gate_accuracy']
                # Конвертируем в минимизацию: чем выше accuracy, тем меньше значение
                gate_penalty = (1.0 - gate_acc) * 1.5 if normalize_scores else (1.0 - gate_acc)
                total_score += gate_penalty * weight
                total_weight += weight
        
        # Mel Quality Score (максимизируем, конвертируем в минимизацию)
        if 'mel_quality_score' in weights and 'mel_quality_score' in metrics:
            mel_quality = metrics['mel_quality_score']
            if 0.0 <= mel_quality <= 1.0:  # Проверка диапазона
                weight = weights['mel_quality_score']
                mel_penalty = (1.0 - mel_quality) if normalize_scores else (1.0 - mel_quality)
                total_score += mel_penalty * weight
                total_weight += weight
        
        # Нормализация финального скора
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            self.logger.warning("Не найдены валидные метрики для композитной целевой функции")
            final_score = float('inf')
        
        # Добавляем штраф за переобучение если настроен
        overfitting_penalty = self.config.get('optimization', {}).get('overfitting_penalty', 0.05)
        if 'val_loss' in metrics and 'train_loss' in metrics and overfitting_penalty > 0:
            overfitting_gap = metrics['val_loss'] - metrics['train_loss']
            if overfitting_gap > 0:
                # Мягкий штраф для TTS (переобучение частично нормально)
                penalty = min(overfitting_gap * overfitting_penalty, 0.5)
                final_score += penalty
        
        self.logger.debug(f"TTS композитная целевая функция: {final_score:.4f} (weights: {total_weight:.2f})")
        return final_score
    
    def calculate_objective_value(self, metrics: Dict[str, float]) -> float:
        """
        Вычисляет значение целевой функции на основе настроек конфигурации
        
        Args:
            metrics: Словарь с метриками обучения
            
        Returns:
            Значение целевой функции
        """
        objective_metric = self.config.get('optimization', {}).get('objective_metric', 'val_loss')
        
        if objective_metric == 'composite_tts_score':
            return self.calculate_composite_tts_objective(metrics)
        elif objective_metric in metrics:
            return metrics[objective_metric]
        else:
            self.logger.warning(f"Метрика {objective_metric} не найдена, используется val_loss")
            return metrics.get('val_loss', float('inf'))
    
    def optimize(self, objective_function, n_trials: int = None) -> Dict[str, Any]:
        """
        Запускает TTS-оптимизированный процесс оптимизации
        
        Args:
            objective_function: Функция для оптимизации
            n_trials: Количество trials (из конфигурации если не указано)
            
        Returns:
            Результаты оптимизации
        """
        if self.study is None:
            self.create_study()
        
        if n_trials is None:
            n_trials = self.config.get('optimization', {}).get('n_trials', 20)
        
        self.logger.info(f"🚀 Запуск TTS оптимизации с {n_trials} trials")
        
        # Добавляем TTS-специфичные callbacks
        callbacks = [
            self._tts_progress_callback,
            self._tts_early_stop_callback
        ]
        
        try:
            self.study.optimize(
                objective_function,
                n_trials=n_trials,
                callbacks=callbacks,
                catch=()  # Не останавливаемся на исключениях в TTS
            )
            
            best_params = self.study.best_params
            best_value = self.study.best_value
            
            self.logger.info(f"🎉 TTS оптимизация завершена!")
            self.logger.info(f"🏆 Лучшие параметры: {best_params}")
            self.logger.info(f"📊 Лучшее значение: {best_value:.4f}")
            
            # Создаем подробный отчет о TTS оптимизации
            results = {
                'best_parameters': best_params,
                'best_value': best_value,
                'n_trials': len(self.study.trials),
                'study_name': self.study.study_name,
                'tts_analysis': self._analyze_tts_optimization_results()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Ошибка при TTS оптимизации: {e}")
            raise
    
    def _tts_progress_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback для отслеживания прогресса TTS оптимизации"""
        n_trials = len(study.trials)
        
        if n_trials % 5 == 0:  # Каждые 5 trials
            best_value = study.best_value if study.best_trial else None
            self.logger.info(f"🔬 TTS прогресс: {n_trials} trials завершено, лучшее значение: {best_value:.4f}")
            
            # Анализируем тренды TTS метрик
            if n_trials >= 10:
                self._analyze_tts_trends(study)
    
    def _tts_early_stop_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback для раннего останова TTS оптимизации при необходимости"""
        n_trials = len(study.trials)
        min_trials = self.tts_config.get('min_training_steps', 20000) // 1000  # Примерная конверсия
        
        # Не останавливаем раньше минимального количества trials для TTS
        if n_trials < max(min_trials, 15):
            return
            
        # Проверяем стагнацию для TTS
        recent_trials = study.trials[-10:]
        values = [t.value for t in recent_trials if t.value is not None]
        
        if len(values) >= 10:
            improvement = min(values[:5]) - min(values[-5:])
            min_improvement = 0.01  # Порог для TTS
            
            if improvement < min_improvement:
                self.logger.info(f"🛑 TTS оптимизация: обнаружена стагнация после {n_trials} trials")
                study.stop()
    
    def _analyze_tts_trends(self, study: optuna.Study):
        """Анализирует тренды в TTS оптимизации"""
        if len(study.trials) < 10:
            return
            
        recent_trials = study.trials[-10:]
        values = [t.value for t in recent_trials if t.value is not None]
        
        if len(values) >= 5:
            trend = np.polyfit(range(len(values)), values, 1)[0]
            
            if trend > 0:
                self.logger.warning(f"📈 TTS тренд: ухудшение целевой функции (slope: {trend:.4f})")
            else:
                self.logger.info(f"📉 TTS тренд: улучшение целевой функции (slope: {trend:.4f})")
    
    def _analyze_tts_optimization_results(self) -> Dict[str, Any]:
        """
        Анализирует результаты TTS оптимизации и возвращает инсайты
        """
        if not self.study or len(self.study.trials) == 0:
            return {"status": "no_trials"}
        
        trials = self.study.trials
        completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if len(completed_trials) == 0:
            return {"status": "no_completed_trials"}
        
        # Базовая статистика
        values = [t.value for t in completed_trials]
        
        analysis = {
            "total_trials": len(trials),
            "completed_trials": len(completed_trials),
            "best_value": min(values),
            "worst_value": max(values),
            "mean_value": np.mean(values),
            "std_value": np.std(values),
            "improvement_rate": self._calculate_improvement_rate(values)
        }
        
        # Анализ важности параметров для TTS
        if len(completed_trials) >= 10:
            try:
                importance = optuna.importance.get_param_importances(self.study)
                analysis["parameter_importance"] = importance
                
                # TTS-специфичные инсайты
                tts_insights = []
                if 'learning_rate' in importance and importance['learning_rate'] > 0.3:
                    tts_insights.append("Learning rate критично важен для данного TTS датасета")
                if 'guided_attention_enabled' in importance and importance['guided_attention_enabled'] > 0.2:
                    tts_insights.append("Guided attention значительно влияет на качество")
                if 'batch_size' in importance and importance['batch_size'] > 0.25:
                    tts_insights.append("Размер батча критичен для стабильности TTS")
                
                analysis["tts_insights"] = tts_insights
                
            except Exception as e:
                self.logger.warning(f"Не удалось вычислить важность параметров: {e}")
        
        # Рекомендации для TTS
        recommendations = []
        best_params = self.study.best_params
        
        if best_params.get('learning_rate', 0) < 0.0005:
            recommendations.append("Рассмотрите увеличение learning_rate для ускорения сходимости")
        if best_params.get('batch_size', 32) < 16:
            recommendations.append("Маленький batch_size может замедлить обучение TTS")
        if not best_params.get('guided_attention_enabled', True):
            recommendations.append("Включение guided attention может улучшить качество alignment")
        
        analysis["tts_recommendations"] = recommendations
        
        return analysis
    
    def _calculate_improvement_rate(self, values: List[float]) -> float:
        """Вычисляет скорость улучшения целевой функции"""
        if len(values) < 5:
            return 0.0
            
        # Сравниваем первую и последнюю треть
        first_third = values[:len(values)//3]
        last_third = values[-len(values)//3:]
        
        if len(first_third) == 0 or len(last_third) == 0:
            return 0.0
            
        first_avg = np.mean(first_third)
        last_avg = np.mean(last_third)
        
        if first_avg == 0:
            return 0.0
            
        improvement_rate = (first_avg - last_avg) / first_avg
        return max(0.0, improvement_rate)  # Только положительные улучшения
    
    def get_study_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику по текущему исследованию
        """
        if not self.study:
            return {"status": "no_study"}
        
        trials = self.study.trials
        
        stats = {
            "study_name": self.study.study_name,
            "total_trials": len(trials),
            "completed_trials": len([t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "failed_trials": len([t for t in trials if t.state == optuna.trial.TrialState.FAIL]),
            "pruned_trials": len([t for t in trials if t.state == optuna.trial.TrialState.PRUNED]),
            "running_trials": len([t for t in trials if t.state == optuna.trial.TrialState.RUNNING])
        }
        
        if self.study.best_trial:
            stats["best_trial"] = {
                "number": self.study.best_trial.number,
                "value": self.study.best_value,
                "params": self.study.best_params
            }
        
        return stats
    
    def report_intermediate_value(self, trial: optuna.Trial, step: int, value: float, metrics: Dict[str, float] = None):
        """
        Отчитывается о промежуточном значении с TTS-специфичными проверками
        
        Args:
            trial: Trial объект
            step: Шаг обучения
            value: Промежуточное значение метрики
            metrics: Дополнительные TTS метрики для логирования
        """
        if trial is None:
            return
            
        try:
            # Всегда отчитываемся о значении для Optuna Dashboard
            trial.report(value, step)
            
            # Сохраняем дополнительные TTS метрики как user attributes
            if metrics:
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                        # Сохраняем промежуточные значения метрик
                        attr_name = f"{metric_name}_step_{step}"
                        trial.set_user_attr(attr_name, float(metric_value))
                        
                        # Сохраняем последнее значение для быстрого доступа
                        trial.set_user_attr(f"last_{metric_name}", float(metric_value))
                        
                        # Для ключевых TTS метрик создаем отдельные series
                        if metric_name in ['attention_alignment_score', 'gate_accuracy', 'mel_quality_score']:
                            trial.set_user_attr(f"{metric_name}_history_{step}", float(metric_value))
            
            # Логируем прогресс каждые 50 шагов для отладки
            if step % 50 == 0:
                self.logger.debug(f"📊 Trial {trial.number}, step {step}: value={value:.4f}")
                if metrics:
                    key_metrics = {k: v for k, v in metrics.items() 
                                 if k in ['val_loss', 'attention_alignment_score', 'gate_accuracy']}
                    if key_metrics:
                        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in key_metrics.items()])
                        self.logger.debug(f"📈 Key TTS metrics: {metrics_str}")
            
            # TTS-специфичные проверки для pruning
            early_pruning_disabled_epochs = self.tts_config.get('early_pruning_disabled_epochs', 100)
            min_steps_for_pruning = early_pruning_disabled_epochs * 10  # Примерно 10 шагов на эпоху
            
            # Проверяем на pruning только после минимального количества шагов
            if step >= min_steps_for_pruning:
                if trial.should_prune():
                    # Дополнительная проверка: не обрезаем если показывают признаки улучшения attention
                    if self._check_tts_improvement_potential(trial, step, value):
                        self.logger.info(f"🎯 TTS trial {trial.number}: отложен pruning из-за потенциала улучшения")
                        return
                    
                    self.logger.info(f"✂️ TTS trial {trial.number} обрезан на шаге {step} (value: {value:.4f})")
                    raise optuna.TrialPruned()
                
        except optuna.TrialPruned:
            raise
        except Exception as e:
            self.logger.warning(f"Ошибка при отчете промежуточного значения: {e}")
    
    def _check_tts_improvement_potential(self, trial: optuna.Trial, step: int, value: float) -> bool:
        """
        Проверяет потенциал улучшения для TTS (например, начало формирования attention)
        """
        # Получаем параметры trial
        params = trial.params
        
        # Если включен guided attention и мы в фазе формирования attention
        if params.get('guided_attention_enabled', False):
            guide_weight = params.get('guide_loss_weight', 1.0)
            if guide_weight > 1.0 and step < 30000:  # Активный guided attention на ранней стадии
                return True
        
        # Проверяем тренд последних значений
        intermediate_values = list(trial.intermediate_values.values())
        if len(intermediate_values) >= 5:
            recent_values = intermediate_values[-5:]
            trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            
            # Если есть тренд к улучшению, даем еще шанс
            if trend < -0.01:  # Значительное улучшение
                return True
        
        return False
    
    def cleanup_study(self):
        """Очистка ресурсов исследования"""
        if self.study:
            self.logger.info(f"Очистка TTS исследования: {self.study.study_name}")
            self.study = None 