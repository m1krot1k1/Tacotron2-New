"""
Optimization Engine для Smart Tuner V2
Автоматический подбор гиперпараметров с использованием Optuna
Теперь с TTS-специфичными возможностями и композитной целевой функцией
"""

import optuna
import yaml
import logging
import numpy as np
import math
import sqlite3
import time
import random
import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler
from sqlalchemy import create_engine, pool
import torch

class OptimizationEngine:
    """
    TTS-оптимизированный движок оптимизации гиперпараметров на основе Optuna
    Автоматически подбирает лучшие гиперпараметры на основе композитных TTS метрик
    """
    
    def __init__(self, config_path: str = "smart_tuner/config.yaml"):
        """
        Инициализация движка оптимизации с улучшенной обработкой SQLite
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Валидация конфигурации
        self._validate_config()
        
        self.study = None
        self.logger = self._setup_logger()
        
        # TTS-специфичные настройки
        self.tts_config = self.config.get('optimization', {}).get('tts_specific', {})
        self.composite_config = self.config.get('optimization', {}).get('composite_objective', {})
        
        # Настройка Optuna с TTS-адаптированными параметрами
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler())
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.logger.info("TTS-оптимизированный OptimizationEngine инициализирован")
        
        self.setup_sqlite_wal()
        
    def _load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации из YAML файла"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Файл конфигурации {self.config_path} не найден")
            raise
        except yaml.YAMLError as e:
            print(f"Ошибка парсинга YAML: {e}")
            raise
    
    def _validate_config(self):
        """
        Валидация конфигурации с проверкой критических параметров TTS.
        Реализует рекомендации из технического задания.
        """
        errors = []
        warnings = []
        
        # 1. Проверка основных секций
        required_sections = ['optimization', 'hyperparameter_search_space', 'training_safety']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Отсутствует обязательная секция: {section}")
        
        # 2. Проверка гиперпараметров
        if 'hyperparameter_search_space' in self.config:
            search_space = self.config['hyperparameter_search_space']
            
            # Проверка learning_rate
            if 'learning_rate' in search_space:
                lr_config = search_space['learning_rate']
                if lr_config.get('max', 0) > 0.001:
                    warnings.append("learning_rate.max > 0.001 может привести к нестабильности TTS")
                if lr_config.get('min', 0) < 1e-6:
                    warnings.append("learning_rate.min < 1e-6 может быть слишком мал для TTS")
            
            # Проверка batch_size
            if 'batch_size' in search_space:
                batch_config = search_space['batch_size']
                if batch_config.get('min', 0) < 16:
                    warnings.append("batch_size.min < 16 может привести к проблемам с attention alignment")
                if batch_config.get('max', 0) > 64:
                    warnings.append("batch_size.max > 64 может вызвать проблемы с памятью")
            
            # Проверка dropout параметров
            dropout_params = ['p_attention_dropout', 'dropout_rate', 'postnet_dropout_rate']
            for param in dropout_params:
                if param in search_space:
                    dropout_config = search_space[param]
                    if dropout_config.get('max', 0) > 0.4:
                        warnings.append(f"{param}.max > 0.4 может ухудшить качество attention")
        
        # 3. Проверка настроек безопасности обучения
        if 'training_safety' in self.config:
            safety_config = self.config['training_safety']
            
            if 'tts_quality_checks' in safety_config:
                quality_checks = safety_config['tts_quality_checks']
                
                # Проверка порогов качества
                if quality_checks.get('min_attention_alignment', 0) > 0.8:
                    warnings.append("min_attention_alignment > 0.8 может быть слишком строгим для начального обучения")
                
                if quality_checks.get('max_validation_loss', 100) < 5.0:
                    warnings.append("max_validation_loss < 5.0 может быть слишком строгим для TTS")
        
        # 4. Проверка настроек оптимизации
        if 'optimization' in self.config:
            opt_config = self.config['optimization']
            
            if opt_config.get('n_trials', 0) < 5:
                warnings.append("n_trials < 5 может быть недостаточно для качественной оптимизации")
            
            if opt_config.get('n_trials', 0) > 100:
                warnings.append("n_trials > 100 может занять очень много времени")
        
        # Вывод результатов валидации
        if errors:
            error_msg = "Критические ошибки конфигурации:\n" + "\n".join(f"- {e}" for e in errors)
            print(error_msg)
            raise ValueError(error_msg)
        
        if warnings:
            warning_msg = "Предупреждения конфигурации:\n" + "\n".join(f"- {w}" for w in warnings)
            print(warning_msg)
        
        print("✅ Конфигурация валидирована успешно")
            
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
        
    def setup_sqlite_wal(self):
        """
        Настройка SQLite для работы в WAL режиме с retry механизмом
        """
        storage_path = "smart_tuner/optuna_studies.db"
        
        # Создаем директорию если не существует
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        
        conn = None
        try:
            conn = sqlite3.connect(storage_path, timeout=30)
            
            # Включаем WAL режим для лучшей конкурентности
            conn.execute('PRAGMA journal_mode=WAL;')
            conn.execute('PRAGMA synchronous=NORMAL;')
            conn.execute('PRAGMA cache_size=10000;')
            conn.execute('PRAGMA temp_store=MEMORY;')
            conn.execute('PRAGMA mmap_size=268435456;')  # 256MB
            conn.execute('PRAGMA busy_timeout=30000;')   # 30 секунд
            
            conn.commit()
            print("✅ SQLite настроен в WAL режиме")
            
        except sqlite3.Error as e:
            print(f"❌ Ошибка настройки SQLite: {e}")
        finally:
            if conn:
                conn.close()
                
    def create_study_with_retry(self, study_name: str = None, max_retries: int = 5) -> optuna.Study:
        """
        Создание Optuna study с retry механизмом
        """
        storage_url = f"sqlite:///smart_tuner/optuna_studies.db"
        
        for attempt in range(max_retries):
            try:
                # Создаем engine с connection pooling
                engine = create_engine(
                    storage_url,
                    poolclass=pool.NullPool,  # Отключаем pooling
                    connect_args={
                        "timeout": 30,
                        "check_same_thread": False,
                        "isolation_level": None  # Autocommit режим
                    }
                )
                
                # Настраиваем sampler с отключенными предупреждениями
                sampler = CmaEsSampler(
                    warn_independent_sampling=False,
                    n_startup_trials=5
                )
                
                self.study = optuna.create_study(
                    study_name=study_name,
                    storage=storage_url,
                    direction="minimize",
                    load_if_exists=True,
                    sampler=sampler
                )
                
                print(f"✅ Optuna study создан: {study_name}")
                return self.study
                
            except Exception as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"⚠️ База заблокирована, ожидание {wait_time:.1f}с...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Ошибка создания study: {e}")
                    raise
                    
        raise Exception("Не удалось создать study после всех попыток")
        
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
                # Используем значение по умолчанию с правильным типом
                default_value = param_config.get('default')
                if default_value is not None:
                    # Преобразуем в правильный тип
                    if param_type == 'float':
                        suggested_params[param_name] = float(default_value)
                    elif param_type == 'int':
                        suggested_params[param_name] = int(default_value)
                    else:
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
            batch_size = int(params['batch_size'])
            learning_rate = float(params['learning_rate'])
            
            # Для TTS меньшие batch sizes требуют меньшие learning rates
            if batch_size <= 16 and learning_rate > 0.003:
                validated_params['learning_rate'] = min(learning_rate, 0.003)
                self.logger.debug(f"Скорректирован learning_rate для batch_size {batch_size}: {validated_params['learning_rate']}")
        
        # Корректировка guided attention в зависимости от эпох
        if 'guided_attention_enabled' in params and 'epochs' in params:
            epochs = int(params['epochs'])
            # Для коротких обучений принудительно включаем guided attention
            if epochs < 150:
                validated_params['guided_attention_enabled'] = True
                if 'guide_loss_weight' in validated_params:
                    validated_params['guide_loss_weight'] = max(float(validated_params.get('guide_loss_weight', 1.0)), 1.5)
        
        # Корректировка dropout параметров для стабильности
        dropout_params = ['p_attention_dropout', 'dropout_rate', 'postnet_dropout_rate']
        for dropout_param in dropout_params:
            if dropout_param in validated_params:
                # Ограничиваем dropout для стабильности TTS
                validated_params[dropout_param] = min(float(validated_params[dropout_param]), 0.6)
        
        return validated_params
    
    def calculate_composite_tts_objective(self, metrics: Dict[str, float]) -> float:
        """
        🎯 УЛУЧШЕННАЯ композитная TTS целевая функция с более умной оценкой качества
        Теперь более сбалансированная и адаптивная к реальным TTS условиям
        """
        if not metrics:
            self.logger.warning("Получены пустые метрики для композитной целевой функции")
            return float('inf')
        
        # 📊 Получаем улучшенные веса из конфигурации
        weights = self.composite_config.get('weights', {
            'validation_loss': 0.5,      # Увеличено - основной показатель
            'attention_alignment_score': 0.2,  # Уменьшено - менее критично
            'gate_accuracy': 0.2,        # Умеренно важно
            'mel_quality_score': 0.1     # Дополнительный показатель
        })
        
        normalize_scores = self.composite_config.get('normalize_scores', True)
        quality_bonus_threshold = self.composite_config.get('quality_bonus_threshold', 0.8)
        progress_weight = self.composite_config.get('progress_weight', 0.1)
        
        total_score = 0.0
        total_weight = 0.0
        components = {}  # Для детального логирования
        
        # 1. 📉 Validation Loss (основной компонент - минимизируем)
        if 'validation_loss' in weights and 'val_loss' in metrics:
            val_loss = metrics['val_loss']
            if val_loss > 0 and val_loss < 1000:  # Санитарная проверка
                weight = weights['validation_loss']
                # 🎯 Улучшенная нормализация для TTS (типичный диапазон 1.0-10.0)
                if normalize_scores:
                    # Используем логарифмическую шкалу для лучшей чувствительности
                    normalized_loss = min(math.log(val_loss + 1) / math.log(11), 3.0)
                else:
                    normalized_loss = val_loss
                    
                total_score += normalized_loss * weight
                total_weight += weight
                components['val_loss'] = normalized_loss * weight
                
        # 2. 🎯 Attention Alignment Score (менее критично - максимизируем)
        if 'attention_alignment_score' in weights and 'attention_alignment_score' in metrics:
            att_score = metrics['attention_alignment_score']
            if 0.0 <= att_score <= 1.0:
                weight = weights['attention_alignment_score']
                # 🎯 Более мягкая функция штрафа
                if normalize_scores:
                    # Квадратичная функция для более мягкого штрафа
                    alignment_penalty = (1.0 - att_score) ** 1.5
                else:
                    alignment_penalty = 1.0 - att_score
                    
                total_score += alignment_penalty * weight
                total_weight += weight
                components['attention'] = alignment_penalty * weight
        
        # 3. 🚪 Gate Accuracy (умеренно важно - максимизируем)
        if 'gate_accuracy' in weights and 'gate_accuracy' in metrics:
            gate_acc = metrics['gate_accuracy']
            if 0.0 <= gate_acc <= 1.0:
                weight = weights['gate_accuracy']
                # 🎯 Более реалистичная функция штрафа
                if normalize_scores:
                    # Используем степенную функцию для плавного штрафа
                    gate_penalty = (1.0 - gate_acc) ** 1.2
                else:
                    gate_penalty = 1.0 - gate_acc
                    
                total_score += gate_penalty * weight
                total_weight += weight
                components['gate'] = gate_penalty * weight
        
        # 4. 🎵 Mel Quality Score (дополнительный - максимизируем)
        if 'mel_quality_score' in weights and 'mel_quality_score' in metrics:
            mel_quality = metrics['mel_quality_score']
            if 0.0 <= mel_quality <= 1.0:
                weight = weights['mel_quality_score']
                mel_penalty = (1.0 - mel_quality) ** 0.8 if normalize_scores else (1.0 - mel_quality)
                total_score += mel_penalty * weight
                total_weight += weight
                components['mel'] = mel_penalty * weight
        
        # 5. 📈 НОВЫЙ компонент: прогресс обучения (бонус за улучшение)
        if progress_weight > 0 and 'training_loss' in metrics and 'initial_training_loss' in metrics:
            training_loss = metrics['training_loss']
            initial_loss = metrics['initial_training_loss']
            
            if initial_loss > 0 and training_loss < initial_loss:
                progress = (initial_loss - training_loss) / initial_loss
                # Бонус за прогресс (уменьшает общий score)
                progress_bonus = max(0, progress - 0.05) * progress_weight  # Бонус только если прогресс > 5%
                total_score -= progress_bonus  # Вычитаем бонус
                components['progress_bonus'] = -progress_bonus
        
        # 📊 Нормализация финального скора
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            self.logger.warning("Не найдены валидные метрики для композитной целевой функции")
            return float('inf')
            
        # 6. ⚖️ УЛУЧШЕННЫЙ штраф за переобучение (более мягкий для TTS)
        overfitting_penalty = self.config.get('optimization', {}).get('overfitting_penalty', 0.03)  # Уменьшено
        if 'val_loss' in metrics and 'train_loss' in metrics and overfitting_penalty > 0:
            val_loss = metrics['val_loss']
            train_loss = metrics['train_loss']
            
            if val_loss > train_loss:
                overfitting_gap = val_loss - train_loss
                # 🎯 Прогрессивный штраф (больше штраф при больших различиях)
                if overfitting_gap > 2.0:  # Критическое переобучение
                    penalty = overfitting_gap * overfitting_penalty * 2.0
                elif overfitting_gap > 1.0:  # Умеренное переобучение
                    penalty = overfitting_gap * overfitting_penalty
                else:  # Незначительное переобучение (нормально для TTS)
                    penalty = overfitting_gap * overfitting_penalty * 0.5
                    
                penalty = min(penalty, 0.3)  # Ограничиваем максимальный штраф
                final_score += penalty
                components['overfitting'] = penalty
        
        # 7. 🏆 НОВЫЙ бонус за высокое качество
        if quality_bonus_threshold > 0:
            # Если несколько метрик превышают порог качества, даем бонус
            quality_metrics = []
            if 'attention_alignment_score' in metrics:
                quality_metrics.append(metrics['attention_alignment_score'])
            if 'gate_accuracy' in metrics:
                quality_metrics.append(metrics['gate_accuracy'])
            if 'mel_quality_score' in metrics:
                quality_metrics.append(metrics['mel_quality_score'])
                
            if quality_metrics:
                avg_quality = sum(quality_metrics) / len(quality_metrics)
                if avg_quality >= quality_bonus_threshold:
                    quality_bonus = (avg_quality - quality_bonus_threshold) * 0.2
                    final_score -= quality_bonus  # Вычитаем бонус
                    components['quality_bonus'] = -quality_bonus
        
        # 📝 Детальное логирование для анализа
        if self.logger.isEnabledFor(logging.DEBUG):
            component_str = ", ".join([f"{k}: {v:.4f}" for k, v in components.items()])
            self.logger.debug(f"TTS композитные компоненты: {component_str}")
            
        self.logger.debug(f"TTS финальная целевая функция: {final_score:.4f} (вес: {total_weight:.2f})")
        
        # 🛡️ Санитарная проверка результата
        if not math.isfinite(final_score) or final_score < 0:
            self.logger.warning(f"Некорректный финальный score: {final_score}, возвращаем высокое значение")
            return 10.0
            
        return max(final_score, 0.01)  # Минимальное положительное значение
    
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
            self.create_study_with_retry()
        
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