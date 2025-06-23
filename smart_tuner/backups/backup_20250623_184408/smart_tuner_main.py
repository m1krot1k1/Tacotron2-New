#!/usr/bin/env python3
"""
Smart Tuner V2 - Main Entry Point
Автоматизированная система обучения Tacotron2 с TTS-специфичными возможностями

Основные функции:
- TTS-оптимизированная оптимизация гиперпараметров 
- Интеллектуальное управление обучением с композитными метриками
- Фазовое обучение с адаптацией к стадиям TTS
- Проактивное устранение проблем обучения
- Автоматическое логирование и экспорт результатов
"""

import os
import sys
import yaml
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорты компонентов Smart Tuner
from smart_tuner.trainer_wrapper import TrainerWrapper
from smart_tuner.optimization_engine import OptimizationEngine
from smart_tuner.early_stop_controller import EarlyStopController
from smart_tuner.alert_manager import AlertManager
from smart_tuner.model_registry import ModelRegistry

# Импорты систем логирования
from training_integration import (
    setup_training_logging, 
    finish_training_logging,
    export_current_training
)

class SmartTunerMain:
    """
    Главный контроллер TTS-оптимизированной системы умного обучения
    """
    
    def __init__(self, config_path: str = "smart_tuner/config.yaml"):
        """
        Инициализация Smart Tuner с TTS конфигурацией
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logger()
        
        # Инициализация компонентов
        self.trainer_wrapper = None
        self.optimization_engine = None
        self.early_stop_controller = None
        self.alert_manager = None
        self.model_registry = None
        
        # TTS-специфичные настройки
        self.tts_config = self.config.get('tts_phase_training', {})
        self.current_phase = "pre_alignment"
        self.training_start_time = None
        
        # Система логирования
        self.training_logger = None
        self.export_system = None
        
        self.logger.info("🚀 Smart Tuner V2 TTS инициализирован")
        
    def _load_config(self) -> Dict[str, Any]:
        """Загрузка TTS конфигурации"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"❌ Файл конфигурации {self.config_path} не найден")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"❌ Ошибка парсинга конфигурации: {e}")
            sys.exit(1)
            
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера для TTS"""
        logger = logging.getLogger('SmartTunerMain')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Консольный handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                '%(asctime)s - 🧠 Smart Tuner - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # Файловый handler для TTS логов
            log_dir = Path("smart_tuner/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / f"smart_tuner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        return logger
        
    def initialize_components(self):
        """Инициализация всех TTS компонентов системы"""
        try:
            self.logger.info("🔧 Инициализация TTS компонентов...")
            
            # Оптимизатор с TTS поддержкой
            self.optimization_engine = OptimizationEngine(self.config_path)
            self.logger.info("✅ TTS OptimizationEngine инициализирован")
            
            # Контроллер раннего останова с TTS диагностикой
            self.early_stop_controller = EarlyStopController(self.config_path)
            self.logger.info("✅ TTS EarlyStopController инициализирован")
            
            # Менеджер алертов
            self.alert_manager = AlertManager(self.config_path)
            self.logger.info("✅ AlertManager инициализирован")
            
            # Реестр моделей
            self.model_registry = ModelRegistry(self.config_path)
            self.logger.info("✅ ModelRegistry инициализирован")
            
            # Обертка тренера с TTS интеграцией (принимает только config)
            self.trainer_wrapper = TrainerWrapper(config=self.config)
            self.logger.info("✅ TTS TrainerWrapper инициализирован")
            
            # Система логирования TTS будет инициализирована при старте обучения
            self.training_logger = None
            self.export_system = None
            self.logger.info("✅ TTS система логирования подготовлена")
            
            self.logger.info("🎉 Все TTS компоненты успешно инициализированы!")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации компонентов: {e}")
            raise
    
    def run_optimization(self) -> Dict[str, Any]:
        """
        Запуск TTS-оптимизированного процесса оптимизации гиперпараметров
        
        Returns:
            Результаты оптимизации
        """
        self.logger.info("🎯 Запуск TTS оптимизации гиперпараметров...")
        
        try:
            # Создаем исследование с TTS настройками
            study = self.optimization_engine.create_study(
                study_name=f"tacotron2_tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Получаем количество trials из конфигурации
            n_trials = self.config.get('optimization', {}).get('n_trials', 30)
            
            def tts_objective_function(trial):
                """TTS-специфичная целевая функция с композитными метриками"""
                self.logger.info(f"🔬 Начало TTS trial {trial.number}")
                
                # Получаем TTS-оптимизированные гиперпараметры
                suggested_params = self.optimization_engine.suggest_hyperparameters(trial)
                
                try:
                    # Запускаем обучение с TTS логированием
                    metrics = self.trainer_wrapper.train_with_params(
                        suggested_params, 
                        trial=trial,
                        tts_phase_training=self.tts_config.get('enabled', True)
                    )
                    
                    if not metrics:
                        self.logger.warning(f"TTS trial {trial.number}: получены пустые метрики")
                        return float('inf')
                    
                    # Вычисляем композитную TTS целевую функцию
                    objective_value = self.optimization_engine.calculate_objective_value(metrics)
                    
                    self.logger.info(f"🎯 TTS trial {trial.number} завершен: {objective_value:.4f}")
                    
                    # Проверяем TTS качественные пороги
                    if self._check_tts_quality_thresholds(metrics):
                        self.logger.info(f"✅ TTS trial {trial.number} прошел проверки качества")
                    else:
                        self.logger.warning(f"⚠️ TTS trial {trial.number} не прошел проверки качества")
                        objective_value += 0.5  # Штраф за низкое качество TTS
                    
                    return objective_value
                    
                except Exception as e:
                    self.logger.error(f"❌ Ошибка в TTS trial {trial.number}: {e}")
                    return float('inf')
            
            # Запускаем TTS оптимизацию
            results = self.optimization_engine.optimize(
                tts_objective_function, 
                n_trials=n_trials
            )
            
            self.logger.info("🎉 TTS оптимизация завершена успешно!")
            
            # Анализируем и сохраняем результаты
            self._save_tts_optimization_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка при TTS оптимизации: {e}")
            raise
    
    def _check_tts_quality_thresholds(self, metrics: Dict[str, float]) -> bool:
        """
        🎯 УЛУЧШЕННАЯ проверка TTS метрик на соответствие качественным требованиям
        Теперь более реалистичная и адаптивная
        """
        if not metrics:
            self.logger.warning("Пустые метрики для проверки качества")
            return False
            
        quality_checks = self.config.get('training_safety', {}).get('tts_quality_checks', {})
        
        # 📊 Собираем все проверки с более реалистичными порогами
        checks = []
        check_details = []
        
        # 1. Проверка attention alignment (более мягкая)
        min_attention = quality_checks.get('min_attention_alignment', 0.4)  # Снижено с 0.6
        current_attention = metrics.get('attention_alignment_score', 0.0)
        attention_check = current_attention >= min_attention
        checks.append(attention_check)
        check_details.append(f"attention_alignment: {current_attention:.3f} >= {min_attention} ({'✅' if attention_check else '❌'})")
        
        # 2. Проверка gate accuracy (более достижимая)
        min_gate = quality_checks.get('min_gate_accuracy', 0.5)  # Снижено с 0.7
        current_gate = metrics.get('gate_accuracy', 0.0)
        gate_check = current_gate >= min_gate
        checks.append(gate_check)
        check_details.append(f"gate_accuracy: {current_gate:.3f} >= {min_gate} ({'✅' if gate_check else '❌'})")
        
        # 3. Проверка validation loss (более разумная)
        max_val_loss = quality_checks.get('max_validation_loss', 25.0)  # Снижено с 50.0
        current_val_loss = metrics.get('val_loss', float('inf'))
        val_loss_check = current_val_loss <= max_val_loss
        checks.append(val_loss_check)
        check_details.append(f"val_loss: {current_val_loss:.3f} <= {max_val_loss} ({'✅' if val_loss_check else '❌'})")
        
        # 4. Проверка mel quality (более достижимая)
        min_mel_quality = quality_checks.get('mel_quality_threshold', 0.3)  # Снижено с 0.5
        current_mel_quality = metrics.get('mel_quality_score', 0.0)
        mel_check = current_mel_quality >= min_mel_quality
        checks.append(mel_check)
        check_details.append(f"mel_quality: {current_mel_quality:.3f} >= {min_mel_quality} ({'✅' if mel_check else '❌'})")
        
        # 5. НОВАЯ проверка прогресса обучения
        min_progress = quality_checks.get('min_training_progress', 0.05)
        training_loss = metrics.get('training_loss', float('inf'))
        initial_loss = metrics.get('initial_training_loss', training_loss)
        progress = (initial_loss - training_loss) / initial_loss if initial_loss > 0 else 0
        progress_check = progress >= min_progress
        checks.append(progress_check)
        check_details.append(f"training_progress: {progress:.3f} >= {min_progress} ({'✅' if progress_check else '❌'})")
        
        # 📊 Статистика проверок
        passed_checks = sum(checks)
        total_checks = len(checks)
        
        # 🎯 НОВАЯ ЛОГИКА: требуем прохождения минимум 60% проверок (вместо 100%)
        min_required_checks = max(1, int(total_checks * 0.6))  # Минимум 60% проверок
        quality_passed = passed_checks >= min_required_checks
        
        # Подробный лог результатов
        self.logger.info(f"🔍 Проверка качества TTS:")
        for detail in check_details:
            self.logger.info(f"  • {detail}")
        
        if quality_passed:
            self.logger.info(f"✅ Качество достаточное: {passed_checks}/{total_checks} проверок пройдено ({passed_checks/total_checks*100:.1f}%)")
        else:
            self.logger.warning(f"⚠️ Проблемы качества: {passed_checks}/{total_checks} проверок пройдено, требуется минимум {min_required_checks}")
        
        return quality_passed
    
    def run_single_training(self, hyperparams: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Запуск адаптивного TTS обучения с автоматической оптимизацией
        
        Args:
            hyperparams: Гиперпараметры для обучения (используются лучшие если не указано)
            
        Returns:
            Результаты обучения
        """
        self.logger.info("🚂 Запуск адаптивного TTS обучения с автоматической оптимизацией...")
        
        self.training_start_time = datetime.now()
        max_restarts = 3
        current_restart = 0
        best_results = None
        best_score = float('inf')
        
        try:
            while current_restart <= max_restarts:
                self.logger.info(f"🔄 Итерация обучения: {current_restart + 1}/{max_restarts + 1}")
                
                # Если это не первый запуск, проводим мини-оптимизацию
                if current_restart > 0:
                    self.logger.info("🔍 Запуск мини-оптимизации для улучшения параметров...")
                    mini_optimization_results = self._run_mini_optimization(n_trials=8)
                    
                    if mini_optimization_results and mini_optimization_results.get('best_params'):
                        hyperparams = mini_optimization_results['best_params']
                        self.logger.info(f"✅ Найдены улучшенные параметры: {hyperparams}")
                
                # Используем лучшие параметры если они не переданы
                if hyperparams is None:
                    hyperparams = self._get_best_hyperparams()
                    
                if not hyperparams:
                    self.logger.warning("Используются параметры по умолчанию из конфигурации")
                    hyperparams = self._get_default_hyperparams()
                
                self.logger.info(f"🎛️ TTS гиперпараметры (итерация {current_restart + 1}): {hyperparams}")
                
                # Запускаем обучение с TTS мониторингом
                results = self.trainer_wrapper.train_with_params(
                    hyperparams,
                    tts_phase_training=self.tts_config.get('enabled', True),
                    single_training=True,
                    restart_iteration=current_restart
                )
                
                if results:
                    # Вычисляем композитную оценку качества
                    current_score = self.optimization_engine.calculate_composite_tts_objective(results)
                    self.logger.info(f"📊 Оценка качества итерации {current_restart + 1}: {current_score:.4f}")
                    
                    # Проверяем, улучшились ли результаты
                    if current_score < best_score:
                        best_score = current_score
                        best_results = results.copy()
                        self.logger.info(f"✅ Новый лучший результат: {best_score:.4f}")
                        
                        # Если результат достаточно хорош, прекращаем
                        if self._check_tts_quality_thresholds(results):
                            self.logger.info("🎉 Достигнуто высокое качество TTS! Завершаем обучение.")
                            break
                    else:
                        self.logger.info(f"⚠️ Результат не улучшился. Лучший: {best_score:.4f}")
                
                # Проверяем, нужен ли перезапуск
                if current_restart < max_restarts:
                    if self._should_restart_training(results):
                        self.logger.info("🔄 Обнаружены проблемы качества. Планируется перезапуск с оптимизацией...")
                        current_restart += 1
                        continue
                    else:
                        self.logger.info("✅ Качество удовлетворительное. Завершаем обучение.")
                        break
                else:
                    self.logger.info("📊 Достигнуто максимальное количество перезапусков.")
                    break
            
            # Используем лучшие результаты
            final_results = best_results if best_results else results
            
            # Финализируем логирование TTS
            if self.training_logger and self.export_system:
                finish_training_logging(
                    self.training_logger, 
                    self.export_system,
                    final_metrics=final_results
                )
            
            training_duration = datetime.now() - self.training_start_time
            self.logger.info(f"🎉 Адаптивное TTS обучение завершено за {training_duration}")
            self.logger.info(f"🏆 Лучшая оценка качества: {best_score:.4f}")
            self.logger.info(f"🔄 Количество перезапусков: {current_restart}")
            
            # Создаем экспорт для AI анализа
            try:
                export_path = export_current_training()
                self.logger.info(f"📤 TTS экспорт создан: {export_path}")
            except Exception as e:
                self.logger.warning(f"Ошибка создания экспорта: {e}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка при адаптивном TTS обучении: {e}")
            
            # Попытка финализации логирования при ошибке
            if self.training_logger:
                try:
                    finish_training_logging(
                        self.training_logger, 
                        self.export_system,
                        final_metrics={'error': str(e)},
                        training_completed=False
                    )
                except:
                    pass
            
            raise
    
    def _get_best_hyperparams(self) -> Optional[Dict[str, Any]]:
        """Получение лучших TTS гиперпараметров из оптимизации"""
        if not self.optimization_engine or not self.optimization_engine.study:
            return None
            
        try:
            if self.optimization_engine.study.best_trial:
                return self.optimization_engine.study.best_params
        except:
            pass
            
        return None
    
    def _run_mini_optimization(self, n_trials: int = 8) -> Dict[str, Any]:
        """
        Запуск мини-оптимизации для улучшения параметров
        
        Args:
            n_trials: Количество trials для мини-оптимизации
            
        Returns:
            Результаты мини-оптимизации
        """
        self.logger.info(f"🔍 Мини-оптимизация: {n_trials} trials")
        
        try:
            study = self.optimization_engine.create_study(
                study_name=f"mini_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            def mini_objective_function(trial):
                """Облегченная целевая функция для мини-оптимизации"""
                suggested_params = self.optimization_engine.suggest_hyperparameters(trial)
                
                # Запускаем короткое обучение (50 эпох)
                suggested_params['epochs'] = 50
                
                try:
                    metrics = self.trainer_wrapper.train_with_params(
                        suggested_params, 
                        trial=trial,
                        mini_optimization=True
                    )
                    
                    if metrics:
                        return self.optimization_engine.calculate_composite_tts_objective(metrics)
                    else:
                        return float('inf')
                        
                except Exception as e:
                    self.logger.warning(f"Ошибка в мини-trial {trial.number}: {e}")
                    return float('inf')
            
            # Запускаем мини-оптимизацию
            study.optimize(mini_objective_function, n_trials=n_trials)
            
            if study.best_trial:
                return {
                    'best_params': study.best_params,
                    'best_score': study.best_value,
                    'n_trials': len(study.trials)
                }
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Ошибка мини-оптимизации: {e}")
            return None
    
    def _should_restart_training(self, results: Dict[str, Any]) -> bool:
        """
        🚀 УЛУЧШЕННАЯ функция определения необходимости перезапуска обучения
        Теперь более умная и адаптивная логика принятия решений
        """
        if not results:
            self.logger.info("📊 Пустые результаты - рекомендуется перезапуск")
            return True
            
        # 📊 Получаем основные метрики
        val_loss = results.get('val_loss', float('inf'))
        attention_score = results.get('attention_alignment_score', 0.0)
        gate_accuracy = results.get('gate_accuracy', 0.0)
        mel_quality = results.get('mel_quality_score', 0.0)
        training_loss = results.get('training_loss', float('inf'))
        
        # 🎯 НОВЫЕ критерии для перезапуска (более мягкие и умные)
        restart_reasons = []
        
        # 1. Критический validation loss
        if val_loss > 30.0:  # Увеличено с 10.0
            restart_reasons.append(f"высокий val_loss ({val_loss:.3f})")
            
        # 2. Очень плохой attention (только если катастрофически плохой)
        if attention_score < 0.3:  # Снижено с 0.6
            restart_reasons.append(f"критический attention ({attention_score:.3f})")
            
        # 3. Очень плохая gate accuracy (только если катастрофически плохая)
        if gate_accuracy < 0.4:  # Снижено с 0.7
            restart_reasons.append(f"критическая gate_accuracy ({gate_accuracy:.3f})")
            
        # 4. Очень плохое mel quality (только если катастрофически плохое)
        if mel_quality < 0.2:  # Снижено с 0.4
            restart_reasons.append(f"критическое mel_quality ({mel_quality:.3f})")
            
        # 5. НОВЫЙ критерий: отсутствие прогресса в обучении
        initial_loss = results.get('initial_training_loss', training_loss)
        if initial_loss > 0:
            progress = (initial_loss - training_loss) / initial_loss
            if progress < 0.02:  # Меньше 2% улучшения
                restart_reasons.append(f"отсутствие прогресса ({progress*100:.1f}%)")
        
        # 6. НОВЫЙ критерий: взрывной градиент
        grad_norm = results.get('grad_norm', 0.0)
        if grad_norm > 200.0:
            restart_reasons.append(f"взрывной градиент ({grad_norm:.1f})")
            
        # 🎯 УМНАЯ ЛОГИКА: перезапуск только при множественных критических проблемах
        critical_issues = len(restart_reasons)
        
        # Определяем нужен ли перезапуск
        should_restart = False
        
        if critical_issues >= 3:
            # Если 3+ критических проблем - точно перезапускаем
            should_restart = True
            self.logger.warning(f"🚨 Критические проблемы ({critical_issues}): {', '.join(restart_reasons)}")
        elif critical_issues == 2:
            # Если 2 проблемы - перезапускаем только если одна из них критическая
            if val_loss > 50.0 or attention_score < 0.2 or gate_accuracy < 0.3:
                should_restart = True
                self.logger.warning(f"⚠️ Серьезные проблемы ({critical_issues}): {', '.join(restart_reasons)}")
            else:
                self.logger.info(f"⚠️ Умеренные проблемы ({critical_issues}): {', '.join(restart_reasons)} - продолжаем")
        elif critical_issues == 1:
            # Если 1 проблема - перезапускаем только если она катастрофическая
            if val_loss > 100.0 or grad_norm > 500.0:
                should_restart = True
                self.logger.warning(f"💥 Катастрофическая проблема: {', '.join(restart_reasons)}")
            else:
                self.logger.info(f"ℹ️ Минорная проблема: {', '.join(restart_reasons)} - продолжаем")
        else:
            # Нет критических проблем
            self.logger.info("✅ Качество обучения приемлемое - продолжаем")
            
        # 📊 Дополнительная информация для анализа
        if not should_restart:
            self.logger.info("📈 Текущие метрики TTS:")
            self.logger.info(f"  • val_loss: {val_loss:.3f}")
            self.logger.info(f"  • attention_score: {attention_score:.3f}")
            self.logger.info(f"  • gate_accuracy: {gate_accuracy:.3f}")
            self.logger.info(f"  • mel_quality: {mel_quality:.3f}")
            if initial_loss > 0:
                self.logger.info(f"  • прогресс: {progress*100:.1f}%")
        
        return should_restart
    
    def _get_default_hyperparams(self) -> Dict[str, Any]:
        """Получение TTS гиперпараметров по умолчанию из конфигурации"""
        search_space = self.config.get('hyperparameter_search_space', {})
        default_params = {}
        
        for param_name, param_config in search_space.items():
            default_value = param_config.get('default')
            if default_value is not None:
                default_params[param_name] = default_value
            elif param_config.get('type') == 'categorical':
                choices = param_config.get('choices', [])
                if choices:
                    default_params[param_name] = choices[0]
                    
        return default_params
    
    def _save_tts_optimization_results(self, results: Dict[str, Any]):
        """Сохранение результатов TTS оптимизации"""
        try:
            results_dir = Path("smart_tuner/optimization_results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = results_dir / f"tts_optimization_{timestamp}.yaml"
            
            # Добавляем метаданные
            results['metadata'] = {
                'timestamp': timestamp,
                'config_path': self.config_path,
                'tts_version': 'Smart Tuner V2 TTS',
                'optimization_type': 'composite_tts_objective'
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
                
            self.logger.info(f"💾 TTS результаты сохранены: {results_file}")
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения результатов: {e}")
    
    def run_monitoring_mode(self):
        """Режим мониторинга TTS обучения"""
        self.logger.info("👁️ Запуск режима TTS мониторинга...")
        
        try:
            # Создаем файлы мониторинга если не существуют
            monitoring_dir = Path("smart_tuner/monitoring") 
            monitoring_dir.mkdir(parents=True, exist_ok=True)
            
            status_file = monitoring_dir / "tts_status.yaml"
            
            while True:
                try:
                    # Проверяем статус TTS обучения
                    tts_status = self._get_tts_training_status()
                    
                    # Сохраняем статус
                    with open(status_file, 'w', encoding='utf-8') as f:
                        yaml.dump(tts_status, f, default_flow_style=False)
                    
                    # Проверяем алерты
                    if self.alert_manager:
                        self.alert_manager.check_training_status(tts_status)
                    
                    time.sleep(30)  # Проверяем каждые 30 секунд
                    
                except KeyboardInterrupt:
                    self.logger.info("🛑 Остановка мониторинга по запросу пользователя")
                    break
                except Exception as e:
                    self.logger.error(f"Ошибка мониторинга: {e}")
                    time.sleep(60)  # Ждем минуту при ошибке
                    
        except Exception as e:
            self.logger.error(f"❌ Ошибка режима мониторинга: {e}")
    
    def _get_tts_training_status(self) -> Dict[str, Any]:
        """Получение статуса TTS обучения"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'mode': 'monitoring',
            'tts_version': 'Smart Tuner V2 TTS'
        }
        
        try:
            # Статус от early stop controller
            if self.early_stop_controller:
                tts_summary = self.early_stop_controller.get_tts_training_summary()
                status.update(tts_summary)
            
            # Статус оптимизации
            if self.optimization_engine:
                opt_stats = self.optimization_engine.get_study_statistics()
                status['optimization'] = opt_stats
                
        except Exception as e:
            status['error'] = str(e)
            
        return status
    
    def cleanup(self):
        """Очистка ресурсов TTS системы"""
        self.logger.info("🧹 Очистка TTS ресурсов...")
        
        try:
            if self.optimization_engine:
                self.optimization_engine.cleanup_study()
                
            if self.early_stop_controller:
                self.early_stop_controller.reset()
                
            if self.alert_manager and hasattr(self.alert_manager, 'cleanup'):
                self.alert_manager.cleanup()
                
            self.logger.info("✅ TTS ресурсы очищены")
            
        except Exception as e:
            self.logger.error(f"Ошибка очистки: {e}")

def main():
    """Главная функция запуска Smart Tuner V2 TTS"""
    parser = argparse.ArgumentParser(description='Smart Tuner V2 - TTS Автоматизированная система обучения')
    parser.add_argument('--config', '-c', 
                       default='smart_tuner/config.yaml',
                       help='Путь к файлу конфигурации')
    parser.add_argument('--mode', '-m',
                       choices=['optimize', 'train', 'monitor'],
                       default='train',
                       help='Режим работы: optimize - оптимизация гиперпараметров, train - обучение, monitor - мониторинг')
    parser.add_argument('--trials', '-t',
                       type=int,
                       help='Количество trials для оптимизации (перекрывает настройку в конфиге)')
    parser.add_argument('--hyperparams', '-p',
                       help='JSON строка с гиперпараметрами для режима train')
    
    args = parser.parse_args()
    
    # Проверяем существование конфигурации
    if not os.path.exists(args.config):
        print(f"❌ Файл конфигурации {args.config} не найден")
        return 1
    
    # Инициализируем Smart Tuner
    smart_tuner = None
    
    try:
        print("🚀 Запуск Smart Tuner V2 TTS...")
        smart_tuner = SmartTunerMain(args.config)
        smart_tuner.initialize_components()
        
        # Обновляем конфигурацию из аргументов
        if args.trials:
            smart_tuner.config.setdefault('optimization', {})['n_trials'] = args.trials
        
        # Выполняем действие в зависимости от режима
        if args.mode == 'optimize':
            print("🎯 Режим: TTS Оптимизация гиперпараметров")
            results = smart_tuner.run_optimization()
            print(f"🎉 Оптимизация завершена! Лучшие параметры: {results.get('best_parameters', {})}")
            
        elif args.mode == 'train':
            print("🚂 Режим: TTS Обучение")
            hyperparams = None
            if args.hyperparams:
                import json
                try:
                    hyperparams = json.loads(args.hyperparams)
                except json.JSONDecodeError:
                    print("❌ Неверный формат JSON для гиперпараметров")
                    return 1
                    
            results = smart_tuner.run_single_training(hyperparams)
            print(f"🎉 TTS Обучение завершено! Финальные метрики: {results}")
            
        elif args.mode == 'monitor':
            print("👁️ Режим: TTS Мониторинг")
            smart_tuner.run_monitoring_mode()
            
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Остановлено пользователем")
        return 0
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        return 1
        
    finally:
        if smart_tuner:
            smart_tuner.cleanup()

if __name__ == "__main__":
    sys.exit(main())

 