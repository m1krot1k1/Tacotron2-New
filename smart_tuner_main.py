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

# Импорты систем логирования (упрощенные)
try:
    from training_integration import (
        setup_training_logging, 
        finish_training_logging,
        export_current_training
    )
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    # Заглушки для функций логирования
    def setup_training_logging(*args, **kwargs):
        return None, None
    def finish_training_logging(*args, **kwargs):
        pass
    def export_current_training(*args, **kwargs):
        return None

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
            
            # Обертка тренера с TTS интеграцией (принимает словарь config)
            self.trainer_wrapper = TrainerWrapper(self.config)
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
                """
                🎯 TTS-оптимизированная целевая функция для Optuna
                Учитывает специфику обучения TTS моделей
                """
                # Инициализируем время начала обучения
                from datetime import datetime
                self.training_start_time = datetime.now()
                
                try:
                    self.logger.info(f"🎯 TTS trial {trial.number} начат")
                    
                    # Получаем предложенные гиперпараметры
                    suggested_params = self.optimization_engine.suggest_hyperparameters(trial)
                    self.logger.info(f"🎛️ TTS параметры для trial {trial.number}: {suggested_params}")
                    
                    # Создаем TensorBoard writer для этого trial
                    from torch.utils.tensorboard import SummaryWriter
                    log_dir = os.path.join("output", "optuna_trials", f"trial_{trial.number}")
                    os.makedirs(log_dir, exist_ok=True)
                    writer = SummaryWriter(log_dir)
                    
                    # Запускаем обучение с предложенными параметрами
                    metrics = self.trainer_wrapper.train_with_params(
                        suggested_params, 
                        trial=trial,
                        writer=writer
                    )
                    
                    # Закрываем writer
                    writer.close()
                    
                    self.logger.info(f"📊 TTS trial {trial.number} получил метрики: {metrics}")
                    
                    # Проверяем качество результатов
                    if self._check_tts_quality_thresholds(metrics):
                        self.logger.info(f"✅ TTS trial {trial.number} прошел проверку качества")
                    else:
                        self.logger.warning(f"⚠️ TTS trial {trial.number} не прошел проверку качества")
                    
                    # Вычисляем композитную оценку
                    composite_score = self.optimization_engine.calculate_composite_tts_objective(metrics)
                    
                    self.logger.info(f"🎯 TTS trial {trial.number} завершен: {composite_score}")
                    return composite_score
                    
                except Exception as e:
                    self.logger.error(f"❌ Ошибка в TTS trial {trial.number}: {e}")
                    import traceback
                    self.logger.error(f"Полный traceback: {traceback.format_exc()}")
                    return float('inf')  # Возвращаем худший возможный результат
            
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
        Теперь более реалистичная и адаптивная + защита от преждевременного завершения
        """
        if not metrics:
            self.logger.warning("Пустые метрики для проверки качества")
            return False
        
        # 🛡️ ДОПОЛНИТЕЛЬНАЯ ЗАЩИТА: минимальное время и шаги обучения
        training_duration = 0
        if hasattr(self, 'training_start_time') and self.training_start_time is not None:
            from datetime import datetime
            training_duration = (datetime.now() - self.training_start_time).total_seconds()
            min_training_time = 600  # 10 минут минимум
            if training_duration < min_training_time:
                self.logger.info(f"⏰ Обучение слишком короткое ({training_duration/60:.1f} мин < {min_training_time/60:.1f} мин). Продолжаем...")
                return False
        
        # Проверяем минимальное количество validation шагов
        validation_step = metrics.get('validation.step', 0)
        min_validation_steps = 3  # Минимум 3 validation шага
        if validation_step < min_validation_steps:
            self.logger.info(f"📊 Недостаточно validation шагов ({validation_step} < {min_validation_steps}). Продолжаем...")
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
        
        # 🎯 БОЛЕЕ СТРОГАЯ ЛОГИКА: требуем прохождения минимум 80% проверок + все критические
        min_required_checks = max(2, int(total_checks * 0.8))  # Минимум 80% проверок
        critical_checks_passed = attention_check and gate_check and val_loss_check  # Критические проверки
        
        quality_passed = passed_checks >= min_required_checks and critical_checks_passed
        
        # Подробный лог результатов
        self.logger.info(f"🔍 Проверка качества TTS:")
        for detail in check_details:
            self.logger.info(f"  • {detail}")
        self.logger.info(f"⏰ Время обучения: {training_duration/60:.1f} мин, validation шагов: {validation_step}")
        
        if quality_passed:
            self.logger.info(f"✅ Качество достаточное: {passed_checks}/{total_checks} проверок пройдено ({passed_checks/total_checks*100:.1f}%)")
        else:
            self.logger.warning(f"⚠️ Проблемы качества: {passed_checks}/{total_checks} проверок пройдено, требуется минимум {min_required_checks} + все критические проверки")
        
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
        
        from datetime import datetime
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
                
                # Создаем TensorBoard writer для single training
                from torch.utils.tensorboard import SummaryWriter
                log_dir = os.path.join("output", "latest", f"single_training_restart_{current_restart}")
                os.makedirs(log_dir, exist_ok=True)
                writer = SummaryWriter(log_dir)
                
                # Запускаем обучение с TTS мониторингом
                results = self.trainer_wrapper.train_with_params(
                    hyperparams,
                    writer=writer,
                    tts_phase_training=self.tts_config.get('enabled', True),
                    single_training=True,
                    restart_iteration=current_restart
                )
                
                # Закрываем writer
                writer.close()
                
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
            
            from datetime import datetime
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
                    # Создаем TensorBoard writer для мини-оптимизации
                    from torch.utils.tensorboard import SummaryWriter
                    log_dir = os.path.join("output", "latest", f"mini_opt_trial_{trial.number}")
                    os.makedirs(log_dir, exist_ok=True)
                    writer = SummaryWriter(log_dir)
                    
                    metrics = self.trainer_wrapper.train_with_params(
                        suggested_params, 
                        trial=trial,
                        writer=writer,
                        mini_optimization=True
                    )
                    
                    # Закрываем writer
                    writer.close()
                    
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
        Теперь более умная и адаптивная логика принятия решений + защита от раннего завершения
        """
        if not results:
            self.logger.info("📊 Пустые результаты - рекомендуется перезапуск")
            return True
            
        # 🛡️ ДОПОЛНИТЕЛЬНАЯ ЗАЩИТА: принудительное продолжение если обучение слишком короткое
        training_duration = 0
        if hasattr(self, 'training_start_time') and self.training_start_time is not None:
            from datetime import datetime
            training_duration = (datetime.now() - self.training_start_time).total_seconds()
            min_training_time = 600  # 10 минут минимум
            if training_duration < min_training_time:
                self.logger.info(f"⏰ Обучение слишком короткое ({training_duration/60:.1f} мин < {min_training_time/60:.1f} мин). ПРИНУДИТЕЛЬНОЕ продолжение...")
                return True  # Принудительно перезапускаем (продолжаем)
        
        # Проверяем минимальное количество validation шагов
        validation_step = results.get('validation.step', 0)
        min_validation_steps = 3  # Минимум 3 validation шага
        if validation_step < min_validation_steps:
            self.logger.info(f"📊 Недостаточно validation шагов ({validation_step} < {min_validation_steps}). ПРИНУДИТЕЛЬНОЕ продолжение...")
            return True  # Принудительно перезапускаем (продолжаем)
        
        # 📊 Получаем ключевые метрики с безопасными значениями по умолчанию
        val_loss = results.get('validation_loss', float('inf'))
        attention_score = results.get('attention_alignment_score', 0.0)
        gate_accuracy = results.get('gate_accuracy', 0.0)
        mel_quality = results.get('mel_quality_score', 0.0)
        training_loss = results.get('training_loss', float('inf'))
        initial_loss = results.get('initial_training_loss', training_loss)
        
        # 🚨 КРИТИЧЕСКИЕ ПРОБЛЕМЫ (обязательный перезапуск)
        critical_problems = []
        
        # 1. Validation loss катастрофически высокий
        if val_loss > 100.0:
            critical_problems.append(f"validation_loss слишком высокий: {val_loss:.2f}")
        
        # 2. Полное отсутствие attention
        if attention_score < 0.05:
            critical_problems.append(f"attention практически отсутствует: {attention_score:.3f}")
        
        # 3. Gate accuracy критически низкий
        if gate_accuracy < 0.1:
            critical_problems.append(f"gate_accuracy критически низкий: {gate_accuracy:.3f}")
        
        # 4. Полное отсутствие прогресса обучения
        progress = (initial_loss - training_loss) / initial_loss if initial_loss > 0 else 0
        if progress < -0.1:  # Обучение ухудшается
            critical_problems.append(f"обучение деградирует: прогресс {progress:.3f}")
        
        if critical_problems:
            self.logger.warning("🚨 Обнаружены критические проблемы:")
            for problem in critical_problems:
                self.logger.warning(f"  • {problem}")
            self.logger.info("🔄 РЕКОМЕНДУЕТСЯ ПЕРЕЗАПУСК")
            return True
        
        # ⚠️ СЕРЬЕЗНЫЕ ПРОБЛЕМЫ (перезапуск при накоплении)
        serious_problems = []
        
        # 1. Validation loss высокий, но не критический
        if 25.0 < val_loss <= 100.0:
            serious_problems.append(f"validation_loss высокий: {val_loss:.2f}")
        
        # 2. Attention слабый
        if 0.05 <= attention_score < 0.3:
            serious_problems.append(f"attention слабый: {attention_score:.3f}")
        
        # 3. Gate accuracy низкий
        if 0.1 <= gate_accuracy < 0.4:
            serious_problems.append(f"gate_accuracy низкий: {gate_accuracy:.3f}")
        
        # 4. Mel quality неудовлетворительный
        if mel_quality < 0.2:
            serious_problems.append(f"mel_quality неудовлетворительный: {mel_quality:.3f}")
        
        # 5. Медленный прогресс
        if 0 <= progress < 0.02:
            serious_problems.append(f"медленный прогресс: {progress:.3f}")
        
        # Перезапуск если много серьезных проблем
        serious_threshold = 3  # Максимум 2 серьезные проблемы
        if len(serious_problems) >= serious_threshold:
            self.logger.warning(f"⚠️ Обнаружено {len(serious_problems)} серьезных проблем:")
            for problem in serious_problems:
                self.logger.warning(f"  • {problem}")
            self.logger.info("🔄 РЕКОМЕНДУЕТСЯ ПЕРЕЗАПУСК из-за накопления проблем")
            return True
        elif serious_problems:
            self.logger.info(f"⚠️ Обнаружено {len(serious_problems)} серьезных проблем (допустимо):")
            for problem in serious_problems:
                self.logger.info(f"  • {problem}")
        
        # 🎯 ПОЛОЖИТЕЛЬНЫЕ ПОКАЗАТЕЛИ (продолжаем обучение)
        good_indicators = []
        
        if val_loss <= 25.0:
            good_indicators.append(f"validation_loss приемлемый: {val_loss:.2f}")
        if attention_score >= 0.3:
            good_indicators.append(f"attention хороший: {attention_score:.3f}")
        if gate_accuracy >= 0.4:
            good_indicators.append(f"gate_accuracy приемлемый: {gate_accuracy:.3f}")
        if mel_quality >= 0.2:
            good_indicators.append(f"mel_quality приемлемый: {mel_quality:.3f}")
        if progress >= 0.02:
            good_indicators.append(f"хороший прогресс: {progress:.3f}")
        
        should_restart = len(good_indicators) < 2  # Нужно минимум 2 хороших показателя
        
        if should_restart:
            self.logger.info(f"🔄 РЕКОМЕНДУЕТСЯ ПЕРЕЗАПУСК: недостаточно хороших показателей ({len(good_indicators)}/5)")
        else:
            self.logger.info(f"✅ ПРОДОЛЖАЕМ ОБУЧЕНИЕ: достаточно хороших показателей ({len(good_indicators)}/5)")
            for indicator in good_indicators:
                self.logger.info(f"  • {indicator}")
        
        # 📊 Дополнительная информация для анализа
        if not should_restart:
            self.logger.info("📈 Текущие метрики TTS:")
            self.logger.info(f"  • val_loss: {val_loss:.3f}")
            self.logger.info(f"  • attention_score: {attention_score:.3f}")
            self.logger.info(f"  • gate_accuracy: {gate_accuracy:.3f}")
            self.logger.info(f"  • mel_quality: {mel_quality:.3f}")
            progress_pct = (initial_loss - training_loss) / initial_loss * 100 if initial_loss > 0 else float('nan')
            self.logger.info(f"  • прогресс: {progress_pct:.1f}%")
            
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
            
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = results_dir / f"tts_optimization_{timestamp}.yaml"
            
            # Подготавливаем данные для сохранения
            save_data = {
                'timestamp': timestamp,
                'best_parameters': results.get('best_params', {}),
                'best_value': results.get('best_value', float('inf')),
                'n_trials': results.get('n_trials', 0),
                'study_name': results.get('study_name', 'unknown'),
                'tts_analysis': results.get('tts_analysis', {}),
                'metadata': {
                    'config_path': self.config_path,
                    'tts_version': 'Smart Tuner V2 TTS',
                    'optimization_type': 'composite_tts_objective'
                }
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                yaml.dump(save_data, f, default_flow_style=False, allow_unicode=True)
            
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
    
    def run_automatic_mode(self, n_trials: int = 15) -> Dict[str, Any]:
        """
        🤖 ПОЛНОСТЬЮ АВТОМАТИЧЕСКИЙ РЕЖИМ
        Сначала оптимизация гиперпараметров, затем обучение с лучшими параметрами
        """
        self.logger.info("🤖 Запуск полностью автоматического режима TTS обучения")
        self.logger.info("=" * 80)
        
        from datetime import datetime
        total_start_time = datetime.now()
        final_results = {}
        
        try:
            # ЭТАП 1: Оптимизация гиперпараметров
            self.logger.info("🎯 ЭТАП 1/2: Оптимизация гиперпараметров")
            self.logger.info("=" * 50)
            
            optimization_results = self.run_optimization()
            
            if optimization_results and optimization_results.get('best_parameters'):
                best_params = optimization_results['best_parameters']
                best_score = optimization_results.get('best_value', float('inf'))
                
                self.logger.info(f"✅ Оптимизация завершена успешно!")
                self.logger.info(f"🏆 Лучшие параметры: {best_params}")
                self.logger.info(f"📊 Лучшая оценка: {best_score:.4f}")
                
                # Сохраняем результаты оптимизации
                self._save_tts_optimization_results(optimization_results)
                final_results['optimization'] = optimization_results
                
                # Пауза между этапами
                self.logger.info("⏳ Пауза между этапами (30 сек)...")
                time.sleep(30)
                
                # ЭТАП 2: Обучение с лучшими параметрами
                self.logger.info("🚀 ЭТАП 2/2: Обучение с лучшими параметрами")
                self.logger.info("=" * 50)
                
                training_results = self.run_single_training(best_params)
                
                if training_results:
                    final_results['training'] = training_results
                    self.logger.info("✅ Обучение с оптимальными параметрами завершено!")
                    
                    # Анализ финального качества
                    final_score = self.optimization_engine.calculate_composite_tts_objective(training_results)
                    improvement = ((best_score - final_score) / best_score * 100) if best_score > 0 else 0
                    
                    self.logger.info(f"📈 Улучшение качества: {improvement:.1f}%")
                    self.logger.info(f"🎯 Финальная оценка: {final_score:.4f}")
                    
                    final_results['improvement_percent'] = improvement
                    final_results['final_score'] = final_score
                else:
                    self.logger.error("❌ Обучение с лучшими параметрами не удалось")
                    final_results['training_error'] = True
            else:
                self.logger.error("❌ Оптимизация не дала результатов")
                final_results['optimization_error'] = True
                
                # Запускаем обучение с параметрами по умолчанию
                self.logger.info("🔄 Запуск обучения с параметрами по умолчанию...")
                default_params = self._get_default_hyperparams()
                training_results = self.run_single_training(default_params)
                final_results['training'] = training_results
            
            # Общая статистика
            from datetime import datetime
            total_duration = datetime.now() - total_start_time
            final_results['total_duration'] = str(total_duration)
            
            self.logger.info("=" * 80)
            self.logger.info("🎉 АВТОМАТИЧЕСКИЙ РЕЖИМ ЗАВЕРШЕН!")
            self.logger.info("=" * 80)
            self.logger.info(f"⏱️ Общее время: {total_duration}")
            
            if 'improvement_percent' in final_results:
                self.logger.info(f"📈 Улучшение: {final_results['improvement_percent']:.1f}%")
            
            # Создаем экспорт итогового результата
            try:
                export_path = export_current_training()
                self.logger.info(f"📤 Итоговый экспорт создан: {export_path}")
                final_results['export_path'] = export_path
            except Exception as e:
                self.logger.warning(f"Ошибка создания экспорта: {e}")
                
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в автоматическом режиме: {e}")
            final_results['error'] = str(e)
            return final_results

def main():
    """Главная функция запуска Smart Tuner V2 TTS"""
    parser = argparse.ArgumentParser(description='Smart Tuner V2 - TTS Автоматизированная система обучения')
    parser.add_argument('--config', '-c', 
                       default='smart_tuner/config.yaml',
                       help='Путь к файлу конфигурации')
    parser.add_argument('--mode', '-m',
                       choices=['optimize', 'train', 'monitor', 'auto'],
                       default='train',
                       help='Режим работы: optimize - оптимизация гиперпараметров, train - обучение, monitor - мониторинг, auto - автоматический режим')
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
            
        elif args.mode == 'auto':
            print("🤖 Режим: TTS Автоматический режим")
            n_trials = args.trials or 15  # Используем переданное значение или по умолчанию 15
            results = smart_tuner.run_automatic_mode(n_trials)
            print(f"🎉 Автоматический режим завершен! Результаты: {results}")
            
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

 