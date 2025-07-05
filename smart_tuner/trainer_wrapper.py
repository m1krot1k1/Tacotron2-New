import logging
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import optuna

# Добавляем корневую директорию в путь для импортов
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# === MLflow: безопасная инициализация ===
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow не найден, метрики не будут логироваться")

# === TensorBoard: безопасная инициализация ===
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logging.warning("TensorBoard не найден, метрики не будут логироваться")

# Импорт утилит для метрик качества
try:
    from training_utils.dynamic_padding import DynamicPaddingCollator
    from training_utils.bucket_batching import BucketBatchSampler
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    logging.warning("Утилиты не найдены")

# Импорт дополнительных компонентов
try:
    from debug_reporter import initialize_debug_reporter, get_debug_reporter
    DEBUG_REPORTER_AVAILABLE = True
except ImportError:
    DEBUG_REPORTER_AVAILABLE = False
    logging.warning("Debug Reporter не найден")

class EnhancedTrainerWrapper:
    """
    🚀 УЛУЧШЕННАЯ обертка для управления процессом обучения Tacotron2.
    Интегрирует все возможности из enhanced_training_main.py:
    - Расширенное логирование (10+ метрик)
    - Фазовое обучение
    - Современные техники (bucket batching, dynamic padding)
    - Адаптивные гиперпараметры
    - Smart Tuner интеграция
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.last_checkpoint_path: Optional[str] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Состояние обучения
        self.current_epoch = 0
        self.global_step = 0
        self.best_validation_loss = float('inf')
        self.training_metrics_history = []
        
        # Фазы обучения (как в enhanced_training_main.py)
        self.training_phases = {
            'pre_alignment': {'max_epoch': 500, 'focus': 'attention_learning'},
            'alignment_learning': {'max_epoch': 2000, 'focus': 'attention_stabilization'},
            'quality_optimization': {'max_epoch': 3000, 'focus': 'quality_improvement'},
            'fine_tuning': {'max_epoch': 3500, 'focus': 'final_polishing'}
        }
        
        # Инициализация логирования
        self.tensorboard_writer = None
        self.tensorboard_logdir = 'logs'
        
        # Инициализация компонентов
        self.debug_reporter = None
        self.telegram_monitor = None
        
        logging.info("🚀 Enhanced TrainerWrapper инициализирован с расширенными возможностями")

    def train_with_params(
        self, 
        params: Dict[str, Any], 
        trial: Optional[optuna.Trial] = None, 
        writer: Optional[Any] = None,  # SummaryWriter
        **kwargs  # Для обратной совместимости
    ) -> Optional[Dict[str, float]]:
        """
        🚀 Запускает один сеанс обучения с заданными параметрами.
        Теперь с расширенными возможностями из enhanced_training_main.py
        """
        logging.info(f"🧪 Запуск ENHANCED обучения с параметрами: {params}")
        
        # Создаем hparams на лету для этого конкретного запуска
        from hparams import create_hparams
        hparams = create_hparams()
        for key, value in params.items():
            if hasattr(hparams, key):
                setattr(hparams, key, value)
        
        # Инициализируем расширенное логирование
        self._setup_enhanced_logging(trial)
        
        # Запускаем основной метод обучения
        return self._train_core_enhanced(hparams, trial, self.last_checkpoint_path, writer)

    def _setup_enhanced_logging(self, trial: Optional[optuna.Trial] = None):
        """Настраивает расширенное логирование как в enhanced_training_main.py"""
        
        # === Очистка старых логов TensorBoard ===
        if TENSORBOARD_AVAILABLE and os.path.exists(self.tensorboard_logdir):
            try:
                for file in os.listdir(self.tensorboard_logdir):
                    if file.startswith('events.out.tfevents'):
                        os.remove(os.path.join(self.tensorboard_logdir, file))
                        logging.info(f"🗑️ Удален старый лог TensorBoard: {file}")
            except Exception as e:
                logging.warning(f"⚠️ Ошибка очистки старых логов: {e}")
        
        # === Инициализация TensorBoard ===
        if TENSORBOARD_AVAILABLE:
            try:
                self.tensorboard_writer = SummaryWriter(self.tensorboard_logdir)
                logging.info(f"✅ TensorBoard writer инициализирован: {self.tensorboard_logdir}")
            except Exception as e:
                logging.error(f"⚠️ Ошибка инициализации TensorBoard: {e}")
                self.tensorboard_writer = None
        
        # === Инициализация MLflow ===
        if MLFLOW_AVAILABLE:
            try:
                # Завершаем предыдущий run если он активен
                try:
                    mlflow.end_run()
                except:
                    pass  # Игнорируем ошибки если run не был активен
                
                # Создаем уникальное имя эксперимента
                experiment_name = f"tacotron2_enhanced_{int(time.time())}"
                mlflow.set_experiment(experiment_name)
                
                # Начинаем новый run
                run_name = f"enhanced_run_{trial.number if trial else 'single'}_{int(time.time())}"
                mlflow.start_run(run_name=run_name)
                logging.info(f"✅ MLflow эксперимент инициализирован: {experiment_name}")
            except Exception as e:
                logging.error(f"⚠️ Ошибка инициализации MLflow: {e}")
        
        # === Инициализация Debug Reporter ===
        if DEBUG_REPORTER_AVAILABLE:
            try:
                self.debug_reporter = initialize_debug_reporter(self.telegram_monitor)
                logging.info("🔍 Debug Reporter инициализирован")
            except Exception as e:
                logging.error(f"⚠️ Ошибка инициализации Debug Reporter: {e}")
        
        # === Инициализация Telegram Monitor ===
        try:
            from smart_tuner.telegram_monitor import TelegramMonitor
            import yaml
            
            # Загружаем конфиг
            config_path = "smart_tuner/config.yaml"
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                logging.warning(f"Не удалось загрузить конфиг: {e}")
                config = {}
            
            telegram_config = config.get('telegram', {})
            bot_token = telegram_config.get('bot_token')
            chat_id = telegram_config.get('chat_id')
            enabled = telegram_config.get('enabled', False)
            
            if bot_token and chat_id and enabled:
                self.telegram_monitor = TelegramMonitor()
                logging.info("📱 Telegram Monitor инициализирован")
            else:
                self.telegram_monitor = None
                logging.warning("📱 Telegram Monitor отключен (неполные настройки)")
        except Exception as e:
            self.telegram_monitor = None
            logging.error(f"Ошибка инициализации Telegram Monitor: {e}")

    def _train_core_enhanced(
        self, 
        hparams: Any, 
        trial: Optional[optuna.Trial] = None, 
        checkpoint_path: Optional[str] = None,
        writer: Optional[Any] = None  # SummaryWriter
    ) -> Optional[Dict[str, float]]:
        """
        🚀 Вызывает улучшенную функцию обучения с расширенными возможностями.
        Использует train.py с расширенным логированием.
        """
        try:
            logging.info("🚀 Импортируем ENHANCED функцию обучения из train.py...")
            
            # Используем train.py с расширенными возможностями
            from train import train as core_train_func
            
            # Устанавливаем директории для этого запуска
            output_directory = os.path.join(
                self.config.get('training', {}).get('base_output_dir', 'output'),
                f"enhanced_trial_{trial.number}" if trial else "enhanced_single_run"
            )
            log_directory = os.path.join(output_directory, "logs")
            os.makedirs(log_directory, exist_ok=True)
            
            logging.info(f"🚀 Запускаем ENHANCED обучение с параметрами:")
            logging.info(f"   - epochs: {hparams.epochs}")
            logging.info(f"   - batch_size: {hparams.batch_size}")
            logging.info(f"   - learning_rate: {hparams.learning_rate}")
            logging.info(f"   - output_directory: {output_directory}")

            # 📱 Инициализируем Telegram Monitor
            telegram_monitor = None
            try:
                from smart_tuner.telegram_monitor import TelegramMonitor
                telegram_monitor = TelegramMonitor()
                logging.info("📱 Telegram Monitor инициализирован")
            except Exception as e:
                logging.warning(f"⚠️ Не удалось инициализировать Telegram Monitor: {e}")

            # Запускаем обучение с расширенными возможностями
            final_metrics = core_train_func(
                output_directory=output_directory,
                log_directory=log_directory,
                checkpoint_path=checkpoint_path,
                warm_start=False,
                ignore_mmi_layers=False,
                ignore_gst_layers=False,
                ignore_tsgst_layers=False,
                n_gpus=1,
                rank=0,
                group_name="",
                hparams=hparams,
                # Интеграция со Smart Tuner
                smart_tuner_trial=trial,
                smart_tuner_logger=self._setup_logger(output_directory),
                tensorboard_writer=writer,
                telegram_monitor=telegram_monitor
            )
            
            logging.info(f"✅ ENHANCED обучение завершено, получены метрики: {final_metrics}")
            
            if final_metrics and final_metrics.get('checkpoint_path'):
                self.last_checkpoint_path = final_metrics['checkpoint_path']
            
            # === Расширенное логирование метрик ===
            self._log_enhanced_metrics(final_metrics, trial)
            
            return final_metrics
        
        except Exception as e:
            import traceback
            logging.error(f"❌ Критическая ошибка в ENHANCED обучении: {e}")
            logging.error(f"Полный traceback: {traceback.format_exc()}")
            
            # Возвращаем частичные метрики, если возможно
            try:
                return {
                    "validation_loss": float('inf'),
                    "iteration": 0,
                    "checkpoint_path": None,
                    "error": str(e)
                }
            except:
                return None

    def _log_enhanced_metrics(self, metrics: Dict[str, Any], trial: Optional[optuna.Trial] = None):
        """Расширенное логирование метрик как в enhanced_training_main.py"""
        
        if not metrics:
            return
        
        # === Логирование в TensorBoard ===
        if self.tensorboard_writer is not None:
            try:
                # Основные метрики
                if 'validation_loss' in metrics:
                    self.tensorboard_writer.add_scalar("val/loss", metrics['validation_loss'], self.global_step)
                if 'training_loss' in metrics:
                    self.tensorboard_writer.add_scalar("train/loss", metrics['training_loss'], self.global_step)
                
                # Дополнительные метрики качества
                if 'attention_alignment_score' in metrics:
                    self.tensorboard_writer.add_scalar("quality/attention_alignment", metrics['attention_alignment_score'], self.global_step)
                if 'gate_accuracy' in metrics:
                    self.tensorboard_writer.add_scalar("quality/gate_accuracy", metrics['gate_accuracy'], self.global_step)
                if 'mel_quality_score' in metrics:
                    self.tensorboard_writer.add_scalar("quality/mel_quality", metrics['mel_quality_score'], self.global_step)
                
                # Гиперпараметры
                if 'learning_rate' in metrics:
                    self.tensorboard_writer.add_scalar("hyperparams/learning_rate", metrics['learning_rate'], self.global_step)
                if 'batch_size' in metrics:
                    self.tensorboard_writer.add_scalar("hyperparams/batch_size", metrics['batch_size'], self.global_step)
                
                self.tensorboard_writer.flush()
                logging.info("✅ Метрики записаны в TensorBoard")
            except Exception as e:
                logging.error(f"⚠️ Ошибка логирования в TensorBoard: {e}")
        
        # === Логирование в MLflow ===
        if MLFLOW_AVAILABLE:
            try:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        mlflow.log_metric(f"enhanced.{key}", value, step=self.global_step)
                logging.info("✅ Метрики записаны в MLflow")
            except Exception as e:
                logging.error(f"⚠️ Ошибка логирования в MLflow: {e}")

    def get_current_training_phase(self) -> str:
        """Определяет текущую фазу обучения"""
        for phase, config in self.training_phases.items():
            if self.current_epoch <= config['max_epoch']:
                return phase
        return 'fine_tuning'

    def adjust_hyperparams_for_phase(self, phase: str, hparams: Any):
        """Адаптирует гиперпараметры для текущей фазы обучения"""
        if phase == 'pre_alignment':
            # Снижаем learning rate для стабильности
            hparams.learning_rate *= 0.5
            hparams.guided_attn_weight = 100.0
        elif phase == 'alignment_learning':
            # Увеличиваем guided attention
            hparams.guided_attn_weight = 200.0
        elif phase == 'quality_optimization':
            # Фокус на качестве
            hparams.learning_rate *= 0.8
        elif phase == 'fine_tuning':
            # Финальная полировка
            hparams.learning_rate *= 0.5
        
        logging.info(f"🔄 Адаптированы гиперпараметры для фазы '{phase}'")

    def _setup_logger(self, output_directory: str) -> logging.Logger:
        """Настраивает стандартный Python логгер для этого запуска."""
        logger = logging.getLogger(f"enhanced_smart_tuner_{os.path.basename(output_directory)}")
        logger.setLevel(logging.INFO)
        
        # Убираем старые обработчики, если есть
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Создаем новый обработчик для файла
        log_file = os.path.join(output_directory, "enhanced_smart_tuner.log")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Логи для ENHANCED запуска будут сохраняться в {log_file}")
        return logger
        
    def cleanup(self):
        """Очистка ресурсов"""
        self.last_checkpoint_path = None
        
        # Закрываем TensorBoard
        if self.tensorboard_writer is not None:
            try:
                self.tensorboard_writer.close()
                logging.info("✅ TensorBoard writer закрыт")
            except Exception as e:
                logging.error(f"⚠️ Ошибка закрытия TensorBoard: {e}")
        
        # Завершаем MLflow
        if MLFLOW_AVAILABLE:
            try:
                mlflow.end_run()
                logging.info("✅ MLflow run завершен")
            except Exception as e:
                logging.error(f"⚠️ Ошибка завершения MLflow run: {e}")
        
        logging.info("🚀 Enhanced TrainerWrapper очищен.")


# Для обратной совместимости
class TrainerWrapper(EnhancedTrainerWrapper):
    """
    Обратная совместимость с старым TrainerWrapper
    """
    pass 