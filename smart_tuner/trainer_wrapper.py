import logging
import os
from typing import Dict, Any, Optional
import torch
import optuna


class TrainerWrapper:
    """
    Упрощенная обертка для управления процессом обучения Tacotron2.
    Интегрируется с Optuna, MLflow и TensorBoard.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.last_checkpoint_path: Optional[str] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("TrainerWrapper инициализирован.")

    def train_with_params(
        self, 
        params: Dict[str, Any], 
        trial: Optional[optuna.Trial] = None, 
        writer: Optional[Any] = None,  # SummaryWriter
        **kwargs  # Для обратной совместимости
    ) -> Optional[Dict[str, float]]:
        """
        Запускает один сеанс обучения с заданными параметрами.
        Это основная точка входа для Smart Tuner.
        """
        logging.info(f"🧪 Запуск обучения с параметрами: {params}")
        
        # Создаем hparams на лету для этого конкретного запуска
        from hparams import create_hparams
        hparams = create_hparams()
        for key, value in params.items():
            if hasattr(hparams, key):
                setattr(hparams, key, value)
        
        # Запускаем основной метод обучения
        return self._train_core(hparams, trial, self.last_checkpoint_path, writer)

    def _train_core(
        self, 
        hparams: Any, 
        trial: Optional[optuna.Trial] = None, 
        checkpoint_path: Optional[str] = None,
        writer: Optional[Any] = None  # SummaryWriter
    ) -> Optional[Dict[str, float]]:
        """
        Вызывает основную функцию обучения из train.py.
        """
        try:
            logging.info("Импортируем функцию train из train.py...")
            
            # 🔥 ИСПРАВЛЕНИЕ ПУТИ: Убеждаемся что loss_function.py в sys.path
            import sys
            import os
            
            # Добавляем родительскую директорию в sys.path для импорта loss_function
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
                logging.info(f"Добавлен в sys.path: {parent_dir}")
            
            from train import train as core_train_func
            logging.info("Функция train успешно импортирована.")
            
            # Устанавливаем директории для этого запуска
            output_directory = os.path.join(
                self.config.get('training', {}).get('base_output_dir', 'output'),
                f"trial_{trial.number}" if trial else "single_run"
            )
            log_directory = os.path.join(output_directory, "logs")
            os.makedirs(log_directory, exist_ok=True)
            
            logging.info(f"🚀 Запускаем обучение с параметрами:")
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
            
            logging.info(f"✅ Обучение завершено, получены метрики: {final_metrics}")
            
            if final_metrics and final_metrics.get('checkpoint_path'):
                self.last_checkpoint_path = final_metrics['checkpoint_path']
            
            return final_metrics
        
        except Exception as e:
            import traceback
            logging.error(f"❌ Критическая ошибка в ядре обучения: {e}")
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

    def _setup_logger(self, output_directory: str) -> logging.Logger:
        """Настраивает стандартный Python логгер для этого запуска."""
        logger = logging.getLogger(f"smart_tuner_train_{os.path.basename(output_directory)}")
        logger.setLevel(logging.INFO)
        
        # Убираем старые обработчики, если есть
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Создаем новый обработчик для файла
        log_file = os.path.join(output_directory, "smart_tuner_train.log")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Логи для этого запуска будут сохраняться в {log_file}")
        return logger
        
    def cleanup(self):
        self.last_checkpoint_path = None
        logging.info("TrainerWrapper очищен.") 