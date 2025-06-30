#!/usr/bin/env python3
"""
🧠 SMART TRAINING LOGGER - Интеллектуальная система логирования TTS
Полноценная система логирования для Smart Tuner с поддержкой MLflow, TensorBoard и файлов.
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


@dataclass
class TrainingMetrics:
    """Структура для хранения метрик обучения"""
    step: int
    epoch: int
    total_loss: float
    mel_loss: float = 0.0
    gate_loss: float = 0.0
    guide_loss: float = 0.0
    learning_rate: float = 0.0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class SmartTrainingLogger:
    """
    🧠 Интеллектуальная система логирования для TTS обучения
    """
    
    def __init__(self, 
                 log_dir: str = "logs",
                 experiment_name: str = "TTS_Training",
                 enable_mlflow: bool = True,
                 enable_tensorboard: bool = True,
                 enable_file_logging: bool = True):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.experiment_name = experiment_name
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        self.enable_tensorboard = enable_tensorboard and TENSORBOARD_AVAILABLE
        self.enable_file_logging = enable_file_logging
        
        # Инициализация компонентов
        self.logger = self._setup_file_logger()
        self.tensorboard_writer = self._setup_tensorboard() if self.enable_tensorboard else None
        self.mlflow_run = self._setup_mlflow() if self.enable_mlflow else None
        
        # Хранение метрик
        self.metrics_history: List[TrainingMetrics] = []
        self.best_loss = float('inf')
        self.start_time = time.time()
        
        self.logger.info(f"🧠 SmartTrainingLogger инициализирован")
        self.logger.info(f"📊 MLflow: {'✅' if self.enable_mlflow else '❌'}")
        self.logger.info(f"📈 TensorBoard: {'✅' if self.enable_tensorboard else '❌'}")
        self.logger.info(f"📝 File logging: {'✅' if self.enable_file_logging else '❌'}")
    
    def _setup_file_logger(self) -> logging.Logger:
        """Настройка файлового логгера"""
        logger = logging.getLogger('SmartTrainingLogger')
        logger.setLevel(logging.INFO)
        
        # Удаляем существующие handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Файловый handler
        if self.enable_file_logging:
            log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Консольный handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_tensorboard(self) -> Optional[SummaryWriter]:
        """Настройка TensorBoard"""
        if not TENSORBOARD_AVAILABLE:
            return None
            
        try:
            tb_dir = self.log_dir / "tensorboard" / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            writer = SummaryWriter(str(tb_dir))
            self.logger.info(f"📈 TensorBoard инициализирован: {tb_dir}")
            return writer
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации TensorBoard: {e}")
            return None
    
    def _setup_mlflow(self) -> Optional[str]:
        """Настройка MLflow"""
        if not MLFLOW_AVAILABLE:
            return None
            
        try:
            # Настройка tracking URI
            mlflow_dir = self.log_dir / "mlflow"
            mlflow_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")
            
            # Создание/получение эксперимента
            mlflow.set_experiment(self.experiment_name)
            
            # Запуск run
            run = mlflow.start_run()
            run_id = run.info.run_id
            self.logger.info(f"📊 MLflow инициализирован: Run ID {run_id}")
            return run_id
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации MLflow: {e}")
            return None
    
    def log_training_step(self, metrics: TrainingMetrics) -> None:
        """
        Логирование одного шага обучения
        
        Args:
            metrics: Метрики для логирования
        """
        try:
            # Сохранение в историю
            self.metrics_history.append(metrics)
            
            # Обновление лучшего результата
            if metrics.total_loss < self.best_loss:
                self.best_loss = metrics.total_loss
                self.logger.info(f"🎯 Новый лучший результат! Loss: {self.best_loss:.6f}")
            
            # Файловое логирование
            if self.enable_file_logging:
                self._log_to_file(metrics)
            
            # TensorBoard логирование
            if self.tensorboard_writer:
                self._log_to_tensorboard(metrics)
            
            # MLflow логирование
            if self.mlflow_run:
                self._log_to_mlflow(metrics)
            
            # Прогресс обучения
            self._log_progress(metrics)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка логирования шага: {e}")
    
    def _log_to_file(self, metrics: TrainingMetrics) -> None:
        """Логирование в файл"""
        elapsed_time = time.time() - self.start_time
        self.logger.info(
            f"Step {metrics.step} | Epoch {metrics.epoch} | "
            f"Loss: {metrics.total_loss:.6f} | "
            f"Mel: {metrics.mel_loss:.6f} | "
            f"Gate: {metrics.gate_loss:.6f} | "
            f"Guide: {metrics.guide_loss:.6f} | "
            f"LR: {metrics.learning_rate:.2e} | "
            f"Time: {elapsed_time:.1f}s"
        )
    
    def _log_to_tensorboard(self, metrics: TrainingMetrics) -> None:
        """Логирование в TensorBoard"""
        try:
            self.tensorboard_writer.add_scalar('Loss/Total', metrics.total_loss, metrics.step)
            self.tensorboard_writer.add_scalar('Loss/Mel', metrics.mel_loss, metrics.step)
            self.tensorboard_writer.add_scalar('Loss/Gate', metrics.gate_loss, metrics.step)
            self.tensorboard_writer.add_scalar('Loss/Guide', metrics.guide_loss, metrics.step)
            self.tensorboard_writer.add_scalar('Learning_Rate', metrics.learning_rate, metrics.step)
            
            # Дополнительные метрики
            elapsed_time = time.time() - self.start_time
            self.tensorboard_writer.add_scalar('Training/Elapsed_Time', elapsed_time, metrics.step)
            self.tensorboard_writer.add_scalar('Training/Best_Loss', self.best_loss, metrics.step)
            
            # Flush для немедленной записи
            self.tensorboard_writer.flush()
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка TensorBoard логирования: {e}")
    
    def _log_to_mlflow(self, metrics: TrainingMetrics) -> None:
        """Логирование в MLflow"""
        try:
            mlflow.log_metric('total_loss', metrics.total_loss, step=metrics.step)
            mlflow.log_metric('mel_loss', metrics.mel_loss, step=metrics.step)
            mlflow.log_metric('gate_loss', metrics.gate_loss, step=metrics.step)
            mlflow.log_metric('guide_loss', metrics.guide_loss, step=metrics.step)
            mlflow.log_metric('learning_rate', metrics.learning_rate, step=metrics.step)
            mlflow.log_metric('best_loss', self.best_loss, step=metrics.step)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка MLflow логирования: {e}")
    
    def _log_progress(self, metrics: TrainingMetrics) -> None:
        """Логирование прогресса обучения"""
        # Каждые 100 шагов выводим детальную информацию
        if metrics.step % 100 == 0:
            elapsed_time = time.time() - self.start_time
            avg_loss = sum(m.total_loss for m in self.metrics_history[-100:]) / min(100, len(self.metrics_history))
            
            self.logger.info(f"📊 Прогресс обучения:")
            self.logger.info(f"   Шаг: {metrics.step}")
            self.logger.info(f"   Эпоха: {metrics.epoch}")
            self.logger.info(f"   Текущий Loss: {metrics.total_loss:.6f}")
            self.logger.info(f"   Средний Loss (100 шагов): {avg_loss:.6f}")
            self.logger.info(f"   Лучший Loss: {self.best_loss:.6f}")
            self.logger.info(f"   Время обучения: {elapsed_time:.1f}s")
    
    def log_epoch_summary(self, epoch: int, avg_loss: float, epoch_time: float) -> None:
        """Логирование итогов эпохи"""
        self.logger.info(f"🎉 Эпоха {epoch} завершена:")
        self.logger.info(f"   Средний Loss: {avg_loss:.6f}")
        self.logger.info(f"   Время эпохи: {epoch_time:.1f}s")
        self.logger.info(f"   Лучший Loss за все время: {self.best_loss:.6f}")
        
        # Логирование в системы
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('Epoch/Average_Loss', avg_loss, epoch)
            self.tensorboard_writer.add_scalar('Epoch/Time', epoch_time, epoch)
        
        if self.mlflow_run:
            mlflow.log_metric('epoch_avg_loss', avg_loss, step=epoch)
            mlflow.log_metric('epoch_time', epoch_time, step=epoch)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """Логирование гиперпараметров"""
        self.logger.info("⚙️ Гиперпараметры обучения:")
        for key, value in hparams.items():
            self.logger.info(f"   {key}: {value}")
        
        # MLflow логирование параметров
        if self.mlflow_run:
            for key, value in hparams.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(key, value)
        
        # TensorBoard логирование (как текст)
        if self.tensorboard_writer:
            hparams_text = '\n'.join([f"{k}: {v}" for k, v in hparams.items()])
            self.tensorboard_writer.add_text('Hyperparameters', hparams_text, 0)
    
    def save_metrics_history(self, filepath: Optional[str] = None) -> str:
        """Сохранение истории метрик в JSON файл"""
        if filepath is None:
            filepath = self.log_dir / f"metrics_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            metrics_data = []
            for m in self.metrics_history:
                metrics_data.append({
                    'step': m.step,
                    'epoch': m.epoch,
                    'total_loss': m.total_loss,
                    'mel_loss': m.mel_loss,
                    'gate_loss': m.gate_loss,
                    'guide_loss': m.guide_loss,
                    'learning_rate': m.learning_rate,
                    'timestamp': m.timestamp
                })
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📊 История метрик сохранена: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения истории метрик: {e}")
            return ""
    
    def close(self) -> None:
        """Закрытие всех логгеров"""
        try:
            # Сохранение финальных метрик
            self.save_metrics_history()
            
            # Закрытие TensorBoard
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
                self.logger.info("📈 TensorBoard закрыт")
            
            # Закрытие MLflow
            if self.mlflow_run:
                mlflow.end_run()
                self.logger.info("📊 MLflow run завершен")
            
            self.logger.info("🏁 SmartTrainingLogger завершил работу")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка закрытия логгера: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Получение итоговой сводки обучения"""
        if not self.metrics_history:
            return {}
        
        total_time = time.time() - self.start_time
        total_steps = len(self.metrics_history)
        
        return {
            'total_steps': total_steps,
            'total_time_seconds': total_time,
            'best_loss': self.best_loss,
            'final_loss': self.metrics_history[-1].total_loss if self.metrics_history else 0,
            'average_loss': sum(m.total_loss for m in self.metrics_history) / total_steps,
            'total_epochs': self.metrics_history[-1].epoch if self.metrics_history else 0,
            'steps_per_second': total_steps / total_time if total_time > 0 else 0
        }


def create_smart_logger(log_dir: str = "logs", 
                       experiment_name: str = "TTS_Training",
                       **kwargs) -> SmartTrainingLogger:
    """
    Фабричная функция для создания умного логгера
    """
    return SmartTrainingLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        **kwargs
    )


def get_training_logger(log_dir: str = "logs", 
                       experiment_name: str = "TTS_Training",
                       **kwargs) -> SmartTrainingLogger:
    """
    🤖 НЕДОСТАЮЩАЯ ФУНКЦИЯ: Получение логгера для тренировки
    
    Эта функция нужна для совместимости со Smart Tuner системой.
    Является алиасом для create_smart_logger.
    """
    return create_smart_logger(log_dir, experiment_name, **kwargs)


def log_training_start(experiment_name: str = "TTS_Training", 
                      hparams: Dict[str, Any] = None, 
                      **kwargs) -> None:
    """
    🤖 НЕДОСТАЮЩАЯ ФУНКЦИЯ: Логирование начала обучения
    
    Эта функция нужна для совместимости со Smart Tuner системой.
    """
    try:
        logger = get_training_logger(experiment_name=experiment_name)
        
        if hparams:
            logger.log_hyperparameters(hparams)
            
        # Логируем начало обучения
        import logging
        logging.info(f"🚀 Начато обучение эксперимента: {experiment_name}")
        
        if hparams:
            logging.info(f"📊 Гиперпараметры: {hparams}")
            
    except Exception as e:
        import logging
        logging.warning(f"⚠️ Ошибка логирования начала обучения: {e}")


def log_training_metrics(metrics: Dict[str, Any], step: int = None, **kwargs) -> None:
    """
    🤖 НЕДОСТАЮЩАЯ ФУНКЦИЯ: Логирование метрик обучения
    
    Эта функция нужна для совместимости со Smart Tuner системой.
    """
    try:
        # Логируем через обычный logging
        import logging
        
        if step is not None:
            logging.info(f"📊 Метрики (шаг {step}): {metrics}")
        else:
            logging.info(f"📊 Метрики: {metrics}")
        
        # Пытаемся использовать MLflow если доступен
        try:
            import mlflow
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value, step=step)
                    
        except ImportError:
            pass  # MLflow недоступен
        except Exception as e:
            logging.warning(f"⚠️ Ошибка MLflow логирования: {e}")
            
    except Exception as e:
        import logging
        logging.warning(f"⚠️ Ошибка логирования метрик: {e}")


def log_param_change(param_name: str, old_value: Any, new_value: Any, reason: str = "", **kwargs) -> None:
    """
    🤖 НЕДОСТАЮЩАЯ ФУНКЦИЯ: Логирование изменений параметров
    
    Эта функция нужна для совместимости со Smart Tuner системой.
    """
    try:
        import logging
        
        # Форматируем сообщение об изменении
        if reason:
            message = f"🔧 Параметр изменен: {param_name} = {old_value} → {new_value} (причина: {reason})"
        else:
            message = f"🔧 Параметр изменен: {param_name} = {old_value} → {new_value}"
            
        logging.info(message)
        
        # Пытаемся использовать MLflow если доступен
        try:
            import mlflow
            # Логируем как параметр
            mlflow.log_param(f"{param_name}_change", f"{old_value}_to_{new_value}")
            
        except ImportError:
            pass  # MLflow недоступен
        except Exception as e:
            logging.warning(f"⚠️ Ошибка MLflow логирования изменения параметра: {e}")
            
    except Exception as e:
        import logging
        logging.warning(f"⚠️ Ошибка логирования изменения параметра: {e}")


def log_training_warning(message: str, **kwargs) -> None:
    """
    🤖 НЕДОСТАЮЩАЯ ФУНКЦИЯ: Логирование предупреждений обучения
    
    Эта функция нужна для совместимости со Smart Tuner системой.
    """
    try:
        import logging
        logging.warning(f"⚠️ {message}")
        
    except Exception as e:
        import logging
        logging.error(f"❌ Ошибка логирования предупреждения: {e}")


if __name__ == "__main__":
    # Тестирование системы логирования
    logger = create_smart_logger(experiment_name="Test_TTS")
    
    # Тестовые гиперпараметры
    test_hparams = {
        'batch_size': 12,
        'learning_rate': 1e-5,
        'epochs': 1000
    }
    logger.log_hyperparameters(test_hparams)
    
    # Тестовые метрики
    for step in range(5):
        metrics = TrainingMetrics(
            step=step,
            epoch=0,
            total_loss=1.0 - step * 0.1,
            mel_loss=0.5 - step * 0.05,
            gate_loss=0.3 - step * 0.03,
            guide_loss=0.2 - step * 0.02,
            learning_rate=1e-5
        )
        logger.log_training_step(metrics)
    
    # Итоги эпохи
    logger.log_epoch_summary(0, 0.7, 120.0)
    
    # Сводка и закрытие
    summary = logger.get_training_summary()
    print(f"📊 Итоговая сводка: {summary}")
    
    logger.close()
    print("✅ Тестирование SmartTrainingLogger завершено!") 