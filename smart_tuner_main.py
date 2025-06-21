#!/usr/bin/env python3
"""Smart Tuner V2 Main"""

import os
import sys
import logging
import argparse
import yaml
import threading
import time
import subprocess
import requests
from smart_tuner.trainer_wrapper import TrainerWrapper
from smart_tuner.optimization_engine import OptimizationEngine
from smart_tuner.alert_manager import AlertManager
from smart_tuner.web_interfaces import WebInterfaceManager
from smart_tuner.early_stop_controller import EarlyStopController
from smart_tuner.log_watcher import LogWatcher
from smart_tuner.metrics_store import MetricsStore
from utils import find_latest_checkpoint, load_hparams, save_hparams

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - (%(name)s) - %(message)s',
    handlers=[
        logging.FileHandler('smart_tuner_main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SmartTunerMain:
    def __init__(self, config_path="smart_tuner/config.yaml"):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Проверяем и запускаем MLflow если нужно
        self.ensure_mlflow_running()
        
        self.trainer = TrainerWrapper(self.config)
        self.optimizer = OptimizationEngine(config_path)
        self.alert_manager = AlertManager(self.config)
        
        # Инициализация веб-интерфейсов
        self.web_manager = WebInterfaceManager(config_path)
        self.web_manager.create_interfaces()
        
        logger.info("SmartTunerMain инициализирован")
        
        # Запуск веб-интерфейсов в фоновом режиме
        self.start_web_interfaces()
        
    def is_mlflow_running(self, port=5000):
        """Проверяет, запущен ли MLflow на указанном порту"""
        try:
            response = requests.get(f"http://localhost:{port}", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def ensure_mlflow_running(self):
        """Убеждается, что MLflow сервер запущен"""
        if not self.is_mlflow_running():
            logger.info("MLflow не запущен. Запускаем...")
            try:
                # Создаем директорию для логов если её нет
                os.makedirs("mlruns", exist_ok=True)
                
                # Запускаем MLflow в фоновом режиме
                subprocess.Popen([
                    sys.executable, "-m", "mlflow", "ui",
                    "--host", "0.0.0.0",
                    "--port", "5000",
                    "--backend-store-uri", f"file://{os.path.abspath('mlruns')}"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Ждем запуска
                for _ in range(10):
                    time.sleep(1)
                    if self.is_mlflow_running():
                        logger.info("✅ MLflow успешно запущен на порту 5000")
                        return
                
                logger.warning("⚠️ MLflow запущен, но проверка подключения не прошла")
                
            except Exception as e:
                logger.error(f"❌ Ошибка запуска MLflow: {e}")
        else:
            logger.info("✅ MLflow уже запущен")
        
    def start_web_interfaces(self):
        """Запуск всех веб-интерфейсов в фоновом режиме."""
        try:
            logger.info("🚀 Запуск веб-интерфейсов Smart Tuner V2...")
            
            # Метод start_all уже запускает все в фоновых потоках,
            # поэтому дополнительный поток не нужен.
            self.web_manager.start_all()
            
            # Небольшая задержка для инициализации серверов
            time.sleep(2)
            
            # Отображение дашборда
            self.web_manager.print_dashboard()
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска веб-интерфейсов: {e}")
        
    def run_proactive_training(self):
        """
        Запускает обучение с проактивным контролем.
        Цикл обучения, который может перезапускаться с новыми параметрами.
        """
        logger.info("🚀 Запуск обучения в проактивном режиме...")
        self.alert_manager.send_info_notification("🚀 Началось проактивное обучение Smart Tuner V2!")

        hparams = load_hparams(self.config['hparams_path'])
        current_checkpoint = find_latest_checkpoint(self.config['checkpoint_path'])
        
        controller = EarlyStopController(self.config_path)
        metrics_store = MetricsStore() # Используем для сбора метрик
        log_watcher = LogWatcher(
            metrics_store=metrics_store,
            tracking_uri=self.config.get('mlflow', {}).get('tracking_uri', 'mlruns')
        )
        
        training_active = True
        while training_active:
            # Запускаем обучение с текущими hparams и с последнего чекпоинта
            process, run_id = self.trainer.start_training(
                hparams_override=hparams, 
                checkpoint_path=current_checkpoint
            )
            if not process:
                logger.error("Не удалось запустить процесс обучения. Прерываем.")
                self.alert_manager.send_error_notification("❌ Не удалось запустить процесс обучения. Проверьте логи.")
                break
            
            # Начинаем следить за логами для нового run_id
            log_watcher.set_run_id(run_id)
            watcher_thread = threading.Thread(target=log_watcher.watch, daemon=True)
            watcher_thread.start()

            # Цикл мониторинга текущего процесса обучения
            while process.poll() is None:
                time.sleep(self.config.get('proactive_measures', {}).get('check_interval', 60))
                
                new_metrics = metrics_store.get_latest_metrics()
                if not new_metrics:
                    continue

                controller.add_metrics(new_metrics)
                
                decision = controller.decide_next_step(hparams)
                action = decision.get('action', 'continue')

                if action == 'stop':
                    logger.info(f"Получено решение 'stop': {decision.get('reason')}")
                    self.alert_manager.send_success_notification(f"✅ Обучение остановлено: {decision.get('reason')}")
                    self.trainer.stop_training()
                    training_active = False
                    break

                if action == 'restart':
                    logger.warning(f"Получено решение 'restart': {decision.get('reason')}")
                    self.alert_manager.send_info_notification(f"🔄 Перезапуск обучения: {decision.get('reason')}")
                    
                    self.trainer.stop_training() # Останавливаем текущий процесс
                    time.sleep(5) # Даем время процессу завершиться

                    hparams = decision['new_params']
                    save_hparams(self.config['hparams_path'], hparams) # Сохраняем новые параметры
                    
                    # Находим последний сохраненный чекпоинт для продолжения
                    current_checkpoint = find_latest_checkpoint(self.config['checkpoint_path'])
                    
                    logger.info(f"Новые параметры для перезапуска: {hparams}")
                    logger.info(f"Продолжаем с чекпоинта: {current_checkpoint}")
                    break # Выходим из внутреннего цикла для перезапуска внешнего
            
            # Если цикл мониторинга завершился, а флаг training_active все еще True,
            # значит, обучение завершилось само по себе (успешно или с ошибкой).
            if process.poll() is not None and training_active:
                logger.info(f"Процесс обучения завершился с кодом {process.returncode}.")
                if process.returncode == 0:
                    self.alert_manager.send_success_notification("🎉 Обучение успешно завершено!")
                else:
                    self.alert_manager.send_error_notification(f"❌ Обучение завершилось с ошибкой (код: {process.returncode}).")
                training_active = False

        logger.info("Проактивное обучение завершено.")

    def run_optimization(self, n_trials=10):
        """Запуск оптимизации гиперпараметров"""
        logger.info(f"Запуск оптимизации с {n_trials} trials")
        
        # Отправляем уведомление о начале оптимизации
        self.alert_manager.send_info_notification(
            "🤖 Smart Tuner V2 - Старт оптимизации\n\n"
            f"🎯 Количество trials: {n_trials}\n"
            f"⏰ Ожидаемое время: ~{n_trials * 15} минут\n\n"
            "📊 Следите за прогрессом в веб-интерфейсах:\n"
            "• MLflow UI для экспериментов\n"
            "• Optimization Engine для Optuna\n"
            "• Metrics Store для статистики"
        )
        
        try:
            best_params = self.optimizer.optimize(self.objective, n_trials)
            
            logger.info(f"Оптимизация завершена! Лучшие параметры: {best_params}")
            
            # Отправляем уведомление о завершении оптимизации
            params_text = "\n".join([f"• {k}: {v}" for k, v in best_params.items()])
            self.alert_manager.send_success_notification(
                "🎉 Оптимизация завершена!\n\n"
                f"🏆 Лучшие параметры:\n{params_text}\n\n"
                "📁 Лучшая модель сохранена в smart_tuner/models/\n"
                "🌐 Все результаты доступны в веб-интерфейсах"
            )
            
            return best_params
            
        except Exception as e:
            logger.error(f"Ошибка при оптимизации: {e}")
            self.alert_manager.send_error_notification(
                f"❌ Ошибка при оптимизации\n\n"
                f"🔴 Ошибка: {str(e)}\n"
                "🌐 Проверьте веб-интерфейсы для диагностики"
            )
            raise
    
    def objective(self, trial):
        """
        Целевая функция для оптимизации Optuna
        
        Args:
            trial: Объект trial от Optuna
            
        Returns:
            Значение целевой функции для минимизации
        """
        try:
            # Получаем предложенные гиперпараметры
            hyperparams = self.optimizer.suggest_hyperparameters(trial)
            logger.info(f"Trial {trial.number}: Тестируем параметры {hyperparams}")
            
            # Отправляем уведомление о начале trial
            params_text = "\n".join([f"• {k}: {v}" for k, v in hyperparams.items()])
            self.alert_manager.send_info_notification(
                f"🧪 Trial #{trial.number} начат\n\n"
                f"🔧 Гиперпараметры:\n{params_text}\n\n"
                f"⏱️ Ожидаемое время: ~15 минут\n"
                "🌐 Прогресс в веб-интерфейсах"
            )
            
            # Запускаем обучение с этими параметрами
            metrics = self.trainer.train_with_params(hyperparams)
            
            if not metrics:
                logger.warning(f"Trial {trial.number}: Получены пустые метрики")
                # В случае сбоя возвращаем большое значение, чтобы Optuna избегала этих параметров
                return float('inf')

            # Получаем целевую метрику из результатов
            objective_metric = self.config.get("optimization", {}).get("objective_metric", "val_loss")
            
            # Отправляем уведомление о завершении trial
            self.alert_manager.send_success_notification(
                f"✅ Trial #{trial.number} завершен\n\n"
                f"🏆 Результат ({objective_metric}): {metrics.get(objective_metric, 'N/A'):.4f}"
            )

            return metrics.get(objective_metric, float('inf'))
            
        except Exception as e:
            logger.error(f"Критическая ошибка в objective function для trial {trial.number}: {e}")
            self.alert_manager.send_error_notification(
                f"💥 Критическая ошибка в Trial #{trial.number}\n\n"
                f"🔴 Ошибка: {str(e)}\n"
                "🌐 Проверьте веб-интерфейсы для диагностики"
            )
            # Сообщаем Optuna о сбое
            raise optuna.exceptions.TrialPruned()


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="Smart Tuner V2 для Tacotron2")
    parser.add_argument('--config', default='smart_tuner/config.yaml', help='Путь к файлу конфигурации')
    parser.add_argument('--optimize', action='store_true', help='Запуск оптимизации гиперпараметров')
    parser.add_argument('--train', action='store_true', help='Запуск одиночного полноценного обучения')
    parser.add_argument('--n_trials', type=int, default=10, help='Количество trials для оптимизации')

    args = parser.parse_args()
    
    tuner = SmartTunerMain(args.config)
    
    if args.optimize:
        tuner.run_optimization(n_trials=args.n_trials)
    elif args.train:
        tuner.run_proactive_training()
    else:
        logger.info("Не выбран режим работы. Используйте --train или --optimize.")
        parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Неперехваченная ошибка в main: {e}", exc_info=True)
        sys.exit(1)

