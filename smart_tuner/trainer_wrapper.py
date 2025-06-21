import logging
import subprocess
import os
import time
import requests
import sys
from datetime import datetime
import mlflow
from smart_tuner.alert_manager import AlertManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - (TrainerWrapper) - %(message)s'
)

class TrainerWrapper:
    """
    Обертка для запуска и управления процессом обучения train.py.
    """
    def __init__(self, config: dict):
        try:
            self.config = config
            logging.info(f"Инициализация TrainerWrapper с config типа: {type(config)}")
            
            self.python_executable = self.config.get('training', {}).get('python_executable', 'python')
            self.train_script = self.config.get('training', {}).get('script_path', 'train.py')
            self.output_dir = self.config.get('output_dir', 'output')
            self.current_process = None
            self.current_run_id = None
            self.alert_manager = AlertManager(config)
            self.ensure_mlflow_running()
            logging.info("TrainerWrapper инициализирован с конфигурацией.")
            
        except Exception as e:
            logging.error(f"Ошибка при инициализации TrainerWrapper: {e}")
            raise

    def _construct_hparams_string(self, hparams_override):
        """Конструирует строку гиперпараметров из разных форматов."""
        if not hparams_override:
            return ""
        
        if isinstance(hparams_override, str):
            return hparams_override
        
        if hasattr(hparams_override, 'values') and callable(getattr(hparams_override, 'values')):
            hparams_dict = hparams_override.values()
        elif isinstance(hparams_override, dict):
            hparams_dict = hparams_override
        else:
            raise TypeError(f"Неподдерживаемый тип для hparams_override: {type(hparams_override)}")

        return ",".join([f"{key}={value}" for key, value in hparams_dict.items()])

    def is_mlflow_running(self, port=5000):
        try:
            requests.get(f"http://localhost:{port}", timeout=3)
            return True
        except requests.ConnectionError:
            return False

    def ensure_mlflow_running(self):
        # Эта функция может быть расширена для автоматического запуска MLflow
        if not self.is_mlflow_running():
            logging.warning("MLflow UI не запущен. Пожалуйста, запустите его через install.sh.")
        else:
            logging.info("✅ MLflow уже запущен")

    def start_training(self, hparams_override=None, checkpoint_path=None):
        """
        Запускает train.py с заданными параметрами.
        Возвращает процесс, MLflow run_id и пути к директориям.
        """
        run_name = f"proactive_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        
        output_dir = os.path.join(self.config.get('output_dir', 'output'), run_name)
        log_dir = os.path.join(output_dir, "logs")
        checkpoint_dir = os.path.join(output_dir, "checkpoint")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        mlflow.set_tracking_uri(self.config.get('mlflow', {}).get('tracking_uri', 'file:./mlruns'))
        mlflow.set_experiment(self.config.get('experiment_name', 'tacotron2_production'))
        
        active_run = mlflow.start_run(run_name=run_name)
        self.current_run_id = active_run.info.run_id
        logging.info(f"Создан MLflow run: {run_name} с ID: {self.current_run_id}")

        command = [
            self.python_executable, self.train_script,
            '--output_directory', output_dir,
            '--log_directory', log_dir,
        ]

        # Собираем все hparams в одну строку
        hparams_dict = {}
        if hasattr(hparams_override, 'values') and callable(getattr(hparams_override, 'values')):
            hparams_dict = hparams_override.values()
        elif isinstance(hparams_override, dict):
            hparams_dict = hparams_override.copy()

        # Гарантируем, что distributed_run выключен
        hparams_dict['distributed_run'] = 'False'
        
        # Правильно форматируем значения для HParams.parse()
        hparams_parts = []
        for key, value in hparams_dict.items():
            if isinstance(value, list):
                # Списки передаем в формате name=[val1,val2,val3]
                formatted_list = "[" + ",".join(str(v) for v in value) + "]"
                hparams_parts.append(f"{key}={formatted_list}")
            else:
                hparams_parts.append(f"{key}={value}")
        
        hparams_str = ",".join(hparams_parts)
        
        command.extend(["--hparams", hparams_str])

        if checkpoint_path:
            command.extend(["--checkpoint_path", checkpoint_path])

        try:
            env = os.environ.copy()
            env['MLFLOW_RUN_ID'] = self.current_run_id
            
            self.current_process = subprocess.Popen(command, env=env)
            logging.info(f"Запущен процесс обучения с PID: {self.current_process.pid}")
            
            # Возвращаем все необходимые данные
            return self.current_process, self.current_run_id, output_dir, log_dir

        except Exception as e:
            logging.error(f"Не удалось запустить процесс обучения: {e}", exc_info=True)
            mlflow.end_run()
            return None, None, None, None

    def stop_training(self):
        """Останавливает текущий процесс обучения."""
        if self.current_process and self.current_process.poll() is None:
            logging.info(f"Останавливаем процесс обучения с PID: {self.current_process.pid}")
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logging.warning("Процесс не завершился, убиваем принудительно.")
                self.current_process.kill()
            logging.info("Процесс обучения остановлен.")
        else:
            logging.info("Нет активного процесса обучения для остановки.")
            
        # Завершаем MLflow run если он еще активен
        if self.current_run_id:
            try:
                # Проверяем, есть ли активный run
                active_run = mlflow.active_run()
                if active_run and active_run.info.run_id == self.current_run_id:
                    mlflow.end_run()
                    logging.info(f"MLflow run {self.current_run_id} завершен.")
            except Exception as e:
                logging.warning(f"Не удалось завершить MLflow run: {e}")
            finally:
                self.current_run_id = None

    def train_with_params(self, hyperparams: dict):
        """Функция для Optuna: запускает обучение и возвращает метрики."""
        # Эта функция требует более сложной реализации для реальной оптимизации
        logging.info(f"Заглушка для train_with_params с параметрами: {hyperparams}")
        return {"val_loss": 1.0} 