import subprocess
import os
import time
import logging
import yaml
import threading

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - (TrainerWrapper) - %(message)s')

# Добавляем обработчик исключений
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Критическая ошибка:", exc_info=(exc_type, exc_value, exc_traceback))

import sys
sys.excepthook = handle_exception

class TrainerWrapper:
    """
    Управляет жизненным циклом процесса обучения (train.py).
    Запускает, останавливает и отслеживает дочерний процесс.
    """
    def __init__(self, config):
        try:
            logging.info(f"Инициализация TrainerWrapper с config типа: {type(config)}")
            
            if not isinstance(config, dict):
                raise TypeError(f"Config должен быть dict, получен {type(config)}")
                
            self.config = config
            self.process = None
            
            logging.info(f"Ключи в конфигурации: {list(config.keys())}")
            
            # Проверяем наличие необходимых секций в конфигурации
            if 'hyperparameter_search_space' not in self.config:
                logging.warning("Секция 'hyperparameter_search_space' не найдена в конфигурации")
                self.config['hyperparameter_search_space'] = {}
                
            if 'resources' not in self.config:
                logging.warning("Секция 'resources' не найдена в конфигурации")
                self.config['resources'] = {
                    'checkpointing': {'path': 'output/default_experiment'}
                }
                
            logging.info("TrainerWrapper инициализирован с конфигурацией.")
            
        except Exception as e:
            logging.error(f"Ошибка при инициализации TrainerWrapper: {e}")
            raise

    def _construct_hparams_string(self, hparams_override=None):
        """Собирает строку гиперпараметров из базовых и переданных."""
        # Базовые параметры для полноценного обучения
        base_hparams = {
            'training_files': 'data/dataset/train.csv',
            'validation_files': 'data/dataset/val.csv',
            'epochs': 500,  # Полноценное обучение
            'save_interval': 1000,  # Сохранение каждые 1000 итераций
            'validate_interval': 100,  # Валидация каждые 100 итераций
        }
        
        # Параметры из hyperparameter_search_space с их default значениями (если есть)
        if 'hyperparameter_search_space' in self.config:
            search_space_defaults = {}
            for k, v in self.config['hyperparameter_search_space'].items():
                if isinstance(v, dict) and 'default' in v:
                    search_space_defaults[k] = v['default']
                elif isinstance(v, dict) and v.get('type') == 'float':
                    # Используем среднее значение между min и max
                    search_space_defaults[k] = (v.get('min', 0.001) + v.get('max', 0.01)) / 2
                elif isinstance(v, dict) and v.get('type') == 'int':
                    # Используем среднее значение между min и max
                    search_space_defaults[k] = int((v.get('min', 8) + v.get('max', 32)) / 2)
                elif isinstance(v, dict) and v.get('type') == 'categorical':
                    # Используем первый элемент из choices
                    choices = v.get('choices', [])
                    if choices:
                        search_space_defaults[k] = choices[0]
            base_hparams.update(search_space_defaults)
        
        # Обновление параметров, если переданы новые
        if hparams_override:
            override_dict = dict(item.split("=") for item in hparams_override.split(","))
            base_hparams.update(override_dict)
            logging.info(f"Применены новые гиперпараметры: {override_dict}")

        return ",".join([f"{k}={v}" for k, v in base_hparams.items()])

    def start_training(self, hparams_override=None, checkpoint_path=None, parent_run_id=None):
        """
        Запускает процесс обучения train.py с параметрами из конфига.
        """
        if self.process and self.process.poll() is None:
            logging.warning("Процесс обучения уже запущен. Попытка перезапуска.")
            self.stop_training()

        # Формирование команды для запуска
        try:
            # Создаем уникальное имя эксперимента для полноценного обучения
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            experiment_name = f"tacotron2_production_{timestamp}"
            
            res_conf = self.config['resources']
            logging.info(f"resources config: {res_conf}")
            
            output_dir = os.path.join(res_conf['checkpointing']['path'], experiment_name)
            logging.info(f"output_dir: {output_dir}, type: {type(output_dir)}")
            
            if not isinstance(output_dir, str):
                raise TypeError(f"output_dir должен быть строкой, получен {type(output_dir)}: {output_dir}")
            
            log_dir = os.path.join(output_dir, 'logs')
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            
            logging.info(f"Создан эксперимент: {experiment_name}")
            
        except Exception as e:
            logging.error(f"Ошибка при формировании путей: {e}")
            raise

        hparams_str = self._construct_hparams_string(hparams_override)

        cmd = [
            'python', 'train.py',
            '--output_directory', output_dir,
            '--log_directory', log_dir,
            f'--hparams={hparams_str}'
        ]
        
        if parent_run_id:
            cmd.extend(['--parent_run_id', parent_run_id])
            logging.info(f"Процесс будет залогирован в существующий MLflow run: {parent_run_id}")

        # Автоматический поиск и продолжение с последнего чекпоинта
        if not checkpoint_path and self.config.get('training', {}).get('continue_from_checkpoint', True):
            # Ищем последний чекпоинт в output директории
            checkpoint_path = self.find_latest_checkpoint('output')
            
        if checkpoint_path:
            cmd.extend(['--checkpoint_path', checkpoint_path])
            logging.info(f"Обучение будет продолжено с чекпоинта: {checkpoint_path}")
        else:
            logging.info("Начинаем обучение с нуля")

        logging.info(f"Запуск команды: {' '.join(cmd)}")
        
        # Устанавливаем переменные окружения для MLflow
        env = os.environ.copy()
        if parent_run_id:
            env['MLFLOW_RUN_ID'] = parent_run_id
            logging.info(f"Установлена переменная окружения MLFLOW_RUN_ID={parent_run_id}")

        # Запуск процесса
        self.process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding='utf-8', 
            bufsize=1,
            env=env
        )
        logging.info(f"Процесс обучения запущен с PID: {self.process.pid}")

        # Создаем и запускаем поток для мониторинга вывода
        monitor_thread = threading.Thread(target=self._monitor_output, args=(self.process,))
        monitor_thread.daemon = True
        monitor_thread.start()

        return self.process

    def _monitor_output(self, process):
        """Читает вывод процесса в реальном времени и логирует его."""
        for line in iter(process.stdout.readline, ''):
            logging.info(f"[train.py] {line.strip()}")
        process.stdout.close()
        return_code = process.wait()
        logging.info(f"Процесс train.py завершился с кодом {return_code}")

    def stop_training(self):
        """
        Корректно останавливает процесс обучения.
        """
        if self.process and self.process.poll() is None:
            logging.info(f"Отправка сигнала SIGINT процессу {self.process.pid} для корректного завершения.")
            self.process.send_signal(subprocess.signal.SIGINT)
            try:
                self.process.wait(timeout=60)
                logging.info("Процесс обучения успешно остановлен.")
            except subprocess.TimeoutExpired:
                logging.warning("Процесс не остановился за 60 секунд. Принудительное завершение (SIGKILL).")
                self.process.kill()
        else:
            logging.info("Процесс обучения не был запущен или уже завершился.")
        self.process = None

    def find_latest_checkpoint(self, base_path):
        """
        Находит последний чекпоинт во всех экспериментах
        """
        if not os.path.isdir(base_path):
            logging.warning(f"Директория для поиска чекпоинтов не найдена: {base_path}")
            return None
            
        latest_checkpoint = None
        latest_time = 0
        
        # Проходим по всем поддиректориям в output
        for experiment_dir in os.listdir(base_path):
            experiment_path = os.path.join(base_path, experiment_dir)
            if not os.path.isdir(experiment_path):
                continue
                
            # Ищем чекпоинты в этом эксперименте
            checkpoints = [f for f in os.listdir(experiment_path) if f.startswith('checkpoint_')]
            
            for checkpoint in checkpoints:
                checkpoint_path = os.path.join(experiment_path, checkpoint)
                checkpoint_time = os.path.getmtime(checkpoint_path)
                
                if checkpoint_time > latest_time:
                    latest_time = checkpoint_time
                    latest_checkpoint = checkpoint_path
        
        if latest_checkpoint:
            logging.info(f"Найден последний чекпоинт: {latest_checkpoint}")
        else:
            logging.info("Чекпоинты не найдены")
            
        return latest_checkpoint

    def find_best_checkpoint(self, path):
        """
        Находит лучший (или последний) чекпоинт в директории.
        """
        if not os.path.isdir(path):
            logging.warning(f"Директория для поиска чекпоинтов не найдена: {path}")
            return None
            
        # TODO: Реализовать логику поиска лучшего чекпоинта по метрикам.
        # Пока что ищет последний по номеру итерации.
        checkpoints = [f for f in os.listdir(path) if f.startswith('checkpoint_') and not f.endswith('.pt')]
        if not checkpoints:
            return None
        
        # Сортировка по номеру итерации
        checkpoints.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
        latest_checkpoint = checkpoints[0]
        
        logging.info(f"Найден последний чекпоинт: {latest_checkpoint}")
        return os.path.join(path, latest_checkpoint)

    def is_running(self):
        """Проверяет, запущен ли процесс обучения."""
        return self.process and self.process.poll() is None
    
    def train_with_params(self, hyperparams):
        """
        Запускает обучение с заданными гиперпараметрами и возвращает метрики
        
        Args:
            hyperparams: Словарь с гиперпараметрами
            
        Returns:
            Словарь с метриками обучения
        """
        try:
            logging.info(f"Запуск обучения с гиперпараметрами: {hyperparams}")
            
            # Преобразуем словарь гиперпараметров в строку
            hparams_str = ",".join([f"{k}={v}" for k, v in hyperparams.items()])
            
            # Запускаем процесс обучения
            process = self.start_training(hparams_override=hparams_str)
            
            if not process:
                logging.error("Не удалось запустить процесс обучения")
                return None
            
            # Ждем завершения процесса
            return_code = process.wait()
            
            if return_code == 0:
                logging.info("Обучение завершено успешно")
                # TODO: Здесь должно быть чтение реальных метрик из MLflow
                # Пока возвращаем тестовые метрики
                metrics = {
                    'train_loss': 1.5,
                    'val_loss': 1.8,
                    'learning_rate': hyperparams.get('learning_rate', 0.001),
                    'batch_size': hyperparams.get('batch_size', 16),
                    'status': 'completed'
                }
                logging.info(f"Получены метрики: {metrics}")
                return metrics
            else:
                logging.error(f"Процесс обучения завершился с кодом ошибки: {return_code}")
                return None
                
        except Exception as e:
            logging.error(f"Ошибка при обучении с параметрами {hyperparams}: {e}")
            return None 