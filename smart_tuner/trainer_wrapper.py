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

    def start_training(self, hparams_override=None, checkpoint_path=None, run_name_prefix="proactive_run"):
        """
        Запускает train.py с заданными параметрами.
        Возвращает процесс, MLflow run_id и пути к директориям.
        """
        run_name = f"{run_name_prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        
        output_dir = os.path.join(self.config.get('output_dir', 'output'), run_name)
        log_dir = os.path.join(output_dir, "logs")
        checkpoint_dir = os.path.join(output_dir, "checkpoint")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Завершаем предыдущий run если он активен
        if mlflow.active_run():
            mlflow.end_run()
            
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
        if isinstance(hparams_override, dict):
            hparams_dict = hparams_override.copy()
        elif hasattr(hparams_override, 'values') and callable(getattr(hparams_override, 'values')):
            # Если это HParams object, получаем все атрибуты
            try:
                hparams_dict = vars(hparams_override)
            except:
                hparams_dict = {}
        else:
            hparams_dict = {}

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

    def train_with_params(self, hyperparams: dict, trial=None, **kwargs):
        """
        Упрощенная функция обучения с заданными гиперпараметрами.
        Работает как для Optuna, так и для одиночного обучения.
        """
        logging.info(f"🧪 Запуск обучения с параметрами: {hyperparams}")
        
        # Определяем префикс для названия
        name_prefix = "trial" if trial else "single_training"
        
        # Запускаем обучение
        process, run_id, output_dir, log_dir = self.start_training(
            hparams_override=hyperparams,
            checkpoint_path=None,
            run_name_prefix=name_prefix
        )
        
        if not process or not run_id:
            logging.error("Не удалось запустить процесс обучения")
            return {"val_loss": float('inf'), "error": "failed_to_start"}
            
        # Мониторинг процесса обучения
        # ------------------------------
        # ⚙️ Конфигурируемые параметры мониторинга
        training_cfg = self.config.get('training', {})
        check_interval = training_cfg.get('metrics_poll_interval', 30)  # сек

        final_metrics = {}
        last_step = 0

        # Клиент для обращения к MLflow
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        def _fetch_latest_mlflow_metrics(rid: str):
            """Возвращает словарь с последними значениями метрик из MLflow."""
            try:
                metric_keys = [
                    "training.loss", "validation.loss", "grad_norm",
                    "attention_alignment_score", "gate_accuracy", "mel_quality_score",
                    "training.step", "validation.step"
                ]
                latest = {}
                for key in metric_keys:
                    try:
                        hist = client.get_metric_history(rid, key)
                    except Exception:
                        hist = []
                    if hist:
                        # MLflow не гарантирует порядок, сортируем по step
                        last_m = max(hist, key=lambda m: m.step)
                        latest[key] = (last_m.value, last_m.step)
                if not latest:
                    return None
                # Конвертируем в формат, дружелюбный к Optimizer/EarlyStopController
                converted = {}
                mapping = {
                    "training.loss": "train_loss",
                    "validation.loss": "val_loss",
                    "grad_norm": "grad_norm",
                    "attention_alignment_score": "attention_alignment_score",
                    "gate_accuracy": "gate_accuracy",
                    "mel_quality_score": "mel_quality_score",
                }
                for ml_name, (val, step) in latest.items():
                    if ml_name in mapping:
                        converted[mapping[ml_name]] = val
                    # Сохраняем step для train/val
                    if ml_name in ("validation.step", "training.step"):
                        converted["step"] = step
                return converted
            except Exception as e:
                logging.debug(f"Ошибка получения метрик из MLflow: {e}")
                return None
        
        try:
            # Ждем завершения процесса с периодическим мониторингом
            while process.poll() is None:
                time.sleep(check_interval)

                real_metrics = _fetch_latest_mlflow_metrics(run_id)
                if not real_metrics:
                    logging.debug("Нет новых метрик в MLflow (пока). Продолжаем ожидание...")
                    continue

                current_step = real_metrics.get("step", last_step)

                # Отчитываемся в Optuna если это trial
                if trial and current_step != last_step:
                    composite_score = self._calculate_composite_score(real_metrics)
                    try:
                        from smart_tuner.optimization_engine import OptimizationEngine
                        opt_engine = OptimizationEngine()
                        opt_engine.report_intermediate_value(trial, current_step, composite_score, real_metrics)
                    except Exception as e:
                        logging.warning(f"Ошибка отчета в Optuna: {e}")

                final_metrics = real_metrics
                last_step = current_step
                
                # ------------------------------
                # 🛑 DEMO-MODE BREAK (удалено)
                # Ранее обучение насильно прерывалось после 200 псевдо-шагов,
                # что приводило к одинаковой продолжительности этапов (~2.5 мин).
                # Теперь эта логика заменена на конфигурируемый предел,
                # который по-умолчанию отключён. Это обеспечивает гибкость —
                # обучение продолжается до естественного завершения train.py
                # либо до срабатывания EarlyStopController.
                max_demo_steps = self.config.get('training', {}).get('max_demo_steps')
                if max_demo_steps is not None and current_step > max_demo_steps:
                    logging.info(
                        f"⏹ Demo-mode: достигнут лимит {max_demo_steps} шагов, прерываем обучение"
                    )
                    break
                
            # Ждем завершения процесса
            if process.poll() is None:
                logging.info("Останавливаем процесс по таймауту")
                self.stop_training()
                
        except Exception as e:
            logging.error(f"Ошибка во время обучения: {e}")
            self.stop_training()
            return {"val_loss": float('inf'), "error": str(e)}
        finally:
            # Убеждаемся, что процесс остановлен
            if process.poll() is None:
                self.stop_training()
        
        # Возвращаем финальные метрики
        if final_metrics:
            logging.info(f"✅ Обучение завершено. Финальные метрики: {final_metrics}")
            return final_metrics
        else:
            logging.warning("❌ Обучение завершено без метрик")
            return {"val_loss": float('inf'), "error": "no_metrics"}

    def _calculate_composite_score(self, metrics: dict) -> float:
        """Вычисляет композитную оценку для Optuna"""
        val_loss = metrics.get('val_loss', 10.0)
        attention_score = metrics.get('attention_alignment_score', 0.0)
        gate_accuracy = metrics.get('gate_accuracy', 0.0)
        mel_quality = metrics.get('mel_quality_score', 0.0)
        
        # Композитная функция (чем меньше, тем лучше)
        composite = (
            val_loss * 0.4 +
            (1 - attention_score) * 0.3 +
            (1 - gate_accuracy) * 0.2 +
            (1 - mel_quality) * 0.1
        )
        
        return composite
    
    def _convert_metrics_for_optuna(self, raw_metrics: dict) -> dict:
        """
        Преобразует метрики из MetricsStore в формат для EarlyStopController.
        Аналогично методу в SmartTunerMain, но адаптирован для TrainerWrapper.
        """
        if not raw_metrics:
            return None
            
        metrics_mapping = {
            'training.loss': 'train_loss',
            'validation.loss': 'val_loss', 
            'grad_norm': 'grad_norm'
        }
        
        converted = {}
        for mlflow_name, advisor_name in metrics_mapping.items():
            if mlflow_name in raw_metrics:
                value = raw_metrics[mlflow_name]
                if isinstance(value, list) and len(value) > 0:
                    converted[advisor_name] = float(value[-1])
                elif isinstance(value, (int, float)):
                    converted[advisor_name] = float(value)
        
        # Проверяем наличие необходимых метрик
        required_metrics = ['train_loss', 'val_loss', 'grad_norm']
        if all(metric in converted for metric in required_metrics):
            return converted
        else:
            missing = [m for m in required_metrics if m not in converted]
            logging.debug(f"Не хватает метрик для EarlyStopController: {missing}")
            return None 