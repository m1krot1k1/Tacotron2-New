import mlflow
import time
import logging
from smart_tuner.metrics_store import MetricsStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - (LogWatcher) - %(message)s')

class LogWatcher:
    """
    Отслеживает MLflow run, собирает метрики и передает их в MetricsStore.
    """
    def __init__(self, metrics_store, tracking_uri):
        if not isinstance(metrics_store, MetricsStore):
            raise TypeError("metrics_store должен быть экземпляром класса MetricsStore")
        self.metrics_store = metrics_store
        self.tracking_uri = tracking_uri
        self.mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
        self.current_run_id = None
        self.last_step_processed = -1
        logging.info(f"LogWatcher инициализирован. URI: {self.tracking_uri}")

    def set_run_id(self, run_id):
        """
        Устанавливает ID эксперимента (run), за которым нужно следить.
        """
        self.current_run_id = run_id
        self.last_step_processed = -1 # Сбрасываем счетчик при смене run
        logging.info(f"LogWatcher теперь отслеживает run_id: {self.current_run_id}")

    def check_for_new_metrics(self):
        """
        Проверяет наличие новых метрик в отслеживаемом run
        и добавляет их в MetricsStore.
        """
        if not self.current_run_id:
            logging.warning("Run ID не установлен. Проверка метрик невозможна.")
            return False

        try:
            run = self.mlflow_client.get_run(self.current_run_id)
            if run.info.status in ["FINISHED", "FAILED", "KILLED"]:
                logging.info(f"Run {self.current_run_id} завершен со статусом {run.info.status}.")
                # Можно добавить финальную проверку метрик
                self._fetch_and_store_metrics()
                return False # Сигнализируем, что отслеживание можно прекратить

            self._fetch_and_store_metrics()
            return True # Отслеживание продолжается

        except Exception as e:
            logging.error(f"Ошибка при проверке метрик для run_id {self.current_run_id}: {e}")
            return False

    def _fetch_and_store_metrics(self):
        """Внутренний метод для получения и сохранения метрик."""
        # Получаем все метрики для run'а
        # В MLflow нет простого способа получить метрики "после" определенного шага,
        # поэтому мы забираем всю историю и фильтруем у себя.
        all_metrics_history = {}
        run = self.mlflow_client.get_run(self.current_run_id)
        
        for metric_key in run.data.metrics.keys():
            history = self.mlflow_client.get_metric_history(self.current_run_id, metric_key)
            for metric in history:
                if metric.step > self.last_step_processed:
                    if metric_key not in all_metrics_history:
                        all_metrics_history[metric_key] = []
                    all_metrics_history[metric_key].append({'step': metric.step, 'value': metric.value})

        if not all_metrics_history:
            return

        # Преобразуем в формат для MetricsStore
        max_step = self.last_step_processed
        processed_data = {}
        
        # Собираем данные по шагам
        steps = set()
        for metrics in all_metrics_history.values():
            for m in metrics:
                steps.add(m['step'])
        
        if not steps:
            return
            
        sorted_steps = sorted(list(steps))
        
        processed_data['step'] = sorted_steps
        
        for metric_key, metrics in all_metrics_history.items():
            metric_values = []
            step_value_map = {m['step']: m['value'] for m in metrics}
            for step in sorted_steps:
                metric_values.append(step_value_map.get(step, None))
            processed_data[metric_key] = metric_values

        self.metrics_store.add_metrics(processed_data)
        new_max_step = max(steps)
        self.last_step_processed = new_max_step
        
        logging.info(f"Обработаны метрики до шага {self.last_step_processed}. Новых записей: {len(sorted_steps)}.")


    def watch(self, poll_interval=10):
        """
        Запускает бесконечный цикл отслеживания с заданным интервалом.
        (Для демонстрации/тестирования)
        """
        if not self.current_run_id:
            logging.error("Невозможно начать отслеживание: run_id не установлен.")
            return

        logging.info(f"Начинаю отслеживание run {self.current_run_id} с интервалом {poll_interval} сек.")
        while True:
            if not self.check_for_new_metrics():
                logging.info("Отслеживание прекращено.")
                break
            time.sleep(poll_interval) 