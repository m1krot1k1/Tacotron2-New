import mlflow
import time
import logging
from smart_tuner.metrics_store import MetricsStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - (LogWatcher) - %(message)s')

class LogWatcher:
    """
    Отслеживает MLflow run, собирает метрики и передает их в MetricsStore.
    Также обнаруживает "зависшие" процессы обучения.
    """
    def __init__(self, metrics_store, tracking_uri, stall_threshold_seconds=600):
        if not isinstance(metrics_store, MetricsStore):
            raise TypeError("metrics_store должен быть экземпляром класса MetricsStore")
        self.metrics_store = metrics_store
        self.tracking_uri = tracking_uri
        self.mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
        self.current_run_id = None
        self.last_step_processed = -1
        self.last_metric_time = None
        self.stall_threshold = stall_threshold_seconds
        logging.info(f"LogWatcher инициализирован. URI: {self.tracking_uri}. Stall threshold: {self.stall_threshold}s")

    def set_run_id(self, run_id):
        """
        Устанавливает ID эксперимента (run), за которым нужно следить.
        """
        self.current_run_id = run_id
        self.last_step_processed = -1 # Сбрасываем счетчик при смене run
        self.last_metric_time = time.time() # Устанавливаем таймер при старте отслеживания
        logging.info(f"LogWatcher теперь отслеживает run_id: {self.current_run_id}")

    def is_stalled(self):
        """
        Проверяет, не "завис" ли процесс обучения.
        Возвращает True, если с момента получения последней метрики прошло слишком много времени.
        """
        if self.last_metric_time is None:
            return False # Еще не начали отслеживать
        
        time_since_last_metric = time.time() - self.last_metric_time
        if time_since_last_metric > self.stall_threshold:
            logging.warning(f"Обнаружено зависание! Нет новых метрик уже {time_since_last_metric:.0f} секунд (порог: {self.stall_threshold}с).")
            return True
        return False

    def check_for_new_metrics(self):
        """
        Проверяет наличие новых метрик в отслеживаемом run
        и добавляет их в MetricsStore.
        """
        if not self.current_run_id:
            logging.warning("Run ID не установлен. Проверка метрик невозможна.")
            return False, 0 # Возвращаем кортеж (статус, кол-во)

        try:
            run = self.mlflow_client.get_run(self.current_run_id)
            if run.info.status in ["FINISHED", "FAILED", "KILLED"]:
                logging.info(f"Run {self.current_run_id} завершен со статусом {run.info.status}.")
                # Можно добавить финальную проверку метрик
                new_metrics_count = self._fetch_and_store_metrics()
                return False, new_metrics_count # Сигнализируем, что отслеживание можно прекратить

            new_metrics_count = self._fetch_and_store_metrics()
            return True, new_metrics_count # Отслеживание продолжается

        except Exception as e:
            logging.error(f"Ошибка при проверке метрик для run_id {self.current_run_id}: {e}")
            return False, 0

    def _fetch_and_store_metrics(self):
        """
        Внутренний метод для получения и сохранения метрик.
        Возвращает количество новых обработанных шагов.
        """
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
            return 0

        # Преобразуем в формат для MetricsStore
        max_step = self.last_step_processed
        processed_data = {}
        
        # Собираем данные по шагам
        steps = set()
        for metrics in all_metrics_history.values():
            for m in metrics:
                steps.add(m['step'])
        
        if not steps:
            return 0
            
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
        
        # Если были новые метрики, обновляем таймер
        num_new_steps = len(sorted_steps)
        if num_new_steps > 0:
            self.last_metric_time = time.time()
            logging.info(f"Обработаны метрики до шага {self.last_step_processed}. Новых записей: {num_new_steps}.")
        
        return num_new_steps

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
            continue_watching, _ = self.check_for_new_metrics()
            if not continue_watching:
                logging.info("Отслеживание прекращено.")
                break
            
            if self.is_stalled():
                logging.error("Обнаружено зависание! Прерываем отслеживание.")
                # В реальной системе здесь должен быть сигнал для TrainerWrapper
                break

            time.sleep(poll_interval) 