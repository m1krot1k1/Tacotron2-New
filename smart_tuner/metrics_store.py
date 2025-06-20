import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - (MetricsStore) - %(message)s')

class MetricsStore:
    """
    Простое хранилище метрик в памяти.
    Использует pandas DataFrame для удобной работы с временными рядами метрик.
    """
    def __init__(self):
        self.history = pd.DataFrame()
        logging.info("MetricsStore инициализирован.")

    def add_metrics(self, new_metrics_dict):
        """
        Добавляет новую порцию метрик в историю.
        
        Args:
            new_metrics_dict (dict): Словарь, где ключ - название метрики, 
                                     а значение - список новых значений.
                                     Например: {'step': [100], 'val_loss': [0.45]}
        """
        if not new_metrics_dict or 'step' not in new_metrics_dict:
            logging.warning("Попытка добавить пустые метрики или метрики без 'step'.")
            return

        new_df = pd.DataFrame(new_metrics_dict)
        
        if self.history.empty:
            self.history = new_df
        else:
            # Используем concat для добавления новых строк
            self.history = pd.concat([self.history, new_df], ignore_index=True)
        
        # Удаляем дубликаты по шагу, оставляя последнее значение
        self.history.drop_duplicates(subset=['step'], keep='last', inplace=True)
        
        logging.debug(f"Метрики добавлены. Текущий размер истории: {len(self.history)} записей.")

    def get_last_n_metrics(self, n):
        """Возвращает последние N записанных метрик."""
        return self.history.tail(n)

    def get_metric_series(self, metric_name):
        """Возвращает временной ряд для конкретной метрики."""
        if metric_name in self.history:
            return self.history[['step', metric_name]].dropna()
        else:
            logging.warning(f"Метрика '{metric_name}' не найдена в истории.")
            return pd.DataFrame()

    def get_latest_metric(self, metric_name):
        """Возвращает последнее значение для конкретной метрики."""
        series = self.get_metric_series(metric_name)
        if not series.empty:
            return series[metric_name].iloc[-1]
        return None

    def __repr__(self):
        return f"<MetricsStore with {len(self.history)} records>" 