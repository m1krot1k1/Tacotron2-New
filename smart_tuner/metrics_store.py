import pandas as pd
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - (MetricsStore) - %(message)s')

class MetricsStore:
    """Хранит историю метрик обучения в памяти."""
    
    def __init__(self):
        self.metrics_history: List[Dict] = []
        logging.info("MetricsStore инициализирован.")

    def add_metrics(self, metrics: Dict):
        """Добавляет новый набор метрик в историю."""
        if isinstance(metrics, dict) and metrics:
            self.metrics_history.append(metrics)
            logging.debug(f"Добавлены метрики: {metrics}")

    def get_history_df(self) -> Optional[pd.DataFrame]:
        """Возвращает всю историю метрик в виде pandas DataFrame."""
        if not self.metrics_history:
            return None
        return pd.DataFrame(self.metrics_history)

    def get_latest_metrics(self) -> Dict:
        """Возвращает последний добавленный набор метрик."""
        if not self.metrics_history:
            return {}
        return self.metrics_history[-1]

    def reset(self):
        """Очищает историю метрик."""
        self.metrics_history = []
        logging.info("MetricsStore был сброшен.")
        
    def __repr__(self):
        return f"<MetricsStore with {len(self.metrics_history)} records>" 