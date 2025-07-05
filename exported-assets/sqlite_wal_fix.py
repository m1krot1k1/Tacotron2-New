
# ФАЙЛ: optimization_engine.py
# ПРОБЛЕМА: SQLite database locked error

import sqlite3
import time
import optuna
from sqlalchemy import create_engine, pool

class RobustOptimizationEngine:
    def __init__(self):
        self.setup_sqlite_wal()

    def setup_sqlite_wal(self):
        '''
        Настройка SQLite для работы в WAL режиме с retry механизмом
        '''
        storage_path = "smart_tuner/optuna_studies.db"

        # Предварительная настройка базы данных
        conn = None
        try:
            conn = sqlite3.connect(storage_path, timeout=30)

            # Включаем WAL режим для лучшей конкурентности
            conn.execute('PRAGMA journal_mode=WAL;')
            conn.execute('PRAGMA synchronous=NORMAL;')
            conn.execute('PRAGMA cache_size=10000;')
            conn.execute('PRAGMA temp_store=MEMORY;')
            conn.execute('PRAGMA mmap_size=268435456;')  # 256MB
            conn.execute('PRAGMA busy_timeout=30000;')   # 30 секунд

            conn.commit()
            print("✅ SQLite настроен в WAL режиме")

        except sqlite3.Error as e:
            print(f"❌ Ошибка настройки SQLite: {e}")
        finally:
            if conn:
                conn.close()

    def create_study_with_retry(self, study_name=None, max_retries=5):
        '''
        Создание Optuna study с retry механизмом
        '''
        storage_url = f"sqlite:///smart_tuner/optuna_studies.db"

        for attempt in range(max_retries):
            try:
                # Создаем engine с connection pooling
                engine = create_engine(
                    storage_url,
                    poolclass=pool.NullPool,  # Отключаем pooling
                    connect_args={
                        "timeout": 30,
                        "check_same_thread": False,
                        "isolation_level": None  # Autocommit режим
                    }
                )

                self.study = optuna.create_study(
                    study_name=study_name,
                    storage=storage_url,
                    direction="minimize",
                    load_if_exists=True
                )

                print(f"✅ Optuna study создан: {study_name}")
                return self.study

            except Exception as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"⚠️ База заблокирована, ожидание {wait_time:.1f}с...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Ошибка создания study: {e}")
                    raise

        raise Exception("Не удалось создать study после всех попыток")
