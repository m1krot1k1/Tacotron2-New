#!/usr/bin/env python3
"""
🗑️ UNIVERSAL LOG CLEANUP MANAGER для Tacotron2-New
Централизованная очистка ВСЕХ логов и дашбордов при запуске обучения

Очищает:
✅ TensorBoard логи (tensorboard_logs/)
✅ MLflow логи (mlruns/)
✅ Unified Logging System (unified_logs/)
✅ Дашборды БД (monitoring.db, dashboard_metrics.db и т.д.)
✅ Smart Tuner логи и БД
✅ Все .log файлы
✅ Графики (plots/)
✅ Checkpoint backup'ы (опционально)
"""

import os
import shutil
import sqlite3
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict

@dataclass
class CleanupStats:
    """Статистика очистки"""
    directories_cleaned: int = 0
    files_removed: int = 0
    databases_cleaned: int = 0
    space_freed_mb: float = 0.0
    cleanup_time_seconds: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class LogCleanupManager:
    """
    🗑️ Универсальный менеджер очистки логов для всех дашбордов
    """
    
    def __init__(self, project_root: str = ".", keep_last_days: int = 7, dry_run: bool = False):
        self.project_root = Path(project_root)
        self.keep_last_days = keep_last_days
        self.dry_run = dry_run
        
        # Настройка логирования
        self.logger = self._setup_logger()
        
        # Конфигурация очистки
        self.cleanup_config = self._get_cleanup_config()
        
        self.logger.info(f"🗑️ LogCleanupManager инициализирован")
        self.logger.info(f"📁 Проект: {self.project_root}")
        self.logger.info(f"📅 Хранить последние: {self.keep_last_days} дней")
        self.logger.info(f"🔍 Dry run: {'Да' if self.dry_run else 'Нет'}")
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger("LogCleanupManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - 🗑️ %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _get_cleanup_config(self) -> Dict:
        """Конфигурация очистки всех компонентов"""
        return {
            # Директории логов
            'log_directories': [
                'tensorboard_logs',
                'mlruns',
                'unified_logs',
                'logs',
                'logs_auto_fixes',
                'smart_tuner/logs',
                'plots',
                'output'  # TensorBoard выходные данные
            ],
            
            # Базы данных дашбордов
            'dashboard_databases': [
                'monitoring.db',
                'dashboard_metrics.db',
                'rollback_states.db',
                'demo_monitoring.db',
                'demo_risk_assessment.db',
                'risk_assessments.db',
                'smart_tuner/optuna_studies.db',
                'smart_tuner/epoch_optimizer_history.db', 
                'smart_tuner/quality_control_history.db',
                'smart_tuner/advisor_kb.db',
                'meta_learning_data/meta_learning_data.db'
            ],
            
            # Лог файлы
            'log_files': [
                '*.log',
                'ultimate_training.log',
                'mlflow.log',
                'tensorboard.log',
                'production_dashboard.log',
                'smart_tuner_web.log',
                'risk_assessment_demo.log'
            ],
            
            # Checkpoint backup (опционально)
            'checkpoint_backups': [
                'checkpoint_backup',
                'checkpoint_compressed',
                'rollback_checkpoints'
            ],
            
            # Временные файлы
            'temp_files': [
                'emergency_restart_*.json',
                '*.backup',
                'temp_*',
                'tmp_*'
            ]
        }
    
    def cleanup_all(self, include_checkpoints: bool = False) -> CleanupStats:
        """
        🗑️ Полная очистка всех логов и дашбордов
        
        Args:
            include_checkpoints: Включить очистку checkpoint backup'ов
            
        Returns:
            Статистика очистки
        """
        start_time = time.time()
        stats = CleanupStats()
        
        self.logger.info("🚀 Начинаю полную очистку логов и дашбордов...")
        if self.dry_run:
            self.logger.info("🔍 DRY RUN режим - изменения не будут применены")
        
        try:
            # 1. Очистка директорий логов
            dir_stats = self._cleanup_log_directories()
            stats.directories_cleaned += dir_stats[0]
            stats.files_removed += dir_stats[1]
            stats.space_freed_mb += dir_stats[2]
            
            # 2. Очистка баз данных дашбордов
            db_stats = self._cleanup_dashboard_databases()
            stats.databases_cleaned += db_stats[0]
            stats.space_freed_mb += db_stats[1]
            
            # 3. Очистка лог файлов
            file_stats = self._cleanup_log_files()
            stats.files_removed += file_stats[0]
            stats.space_freed_mb += file_stats[1]
            
            # 4. Очистка временных файлов
            temp_stats = self._cleanup_temp_files()
            stats.files_removed += temp_stats[0]
            stats.space_freed_mb += temp_stats[1]
            
            # 5. Очистка checkpoint backup'ов (опционально)
            if include_checkpoints:
                checkpoint_stats = self._cleanup_checkpoint_backups()
                stats.directories_cleaned += checkpoint_stats[0]
                stats.files_removed += checkpoint_stats[1]
                stats.space_freed_mb += checkpoint_stats[2]
            
            stats.cleanup_time_seconds = time.time() - start_time
            
            # Отчет
            self._print_cleanup_report(stats)
            self._save_cleanup_report(stats)
            
            return stats
            
        except Exception as e:
            error_msg = f"Критическая ошибка очистки: {e}"
            self.logger.error(error_msg)
            stats.errors.append(error_msg)
            return stats
    
    def _cleanup_log_directories(self) -> Tuple[int, int, float]:
        """Очистка директорий логов"""
        self.logger.info("📁 Очистка директорий логов...")
        
        dirs_cleaned = 0
        files_removed = 0
        space_freed = 0.0
        cutoff_date = datetime.now() - timedelta(days=self.keep_last_days)
        
        for dir_name in self.cleanup_config['log_directories']:
            dir_path = self.project_root / dir_name
            
            if not dir_path.exists():
                continue
                
            self.logger.info(f"  🧹 Обрабатываю директорию: {dir_path}")
            
            try:
                # Подсчет размера до очистки
                dir_size_before = self._get_directory_size(dir_path)
                
                if dir_name in ['tensorboard_logs', 'mlruns']:
                    # Для TensorBoard и MLflow сохраняем последние runs
                    removed_count = self._cleanup_timestamped_directories(dir_path, cutoff_date)
                    files_removed += removed_count
                else:
                    # Полная очистка остальных директорий
                    if not self.dry_run:
                        shutil.rmtree(dir_path)
                        dir_path.mkdir(exist_ok=True)
                    
                    files_removed += len(list(dir_path.rglob('*'))) if dir_path.exists() else 0
                
                # Подсчет освобожденного места
                dir_size_after = self._get_directory_size(dir_path) if dir_path.exists() else 0
                space_freed += (dir_size_before - dir_size_after) / (1024 * 1024)  # MB
                
                dirs_cleaned += 1
                self.logger.info(f"    ✅ Очищено: {dir_name}")
                
            except Exception as e:
                error_msg = f"Ошибка очистки {dir_path}: {e}"
                self.logger.error(f"    ❌ {error_msg}")
        
        return dirs_cleaned, files_removed, space_freed
    
    def _cleanup_dashboard_databases(self) -> Tuple[int, float]:
        """Очистка баз данных дашбордов"""
        self.logger.info("🗄️ Очистка баз данных дашбордов...")
        
        databases_cleaned = 0
        space_freed = 0.0
        
        for db_path_str in self.cleanup_config['dashboard_databases']:
            db_path = self.project_root / db_path_str
            
            if not db_path.exists():
                continue
            
            self.logger.info(f"  🧹 Очищаю БД: {db_path}")
            
            try:
                # Размер до очистки
                size_before = db_path.stat().st_size / (1024 * 1024)  # MB
                
                if not self.dry_run:
                    # Очистка SQLite БД (оставляем структуру, удаляем данные)
                    self._clean_sqlite_database(db_path)
                
                # Размер после очистки  
                size_after = db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0
                space_freed += (size_before - size_after)
                
                databases_cleaned += 1
                self.logger.info(f"    ✅ БД очищена: {db_path.name}")
                
            except Exception as e:
                error_msg = f"Ошибка очистки БД {db_path}: {e}"
                self.logger.error(f"    ❌ {error_msg}")
        
        return databases_cleaned, space_freed
    
    def _cleanup_log_files(self) -> Tuple[int, float]:
        """Очистка лог файлов"""
        self.logger.info("📄 Очистка лог файлов...")
        
        files_removed = 0
        space_freed = 0.0
        
        for pattern in self.cleanup_config['log_files']:
            log_files = list(self.project_root.glob(pattern))
            
            for log_file in log_files:
                if not log_file.is_file():
                    continue
                
                try:
                    size_mb = log_file.stat().st_size / (1024 * 1024)
                    
                    if not self.dry_run:
                        log_file.unlink()
                    
                    space_freed += size_mb
                    files_removed += 1
                    self.logger.info(f"  🗑️ Удален лог: {log_file.name}")
                    
                except Exception as e:
                    error_msg = f"Ошибка удаления {log_file}: {e}"
                    self.logger.error(f"  ❌ {error_msg}")
        
        return files_removed, space_freed
    
    def _cleanup_temp_files(self) -> Tuple[int, float]:
        """Очистка временных файлов"""
        self.logger.info("🧹 Очистка временных файлов...")
        
        files_removed = 0
        space_freed = 0.0
        
        for pattern in self.cleanup_config['temp_files']:
            temp_files = list(self.project_root.glob(pattern))
            
            for temp_file in temp_files:
                try:
                    size_mb = temp_file.stat().st_size / (1024 * 1024)
                    
                    if not self.dry_run:
                        temp_file.unlink()
                    
                    space_freed += size_mb
                    files_removed += 1
                    self.logger.info(f"  🗑️ Удален временный файл: {temp_file.name}")
                    
                except Exception as e:
                    error_msg = f"Ошибка удаления {temp_file}: {e}"
                    self.logger.error(f"  ❌ {error_msg}")
        
        return files_removed, space_freed
    
    def _cleanup_checkpoint_backups(self) -> Tuple[int, int, float]:
        """Очистка checkpoint backup'ов"""
        self.logger.info("💾 Очистка checkpoint backup'ов...")
        
        dirs_cleaned = 0
        files_removed = 0
        space_freed = 0.0
        
        for backup_dir_name in self.cleanup_config['checkpoint_backups']:
            backup_path = self.project_root / backup_dir_name
            
            if not backup_path.exists():
                continue
            
            try:
                size_before = self._get_directory_size(backup_path)
                file_count = len(list(backup_path.rglob('*')))
                
                if not self.dry_run:
                    shutil.rmtree(backup_path)
                    backup_path.mkdir(exist_ok=True)
                
                space_freed += size_before / (1024 * 1024)  # MB
                files_removed += file_count
                dirs_cleaned += 1
                
                self.logger.info(f"  🗑️ Очищен backup: {backup_dir_name}")
                
            except Exception as e:
                error_msg = f"Ошибка очистки backup {backup_path}: {e}"
                self.logger.error(f"  ❌ {error_msg}")
        
        return dirs_cleaned, files_removed, space_freed
    
    def _cleanup_timestamped_directories(self, parent_dir: Path, cutoff_date: datetime) -> int:
        """Очистка директорий с timestamp (оставляем последние)"""
        removed_count = 0
        
        try:
            subdirs = [d for d in parent_dir.iterdir() if d.is_dir()]
            
            for subdir in subdirs:
                try:
                    # Проверяем дату модификации
                    mod_time = datetime.fromtimestamp(subdir.stat().st_mtime)
                    
                    if mod_time < cutoff_date:
                        if not self.dry_run:
                            shutil.rmtree(subdir)
                        removed_count += 1
                        self.logger.info(f"    🗑️ Удалена старая директория: {subdir.name}")
                        
                except Exception as e:
                    self.logger.error(f"    ❌ Ошибка обработки {subdir}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Ошибка обработки {parent_dir}: {e}")
        
        return removed_count
    
    def _clean_sqlite_database(self, db_path: Path):
        """Очистка SQLite базы данных (сохранение структуры)"""
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Получаем список всех таблиц
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            # Очищаем данные из всех таблиц
            for table in tables:
                table_name = table[0]
                if table_name != 'sqlite_sequence':  # Системная таблица
                    cursor.execute(f"DELETE FROM {table_name}")
            
            # Сброс последовательностей
            cursor.execute("DELETE FROM sqlite_sequence")
            
            conn.commit()
            conn.close()
            
            # Сжатие БД
            conn = sqlite3.connect(str(db_path))
            conn.execute("VACUUM")
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Ошибка очистки SQLite БД {db_path}: {e}")
    
    def _get_directory_size(self, dir_path: Path) -> int:
        """Получение размера директории в байтах"""
        try:
            return sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
        except:
            return 0
    
    def _print_cleanup_report(self, stats: CleanupStats):
        """Печать отчета об очистке"""
        self.logger.info("=" * 60)
        self.logger.info("📊 ОТЧЕТ ОБ ОЧИСТКЕ ЛОГОВ И ДАШБОРДОВ")
        self.logger.info("=" * 60)
        self.logger.info(f"📁 Директорий очищено: {stats.directories_cleaned}")
        self.logger.info(f"📄 Файлов удалено: {stats.files_removed}")
        self.logger.info(f"🗄️ БД очищено: {stats.databases_cleaned}")
        self.logger.info(f"💾 Освобождено места: {stats.space_freed_mb:.2f} MB")
        self.logger.info(f"⏱️ Время очистки: {stats.cleanup_time_seconds:.2f} сек")
        
        if stats.errors:
            self.logger.info(f"❌ Ошибок: {len(stats.errors)}")
            for error in stats.errors:
                self.logger.error(f"  • {error}")
        else:
            self.logger.info("✅ Очистка завершена без ошибок")
        
        self.logger.info("=" * 60)
    
    def _save_cleanup_report(self, stats: CleanupStats):
        """Сохранение отчета об очистке"""
        try:
            report_file = self.project_root / f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'stats': asdict(stats),
                'config': self.cleanup_config,
                'dry_run': self.dry_run
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📄 Отчет сохранен: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения отчета: {e}")

def main():
    """Точка входа для прямого запуска"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal Log Cleanup Manager')
    parser.add_argument('--project-root', default='.', help='Корневая директория проекта')
    parser.add_argument('--keep-days', type=int, default=7, help='Количество дней для хранения логов')
    parser.add_argument('--include-checkpoints', action='store_true', help='Включить очистку checkpoint backup')
    parser.add_argument('--dry-run', action='store_true', help='Режим симуляции без изменений')
    
    args = parser.parse_args()
    
    # Создание и запуск менеджера очистки
    cleanup_manager = LogCleanupManager(
        project_root=args.project_root,
        keep_last_days=args.keep_days,
        dry_run=args.dry_run
    )
    
    # Выполнение очистки
    stats = cleanup_manager.cleanup_all(include_checkpoints=args.include_checkpoints)
    
    # Возврат кода завершения
    return 0 if not stats.errors else 1

if __name__ == "__main__":
    exit(main()) 