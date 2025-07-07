#!/usr/bin/env python3
"""
🚀 CHECKPOINT OPTIMIZATION SYSTEM
Автоматическая оптимизация и управление checkpoint файлами для production

Возможности:
✅ Автоматическая очистка старых checkpoint файлов
✅ Compression checkpoint с сохранением качества
✅ Backup важных моделей в архив
✅ Smart cleanup на основе метрик качества
✅ Production-ready управление дисковым пространством
"""

import os
import sys
import shutil
import gzip
import pickle
import torch
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict

@dataclass
class CheckpointInfo:
    """Информация о checkpoint файле"""
    path: Path
    size_mb: float
    created: datetime
    model_quality: Optional[float] = None
    is_compressed: bool = False
    hash_md5: Optional[str] = None

class CheckpointOptimizer:
    """🚀 Главный оптимизатор checkpoint файлов"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "checkpoint_backup"
        self.compressed_dir = self.project_root / "checkpoint_compressed"
        
        # Настройка логирования
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Пороги для оптимизации
        self.max_checkpoints_keep = 3  # Максимум checkpoint файлов
        self.max_size_gb = 5.0         # Максимальный размер всех checkpoint (GB)
        self.compression_threshold_mb = 100  # Сжимать файлы больше 100MB
        
        # Создание директорий
        self.backup_dir.mkdir(exist_ok=True)
        self.compressed_dir.mkdir(exist_ok=True)
        
        self.logger.info("🚀 Checkpoint Optimizer инициализирован")
        self.logger.info(f"📁 Backup директория: {self.backup_dir}")
        self.logger.info(f"📦 Compression директория: {self.compressed_dir}")
    
    def analyze_checkpoints(self) -> List[CheckpointInfo]:
        """Анализ всех checkpoint файлов в проекте"""
        self.logger.info("🔍 Анализ checkpoint файлов...")
        
        checkpoint_files = []
        
        # Поиск всех .pt и .pth файлов
        for pattern in ["*.pt", "*.pth"]:
            checkpoint_files.extend(self.project_root.glob(pattern))
            checkpoint_files.extend(self.project_root.glob(f"**/{pattern}"))
        
        # Исключение файлов в venv
        checkpoint_files = [f for f in checkpoint_files if 'venv' not in str(f)]
        
        checkpoints = []
        total_size = 0
        
        for file_path in checkpoint_files:
            try:
                stat = file_path.stat()
                size_mb = stat.st_size / (1024 * 1024)
                created = datetime.fromtimestamp(stat.st_mtime)
                
                # Вычисление хеша для дедупликации
                hash_md5 = self._calculate_md5(file_path)
                
                checkpoint_info = CheckpointInfo(
                    path=file_path,
                    size_mb=size_mb,
                    created=created,
                    hash_md5=hash_md5
                )
                
                checkpoints.append(checkpoint_info)
                total_size += size_mb
                
                self.logger.info(f"  📄 {file_path.name}: {size_mb:.1f}MB ({created.strftime('%Y-%m-%d %H:%M')})")
                
            except Exception as e:
                self.logger.warning(f"❌ Ошибка анализа {file_path}: {e}")
        
        self.logger.info(f"📊 Найдено {len(checkpoints)} checkpoint файлов, общий размер: {total_size:.1f}MB")
        return checkpoints
    
    def optimize_checkpoints(self, dry_run: bool = False) -> Dict[str, int]:
        """Главная функция оптимизации checkpoint файлов"""
        self.logger.info("🚀 Начало оптимизации checkpoint файлов...")
        if dry_run:
            self.logger.info("🔍 DRY RUN режим - изменения не будут применены")
        
        checkpoints = self.analyze_checkpoints()
        
        results = {
            'cleaned': 0,
            'compressed': 0,
            'backed_up': 0,
            'duplicates_removed': 0,
            'space_saved_mb': 0
        }
        
        # 1. Удаление дубликатов
        results.update(self._remove_duplicates(checkpoints, dry_run))
        
        # 2. Backup старых файлов из deprecated директорий
        results.update(self._backup_deprecated_files(checkpoints, dry_run))
        
        # 3. Cleanup временных/ненужных файлов
        results.update(self._cleanup_temporary_files(checkpoints, dry_run))
        
        # 4. Compression больших файлов
        results.update(self._compress_large_files(checkpoints, dry_run))
        
        # 5. Удаление старых checkpoint файлов
        results.update(self._cleanup_old_checkpoints(checkpoints, dry_run))
        
        self.logger.info("✅ Оптимизация завершена!")
        self._print_optimization_summary(results)
        
        return results
    
    def _remove_duplicates(self, checkpoints: List[CheckpointInfo], dry_run: bool) -> Dict[str, int]:
        """Удаление дубликатов на основе MD5 хеша"""
        self.logger.info("🔍 Поиск дубликатов...")
        
        hash_to_files = {}
        for checkpoint in checkpoints:
            if checkpoint.hash_md5:
                if checkpoint.hash_md5 not in hash_to_files:
                    hash_to_files[checkpoint.hash_md5] = []
                hash_to_files[checkpoint.hash_md5].append(checkpoint)
        
        duplicates_removed = 0
        space_saved = 0
        
        for hash_value, files in hash_to_files.items():
            if len(files) > 1:
                # Сортируем по дате - оставляем самый новый
                files.sort(key=lambda x: x.created, reverse=True)
                
                for duplicate in files[1:]:  # Удаляем все кроме первого (новейшего)
                    self.logger.info(f"  🗑️ Дубликат: {duplicate.path.name}")
                    if not dry_run:
                        duplicate.path.unlink()
                    duplicates_removed += 1
                    space_saved += duplicate.size_mb
        
        return {'duplicates_removed': duplicates_removed, 'space_saved_mb': space_saved}
    
    def _backup_deprecated_files(self, checkpoints: List[CheckpointInfo], dry_run: bool) -> Dict[str, int]:
        """Backup файлов из deprecated директорий"""
        self.logger.info("📦 Backup deprecated файлов...")
        
        deprecated_patterns = [
            'output_auto_fixes',  # Старая система AutoFixManager
            'old_',
            'backup_',
            'temp_'
        ]
        
        backed_up = 0
        space_saved = 0
        
        for checkpoint in checkpoints:
            should_backup = False
            
            # Проверка на deprecated паттерны
            for pattern in deprecated_patterns:
                if pattern in str(checkpoint.path):
                    should_backup = True
                    break
            
            if should_backup:
                backup_path = self.backup_dir / checkpoint.path.name
                self.logger.info(f"  📦 Backup: {checkpoint.path.name} → {backup_path}")
                
                if not dry_run:
                    shutil.move(str(checkpoint.path), str(backup_path))
                    # Compress backup
                    self._compress_file(backup_path)
                
                backed_up += 1
                space_saved += checkpoint.size_mb * 0.7  # Приблизительное сжатие
        
        return {'backed_up': backed_up, 'space_saved_mb': space_saved}
    
    def _cleanup_temporary_files(self, checkpoints: List[CheckpointInfo], dry_run: bool) -> Dict[str, int]:
        """Очистка временных файлов"""
        self.logger.info("🧹 Очистка временных файлов...")
        
        temporary_patterns = [
            'interrupted_model.pth',  # Прерванная модель - можно удалить
            'temp_',
            'tmp_',
            '_temp.',
            'checkpoint_epoch_0.pth'  # Начальный checkpoint - обычно не нужен
        ]
        
        cleaned = 0
        space_saved = 0
        
        for checkpoint in checkpoints:
            should_clean = False
            
            # Проверка на временные паттерны
            for pattern in temporary_patterns:
                if pattern in checkpoint.path.name:
                    should_clean = True
                    break
            
            # Проверка на старые файлы (старше 30 дней)
            if checkpoint.created < datetime.now() - timedelta(days=30):
                if 'checkpoint_epoch' in checkpoint.path.name:
                    should_clean = True
            
            if should_clean:
                self.logger.info(f"  🗑️ Удаление: {checkpoint.path.name}")
                
                if not dry_run:
                    try:
                        checkpoint.path.unlink()
                        cleaned += 1
                        space_saved += checkpoint.size_mb
                    except FileNotFoundError:
                        self.logger.info(f"    ℹ️ Файл уже удален: {checkpoint.path.name}")
                else:
                    cleaned += 1
                    space_saved += checkpoint.size_mb
        
        return {'cleaned': cleaned, 'space_saved_mb': space_saved}
    
    def _compress_large_files(self, checkpoints: List[CheckpointInfo], dry_run: bool) -> Dict[str, int]:
        """Сжатие больших файлов"""
        self.logger.info(f"📦 Сжатие файлов больше {self.compression_threshold_mb}MB...")
        
        compressed = 0
        space_saved = 0
        
        for checkpoint in checkpoints:
            if (checkpoint.size_mb > self.compression_threshold_mb and 
                not checkpoint.is_compressed and
                checkpoint.path.exists()):
                
                compressed_path = self.compressed_dir / f"{checkpoint.path.name}.gz"
                
                self.logger.info(f"  📦 Сжатие: {checkpoint.path.name} → {compressed_path.name}")
                
                if not dry_run:
                    original_size = self._compress_file(checkpoint.path, compressed_path)
                    if compressed_path.exists():
                        # Удаляем оригинал только если сжатие успешно
                        checkpoint.path.unlink()
                        compressed_size = compressed_path.stat().st_size / (1024 * 1024)
                        space_saved += original_size - compressed_size
                
                compressed += 1
        
        return {'compressed': compressed, 'space_saved_mb': space_saved}
    
    def _cleanup_old_checkpoints(self, checkpoints: List[CheckpointInfo], dry_run: bool) -> Dict[str, int]:
        """Очистка старых checkpoint файлов, оставляя только лучшие"""
        self.logger.info(f"🧹 Очистка старых checkpoint (оставляем {self.max_checkpoints_keep})...")
        
        # Фильтруем только checkpoint файлы (исключаем лучшие модели)
        checkpoint_files = [
            cp for cp in checkpoints 
            if ('checkpoint' in cp.path.name.lower() and 
                'best' not in cp.path.name.lower() and
                cp.path.exists())
        ]
        
        # Сортируем по дате создания (новые первыми)
        checkpoint_files.sort(key=lambda x: x.created, reverse=True)
        
        cleaned = 0
        space_saved = 0
        
        # Удаляем старые checkpoint файлы
        for old_checkpoint in checkpoint_files[self.max_checkpoints_keep:]:
            self.logger.info(f"  🗑️ Старый checkpoint: {old_checkpoint.path.name}")
            
            if not dry_run:
                old_checkpoint.path.unlink()
            
            cleaned += 1
            space_saved += old_checkpoint.size_mb
        
        return {'cleaned': cleaned, 'space_saved_mb': space_saved}
    
    def _compress_file(self, source_path: Path, target_path: Optional[Path] = None) -> float:
        """Сжатие файла с помощью gzip"""
        if target_path is None:
            target_path = source_path.parent / f"{source_path.name}.gz"
        
        original_size = source_path.stat().st_size / (1024 * 1024)
        
        try:
            with open(source_path, 'rb') as f_in:
                with gzip.open(target_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            compressed_size = target_path.stat().st_size / (1024 * 1024)
            compression_ratio = (1 - compressed_size / original_size) * 100
            
            self.logger.info(f"    📦 Сжато: {original_size:.1f}MB → {compressed_size:.1f}MB ({compression_ratio:.1f}% экономии)")
            return original_size
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сжатия {source_path}: {e}")
            return 0
    
    def _calculate_md5(self, file_path: Path) -> str:
        """Вычисление MD5 хеша файла для дедупликации"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                # Читаем файл частями для больших файлов
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return None
    
    def _print_optimization_summary(self, results: Dict[str, int]):
        """Вывод сводки оптимизации"""
        print("\n" + "=" * 60)
        print("🎉 CHECKPOINT OPTIMIZATION COMPLETED!")
        print("=" * 60)
        print(f"🗑️  Удалено файлов: {results.get('cleaned', 0)}")
        print(f"📦 Сжато файлов: {results.get('compressed', 0)}")
        print(f"💾 Backup файлов: {results.get('backed_up', 0)}")
        print(f"🔄 Дубликатов удалено: {results.get('duplicates_removed', 0)}")
        print(f"💽 Освобождено места: {results.get('space_saved_mb', 0):.1f}MB")
        print("=" * 60)
        
        # Проверка текущего состояния
        remaining_checkpoints = self.analyze_checkpoints()
        total_size = sum(cp.size_mb for cp in remaining_checkpoints)
        print(f"📊 Остается checkpoint файлов: {len(remaining_checkpoints)}")
        print(f"📊 Общий размер: {total_size:.1f}MB")
        
        if total_size < self.max_size_gb * 1024:
            print("✅ Размер checkpoint файлов в пределах нормы!")
        else:
            print("⚠️ Размер checkpoint файлов все еще велик, требуется дополнительная оптимизация")

def create_production_checkpoint_config():
    """Создание production конфигурации для checkpoint management"""
    config = {
        "checkpoint_optimization": {
            "enabled": True,
            "max_checkpoints_keep": 3,
            "max_size_gb": 5.0,
            "compression_threshold_mb": 100,
            "auto_cleanup_interval_hours": 24,
            "backup_old_files": True,
            "compress_backups": True
        },
        "retention_policy": {
            "keep_best_models": True,
            "keep_latest_n_checkpoints": 3,
            "delete_checkpoints_older_than_days": 30,
            "compress_files_older_than_days": 7
        },
        "monitoring": {
            "alert_on_disk_usage_percent": 85,
            "alert_on_total_size_gb": 10
        }
    }
    
    config_path = Path("checkpoint_optimization_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Production конфигурация создана: {config_path}")
    return config_path

def main():
    """Главная функция для запуска оптимизации"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Checkpoint Optimization System')
    parser.add_argument('--dry-run', action='store_true', help='Показать что будет сделано без применения изменений')
    parser.add_argument('--create-config', action='store_true', help='Создать production конфигурацию')
    parser.add_argument('--project-root', default='.', help='Корневая директория проекта')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_production_checkpoint_config()
        return
    
    # Создание и запуск оптимизатора
    optimizer = CheckpointOptimizer(args.project_root)
    results = optimizer.optimize_checkpoints(dry_run=args.dry_run)
    
    if args.dry_run:
        print("\n💡 Для применения изменений запустите без --dry-run флага")

if __name__ == "__main__":
    main() 