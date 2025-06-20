"""
Model Registry для Smart Tuner V2
Управление версиями и хранение лучших моделей
"""

import yaml
import logging
import shutil
import json
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import hashlib
import os

class ModelRegistry:
    """
    Реестр моделей для управления версиями и хранения лучших результатов
    Автоматически сохраняет лучшие модели и ведет историю экспериментов
    """
    
    def __init__(self, config_path: str = "smart_tuner/config.yaml"):
        """
        Инициализация реестра моделей
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logger()
        
        # Настройки реестра
        self.registry_path = Path(self.config.get('model_registry', {}).get('path', 'smart_tuner/models'))
        self.max_models = self.config.get('model_registry', {}).get('max_models', 5)
        self.best_model_name = self.config.get('model_registry', {}).get('best_model_name', 'best_model.pt')
        
        # Создание директорий
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / 'registry.json'
        
        # Загрузка существующих метаданных
        self.models_metadata = self._load_metadata()
        
        self.logger.info(f"Реестр моделей инициализирован: {self.registry_path}")
        
    def _load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации из YAML файла"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Файл конфигурации {self.config_path} не найден")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Ошибка парсинга YAML: {e}")
            raise
            
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger('ModelRegistry')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Загрузка метаданных реестра"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Ошибка загрузки метаданных: {e}")
                
        return {
            'models': [],
            'best_model': None,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
    def _save_metadata(self):
        """Сохранение метаданных реестра"""
        self.models_metadata['last_updated'] = datetime.now().isoformat()
        
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.models_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения метаданных: {e}")
            
    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Вычисление хеша файла модели
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            MD5 хеш файла
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Ошибка вычисления хеша: {e}")
            return ""
            
    def register_model(self, model_path: str, metrics: Dict[str, float], 
                      hyperparameters: Dict[str, Any], 
                      experiment_name: str = None) -> str:
        """
        Регистрация новой модели в реестре
        
        Args:
            model_path: Путь к файлу модели
            metrics: Метрики модели
            hyperparameters: Гиперпараметры обучения
            experiment_name: Имя эксперимента
            
        Returns:
            ID зарегистрированной модели
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
            
        # Генерация ID модели
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_id = f"model_{timestamp}"
        
        # Создание директории для модели
        model_dir = self.registry_path / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Копирование файла модели
        target_model_path = model_dir / model_path.name
        shutil.copy2(model_path, target_model_path)
        
        # Вычисление хеша
        file_hash = self._calculate_file_hash(target_model_path)
        
        # Создание метаданных модели
        model_metadata = {
            'id': model_id,
            'original_path': str(model_path),
            'registry_path': str(target_model_path),
            'file_hash': file_hash,
            'file_size': target_model_path.stat().st_size,
            'metrics': metrics,
            'hyperparameters': hyperparameters,
            'experiment_name': experiment_name or f"experiment_{timestamp}",
            'created_at': datetime.now().isoformat(),
            'is_best': False
        }
        
        # Сохранение метаданных модели
        model_metadata_file = model_dir / 'metadata.json'
        with open(model_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(model_metadata, f, indent=2, ensure_ascii=False)
            
        # Добавление в реестр
        self.models_metadata['models'].append(model_metadata)
        
        # Проверка на лучшую модель
        self._update_best_model(model_metadata)
        
        # Очистка старых моделей
        self._cleanup_old_models()
        
        # Сохранение метаданных реестра
        self._save_metadata()
        
        self.logger.info(f"Модель зарегистрирована: {model_id}")
        return model_id
        
    def _update_best_model(self, new_model: Dict[str, Any]):
        """
        Обновление лучшей модели
        
        Args:
            new_model: Метаданные новой модели
        """
        # Критерий для определения лучшей модели
        primary_metric = self.config.get('model_registry', {}).get('primary_metric', 'val_loss')
        minimize = self.config.get('model_registry', {}).get('minimize_metric', True)
        
        if primary_metric not in new_model['metrics']:
            self.logger.warning(f"Метрика {primary_metric} не найдена в модели {new_model['id']}")
            return
            
        new_metric_value = new_model['metrics'][primary_metric]
        current_best = self.models_metadata.get('best_model')
        
        is_better = False
        
        if current_best is None:
            is_better = True
        else:
            current_best_value = current_best['metrics'].get(primary_metric)
            if current_best_value is not None:
                if minimize:
                    is_better = new_metric_value < current_best_value
                else:
                    is_better = new_metric_value > current_best_value
                    
        if is_better:
            # Обновление флагов
            for model in self.models_metadata['models']:
                model['is_best'] = False
                
            new_model['is_best'] = True
            self.models_metadata['best_model'] = new_model.copy()
            
            # Копирование лучшей модели
            self._copy_best_model(new_model)
            
            self.logger.info(
                f"Новая лучшая модель: {new_model['id']} "
                f"({primary_metric}: {new_metric_value:.4f})"
            )
            
    def _copy_best_model(self, best_model: Dict[str, Any]):
        """
        Копирование лучшей модели в корень реестра
        
        Args:
            best_model: Метаданные лучшей модели
        """
        source_path = Path(best_model['registry_path'])
        target_path = self.registry_path / self.best_model_name
        
        try:
            shutil.copy2(source_path, target_path)
            self.logger.info(f"Лучшая модель скопирована: {target_path}")
        except Exception as e:
            self.logger.error(f"Ошибка копирования лучшей модели: {e}")
            
    def _cleanup_old_models(self):
        """Очистка старых моделей, превышающих лимит"""
        if len(self.models_metadata['models']) <= self.max_models:
            return
            
        # Сортировка по дате создания (старые первыми)
        sorted_models = sorted(
            self.models_metadata['models'],
            key=lambda x: x['created_at']
        )
        
        # Удаление старых моделей (кроме лучшей)
        models_to_remove = []
        for model in sorted_models[:-self.max_models]:
            if not model.get('is_best', False):
                models_to_remove.append(model)
                
        for model in models_to_remove:
            self._remove_model(model['id'])
            
    def _remove_model(self, model_id: str):
        """
        Удаление модели из реестра
        
        Args:
            model_id: ID модели для удаления
        """
        # Поиск модели
        model_to_remove = None
        for i, model in enumerate(self.models_metadata['models']):
            if model['id'] == model_id:
                model_to_remove = model
                break
                
        if model_to_remove is None:
            self.logger.warning(f"Модель {model_id} не найдена для удаления")
            return
            
        # Удаление файлов
        model_dir = self.registry_path / model_id
        if model_dir.exists():
            try:
                shutil.rmtree(model_dir)
                self.logger.info(f"Удалена директория модели: {model_dir}")
            except Exception as e:
                self.logger.error(f"Ошибка удаления директории {model_dir}: {e}")
                
        # Удаление из метаданных
        self.models_metadata['models'] = [
            m for m in self.models_metadata['models'] 
            if m['id'] != model_id
        ]
        
        self.logger.info(f"Модель {model_id} удалена из реестра")
        
    def get_best_model_path(self) -> Optional[str]:
        """
        Получение пути к лучшей модели
        
        Returns:
            Путь к лучшей модели или None
        """
        best_model_path = self.registry_path / self.best_model_name
        
        if best_model_path.exists():
            return str(best_model_path)
        else:
            return None
            
    def get_best_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Получение информации о лучшей модели
        
        Returns:
            Метаданные лучшей модели или None
        """
        return self.models_metadata.get('best_model')
        
    def get_all_models(self) -> List[Dict[str, Any]]:
        """
        Получение списка всех моделей
        
        Returns:
            Список метаданных всех моделей
        """
        return self.models_metadata['models'].copy()
        
    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение модели по ID
        
        Args:
            model_id: ID модели
            
        Returns:
            Метаданные модели или None
        """
        for model in self.models_metadata['models']:
            if model['id'] == model_id:
                return model.copy()
        return None
        
    def export_model(self, model_id: str, export_path: str) -> bool:
        """
        Экспорт модели в указанное место
        
        Args:
            model_id: ID модели
            export_path: Путь для экспорта
            
        Returns:
            True, если экспорт успешен
        """
        model = self.get_model_by_id(model_id)
        if model is None:
            self.logger.error(f"Модель {model_id} не найдена")
            return False
            
        source_path = Path(model['registry_path'])
        target_path = Path(export_path)
        
        try:
            # Создание директории если нужно
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Копирование модели
            shutil.copy2(source_path, target_path)
            
            # Копирование метаданных
            metadata_target = target_path.parent / f"{target_path.stem}_metadata.json"
            with open(metadata_target, 'w', encoding='utf-8') as f:
                json.dump(model, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Модель {model_id} экспортирована в {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка экспорта модели {model_id}: {e}")
            return False
            
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Получение статистики реестра
        
        Returns:
            Статистика реестра
        """
        models = self.models_metadata['models']
        
        if not models:
            return {
                'total_models': 0,
                'best_model': None,
                'registry_size_mb': 0,
                'created_at': self.models_metadata.get('created_at'),
                'last_updated': self.models_metadata.get('last_updated')
            }
            
        # Вычисление размера реестра
        total_size = 0
        for model in models:
            total_size += model.get('file_size', 0)
            
        # Поиск лучшей модели
        best_model = self.models_metadata.get('best_model')
        
        stats = {
            'total_models': len(models),
            'best_model': {
                'id': best_model['id'],
                'metrics': best_model['metrics'],
                'experiment_name': best_model['experiment_name']
            } if best_model else None,
            'registry_size_mb': total_size / (1024 * 1024),
            'created_at': self.models_metadata.get('created_at'),
            'last_updated': self.models_metadata.get('last_updated'),
            'experiments': list(set(m.get('experiment_name', 'Unknown') for m in models))
        }
        
        return stats
        
    def cleanup_registry(self, keep_best: bool = True):
        """
        Полная очистка реестра
        
        Args:
            keep_best: Сохранить ли лучшую модель
        """
        models_to_keep = []
        
        if keep_best and self.models_metadata.get('best_model'):
            best_model_id = self.models_metadata['best_model']['id']
            models_to_keep = [m for m in self.models_metadata['models'] 
                            if m['id'] == best_model_id]
            
        # Удаление всех остальных моделей
        for model in self.models_metadata['models']:
            if model['id'] not in [m['id'] for m in models_to_keep]:
                self._remove_model(model['id'])
                
        # Обновление метаданных
        self.models_metadata['models'] = models_to_keep
        if not keep_best:
            self.models_metadata['best_model'] = None
            
        self._save_metadata()
        
        self.logger.info(f"Реестр очищен, сохранено моделей: {len(models_to_keep)}") 