#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 ADVANCED MODEL CHECKPOINTING SYSTEM
Интеллектуальная система сохранения checkpoint'ов с автоматическим восстановлением

Решает критические проблемы из exported-assets:
- Отсутствие автовосстановления при сбоях NaN/Inf
- Простые критерии сохранения (только validation loss)  
- Нет защиты от критических ошибок
- Фрагментированность checkpoint систем
- Отсутствие управления дисковым пространством

Компоненты:
1. IntelligentCheckpointManager - умное управление checkpoint'ами
2. MultiCriteriaModelSelector - выбор лучших моделей по множественным критериям
3. AutoRecoverySystem - автоматическое восстановление при сбоях
4. CheckpointHealthAnalyzer - анализ качества checkpoint'ов
"""

import os
import time
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import torch
import torch.nn as nn

# Интеграция с existing системами
try:
    from unified_logging_system import UnifiedLoggingSystem
    UNIFIED_LOGGING_AVAILABLE = True
except ImportError:
    UNIFIED_LOGGING_AVAILABLE = False

class CheckpointQuality(Enum):
    """Уровни качества checkpoint'ов"""
    EXCELLENT = "excellent"    # Лучшие модели для production
    GOOD = "good"             # Хорошие модели для тестирования
    ACCEPTABLE = "acceptable"  # Приемлемые модели
    POOR = "poor"             # Плохие модели (кандидаты на удаление)
    CRITICAL = "critical"     # Критические ошибки (немедленное удаление)

@dataclass
class CheckpointMetrics:
    """Комплексные метрики checkpoint'а"""
    # Основные метрики обучения
    epoch: int
    global_step: int
    validation_loss: float
    training_loss: float
    learning_rate: float
    
    # TTS-специфичные метрики
    attention_diagonality: float = 0.5
    gate_accuracy: float = 0.8
    mel_reconstruction_quality: float = 0.7
    attention_stability: float = 0.6
    
    # Метрики стабильности
    gradient_norm: float = 2.0
    gradient_stability: float = 1.0
    loss_trend: float = -0.1
    convergence_score: float = 0.5
    
    # Метаданные
    timestamp: str = ""
    training_time: float = 0.0
    model_size_mb: float = 100.0
    memory_usage_mb: float = 2048.0
    
    # Флаги проблем
    has_nan_weights: bool = False
    has_gradient_explosion: bool = False
    has_attention_collapse: bool = False
    is_stable: bool = True

@dataclass
class CheckpointInfo:
    """Информация о checkpoint'е"""
    path: str
    metrics: CheckpointMetrics
    quality: CheckpointQuality
    health_score: float
    is_best: bool = False
    is_emergency_backup: bool = False
    file_hash: str = ""
    creation_time: float = 0.0

class MultiCriteriaModelSelector:
    """
    Интеллектуальный селектор лучших моделей на основе множественных критериев
    Превосходит простое сравнение по validation loss
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Веса критериев (настроены для TTS задач)
        self.weights = weights or {
            'validation_loss': 0.25,      # Основной критерий
            'attention_quality': 0.20,    # Критично для TTS
            'stability': 0.15,            # Стабильность обучения
            'convergence': 0.15,          # Скорость сходимости  
            'mel_quality': 0.10,          # Качество mel-спектрограмм
            'gradient_health': 0.10,      # Здоровье градиентов
            'gate_accuracy': 0.05         # Точность gate предсказаний
        }
        
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        if UNIFIED_LOGGING_AVAILABLE:
            return UnifiedLoggingSystem().register_component("MultiCriteriaSelector")
        else:
            logger = logging.getLogger("MultiCriteriaSelector") 
            logger.setLevel(logging.INFO)
            return logger
    
    def calculate_model_score(self, metrics: CheckpointMetrics) -> float:
        """
        Вычисление комплексного score модели на основе множественных критериев
        
        Returns:
            float: Комплексный score (0.0 - 1.0, где 1.0 = идеальная модель)
        """
        try:
            # Нормализация компонентов score
            components = {}
            
            # 1. Validation Loss (инвертированный и нормализованный)
            val_loss_norm = max(0.0, min(1.0, (10.0 - metrics.validation_loss) / 10.0))
            components['validation_loss'] = val_loss_norm
            
            # 2. Attention Quality (диагональность)
            components['attention_quality'] = min(1.0, metrics.attention_diagonality / 0.8)
            
            # 3. Stability Score
            stability_score = self._calculate_stability_score(metrics)
            components['stability'] = stability_score
            
            # 4. Convergence Score  
            components['convergence'] = min(1.0, max(0.0, metrics.convergence_score))
            
            # 5. Mel Quality
            components['mel_quality'] = min(1.0, max(0.0, metrics.mel_reconstruction_quality))
            
            # 6. Gradient Health
            grad_health = 1.0 - min(1.0, abs(metrics.gradient_norm - 2.0) / 10.0)
            components['gradient_health'] = grad_health
            
            # 7. Gate Accuracy
            components['gate_accuracy'] = min(1.0, max(0.0, metrics.gate_accuracy))
            
            # Взвешенная сумма компонентов
            total_score = sum(
                self.weights[key] * value 
                for key, value in components.items()
            )
            
            # Применение штрафов за критические проблемы
            if metrics.has_nan_weights:
                total_score *= 0.1  # Критический штраф
            if metrics.has_gradient_explosion:
                total_score *= 0.3  # Серьезный штраф
            if metrics.has_attention_collapse:
                total_score *= 0.2  # Серьезный штраф
                
            return max(0.0, min(1.0, total_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating model score: {e}")
            return 0.0
    
    def _calculate_stability_score(self, metrics: CheckpointMetrics) -> float:
        """Вычисление score стабильности обучения"""
        stability_factors = []
        
        # Стабильность градиентов
        grad_stability = max(0.0, 1.0 - metrics.gradient_stability / 5.0)
        stability_factors.append(grad_stability)
        
        # Тренд loss (отрицательный = хорошо)
        loss_trend_score = max(0.0, 1.0 + metrics.loss_trend) if metrics.loss_trend < 0 else 0.5
        stability_factors.append(loss_trend_score)
        
        # Attention стабильность
        stability_factors.append(min(1.0, metrics.attention_stability))
        
        # Общий флаг стабильности
        stability_factors.append(1.0 if metrics.is_stable else 0.3)
        
        return np.mean(stability_factors)
    
    def rank_checkpoints(self, checkpoints: List[CheckpointInfo]) -> List[CheckpointInfo]:
        """Ранжирование checkpoint'ов по качеству"""
        if not checkpoints:
            return []
        
        # Вычисление score для каждого checkpoint'а
        for checkpoint in checkpoints:
            score = self.calculate_model_score(checkpoint.metrics)
            checkpoint.health_score = score
            
        # Сортировка по score (убывание)
        ranked = sorted(checkpoints, key=lambda x: x.health_score, reverse=True)
        
        # Маркировка лучшего checkpoint'а
        if ranked:
            ranked[0].is_best = True
            
        self.logger.info(f"Ranked {len(checkpoints)} checkpoints. Best score: {ranked[0].health_score:.4f}")
        return ranked

class CheckpointHealthAnalyzer:
    """Анализатор здоровья checkpoint'ов для детекции проблем"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        if UNIFIED_LOGGING_AVAILABLE:
            return UnifiedLoggingSystem().register_component("CheckpointHealthAnalyzer")
        else:
            logger = logging.getLogger("CheckpointHealthAnalyzer") 
            logger.setLevel(logging.INFO)
            return logger
    
    def analyze_checkpoint_health(self, checkpoint_path: str) -> Tuple[CheckpointQuality, List[str]]:
        """
        Анализ здоровья checkpoint'а
        
        Returns:
            Tuple[CheckpointQuality, List[str]]: Качество и список обнаруженных проблем
        """
        issues = []
        
        try:
            # Загрузка checkpoint'а
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Проверка базовой структуры
            required_keys = ['model_state_dict', 'epoch', 'global_step']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                issues.append(f"Отсутствуют ключевые поля: {missing_keys}")
                return CheckpointQuality.CRITICAL, issues
            
            # Анализ весов модели
            state_dict = checkpoint['model_state_dict']
            weight_issues = self._analyze_model_weights(state_dict)
            issues.extend(weight_issues)
            
            # Анализ метрик обучения
            metrics_issues = self._analyze_training_metrics(checkpoint)
            issues.extend(metrics_issues)
            
            # Определение качества на основе проблем
            quality = self._determine_quality_from_issues(issues)
            
            if issues:
                self.logger.warning(f"Checkpoint health issues found: {len(issues)} problems")
            else:
                self.logger.info("Checkpoint health: No critical issues detected")
                
            return quality, issues
            
        except Exception as e:
            error_msg = f"Критическая ошибка анализа checkpoint: {e}"
            issues.append(error_msg)
            self.logger.error(error_msg)
            return CheckpointQuality.CRITICAL, issues
    
    def _analyze_model_weights(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        """Анализ весов модели на наличие проблем"""
        issues = []
        
        for name, tensor in state_dict.items():
            try:
                # Проверка на NaN
                if torch.isnan(tensor).any():
                    issues.append(f"NaN веса в слое: {name}")
                
                # Проверка на Inf  
                if torch.isinf(tensor).any():
                    issues.append(f"Inf веса в слое: {name}")
                
                # Проверка на экстремальные значения
                tensor_max = tensor.abs().max().item()
                if tensor_max > 100.0:
                    issues.append(f"Экстремально большие веса в {name}: {tensor_max:.2f}")
                    
                # Проверка на "мертвые" веса
                if tensor_max < 1e-8:
                    issues.append(f"Мертвые веса в {name}: max={tensor_max:.2e}")
                    
            except Exception as e:
                issues.append(f"Ошибка анализа весов {name}: {e}")
                
        return issues
    
    def _analyze_training_metrics(self, checkpoint: Dict[str, Any]) -> List[str]:
        """Анализ метрик обучения"""
        issues = []
        
        # Проверка базовых метрик
        if 'validation_loss' in checkpoint:
            val_loss = checkpoint['validation_loss']
            if val_loss > 50.0:
                issues.append(f"Критически высокий validation loss: {val_loss:.2f}")
            elif val_loss > 20.0:
                issues.append(f"Высокий validation loss: {val_loss:.2f}")
                
        return issues
    
    def _determine_quality_from_issues(self, issues: List[str]) -> CheckpointQuality:
        """Определение качества checkpoint'а на основе обнаруженных проблем"""
        if not issues:
            return CheckpointQuality.EXCELLENT
        
        critical_keywords = ['NaN', 'Inf', 'Критически', 'мертвые']
        serious_keywords = ['Высокий', 'Низкое', 'Экстремально']
        
        critical_count = sum(1 for issue in issues if any(kw in issue for kw in critical_keywords))
        serious_count = sum(1 for issue in issues if any(kw in issue for kw in serious_keywords))
        
        if critical_count > 0:
            return CheckpointQuality.CRITICAL
        elif serious_count >= 3:
            return CheckpointQuality.POOR
        elif serious_count >= 1:
            return CheckpointQuality.ACCEPTABLE
        else:
            return CheckpointQuality.GOOD

class AutoRecoverySystem:
    """Система автоматического восстановления при критических сбоях"""
    
    def __init__(self, checkpoint_manager):
        self.checkpoint_manager = checkpoint_manager
        self.logger = self._setup_logger()
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        
    def _setup_logger(self):
        if UNIFIED_LOGGING_AVAILABLE:
            return UnifiedLoggingSystem().register_component("AutoRecoverySystem")
        else:
            logger = logging.getLogger("AutoRecoverySystem")
            logger.setLevel(logging.INFO)
            return logger
    
    def detect_critical_failure(self, metrics: CheckpointMetrics) -> bool:
        """Детекция критических сбоев в обучении"""
        critical_conditions = []
        
        # 1. NaN/Inf в метриках
        if (not np.isfinite(metrics.validation_loss) or 
            not np.isfinite(metrics.training_loss)):
            critical_conditions.append("NaN/Inf в loss функциях")
        
        # 2. Взрыв градиентов
        if metrics.gradient_norm > 1000.0:
            critical_conditions.append(f"Взрыв градиентов: {metrics.gradient_norm:.2f}")
        
        # 3. Полный коллапс attention
        if metrics.attention_diagonality < 0.01:
            critical_conditions.append(f"Коллапс attention: {metrics.attention_diagonality:.4f}")
        
        # 4. Экстремально высокий loss
        if metrics.validation_loss > 1000.0:
            critical_conditions.append(f"Экстремальный loss: {metrics.validation_loss:.2f}")
        
        if critical_conditions:
            self.logger.error(f"🚨 КРИТИЧЕСКИЙ СБОЙ ОБНАРУЖЕН: {critical_conditions}")
            return True
            
        return False
    
    def attempt_recovery(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> bool:
        """Попытка автоматического восстановления"""
        if self.recovery_attempts >= self.max_recovery_attempts:
            self.logger.error(f"❌ Превышено максимальное количество попыток восстановления")
            return False
        
        self.recovery_attempts += 1
        self.logger.warning(f"🔄 Попытка автовосстановления #{self.recovery_attempts}")
        
        # Поиск последнего хорошего checkpoint'а
        good_checkpoint = self.checkpoint_manager.get_best_checkpoint(
            min_quality=CheckpointQuality.ACCEPTABLE
        )
        
        if not good_checkpoint:
            self.logger.error("❌ Не найден подходящий checkpoint для восстановления")
            return False
        
        try:
            # Загрузка хорошего checkpoint'а
            self.logger.info(f"📁 Загрузка checkpoint для восстановления: {good_checkpoint.path}")
            checkpoint = torch.load(good_checkpoint.path, map_location='cpu', weights_only=False)
            
            # Восстановление состояния модели
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Восстановление оптимизатора с уменьшенным learning rate
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Уменьшение learning rate для стабилизации
                for param_group in optimizer.param_groups:
                    original_lr = param_group['lr']
                    param_group['lr'] = original_lr * 0.1  # Уменьшаем в 10 раз
                    
            self.logger.info("✅ Автовосстановление выполнено успешно")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка автовосстановления: {e}")
            return False

class IntelligentCheckpointManager:
    """Главный менеджер интеллектуального управления checkpoint'ами"""
    
    def __init__(self, 
                 checkpoint_dir: str = "intelligent_checkpoints",
                 max_checkpoints: int = 10,
                 min_save_interval: int = 500):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.min_save_interval = min_save_interval
        self.last_save_step = 0
        
        # Инициализация компонентов
        self.selector = MultiCriteriaModelSelector()
        self.health_analyzer = CheckpointHealthAnalyzer()
        self.auto_recovery = AutoRecoverySystem(self)
        
        # Метаданные checkpoint'ов
        self.checkpoints: List[CheckpointInfo] = []
        self.metadata_file = self.checkpoint_dir / "checkpoints_metadata.json"
        
        # Загрузка существующих метаданных
        self._load_metadata()
        
        self.logger = self._setup_logger()
        self.logger.info(f"🎯 Intelligent Checkpoint Manager инициализирован: {self.checkpoint_dir}")
    
    def _setup_logger(self):
        if UNIFIED_LOGGING_AVAILABLE:
            return UnifiedLoggingSystem().register_component("IntelligentCheckpointManager")
        else:
            logger = logging.getLogger("IntelligentCheckpointManager")
            logger.setLevel(logging.INFO)
            return logger
    
    def save_checkpoint(self, 
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       metrics: CheckpointMetrics,
                       force_save: bool = False,
                       is_emergency: bool = False) -> Optional[str]:
        """Интеллектуальное сохранение checkpoint'а"""
        try:
            current_step = metrics.global_step
            
            # Проверка интервала сохранения
            if not force_save and not is_emergency:
                if current_step - self.last_save_step < self.min_save_interval:
                    return None
            
            # Анализ качества checkpoint'а ПЕРЕД сохранением
            quality = self._predict_checkpoint_quality(metrics)
            
            # Пропуск сохранения плохих checkpoint'ов (кроме emergency)
            if not is_emergency and quality == CheckpointQuality.CRITICAL:
                self.logger.warning(f"⚠️ Пропуск сохранения критически плохого checkpoint")
                return None
            
            # Генерация имени файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if is_emergency:
                filename = f"emergency_checkpoint_step_{current_step}_{timestamp}.pt"
            else:
                filename = f"checkpoint_step_{current_step}_{timestamp}.pt"
                
            checkpoint_path = self.checkpoint_dir / filename
            
            # Создание checkpoint данных
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': metrics.epoch,
                'global_step': metrics.global_step,
                'metrics': asdict(metrics),
                'timestamp': timestamp,
                'torch_version': torch.__version__
            }
            
            # Сохранение checkpoint'а
            torch.save(checkpoint_data, checkpoint_path)
            
            # Создание информации о checkpoint'е
            file_hash = self._calculate_file_hash(checkpoint_path)
            checkpoint_info = CheckpointInfo(
                path=str(checkpoint_path),
                metrics=metrics,
                quality=quality,
                health_score=0.0,  # Будет вычислен позже
                is_emergency_backup=is_emergency,
                file_hash=file_hash,
                creation_time=time.time()
            )
            
            # Добавление в список и пересчет рейтингов
            self.checkpoints.append(checkpoint_info)
            self._update_checkpoint_rankings()
            
            # Управление дисковым пространством
            self._manage_storage()
            
            # Сохранение метаданных
            self._save_metadata()
            
            self.last_save_step = current_step
            
            status_icon = "🚨" if is_emergency else "💾"
            self.logger.info(f"{status_icon} Checkpoint сохранен: {filename} (качество: {quality.value})")
            
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения checkpoint: {e}")
            return None
    
    def get_best_checkpoint(self, 
                           min_quality: CheckpointQuality = CheckpointQuality.ACCEPTABLE) -> Optional[CheckpointInfo]:
            """Получение лучшего checkpoint'а по комплексным критериям"""
            # Фильтрация по качеству
            quality_order = [
                CheckpointQuality.EXCELLENT,
                CheckpointQuality.GOOD, 
                CheckpointQuality.ACCEPTABLE,
                CheckpointQuality.POOR,
                CheckpointQuality.CRITICAL
            ]
            min_quality_idx = quality_order.index(min_quality)
            
            suitable_checkpoints = [
                cp for cp in self.checkpoints
                if quality_order.index(cp.quality) <= min_quality_idx
            ]
            
            if not suitable_checkpoints:
                return None
            
            # Ранжирование и выбор лучшего
            ranked = self.selector.rank_checkpoints(suitable_checkpoints)
            return ranked[0] if ranked else None
    
    def check_and_recover(self, 
                         model: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         current_metrics: CheckpointMetrics) -> bool:
        """Проверка на критические сбои и автоматическое восстановление"""
        if self.auto_recovery.detect_critical_failure(current_metrics):
            # Сохранение экстренного checkpoint'а
            emergency_path = self.save_checkpoint(
                model, optimizer, current_metrics, 
                force_save=True, is_emergency=True
            )
            
            if emergency_path:
                self.logger.info(f"🚨 Экстренный checkpoint сохранен: {emergency_path}")
            
            # Попытка автоматического восстановления
            return self.auto_recovery.attempt_recovery(model, optimizer)
        
        return False
    
    def _predict_checkpoint_quality(self, metrics: CheckpointMetrics) -> CheckpointQuality:
        """Предсказание качества checkpoint'а на основе метрик"""
        score = self.selector.calculate_model_score(metrics)
        
        # Проверка критических проблем
        if (metrics.has_nan_weights or 
            metrics.has_gradient_explosion or 
            not np.isfinite(metrics.validation_loss)):
            return CheckpointQuality.CRITICAL
        
        # Классификация по score
        if score >= 0.8:
            return CheckpointQuality.EXCELLENT
        elif score >= 0.6:
            return CheckpointQuality.GOOD
        elif score >= 0.4:
            return CheckpointQuality.ACCEPTABLE
        elif score >= 0.2:
            return CheckpointQuality.POOR
        else:
            return CheckpointQuality.CRITICAL
    
    def _update_checkpoint_rankings(self):
        """Обновление рейтингов всех checkpoint'ов"""
        self.checkpoints = self.selector.rank_checkpoints(self.checkpoints)
    
    def _manage_storage(self):
        """Управление дисковым пространством"""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Сначала определяем приоритеты сохранения
        # 1. Лучший checkpoint (по рейтингу)
        # 2. Недавние хорошие checkpoint'ы  
        # 3. Emergency backup'ы (только самые свежие)
        
        # Сортировка: сначала лучшие, потом по времени
        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: (
                x.quality == CheckpointQuality.CRITICAL,  # Critical - в конец
                -x.health_score,  # Лучший score - в начало
                -x.creation_time  # Новые - в начало
            )
        )
        
        # Сохраняем checkpoint'ы с учетом приоритетов и строгого лимита
        to_keep = []
        emergency_count = 0
        max_emergency = min(2, self.max_checkpoints - 1)  # Оставляем место для обычных checkpoint'ов
        
        for checkpoint in sorted_checkpoints:
            # Проверяем общий лимит
            if len(to_keep) >= self.max_checkpoints:
                break
                
            # Всегда сохраняем лучший checkpoint (если он не критический)
            if len(to_keep) == 0 and checkpoint.quality != CheckpointQuality.CRITICAL:
                checkpoint.is_best = True
                to_keep.append(checkpoint)
                continue
            
            # Ограничиваем количество emergency backup'ов
            if checkpoint.is_emergency_backup:
                if emergency_count < max_emergency and len(to_keep) < self.max_checkpoints:
                    to_keep.append(checkpoint)
                    emergency_count += 1
                continue
            
            # Добавляем обычные checkpoint'ы до лимита
            if len(to_keep) < self.max_checkpoints:
                to_keep.append(checkpoint)
        
        # Удаляем лишние checkpoint'ы
        to_remove = [cp for cp in self.checkpoints if cp not in to_keep]
        
        for checkpoint in to_remove:
            try:
                # Удаление файла
                if os.path.exists(checkpoint.path):
                    os.remove(checkpoint.path)
                    self.logger.info(f"🗑️ Удален checkpoint: {os.path.basename(checkpoint.path)} (качество: {checkpoint.quality.value})")
                
                # Удаление из списка
                self.checkpoints.remove(checkpoint)
                
            except Exception as e:
                self.logger.error(f"❌ Ошибка удаления checkpoint: {e}")
        
        # Обновляем рейтинги оставшихся checkpoint'ов
        if self.checkpoints:
            self.checkpoints = self.selector.rank_checkpoints(self.checkpoints)
            self.logger.info(f"💾 Сохранено {len(self.checkpoints)}/{self.max_checkpoints} checkpoint'ов")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Вычисление MD5 хеша файла"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def _load_metadata(self):
        """Загрузка метаданных checkpoint'ов"""
        if not self.metadata_file.exists():
            return
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.checkpoints = []
            for item in data.get('checkpoints', []):
                # Восстановление CheckpointMetrics
                metrics_data = item['metrics']
                metrics = CheckpointMetrics(**metrics_data)
                
                # Создание CheckpointInfo
                checkpoint_info = CheckpointInfo(
                    path=item['path'],
                    metrics=metrics,
                    quality=CheckpointQuality(item['quality']),
                    health_score=item['health_score'],
                    is_best=item.get('is_best', False),
                    is_emergency_backup=item.get('is_emergency_backup', False),
                    file_hash=item.get('file_hash', ''),
                    creation_time=item.get('creation_time', 0.0)
                )
                
                # Проверка существования файла
                if os.path.exists(checkpoint_info.path):
                    self.checkpoints.append(checkpoint_info)
            
            self.logger.info(f"📁 Загружено {len(self.checkpoints)} checkpoint'ов")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки метаданных: {e}")
            self.checkpoints = []
    
    def _save_metadata(self):
        """Сохранение метаданных checkpoint'ов"""
        try:
            data = {
                'checkpoints': [],
                'last_updated': datetime.now().isoformat(),
                'manager_version': '1.0'
            }
            
            for checkpoint in self.checkpoints:
                checkpoint_data = {
                    'path': checkpoint.path,
                    'metrics': asdict(checkpoint.metrics),
                    'quality': checkpoint.quality.value,
                    'health_score': checkpoint.health_score,
                    'is_best': checkpoint.is_best,
                    'is_emergency_backup': checkpoint.is_emergency_backup,
                    'file_hash': checkpoint.file_hash,
                    'creation_time': checkpoint.creation_time
                }
                data['checkpoints'].append(checkpoint_data)
            
            # Сохранение в JSON
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения метаданных: {e}")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Получение отчета о состоянии checkpoint'ов"""
        if not self.checkpoints:
            return {
                'total_checkpoints': 0,
                'status': 'empty',
                'message': 'Нет сохраненных checkpoint\'ов'
            }
        
        # Статистика по качеству
        quality_stats = {}
        for quality in CheckpointQuality:
            count = sum(1 for cp in self.checkpoints if cp.quality == quality)
            quality_stats[quality.value] = count
        
        # Лучший checkpoint
        best = self.get_best_checkpoint()
        
        # Общая статистика
        total_size_mb = sum(
            os.path.getsize(cp.path) / (1024 * 1024) 
            for cp in self.checkpoints 
            if os.path.exists(cp.path)
        )
        
        return {
            'total_checkpoints': len(self.checkpoints),
            'quality_distribution': quality_stats,
            'best_checkpoint': {
                'path': best.path if best else None,
                'score': best.health_score if best else None,
                'quality': best.quality.value if best else None
            },
            'total_size_mb': round(total_size_mb, 2),
            'storage_usage': f"{len(self.checkpoints)}/{self.max_checkpoints}",
            'emergency_backups': sum(1 for cp in self.checkpoints if cp.is_emergency_backup),
            'status': 'healthy' if best and best.quality != CheckpointQuality.CRITICAL else 'warning'
        }

# Интеграционная функция для простого использования
def create_checkpoint_manager(checkpoint_dir: str = "intelligent_checkpoints",
                            max_checkpoints: int = 10) -> IntelligentCheckpointManager:
    """Создание настроенного checkpoint manager'а"""
    return IntelligentCheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=max_checkpoints
    )

# Пример использования
if __name__ == "__main__":
    # Демонстрация использования Advanced Model Checkpointing System
    
    # Создание manager'а
    checkpoint_manager = create_checkpoint_manager(
        checkpoint_dir="demo_checkpoints",
        max_checkpoints=5
    )
    
    print("🎯 Advanced Model Checkpointing System демо")
    print(f"📁 Директория checkpoint'ов: {checkpoint_manager.checkpoint_dir}")
    
    # Получение статуса
    status = checkpoint_manager.get_status_report()
    print(f"📊 Текущий статус: {status}")
    
    print("\n✅ Advanced Model Checkpointing System готов к использованию!")
    print("\n🔧 Интеграция с обучением:")
    print("1. Создайте IntelligentCheckpointManager")
    print("2. Используйте save_checkpoint() во время обучения")  
    print("3. Используйте check_and_recover() для автовосстановления")
    print("4. Используйте get_best_checkpoint() для загрузки лучшей модели") 