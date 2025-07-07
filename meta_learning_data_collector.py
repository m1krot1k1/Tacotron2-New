#!/usr/bin/env python3
"""
📚 META-LEARNING DATA COLLECTOR
Система накопления и обогащения обучающих данных для Meta-Learning Engine

Возможности:
✅ Автоматический сбор данных из реальных эпизодов обучения
✅ Синтетическая генерация дополнительных сценариев
✅ Интеллектуальная фильтрация и качественный отбор данных
✅ Аугментация данных для редких случаев
✅ Валидация и верификация качества данных
✅ Экспорт в различные форматы для обучения
"""

import os
import json
import time
import pickle
import numpy as np
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging

# Интеграция с Meta-Learning Engine
try:
    from meta_learning_engine import MetaLearningEngine, TrainingEpisode, LearningStrategy, TrainingPhase
    META_LEARNING_AVAILABLE = True
except ImportError:
    META_LEARNING_AVAILABLE = False
    # Fallback enum
    class TrainingPhase:
        PRE_ALIGNMENT = "pre_alignment"
        ALIGNMENT_LEARNING = "alignment"
        REFINEMENT = "refinement"
        CONVERGENCE = "convergence"

# Интеграция с системами мониторинга
try:
    from unified_logging_system import UnifiedLoggingSystem
    UNIFIED_LOGGING_AVAILABLE = True
except ImportError:
    UNIFIED_LOGGING_AVAILABLE = False

@dataclass
class DataCollectionConfig:
    """Конфигурация сбора данных"""
    min_episodes_target: int = 500  # Минимальная цель для качественного обучения
    max_episodes_storage: int = 2000  # Максимум в хранилище
    quality_threshold: float = 0.7  # Порог качества для включения данных
    diversity_weight: float = 0.3  # Вес разнообразия при отборе
    synthetic_data_ratio: float = 0.4  # Доля синтетических данных
    data_augmentation_enabled: bool = True
    real_time_collection: bool = True
    export_formats: List[str] = None
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ['json', 'pickle', 'sqlite']

@dataclass
class DataQualityMetrics:
    """Метрики качества данных"""
    total_episodes: int
    successful_episodes: int
    failed_episodes: int
    phase_distribution: Dict[str, int]
    strategy_distribution: Dict[str, int]
    quality_score: float
    diversity_score: float
    completeness_score: float
    data_freshness: float  # Насколько свежие данные (0-1)

class SyntheticDataGenerator:
    """🤖 Генератор синтетических данных для обучения"""
    
    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".SyntheticGenerator")
        
        # Параметры генерации для разных сценариев
        self.scenario_templates = {
            'normal_training': {
                'loss_range': (2.0, 8.0),
                'attention_range': (0.4, 0.8),
                'gradient_range': (0.5, 3.0),
                'success_probability': 0.7
            },
            'difficult_convergence': {
                'loss_range': (5.0, 15.0),
                'attention_range': (0.1, 0.4),
                'gradient_range': (1.0, 8.0),
                'success_probability': 0.3
            },
            'fast_learning': {
                'loss_range': (1.0, 4.0),
                'attention_range': (0.6, 0.9),
                'gradient_range': (0.3, 1.5),
                'success_probability': 0.9
            },
            'instability_recovery': {
                'loss_range': (10.0, 50.0),
                'attention_range': (0.05, 0.2),
                'gradient_range': (5.0, 20.0),
                'success_probability': 0.2
            }
        }
    
    def generate_episodes(self, n_episodes: int, scenario_mix: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Генерация синтетических эпизодов обучения"""
        if scenario_mix is None:
            scenario_mix = {
                'normal_training': 0.6,
                'difficult_convergence': 0.2,
                'fast_learning': 0.15,
                'instability_recovery': 0.05
            }
        
        episodes = []
        
        for i in range(n_episodes):
            # Выбор сценария
            scenario = np.random.choice(
                list(scenario_mix.keys()),
                p=list(scenario_mix.values())
            )
            
            episode = self._generate_single_episode(scenario, i)
            episodes.append(episode)
        
        self.logger.info(f"🤖 Сгенерировано {n_episodes} синтетических эпизодов")
        return episodes
    
    def _generate_single_episode(self, scenario: str, episode_id: int) -> Dict[str, Any]:
        """Генерация одного эпизода"""
        template = self.scenario_templates[scenario]
        
        # Базовые метрики
        initial_loss = np.random.uniform(*template['loss_range'])
        initial_attention = np.random.uniform(*template['attention_range'])
        
        # Симуляция прогресса обучения
        success = np.random.random() < template['success_probability']
        
        if success:
            # Улучшение метрик
            final_loss = initial_loss * np.random.uniform(0.3, 0.8)
            final_attention = min(0.95, initial_attention * np.random.uniform(1.5, 3.0))
            improvement_score = np.random.uniform(0.2, 0.8)
        else:
            # Ухудшение или стагнация
            final_loss = initial_loss * np.random.uniform(0.9, 2.0)
            final_attention = initial_attention * np.random.uniform(0.5, 1.1)
            improvement_score = np.random.uniform(-0.3, 0.1)
        
        # Выбор стратегии
        strategies = ['AGGRESSIVE', 'CONSERVATIVE', 'BALANCED', 'ADAPTIVE']
        strategy = np.random.choice(strategies)
        
        # Выбор фазы
        phases = ['PRE_ALIGNMENT', 'ALIGNMENT_LEARNING', 'REFINEMENT', 'CONVERGENCE']
        initial_phase = np.random.choice(phases[:3])  # Финальную фазу не выбираем как начальную
        
        if success:
            final_phase_idx = min(len(phases) - 1, phases.index(initial_phase) + np.random.randint(1, 3))
            final_phase = phases[final_phase_idx]
        else:
            final_phase = initial_phase  # Нет прогресса
        
        # Создание решений
        decisions = self._generate_decisions(scenario, strategy, success)
        
        # Параметры, которые менялись
        parameters_changed = self._generate_parameter_changes(strategy, success)
        
        episode = {
            'episode_id': f"synthetic_{scenario}_{episode_id}",
            'start_time': time.time() - np.random.uniform(3600, 86400),  # Последние 24 часа
            'end_time': time.time() - np.random.uniform(0, 3600),
            'initial_phase': initial_phase,
            'final_phase': final_phase,
            'initial_loss': initial_loss,
            'initial_attention_quality': initial_attention,
            'final_loss': final_loss,
            'final_attention_quality': final_attention,
            'strategy_used': strategy,
            'decisions_made': decisions,
            'parameters_changed': parameters_changed,
            'success': success,
            'improvement_score': improvement_score,
            'convergence_achieved': success and final_phase == 'CONVERGENCE',
            'model_architecture': 'tacotron2',
            'dataset_size': np.random.randint(1000, 10000),
            'total_steps': np.random.randint(500, 5000),
            'scenario_type': scenario,  # Метка для анализа
            'synthetic': True
        }
        
        return episode
    
    def _generate_decisions(self, scenario: str, strategy: str, success: bool) -> List[Dict[str, Any]]:
        """Генерация решений для эпизода"""
        decisions = []
        
        num_decisions = np.random.randint(3, 12)
        
        for i in range(num_decisions):
            decision_types = ['lr_adjustment', 'batch_size_change', 'attention_weight_change', 'optimization_change']
            decision_type = np.random.choice(decision_types)
            
            if decision_type == 'lr_adjustment':
                old_lr = np.random.uniform(1e-5, 1e-3)
                if strategy == 'AGGRESSIVE':
                    new_lr = old_lr * np.random.uniform(1.2, 2.0)
                elif strategy == 'CONSERVATIVE':
                    new_lr = old_lr * np.random.uniform(0.5, 0.9)
                else:
                    new_lr = old_lr * np.random.uniform(0.8, 1.3)
                
                decision = {
                    'timestamp': time.time() - np.random.uniform(0, 3600),
                    'type': decision_type,
                    'old_value': old_lr,
                    'new_value': new_lr,
                    'reason': f'{strategy.lower()}_strategy'
                }
            
            elif decision_type == 'attention_weight_change':
                old_weight = np.random.uniform(1.0, 10.0)
                new_weight = old_weight * np.random.uniform(0.7, 1.5)
                
                decision = {
                    'timestamp': time.time() - np.random.uniform(0, 3600),
                    'type': decision_type,
                    'old_value': old_weight,
                    'new_value': new_weight,
                    'reason': 'attention_quality_improvement'
                }
            
            else:
                decision = {
                    'timestamp': time.time() - np.random.uniform(0, 3600),
                    'type': decision_type,
                    'reason': f'{strategy.lower()}_optimization'
                }
            
            decisions.append(decision)
        
        return decisions
    
    def _generate_parameter_changes(self, strategy: str, success: bool) -> Dict[str, float]:
        """Генерация изменений параметров"""
        changes = {}
        
        # Изменения learning rate
        if np.random.random() < 0.8:
            base_lr = np.random.uniform(1e-5, 1e-3)
            if strategy == 'AGGRESSIVE':
                changes['learning_rate'] = base_lr * np.random.uniform(1.5, 3.0)
            else:
                changes['learning_rate'] = base_lr * np.random.uniform(0.3, 1.2)
        
        # Изменения guided attention weight
        if np.random.random() < 0.6:
            changes['guided_attention_weight'] = np.random.uniform(2.0, 12.0)
        
        # Изменения batch size
        if np.random.random() < 0.4:
            changes['batch_size'] = np.random.choice([16, 32, 64, 128])
        
        return changes

class DataQualityAnalyzer:
    """📊 Анализатор качества данных"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".QualityAnalyzer")
    
    def analyze_data_quality(self, episodes: List[Dict[str, Any]]) -> DataQualityMetrics:
        """Анализ качества накопленных данных"""
        if not episodes:
            return DataQualityMetrics(0, 0, 0, {}, {}, 0.0, 0.0, 0.0, 0.0)
        
        # Базовая статистика
        total_episodes = len(episodes)
        successful_episodes = sum(1 for ep in episodes if ep.get('success', False))
        failed_episodes = total_episodes - successful_episodes
        
        # Распределение по фазам
        phase_distribution = Counter(ep.get('initial_phase', 'unknown') for ep in episodes)
        
        # Распределение по стратегиям
        strategy_distribution = Counter(ep.get('strategy_used', 'unknown') for ep in episodes)
        
        # Расчет показателей качества
        quality_score = self._calculate_quality_score(episodes)
        diversity_score = self._calculate_diversity_score(episodes)
        completeness_score = self._calculate_completeness_score(episodes)
        data_freshness = self._calculate_data_freshness(episodes)
        
        metrics = DataQualityMetrics(
            total_episodes=total_episodes,
            successful_episodes=successful_episodes,
            failed_episodes=failed_episodes,
            phase_distribution=dict(phase_distribution),
            strategy_distribution=dict(strategy_distribution),
            quality_score=quality_score,
            diversity_score=diversity_score,
            completeness_score=completeness_score,
            data_freshness=data_freshness
        )
        
        self.logger.info(f"📊 Анализ качества данных: {total_episodes} эпизодов, качество={quality_score:.2f}")
        return metrics
    
    def _calculate_quality_score(self, episodes: List[Dict[str, Any]]) -> float:
        """Расчет общей оценки качества данных"""
        if not episodes:
            return 0.0
        
        quality_factors = []
        
        # Фактор 1: Соотношение успешных и неудачных эпизодов
        success_rate = sum(ep.get('success', False) for ep in episodes) / len(episodes)
        balanced_success = 1.0 - abs(success_rate - 0.6)  # Оптимум 60% успеха
        quality_factors.append(balanced_success)
        
        # Фактор 2: Полнота данных
        complete_episodes = sum(
            1 for ep in episodes 
            if all(key in ep for key in ['initial_loss', 'final_loss', 'strategy_used', 'decisions_made'])
        )
        completeness = complete_episodes / len(episodes)
        quality_factors.append(completeness)
        
        # Фактор 3: Разнообразие стратегий
        strategies = set(ep.get('strategy_used', '') for ep in episodes)
        strategy_diversity = min(1.0, len(strategies) / 4)  # 4 основные стратегии
        quality_factors.append(strategy_diversity)
        
        # Фактор 4: Реалистичность метрик
        realistic_episodes = sum(
            1 for ep in episodes
            if (0.1 <= ep.get('initial_loss', 0) <= 100 and 
                0.0 <= ep.get('initial_attention_quality', 0) <= 1.0)
        )
        realism = realistic_episodes / len(episodes)
        quality_factors.append(realism)
        
        return np.mean(quality_factors)
    
    def _calculate_diversity_score(self, episodes: List[Dict[str, Any]]) -> float:
        """Расчет оценки разнообразия данных"""
        if not episodes:
            return 0.0
        
        diversity_factors = []
        
        # Разнообразие фаз
        phases = [ep.get('initial_phase', '') for ep in episodes]
        phase_entropy = self._calculate_entropy(phases)
        diversity_factors.append(phase_entropy)
        
        # Разнообразие стратегий
        strategies = [ep.get('strategy_used', '') for ep in episodes]
        strategy_entropy = self._calculate_entropy(strategies)
        diversity_factors.append(strategy_entropy)
        
        # Разнообразие результатов
        improvement_scores = [ep.get('improvement_score', 0) for ep in episodes]
        score_variance = np.var(improvement_scores) if improvement_scores else 0
        normalized_variance = min(1.0, score_variance / 0.25)  # Нормализация
        diversity_factors.append(normalized_variance)
        
        return np.mean(diversity_factors)
    
    def _calculate_completeness_score(self, episodes: List[Dict[str, Any]]) -> float:
        """Расчет полноты данных"""
        if not episodes:
            return 0.0
        
        required_fields = [
            'episode_id', 'initial_loss', 'final_loss', 'initial_attention_quality',
            'final_attention_quality', 'strategy_used', 'success', 'improvement_score'
        ]
        
        complete_episodes = 0
        for episode in episodes:
            if all(field in episode and episode[field] is not None for field in required_fields):
                complete_episodes += 1
        
        return complete_episodes / len(episodes)
    
    def _calculate_data_freshness(self, episodes: List[Dict[str, Any]]) -> float:
        """Расчет свежести данных"""
        if not episodes:
            return 0.0
        
        current_time = time.time()
        episode_ages = []
        
        for episode in episodes:
            episode_time = episode.get('end_time', current_time)
            age_hours = (current_time - episode_time) / 3600
            freshness = max(0, 1.0 - age_hours / (24 * 7))  # Максимум неделя
            episode_ages.append(freshness)
        
        return np.mean(episode_ages)
    
    def _calculate_entropy(self, values: List[str]) -> float:
        """Расчет энтропии для оценки разнообразия"""
        if not values:
            return 0.0
        
        counts = Counter(values)
        total = len(values)
        
        entropy = 0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Нормализация к максимальной энтропии
        max_entropy = np.log2(len(counts)) if counts else 1
        return entropy / max_entropy if max_entropy > 0 else 0

class MetaLearningDataCollector:
    """📚 Главный класс сбора данных для Meta-Learning Engine"""
    
    def __init__(self, config: Optional[DataCollectionConfig] = None, data_dir: str = "meta_learning_data"):
        self.config = config or DataCollectionConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Компоненты системы
        self.synthetic_generator = SyntheticDataGenerator(self.config)
        self.quality_analyzer = DataQualityAnalyzer()
        
        # Хранилище данных
        self.episodes_storage = []
        self.real_episodes_count = 0
        self.synthetic_episodes_count = 0
        
        # База данных для persistence
        self.db_path = self.data_dir / "meta_learning_data.db"
        self._init_database()
        
        # Логирование
        if UNIFIED_LOGGING_AVAILABLE:
            self.logger = UnifiedLoggingSystem().register_component("MetaLearningDataCollector")
        else:
            self.logger = logging.getLogger(__name__)
        
        # Загрузка существующих данных
        self._load_existing_data()
        
        self.logger.info(f"📚 Meta-Learning Data Collector инициализирован: {len(self.episodes_storage)} эпизодов")
    
    def _init_database(self):
        """Инициализация базы данных"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT UNIQUE,
                    data BLOB,
                    success INTEGER,
                    improvement_score REAL,
                    strategy_used TEXT,
                    synthetic INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_success ON episodes(success)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_synthetic ON episodes(synthetic)
            ''')
    
    def _load_existing_data(self):
        """Загрузка существующих данных из базы"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT data, synthetic FROM episodes ORDER BY created_at')
                
                for data_blob, is_synthetic in cursor.fetchall():
                    try:
                        episode = pickle.loads(data_blob)
                        self.episodes_storage.append(episode)
                        
                        if is_synthetic:
                            self.synthetic_episodes_count += 1
                        else:
                            self.real_episodes_count += 1
                            
                    except Exception as e:
                        self.logger.warning(f"⚠️ Ошибка загрузки эпизода: {e}")
                        
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка загрузки данных: {e}")
    
    def add_real_episode(self, episode_data: Dict[str, Any]):
        """Добавление реального эпизода обучения"""
        # Валидация данных
        if not self._validate_episode_data(episode_data):
            self.logger.warning("⚠️ Невалидные данные эпизода - пропускаем")
            return
        
        # Обогащение данных
        enriched_episode = self._enrich_episode_data(episode_data)
        
        # Сохранение в storage
        self.episodes_storage.append(enriched_episode)
        self.real_episodes_count += 1
        
        # Сохранение в базу данных
        self._save_episode_to_db(enriched_episode, synthetic=False)
        
        # Управление размером storage
        self._manage_storage_size()
        
        self.logger.info(f"📝 Добавлен реальный эпизод: {enriched_episode.get('episode_id', 'unknown')}")
    
    def generate_synthetic_data(self, target_episodes: Optional[int] = None):
        """Генерация синтетических данных"""
        if target_episodes is None:
            current_total = len(self.episodes_storage)
            target_episodes = max(0, self.config.min_episodes_target - current_total)
        
        if target_episodes <= 0:
            self.logger.info("📚 Достаточно данных - синтетическая генерация не требуется")
            return
        
        # Анализ текущих данных для определения нужных сценариев
        current_quality = self.quality_analyzer.analyze_data_quality(self.episodes_storage)
        scenario_mix = self._determine_scenario_mix(current_quality)
        
        # Генерация эпизодов
        synthetic_episodes = self.synthetic_generator.generate_episodes(target_episodes, scenario_mix)
        
        # Добавление в storage
        for episode in synthetic_episodes:
            self.episodes_storage.append(episode)
            self.synthetic_episodes_count += 1
            self._save_episode_to_db(episode, synthetic=True)
        
        self.logger.info(f"🤖 Сгенерировано {len(synthetic_episodes)} синтетических эпизодов")
    
    def get_training_dataset(self, format_type: str = 'episodes') -> Any:
        """Получение датасета для обучения Meta-Learning Engine"""
        if format_type == 'episodes':
            return self.episodes_storage.copy()
        elif format_type == 'processed':
            return self._process_episodes_for_training()
        elif format_type == 'quality_metrics':
            return self.quality_analyzer.analyze_data_quality(self.episodes_storage)
        else:
            raise ValueError(f"Неподдерживаемый формат: {format_type}")
    
    def enhance_data_quality(self):
        """Улучшение качества данных"""
        self.logger.info("🔧 Улучшение качества данных...")
        
        # Анализ текущего качества
        current_quality = self.quality_analyzer.analyze_data_quality(self.episodes_storage)
        
        improvements_needed = []
        
        # Проверка достаточности данных
        if current_quality.total_episodes < self.config.min_episodes_target:
            needed = self.config.min_episodes_target - current_quality.total_episodes
            improvements_needed.append(f"Нужно еще {needed} эпизодов")
            self.generate_synthetic_data(needed)
        
        # Проверка баланса успеха/неудачи
        success_rate = current_quality.successful_episodes / max(1, current_quality.total_episodes)
        if success_rate < 0.3 or success_rate > 0.8:
            improvements_needed.append("Нужен лучший баланс успеха/неудачи")
            self._balance_success_ratio()
        
        # Проверка разнообразия фаз
        phase_counts = current_quality.phase_distribution
        min_phase_count = max(1, current_quality.total_episodes // 10)
        for phase in ['PRE_ALIGNMENT', 'ALIGNMENT_LEARNING', 'REFINEMENT', 'CONVERGENCE']:
            if phase_counts.get(phase, 0) < min_phase_count:
                improvements_needed.append(f"Мало данных для фазы {phase}")
                self._generate_phase_specific_data(phase, min_phase_count)
        
        if improvements_needed:
            self.logger.info(f"🔧 Применены улучшения: {', '.join(improvements_needed)}")
        else:
            self.logger.info("✅ Качество данных удовлетворительное")
    
    def export_data(self, export_format: str = 'json', file_path: Optional[str] = None) -> str:
        """Экспорт данных в различные форматы"""
        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = self.data_dir / f"meta_learning_data_{timestamp}.{export_format}"
        else:
            file_path = Path(file_path)
        
        try:
            if export_format == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'episodes': self.episodes_storage,
                        'metadata': {
                            'total_episodes': len(self.episodes_storage),
                            'real_episodes': self.real_episodes_count,
                            'synthetic_episodes': self.synthetic_episodes_count,
                            'export_timestamp': datetime.now().isoformat()
                        }
                    }, f, indent=2, default=str, ensure_ascii=False)
                    
            elif export_format == 'pickle':
                with open(file_path, 'wb') as f:
                    pickle.dump(self.episodes_storage, f)
                    
            else:
                raise ValueError(f"Неподдерживаемый формат экспорта: {export_format}")
            
            self.logger.info(f"📤 Данные экспортированы: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка экспорта: {e}")
            raise
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Получение статистики по данным"""
        quality_metrics = self.quality_analyzer.analyze_data_quality(self.episodes_storage)
        
        return {
            'data_counts': {
                'total_episodes': len(self.episodes_storage),
                'real_episodes': self.real_episodes_count,
                'synthetic_episodes': self.synthetic_episodes_count,
                'target_episodes': self.config.min_episodes_target
            },
            'quality_metrics': asdict(quality_metrics),
            'progress_to_target': len(self.episodes_storage) / self.config.min_episodes_target,
            'recommendations': self._get_data_recommendations(quality_metrics)
        }
    
    def _validate_episode_data(self, episode_data: Dict[str, Any]) -> bool:
        """Валидация данных эпизода"""
        required_fields = ['episode_id', 'success']
        
        for field in required_fields:
            if field not in episode_data:
                return False
        
        # Проверка диапазонов значений
        if 'initial_loss' in episode_data:
            if not (0.01 <= episode_data['initial_loss'] <= 1000):
                return False
        
        if 'initial_attention_quality' in episode_data:
            if not (0.0 <= episode_data['initial_attention_quality'] <= 1.0):
                return False
        
        return True
    
    def _enrich_episode_data(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Обогащение данных эпизода"""
        enriched = episode_data.copy()
        
        # Добавление timestamp если отсутствует
        if 'end_time' not in enriched:
            enriched['end_time'] = time.time()
        
        # Расчет improvement_score если отсутствует
        if 'improvement_score' not in enriched and 'initial_loss' in enriched and 'final_loss' in enriched:
            loss_improvement = (enriched['initial_loss'] - enriched['final_loss']) / enriched['initial_loss']
            attention_improvement = enriched.get('final_attention_quality', 0) - enriched.get('initial_attention_quality', 0)
            enriched['improvement_score'] = loss_improvement * 0.7 + attention_improvement * 0.3
        
        # Метка реальных данных
        enriched['synthetic'] = False
        
        return enriched
    
    def _save_episode_to_db(self, episode: Dict[str, Any], synthetic: bool):
        """Сохранение эпизода в базу данных"""
        try:
            episode_blob = pickle.dumps(episode)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO episodes 
                    (episode_id, data, success, improvement_score, strategy_used, synthetic)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    episode.get('episode_id', ''),
                    episode_blob,
                    int(episode.get('success', False)),
                    episode.get('improvement_score', 0.0),
                    episode.get('strategy_used', ''),
                    int(synthetic)
                ))
                
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка сохранения в БД: {e}")
    
    def _manage_storage_size(self):
        """Управление размером storage"""
        if len(self.episodes_storage) > self.config.max_episodes_storage:
            # Удаляем старые эпизоды, отдавая предпочтение синтетическим
            episodes_to_remove = len(self.episodes_storage) - self.config.max_episodes_storage
            
            # Сортируем по времени и типу (синтетические первыми)
            sorted_episodes = sorted(
                enumerate(self.episodes_storage), 
                key=lambda x: (not x[1].get('synthetic', False), x[1].get('end_time', 0))
            )
            
            for i in range(episodes_to_remove):
                idx, episode = sorted_episodes[i]
                if episode.get('synthetic', False):
                    self.synthetic_episodes_count -= 1
                else:
                    self.real_episodes_count -= 1
            
            # Удаляем эпизоды
            indices_to_remove = [sorted_episodes[i][0] for i in range(episodes_to_remove)]
            self.episodes_storage = [
                ep for i, ep in enumerate(self.episodes_storage) 
                if i not in indices_to_remove
            ]
    
    def _determine_scenario_mix(self, quality_metrics: DataQualityMetrics) -> Dict[str, float]:
        """Определение микса сценариев для генерации"""
        base_mix = {
            'normal_training': 0.6,
            'difficult_convergence': 0.2,
            'fast_learning': 0.15,
            'instability_recovery': 0.05
        }
        
        # Корректировка на основе текущих данных
        if quality_metrics.successful_episodes / max(1, quality_metrics.total_episodes) < 0.4:
            # Мало успешных эпизодов - больше fast_learning
            base_mix['fast_learning'] += 0.2
            base_mix['normal_training'] -= 0.1
            base_mix['difficult_convergence'] -= 0.1
        
        # Нормализация
        total = sum(base_mix.values())
        return {k: v / total for k, v in base_mix.items()}
    
    def _balance_success_ratio(self):
        """Балансировка соотношения успеха/неудачи"""
        current_success_rate = self.real_episodes_count and sum(
            1 for ep in self.episodes_storage if ep.get('success', False)
        ) / len(self.episodes_storage)
        
        target_success_rate = 0.6
        
        if current_success_rate < target_success_rate:
            # Генерируем больше успешных эпизодов
            scenario_mix = {
                'normal_training': 0.7,
                'fast_learning': 0.3,
                'difficult_convergence': 0.0,
                'instability_recovery': 0.0
            }
        else:
            # Генерируем больше сложных эпизодов
            scenario_mix = {
                'normal_training': 0.3,
                'fast_learning': 0.1,
                'difficult_convergence': 0.4,
                'instability_recovery': 0.2
            }
        
        balance_episodes = min(50, self.config.min_episodes_target // 10)
        synthetic_episodes = self.synthetic_generator.generate_episodes(balance_episodes, scenario_mix)
        
        for episode in synthetic_episodes:
            self.episodes_storage.append(episode)
            self.synthetic_episodes_count += 1
    
    def _generate_phase_specific_data(self, phase: str, count: int):
        """Генерация данных для конкретной фазы"""
        # Настройка генератора под конкретную фазу
        phase_scenarios = {
            'PRE_ALIGNMENT': ['instability_recovery', 'difficult_convergence'],
            'ALIGNMENT_LEARNING': ['normal_training', 'difficult_convergence'],
            'REFINEMENT': ['normal_training', 'fast_learning'],
            'CONVERGENCE': ['fast_learning', 'normal_training']
        }
        
        scenarios = phase_scenarios.get(phase, ['normal_training'])
        scenario_mix = {scenario: 1.0 / len(scenarios) for scenario in scenarios}
        
        episodes = self.synthetic_generator.generate_episodes(count, scenario_mix)
        
        # Принудительно устанавливаем нужную фазу
        for episode in episodes:
            episode['initial_phase'] = phase
            self.episodes_storage.append(episode)
            self.synthetic_episodes_count += 1
    
    def _get_data_recommendations(self, quality_metrics: DataQualityMetrics) -> List[str]:
        """Получение рекомендаций по улучшению данных"""
        recommendations = []
        
        if quality_metrics.total_episodes < self.config.min_episodes_target:
            recommendations.append(f"Накопите еще {self.config.min_episodes_target - quality_metrics.total_episodes} эпизодов")
        
        if quality_metrics.quality_score < 0.7:
            recommendations.append("Улучшите качество данных через валидацию и очистку")
        
        if quality_metrics.diversity_score < 0.6:
            recommendations.append("Увеличьте разнообразие сценариев обучения")
        
        if quality_metrics.data_freshness < 0.5:
            recommendations.append("Обновите данные более свежими эпизодами")
        
        return recommendations


def run_data_collection_demo():
    """Демонстрация системы сбора данных"""
    print("📚 META-LEARNING DATA COLLECTOR DEMO")
    print("=" * 50)
    
    # Создание коллектора
    config = DataCollectionConfig(min_episodes_target=100, max_episodes_storage=200)
    collector = MetaLearningDataCollector(config)
    
    print(f"📊 Текущее состояние: {len(collector.episodes_storage)} эпизодов")
    
    # Генерация данных
    print("\n🤖 Генерация синтетических данных...")
    collector.generate_synthetic_data(50)
    
    # Улучшение качества
    print("\n🔧 Улучшение качества данных...")
    collector.enhance_data_quality()
    
    # Статистика
    print("\n📊 СТАТИСТИКА ДАННЫХ:")
    stats = collector.get_data_statistics()
    
    print(f"   • Всего эпизодов: {stats['data_counts']['total_episodes']}")
    print(f"   • Реальных: {stats['data_counts']['real_episodes']}")
    print(f"   • Синтетических: {stats['data_counts']['synthetic_episodes']}")
    print(f"   • Прогресс к цели: {stats['progress_to_target']*100:.1f}%")
    print(f"   • Качество данных: {stats['quality_metrics']['quality_score']:.2f}")
    print(f"   • Разнообразие: {stats['quality_metrics']['diversity_score']:.2f}")
    
    # Рекомендации
    if stats['recommendations']:
        print("\n💡 РЕКОМЕНДАЦИИ:")
        for rec in stats['recommendations']:
            print(f"   • {rec}")
    
    # Экспорт
    print("\n📤 Экспорт данных...")
    export_path = collector.export_data('json')
    print(f"   • Данные экспортированы в: {export_path}")
    
    print("\n✅ Демонстрация завершена успешно!")
    return collector


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_data_collection_demo() 