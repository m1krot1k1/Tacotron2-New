#!/usr/bin/env python3
"""
🧠 META-LEARNING ENGINE - Система обучения на опыте
=======================================================

Интеллектуальная система, которая обучается на предыдущем опыте и адаптирует
стратегии обучения для повышения эффективности.

Компоненты:
1. EpisodicMemory - память о предыдущих эпизодах обучения
2. PatternAnalyzer - анализ паттернов успеха и неудач
3. StrategyAdaptor - адаптация стратегий на основе опыта
4. MAMLOptimizer - Model-Agnostic Meta-Learning для быстрой адаптации

Автор: Enhanced Tacotron2 AI System
Версия: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import logging
from datetime import datetime
import sqlite3
import time

# Интеграция с существующими системами
try:
    from unified_logging_system import UnifiedLoggingSystem
    UNIFIED_LOGGING_AVAILABLE = True
except ImportError:
    UNIFIED_LOGGING_AVAILABLE = False

try:
    from context_aware_training_manager import TrainingPhase
    CONTEXT_MANAGER_AVAILABLE = True
except ImportError:
    # Fallback enum
    class TrainingPhase(Enum):
        PRE_ALIGNMENT = "pre_alignment"
        ALIGNMENT_LEARNING = "alignment"
        REFINEMENT = "refinement"
        CONVERGENCE = "convergence"
    CONTEXT_MANAGER_AVAILABLE = False


class LearningStrategy(Enum):
    """Стратегии обучения"""
    AGGRESSIVE = "aggressive"      # Быстрые изменения параметров
    CONSERVATIVE = "conservative"  # Осторожные изменения
    BALANCED = "balanced"         # Сбалансированный подход
    ADAPTIVE = "adaptive"         # Адаптивная стратегия


@dataclass
class TrainingEpisode:
    """Эпизод обучения для episodic memory"""
    episode_id: str
    start_time: float
    end_time: float
    initial_phase: TrainingPhase
    final_phase: TrainingPhase
    
    # Исходные метрики
    initial_loss: float
    initial_attention_quality: float
    
    # Финальные метрики
    final_loss: float
    final_attention_quality: float
    
    # Действия и решения
    strategy_used: LearningStrategy
    decisions_made: List[Dict[str, Any]]
    parameters_changed: Dict[str, float]
    
    # Результат
    success: bool
    improvement_score: float
    convergence_achieved: bool
    
    # Метаданные
    model_architecture: str = "tacotron2"
    dataset_size: int = 0
    total_steps: int = 0


@dataclass
class MetaLearningState:
    """Состояние meta-learning системы"""
    total_episodes: int
    successful_episodes: int
    preferred_strategies: Dict[str, float]  # strategy -> success_rate
    learned_patterns: Dict[str, Any]
    adaptation_history: List[Dict[str, Any]]
    last_updated: str


class EpisodicMemory:
    """🧠 Episodic Memory - память о предыдущих эпизодах обучения"""
    
    def __init__(self, memory_dir: str = "meta_learning_memory", max_episodes: int = 1000):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_episodes = max_episodes
        self.episodes: List[TrainingEpisode] = []
        self.db_path = self.memory_dir / "episodes.db"
        
        self.logger = self._setup_logger()
        self._init_database()
        self._load_episodes()
        
        self.logger.info(f"🧠 Episodic Memory инициализирована: {len(self.episodes)} эпизодов")
    
    def _setup_logger(self):
        if UNIFIED_LOGGING_AVAILABLE:
            return UnifiedLoggingSystem().register_component("EpisodicMemory")
        else:
            logger = logging.getLogger("EpisodicMemory")
            logger.setLevel(logging.INFO)
            return logger
    
    def _init_database(self):
        """Инициализация SQLite базы данных"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    data BLOB,
                    success INTEGER,
                    improvement_score REAL,
                    strategy_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_success ON episodes(success)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_strategy ON episodes(strategy_used)
            ''')
    
    def _load_episodes(self):
        """Загрузка эпизодов из базы данных"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT data FROM episodes ORDER BY created_at')
                
                for (episode_data,) in cursor.fetchall():
                    try:
                        episode_dict = pickle.loads(episode_data)
                        episode = TrainingEpisode(**episode_dict)
                        self.episodes.append(episode)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Ошибка загрузки эпизода: {e}")
                        
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка загрузки эпизодов: {e}")
    
    def _cleanup_old_episodes(self):
        """Очистка старых эпизодов при превышении лимита"""
        if len(self.episodes) <= self.max_episodes:
            return
        
        # Сортировка по времени создания (старые первыми)
        sorted_episodes = sorted(self.episodes, key=lambda x: x.start_time)
        
        # Удаление старых эпизодов
        episodes_to_remove = sorted_episodes[:len(self.episodes) - self.max_episodes]
        
        for episode in episodes_to_remove:
            try:
                # Удаление из базы данных
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('DELETE FROM episodes WHERE episode_id = ?', (episode.episode_id,))
                
                # Удаление из памяти
                self.episodes.remove(episode)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Ошибка удаления старого эпизода: {e}")
        
        self.logger.info(f"🗑️ Удалено {len(episodes_to_remove)} старых эпизодов")
    
    def add_episode(self, episode: TrainingEpisode):
        """Добавление нового эпизода в память"""
        try:
            # Сериализация эпизода
            episode_data = pickle.dumps(asdict(episode))
            
            # Сохранение в базу данных
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO episodes 
                    (episode_id, data, success, improvement_score, strategy_used)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    episode.episode_id,
                    episode_data,
                    int(episode.success),
                    episode.improvement_score,
                    episode.strategy_used.value
                ))
            
            # Добавление в память
            self.episodes.append(episode)
            
            # Управление размером памяти
            if len(self.episodes) > self.max_episodes:
                self._cleanup_old_episodes()
            
            self.logger.info(f"📝 Добавлен эпизод: {episode.episode_id} (успех: {episode.success})")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка добавления эпизода: {e}")
    
    def get_similar_episodes(self, 
                           current_phase: TrainingPhase,
                           current_loss: float,
                           current_attention: float,
                           top_k: int = 5) -> List[TrainingEpisode]:
        """Поиск похожих эпизодов для обучения"""
        if not self.episodes:
            return []
        
        # Вычисление similarity score для каждого эпизода
        scored_episodes = []
        
        for episode in self.episodes:
            # Similarity по phase
            phase_similarity = 1.0 if episode.initial_phase == current_phase else 0.5
            
            # Similarity по метрикам
            loss_diff = abs(episode.initial_loss - current_loss)
            loss_similarity = 1.0 / (1.0 + loss_diff / 10.0)  # Нормализация
            
            attention_diff = abs(episode.initial_attention_quality - current_attention)
            attention_similarity = 1.0 / (1.0 + attention_diff * 2.0)
            
            # Общий score с весами
            total_similarity = (
                0.4 * phase_similarity +
                0.3 * loss_similarity +
                0.3 * attention_similarity
            )
            
            # Бонус за успешные эпизоды
            if episode.success:
                total_similarity *= 1.2
                
            scored_episodes.append((total_similarity, episode))
        
        # Сортировка по убыванию similarity
        scored_episodes.sort(reverse=True, key=lambda x: x[0])
        
        # Возврат top-k эпизодов
        similar_episodes = [episode for _, episode in scored_episodes[:top_k]]
        
        self.logger.info(f"🔍 Найдено {len(similar_episodes)} похожих эпизодов")
        return similar_episodes
    
    def get_success_statistics(self) -> Dict[str, Any]:
        """Статистика успешности по стратегиям и фазам"""
        if not self.episodes:
            return {}
        
        stats = {
            'total_episodes': len(self.episodes),
            'successful_episodes': sum(1 for ep in self.episodes if ep.success),
            'strategies': defaultdict(lambda: {'total': 0, 'successful': 0}),
            'phases': defaultdict(lambda: {'total': 0, 'successful': 0, 'avg_improvement': 0.0})
        }
        
        # Статистика по стратегиям
        for episode in self.episodes:
            strategy = episode.strategy_used.value
            stats['strategies'][strategy]['total'] += 1
            if episode.success:
                stats['strategies'][strategy]['successful'] += 1
        
        # Статистика по фазам
        phase_improvements = defaultdict(list)
        for episode in self.episodes:
            phase = episode.initial_phase.value
            stats['phases'][phase]['total'] += 1
            if episode.success:
                stats['phases'][phase]['successful'] += 1
            phase_improvements[phase].append(episode.improvement_score)
        
        # Средние улучшения по фазам
        for phase, improvements in phase_improvements.items():
            if improvements:
                stats['phases'][phase]['avg_improvement'] = np.mean(improvements)
        
        return dict(stats)


class PatternAnalyzer:
    """🔍 Pattern Analyzer - анализ паттернов успеха и неудач"""
    
    def __init__(self, episodic_memory: EpisodicMemory):
        self.memory = episodic_memory
        self.logger = self._setup_logger()
        
        # Обученные паттерны
        self.success_patterns = {}
        self.failure_patterns = {}
        self.decision_patterns = {}
        
    def _setup_logger(self):
        if UNIFIED_LOGGING_AVAILABLE:
            return UnifiedLoggingSystem().register_component("PatternAnalyzer")
        else:
            logger = logging.getLogger("PatternAnalyzer")
            logger.setLevel(logging.INFO)
            return logger
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Анализ паттернов в эпизодической памяти"""
        episodes = self.memory.episodes
        if len(episodes) < 10:  # Недостаточно данных для анализа
            return {}
        
        successful_episodes = [ep for ep in episodes if ep.success]
        failed_episodes = [ep for ep in episodes if not ep.success]
        
        patterns = {
            'success_patterns': self._analyze_success_patterns(successful_episodes),
            'failure_patterns': self._analyze_failure_patterns(failed_episodes),
            'decision_patterns': self._analyze_decision_patterns(episodes),
            'temporal_patterns': self._analyze_temporal_patterns(episodes)
        }
        
        self.logger.info(f"🔍 Проанализировано паттернов: {len(patterns)} категорий")
        return patterns
    
    def _analyze_success_patterns(self, successful_episodes: List[TrainingEpisode]) -> Dict[str, Any]:
        """Анализ паттернов успешных эпизодов"""
        if not successful_episodes:
            return {}
        
        patterns = {}
        
        # Паттерны по стратегиям
        strategy_counts = defaultdict(int)
        for ep in successful_episodes:
            strategy_counts[ep.strategy_used.value] += 1
        
        if strategy_counts:
            most_successful_strategy = max(strategy_counts.items(), key=lambda x: x[1])
            patterns['best_strategy'] = {
                'strategy': most_successful_strategy[0],
                'count': most_successful_strategy[1],
                'success_rate': most_successful_strategy[1] / len(successful_episodes)
            }
        
        # Паттерны по параметрам
        lr_changes = [ep.parameters_changed.get('learning_rate', 0) for ep in successful_episodes]
        if lr_changes:
            patterns['lr_change_pattern'] = {
                'mean': np.mean(lr_changes),
                'std': np.std(lr_changes),
                'median': np.median(lr_changes)
            }
        
        return patterns
    
    def _analyze_failure_patterns(self, failed_episodes: List[TrainingEpisode]) -> Dict[str, Any]:
        """Анализ паттернов неудачных эпизодов"""
        if not failed_episodes:
            return {}
        
        patterns = {}
        
        # Частые причины неудач
        failure_strategies = defaultdict(int)
        for ep in failed_episodes:
            failure_strategies[ep.strategy_used.value] += 1
        
        if failure_strategies:
            worst_strategy = max(failure_strategies.items(), key=lambda x: x[1])
            patterns['worst_strategy'] = {
                'strategy': worst_strategy[0],
                'count': worst_strategy[1]
            }
        
        return patterns
    
    def _analyze_decision_patterns(self, episodes: List[TrainingEpisode]) -> Dict[str, Any]:
        """Анализ паттернов принятия решений"""
        decision_outcomes = defaultdict(list)
        
        for episode in episodes:
            for decision in episode.decisions_made:
                decision_type = decision.get('type', 'unknown')
                decision_outcomes[decision_type].append(episode.success)
        
        patterns = {}
        for decision_type, outcomes in decision_outcomes.items():
            if len(outcomes) >= 3:  # Минимум 3 примера
                success_rate = sum(outcomes) / len(outcomes)
                patterns[decision_type] = {
                    'total_decisions': len(outcomes),
                    'success_rate': success_rate,
                    'confidence': 'high' if len(outcomes) >= 10 else 'low'
                }
        
        return patterns
    
    def _analyze_temporal_patterns(self, episodes: List[TrainingEpisode]) -> Dict[str, Any]:
        """Анализ временных паттернов в эпизодах"""
        if len(episodes) < 5:
            return {}
        
        # Сортировка по времени
        sorted_episodes = sorted(episodes, key=lambda x: x.start_time)
        
        # Анализ тренда успешности
        success_trend = []
        window_size = min(5, len(sorted_episodes) // 2)
        
        for i in range(len(sorted_episodes) - window_size + 1):
            window_episodes = sorted_episodes[i:i + window_size]
            success_rate = sum(1 for ep in window_episodes if ep.success) / len(window_episodes)
            success_trend.append(success_rate)
        
        patterns = {}
        
        if len(success_trend) >= 2:
            # Общий тренд
            trend_slope = (success_trend[-1] - success_trend[0]) / len(success_trend)
            
            if trend_slope > 0.05:
                patterns['success_trend'] = 'improving'
            elif trend_slope < -0.05:
                patterns['success_trend'] = 'declining'
            else:
                patterns['success_trend'] = 'stable'
            
            patterns['trend_slope'] = trend_slope
        
        return patterns


class StrategyAdaptor:
    """🎯 Strategy Adaptor - адаптация стратегий на основе опыта"""
    
    def __init__(self, pattern_analyzer: PatternAnalyzer):
        self.pattern_analyzer = pattern_analyzer
        self.logger = self._setup_logger()
        
        # Текущие стратегии
        self.current_strategies = {
            'learning_rate_adaptation': 'balanced',
            'attention_weight_adaptation': 'conservative',
            'loss_weight_adaptation': 'adaptive'
        }
        
        # История адаптаций
        self.adaptation_history = []
        
    def _setup_logger(self):
        if UNIFIED_LOGGING_AVAILABLE:
            return UnifiedLoggingSystem().register_component("StrategyAdaptor")
        else:
            logger = logging.getLogger("StrategyAdaptor")
            logger.setLevel(logging.INFO)
            return logger
    
    def adapt_strategies(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Адаптация стратегий на основе текущего контекста и опыта"""
        patterns = self.pattern_analyzer.analyze_patterns()
        
        if not patterns:
            self.logger.warning("⚠️ Недостаточно данных для адаптации стратегий")
            return self.current_strategies.copy()
        
        adapted_strategies = self.current_strategies.copy()
        adaptations_made = []
        
        # Адаптация на основе успешных паттернов
        success_patterns = patterns.get('success_patterns', {})
        if 'best_strategy' in success_patterns:
            best_strategy_info = success_patterns['best_strategy']
            if best_strategy_info['success_rate'] > 0.7:  # Высокая успешность
                best_strategy = best_strategy_info['strategy']
                
                # Применяем лучшую стратегию
                for key in adapted_strategies:
                    if adapted_strategies[key] != best_strategy:
                        adapted_strategies[key] = best_strategy
                        adaptations_made.append(f"{key} -> {best_strategy}")
        
        # Адаптация на основе контекста
        current_phase = current_context.get('phase', TrainingPhase.PRE_ALIGNMENT)
        current_loss = current_context.get('loss', 10.0)
        
        # Фазо-специфичные адаптации
        if current_phase == TrainingPhase.PRE_ALIGNMENT and current_loss > 15.0:
            adapted_strategies['learning_rate_adaptation'] = 'aggressive'
            adaptations_made.append("LR adaptation -> aggressive (высокий initial loss)")
        
        elif current_phase == TrainingPhase.CONVERGENCE and current_loss < 3.0:
            adapted_strategies['learning_rate_adaptation'] = 'conservative'
            adaptations_made.append("LR adaptation -> conservative (близко к конвергенции)")
        
        # Логирование адаптаций
        if adaptations_made:
            adaptation_record = {
                'timestamp': datetime.now().isoformat(),
                'adaptations': adaptations_made,
                'context': current_context,
                'patterns_used': list(patterns.keys())
            }
            self.adaptation_history.append(adaptation_record)
            
            self.logger.info(f"🎯 Адаптированы стратегии: {', '.join(adaptations_made)}")
        
        self.current_strategies = adapted_strategies
        return adapted_strategies.copy()


class MetaLearningEngine:
    """🧠 Meta-Learning Engine - главный компонент системы мета-обучения"""
    
    def __init__(self, 
                 memory_dir: str = "meta_learning_memory",
                 max_episodes: int = 1000):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализация компонентов
        self.episodic_memory = EpisodicMemory(memory_dir, max_episodes)
        self.pattern_analyzer = PatternAnalyzer(self.episodic_memory)
        self.strategy_adaptor = StrategyAdaptor(self.pattern_analyzer)
        
        # Состояние системы
        self.state = self._load_state()
        self.current_episode = None
        
        self.logger = self._setup_logger()
        self.logger.info("🧠 Meta-Learning Engine инициализирован")
    
    def _setup_logger(self):
        if UNIFIED_LOGGING_AVAILABLE:
            return UnifiedLoggingSystem().register_component("MetaLearningEngine")
        else:
            logger = logging.getLogger("MetaLearningEngine")
            logger.setLevel(logging.INFO)
            return logger
    
    def start_episode(self, 
                     initial_context: Dict[str, Any],
                     episode_id: Optional[str] = None) -> str:
        """Начало нового эпизода обучения"""
        if episode_id is None:
            episode_id = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_episode = TrainingEpisode(
            episode_id=episode_id,
            start_time=time.time(),
            end_time=0.0,
            initial_phase=initial_context.get('phase', TrainingPhase.PRE_ALIGNMENT),
            final_phase=TrainingPhase.PRE_ALIGNMENT,
            initial_loss=initial_context.get('loss', 10.0),
            initial_attention_quality=initial_context.get('attention_quality', 0.1),
            final_loss=0.0,
            final_attention_quality=0.0,
            strategy_used=LearningStrategy.BALANCED,
            decisions_made=[],
            parameters_changed={},
            success=False,
            improvement_score=0.0,
            convergence_achieved=False,
            total_steps=0
        )
        
        self.logger.info(f"🚀 Начат эпизод обучения: {episode_id}")
        return episode_id
    
    def record_decision(self, decision: Dict[str, Any]):
        """Запись решения в текущий эпизод"""
        if self.current_episode:
            self.current_episode.decisions_made.append({
                **decision,
                'timestamp': time.time()
            })
    
    def get_recommended_strategy(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Получение рекомендуемой стратегии на основе опыта"""
        # Адаптация стратегий на основе паттернов
        adapted_strategies = self.strategy_adaptor.adapt_strategies(current_context)
        
        # Поиск похожих эпизодов
        similar_episodes = self.episodic_memory.get_similar_episodes(
            current_phase=current_context.get('phase', TrainingPhase.PRE_ALIGNMENT),
            current_loss=current_context.get('loss', 10.0),
            current_attention=current_context.get('attention_quality', 0.1)
        )
        
        recommendations = {
            'strategies': adapted_strategies,
            'similar_episodes_count': len(similar_episodes),
            'confidence': 'high' if len(similar_episodes) >= 3 else 'medium',
            'learning_insights': self._extract_insights(similar_episodes)
        }
        
        return recommendations
    
    def _extract_insights(self, similar_episodes: List[TrainingEpisode]) -> List[str]:
        """Извлечение insights из похожих эпизодов"""
        if not similar_episodes:
            return ["Нет опыта для данной ситуации"]
        
        insights = []
        
        # Успешность похожих эпизодов
        successful_episodes = [ep for ep in similar_episodes if ep.success]
        success_rate = len(successful_episodes) / len(similar_episodes)
        
        if success_rate > 0.7:
            insights.append(f"Высокая вероятность успеха ({success_rate:.1%}) в похожих ситуациях")
        elif success_rate < 0.3:
            insights.append(f"Низкая вероятность успеха ({success_rate:.1%}) - требуется осторожность")
        
        # Рекомендации по стратегии
        if successful_episodes:
            strategies = [ep.strategy_used for ep in successful_episodes]
            most_common = max(set(strategies), key=strategies.count)
            insights.append(f"Рекомендуемая стратегия: {most_common.value}")
        
        return insights
    
    def end_episode(self, final_context: Dict[str, Any]) -> bool:
        """Завершение текущего эпизода"""
        if not self.current_episode:
            self.logger.warning("⚠️ Нет активного эпизода для завершения")
            return False
        
        # Обновление финальных метрик
        self.current_episode.end_time = time.time()
        self.current_episode.final_phase = final_context.get('phase', TrainingPhase.PRE_ALIGNMENT)
        self.current_episode.final_loss = final_context.get('loss', self.current_episode.initial_loss)
        self.current_episode.final_attention_quality = final_context.get('attention_quality', 
                                                                        self.current_episode.initial_attention_quality)
        self.current_episode.total_steps = final_context.get('total_steps', 0)
        
        # Оценка успешности и улучшения
        loss_improvement = (self.current_episode.initial_loss - self.current_episode.final_loss) / self.current_episode.initial_loss
        attention_improvement = self.current_episode.final_attention_quality - self.current_episode.initial_attention_quality
        
        self.current_episode.improvement_score = loss_improvement * 0.6 + attention_improvement * 0.4
        self.current_episode.success = (
            loss_improvement > 0.1 and  # Минимум 10% улучшение loss
            attention_improvement > 0.05 and  # Минимум 5% улучшение attention
            self.current_episode.final_loss < 10.0  # Разумный финальный loss
        )
        
        # Сохранение эпизода в память
        self.episodic_memory.add_episode(self.current_episode)
        
        # Обновление состояния
        self.state.total_episodes += 1
        if self.current_episode.success:
            self.state.successful_episodes += 1
        
        episode_id = self.current_episode.episode_id
        success = self.current_episode.success
        
        self.current_episode = None
        self._save_state()
        
        self.logger.info(f"🏁 Эпизод завершен: {episode_id} (успех: {success})")
        return success
    
    def _load_state(self) -> MetaLearningState:
        """Загрузка состояния мета-обучения"""
        state_file = self.memory_dir / "meta_learning_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                return MetaLearningState(
                    total_episodes=data.get('total_episodes', 0),
                    successful_episodes=data.get('successful_episodes', 0),
                    preferred_strategies=data.get('preferred_strategies', {}),
                    learned_patterns=data.get('learned_patterns', {}),
                    adaptation_history=data.get('adaptation_history', []),
                    last_updated=data.get('last_updated', datetime.now().isoformat())
                )
            except Exception as e:
                self.logger.warning(f"⚠️ Ошибка загрузки состояния: {e}")
        
        # Состояние по умолчанию
        return MetaLearningState(
            total_episodes=0,
            successful_episodes=0,
            preferred_strategies={},
            learned_patterns={},
            adaptation_history=[],
            last_updated=datetime.now().isoformat()
        )
    
    def _save_state(self):
        """Сохранение состояния мета-обучения"""
        state_file = self.memory_dir / "meta_learning_state.json"
        
        try:
            self.state.last_updated = datetime.now().isoformat()
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.state), f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения состояния: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Получение статистики обучения системы"""
        memory_stats = self.episodic_memory.get_success_statistics()
        
        success_rate = 0.0
        if self.state.total_episodes > 0:
            success_rate = self.state.successful_episodes / self.state.total_episodes
        
        return {
            'meta_learning_stats': {
                'total_episodes': self.state.total_episodes,
                'successful_episodes': self.state.successful_episodes,
                'success_rate': success_rate,
                'adaptations_made': len(self.strategy_adaptor.adaptation_history)
            },
            'memory_stats': memory_stats,
            'current_strategies': self.strategy_adaptor.current_strategies.copy(),
            'system_maturity': self._assess_system_maturity()
        }
    
    def _assess_system_maturity(self) -> str:
        """Оценка зрелости системы мета-обучения"""
        episodes = self.state.total_episodes
        
        if episodes < 10:
            return "novice"  # Новичок
        elif episodes < 50:
            return "learning"  # Обучается
        elif episodes < 100:
            return "experienced"  # Опытная
        else:
            return "expert"  # Эксперт


# Функции создания и интеграции
def create_meta_learning_engine(memory_dir: str = "meta_learning_memory") -> MetaLearningEngine:
    """Создание настроенного Meta-Learning Engine"""
    return MetaLearningEngine(memory_dir=memory_dir)


if __name__ == "__main__":
    # Демонстрация использования
    print("🧠 Meta-Learning Engine демо")
    
    # Создание engine
    meta_engine = create_meta_learning_engine()
    
    # Симуляция эпизода обучения
    initial_context = {
        'phase': TrainingPhase.PRE_ALIGNMENT,
        'loss': 15.0,
        'attention_quality': 0.1
    }
    
    episode_id = meta_engine.start_episode(initial_context)
    print(f"📝 Начат эпизод: {episode_id}")
    
    # Получение рекомендаций
    recommendations = meta_engine.get_recommended_strategy(initial_context)
    print(f"💡 Рекомендации: {recommendations}")
    
    # Завершение эпизода
    final_context = {
        'phase': TrainingPhase.ALIGNMENT_LEARNING,
        'loss': 8.0,
        'attention_quality': 0.4,
        'total_steps': 1000
    }
    
    success = meta_engine.end_episode(final_context)
    print(f"🏁 Эпизод завершен успешно: {success}")
    
    print("✅ Meta-Learning Engine готов к использованию!") 