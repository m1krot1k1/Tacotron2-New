#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Epoch Optimizer для Smart Tuner TTS
Интеллектуальная система определения оптимального количества эпох на основе:
1. Современных исследований TTS (2024-2025)
2. Анализа датасета и его характеристик
3. Мониторинга качества обучения в реальном времени
4. Предотвращения переобучения и недообучения
"""

import numpy as np
import yaml
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import sqlite3
from datetime import datetime

class IntelligentEpochOptimizer:
    """
    Интеллектуальный оптимизатор эпох для TTS обучения.
    
    Основан на исследованиях:
    - XTTS: 10-30 эпох для качественных голосов
    - Tacotron2: 500-5000 эпох в зависимости от датасета
    - VoiceStar: адаптивное количество эпох с контролем качества
    """
    
    def __init__(self, config_path: str = "smart_tuner/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.logger = logging.getLogger(__name__)
        
        # Настройки оптимизатора
        self.optimizer_config = self.config.get('adaptive_learning', {})
        self.quality_config = self.config.get('quality_monitoring', {})
        self.overfitting_config = self.config.get('overfitting_detection', {})
        
        # База данных для хранения истории оптимизации
        self.db_path = "smart_tuner/epoch_optimizer_history.db"
        self._init_database()
        
        # Метрики обучения
        self.training_history = []
        self.quality_metrics = []
        self.dataset_analysis = {}
        
        # Состояние оптимизатора
        self.current_phase = "initialization"
        self.optimal_epochs_estimate = None
        self.confidence_score = 0.0
        
        # Отслеживание изменений параметров
        self.parameter_changes = []
        self.optimization_decisions = []
        
    def _init_database(self):
        """Инициализирует базу данных для хранения истории оптимизации."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                dataset_size_hours REAL,
                dataset_quality_score REAL,
                voice_complexity TEXT,
                recommended_epochs INTEGER,
                actual_epochs INTEGER,
                final_quality_score REAL,
                training_time_minutes REAL,
                success_rating INTEGER,
                notes TEXT
            )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info(f"База данных оптимизатора эпох инициализирована: {self.db_path}")
        except Exception as e:
            self.logger.error(f"Ошибка инициализации базы данных: {e}")
    
    def analyze_dataset(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Анализирует датасет для определения оптимального количества эпох."""
        analysis = {
            "dataset_size_category": self._categorize_dataset_size(dataset_info),
            "quality_assessment": self._assess_dataset_quality(dataset_info),
            "complexity_analysis": self._analyze_voice_complexity(dataset_info),
            "recommended_epochs_range": None,
            "confidence": 0.0,
            "risk_factors": []
        }
        
        # Определяем рекомендуемое количество эпох
        epochs_recommendation = self._calculate_optimal_epochs(analysis)
        analysis.update(epochs_recommendation)
        
        self.dataset_analysis = analysis
        return analysis
    
    def _categorize_dataset_size(self, dataset_info: Dict[str, Any]) -> str:
        """Категоризирует размер датасета."""
        total_duration = dataset_info.get('total_duration_hours', 0)
        
        if total_duration < 0.5:
            return "very_small"
        elif total_duration < 1.0:
            return "small"
        elif total_duration < 3.0:
            return "medium"
        elif total_duration < 10.0:
            return "large"
        else:
            return "very_large"
    
    def _assess_dataset_quality(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Оценивает качество датасета."""
        quality_metrics = dataset_info.get('quality_metrics', {})
        
        noise_level = quality_metrics.get('background_noise_level', 0.5)
        consistency = quality_metrics.get('voice_consistency', 0.7)
        clarity = quality_metrics.get('speech_clarity', 0.8)
        
        quality_score = (
            (1 - noise_level) * 0.3 +
            consistency * 0.4 +
            clarity * 0.3
        )
        
        return {
            "overall_score": quality_score,
            "noise_level": noise_level,
            "consistency": consistency,
            "clarity": clarity,
            "category": self._quality_category(quality_score)
        }
    
    def _quality_category(self, score: float) -> str:
        """Определяет категорию качества."""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.5:
            return "fair"
        else:
            return "poor"
    
    def _analyze_voice_complexity(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Анализирует сложность голоса."""
        voice_features = dataset_info.get('voice_features', {})
        
        has_accent = voice_features.get('has_accent', False)
        emotional_range = voice_features.get('emotional_range', 'neutral')
        speaking_style = voice_features.get('speaking_style', 'normal')
        pitch_range = voice_features.get('pitch_range_semitones', 12)
        
        complexity_score = 0.0
        
        if has_accent:
            complexity_score += 0.2
        
        if emotional_range == "expressive":
            complexity_score += 0.3
        elif emotional_range == "varied":
            complexity_score += 0.2
            
        if speaking_style == "varied":
            complexity_score += 0.2
        elif speaking_style in ["fast", "slow"]:
            complexity_score += 0.1
            
        pitch_complexity = min(pitch_range / 24, 1.0)
        complexity_score += pitch_complexity * 0.3
        
        return {
            "complexity_score": min(complexity_score, 1.0),
            "category": self._complexity_category(complexity_score),
            "factors": {
                "accent": has_accent,
                "emotional_range": emotional_range,
                "speaking_style": speaking_style,
                "pitch_range": pitch_range
            }
        }
    
    def _complexity_category(self, score: float) -> str:
        """Определяет категорию сложности."""
        if score >= 0.7:
            return "very_complex"
        elif score >= 0.5:
            return "complex"
        elif score >= 0.3:
            return "moderate"
        else:
            return "simple"
    
    def _calculate_optimal_epochs(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Вычисляет оптимальное количество эпох на основе анализа."""
        size_category = analysis["dataset_size_category"]
        quality_score = analysis["quality_assessment"]["overall_score"]
        complexity_score = analysis["complexity_analysis"]["complexity_score"]
        
        # Базовые значения на основе размера датасета
        base_epochs = {
            "very_small": 4000,
            "small": 3000,
            "medium": 2000,
            "large": 1500,
            "very_large": 1000
        }
        
        base = base_epochs[size_category]
        
        # Корректировка на основе качества
        quality_multiplier = 1.0
        if quality_score >= 0.9:
            quality_multiplier = 0.8
        elif quality_score >= 0.7:
            quality_multiplier = 0.9
        elif quality_score < 0.5:
            quality_multiplier = 1.3
            
        # Корректировка на основе сложности
        complexity_multiplier = 1.0 + (complexity_score * 0.5)
        
        # Вычисляем оптимальное количество
        optimal = int(base * quality_multiplier * complexity_multiplier)
        
        # Ограничиваем разумными пределами
        min_epochs = 500
        max_epochs = 5000
        optimal = max(min_epochs, min(max_epochs, optimal))
        
        # Диапазон с запасом
        range_min = max(min_epochs, int(optimal * 0.8))
        range_max = min(max_epochs, int(optimal * 1.2))
        
        # Вычисляем уверенность
        confidence = self._calculate_confidence(analysis)
        
        return {
            "recommended_epochs_range": (range_min, range_max),
            "optimal_epochs": optimal,
            "confidence": confidence,
            "reasoning": self._generate_reasoning(analysis, optimal)
        }
    
    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Вычисляет уверенность в рекомендации."""
        quality_score = analysis["quality_assessment"]["overall_score"]
        
        base_confidence = quality_score * 0.6
        
        size_category = analysis["dataset_size_category"]
        if size_category in ["medium", "large"]:
            base_confidence += 0.2
        elif size_category == "very_large":
            base_confidence += 0.3
            
        complexity_score = analysis["complexity_analysis"]["complexity_score"]
        if complexity_score > 0.8:
            base_confidence -= 0.1
            
        return min(1.0, max(0.1, base_confidence))
    
    def _generate_reasoning(self, analysis: Dict[str, Any], optimal_epochs: int) -> str:
        """Генерирует объяснение рекомендации."""
        size_cat = analysis["dataset_size_category"]
        quality_cat = analysis["quality_assessment"]["category"]
        complexity_cat = analysis["complexity_analysis"]["category"]
        
        reasoning = f"Рекомендация {optimal_epochs} эпох основана на:\n"
        reasoning += f"• Размер датасета: {size_cat}\n"
        reasoning += f"• Качество: {quality_cat}\n"
        reasoning += f"• Сложность голоса: {complexity_cat}\n"
        
        return reasoning
    
    def monitor_training_progress(self, epoch: int, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Мониторит прогресс обучения и корректирует оценки.
        
        Args:
            epoch: Текущая эпоха
            metrics: Метрики обучения
            
        Returns:
            Рекомендации по продолжению обучения
        """
        # Добавляем метрики в историю
        self.training_history.append({
            "epoch": epoch,
            "metrics": metrics.copy(),
            "timestamp": datetime.now()
        })
        
        # Анализируем прогресс
        progress_analysis = self._analyze_training_progress()
        
        # Обновляем оценку оптимальных эпох
        self._update_epoch_estimate(progress_analysis)
        
        # Генерируем рекомендации
        recommendations = self._generate_training_recommendations(epoch, progress_analysis)
        
        return {
            "progress_analysis": progress_analysis,
            "recommendations": recommendations,
            "updated_estimate": self.optimal_epochs_estimate,
            "confidence": self.confidence_score
        }
    
    def _analyze_training_progress(self) -> Dict[str, Any]:
        """Анализирует прогресс обучения."""
        if len(self.training_history) < 10:
            return {"status": "insufficient_data", "trend": "unknown"}
        
        # Анализируем последние 50 эпох
        recent_history = self.training_history[-50:]
        
        # Тренды loss
        val_losses = [h["metrics"].get("val_loss", float('inf')) for h in recent_history]
        train_losses = [h["metrics"].get("train_loss", float('inf')) for h in recent_history]
        
        # Тренды качества
        attention_scores = [h["metrics"].get("attention_alignment_score", 0) for h in recent_history]
        gate_accuracies = [h["metrics"].get("gate_accuracy", 0) for h in recent_history]
        
        analysis = {
            "loss_trends": self._analyze_loss_trends(train_losses, val_losses),
            "quality_trends": self._analyze_quality_trends(attention_scores, gate_accuracies),
            "overfitting_risk": self._assess_overfitting_risk(train_losses, val_losses),
            "convergence_status": self._assess_convergence(recent_history)
        }
        
        return analysis
    
    def _analyze_loss_trends(self, train_losses: List[float], val_losses: List[float]) -> Dict[str, Any]:
        """Анализирует тренды loss."""
        if len(train_losses) < 5:
            return {"trend": "insufficient_data"}
        
        # Вычисляем тренды
        train_trend = np.polyfit(range(len(train_losses)), train_losses, 1)[0]
        val_trend = np.polyfit(range(len(val_losses)), val_losses, 1)[0]
        
        # Стабильность (стандартное отклонение последних 10 значений)
        train_stability = np.std(train_losses[-10:]) if len(train_losses) >= 10 else float('inf')
        val_stability = np.std(val_losses[-10:]) if len(val_losses) >= 10 else float('inf')
        
        return {
            "train_trend": "decreasing" if train_trend < -0.001 else "stable" if abs(train_trend) < 0.001 else "increasing",
            "val_trend": "decreasing" if val_trend < -0.001 else "stable" if abs(val_trend) < 0.001 else "increasing",
            "train_stability": train_stability,
            "val_stability": val_stability,
            "trend_values": {"train": train_trend, "val": val_trend}
        }
    
    def _analyze_quality_trends(self, attention_scores: List[float], gate_accuracies: List[float]) -> Dict[str, Any]:
        """Анализирует тренды качества."""
        if len(attention_scores) < 5:
            return {"trend": "insufficient_data"}
        
        # Средние значения за последние периоды
        recent_attention = np.mean(attention_scores[-10:]) if len(attention_scores) >= 10 else 0
        early_attention = np.mean(attention_scores[:10]) if len(attention_scores) >= 10 else 0
        
        recent_gate = np.mean(gate_accuracies[-10:]) if len(gate_accuracies) >= 10 else 0
        early_gate = np.mean(gate_accuracies[:10]) if len(gate_accuracies) >= 10 else 0
        
        return {
            "attention_improvement": recent_attention - early_attention,
            "gate_improvement": recent_gate - early_gate,
            "current_attention": recent_attention,
            "current_gate": recent_gate,
            "quality_status": self._determine_quality_status(recent_attention, recent_gate)
        }
    
    def _determine_quality_status(self, attention_score: float, gate_accuracy: float) -> str:
        """Определяет статус качества."""
        if attention_score >= 0.9 and gate_accuracy >= 0.8:
            return "excellent"
        elif attention_score >= 0.8 and gate_accuracy >= 0.7:
            return "good"
        elif attention_score >= 0.6 and gate_accuracy >= 0.5:
            return "fair"
        else:
            return "poor"
    
    def _assess_overfitting_risk(self, train_losses: List[float], val_losses: List[float]) -> Dict[str, Any]:
        """Оценивает риск переобучения."""
        if len(train_losses) < 10 or len(val_losses) < 10:
            return {"risk": "unknown", "confidence": 0.0}
        
        # Последние значения
        recent_train = np.mean(train_losses[-10:])
        recent_val = np.mean(val_losses[-10:])
        
        # Ранние значения
        early_train = np.mean(train_losses[:10])
        early_val = np.mean(val_losses[:10])
        
        # Разрыв между train и val loss
        current_gap = recent_val - recent_train
        initial_gap = early_val - early_train
        
        # Тренд validation loss
        val_trend = np.polyfit(range(len(val_losses[-20:])), val_losses[-20:], 1)[0]
        
        risk_score = 0.0
        
        # Увеличивающийся разрыв
        if current_gap > initial_gap * 1.5:
            risk_score += 0.3
            
        # Растущий validation loss
        if val_trend > 0.001:
            risk_score += 0.4
            
        # Очень низкий train loss при высоком val loss
        if recent_train < 0.1 and current_gap > 0.5:
            risk_score += 0.3
        
        risk_level = "high" if risk_score > 0.6 else "medium" if risk_score > 0.3 else "low"
        
        return {
            "risk": risk_level,
            "score": risk_score,
            "gap": current_gap,
            "val_trend": val_trend
        }
    
    def _assess_convergence(self, recent_history: List[Dict]) -> Dict[str, Any]:
        """Оценивает сходимость обучения."""
        if len(recent_history) < 20:
            return {"status": "insufficient_data"}
        
        # Анализируем стабильность метрик
        val_losses = [h["metrics"].get("val_loss", float('inf')) for h in recent_history[-20:]]
        
        # Коэффициент вариации (стандартное отклонение / среднее)
        cv = np.std(val_losses) / np.mean(val_losses) if np.mean(val_losses) > 0 else float('inf')
        
        # Тренд
        trend = np.polyfit(range(len(val_losses)), val_losses, 1)[0]
        
        if cv < 0.02 and abs(trend) < 0.001:
            status = "converged"
        elif cv < 0.05 and abs(trend) < 0.005:
            status = "converging"
        else:
            status = "not_converged"
        
        return {
            "status": status,
            "stability": cv,
            "trend": trend
        }
    
    def _update_epoch_estimate(self, progress_analysis: Dict[str, Any]):
        """Обновляет оценку оптимальных эпох на основе прогресса."""
        if not self.optimal_epochs_estimate:
            return
        
        # Факторы для корректировки
        overfitting_risk = progress_analysis.get("overfitting_risk", {}).get("risk", "low")
        convergence_status = progress_analysis.get("convergence_status", {}).get("status", "not_converged")
        quality_status = progress_analysis.get("quality_trends", {}).get("quality_status", "poor")
        
        adjustment_factor = 1.0
        
        # Корректировка на основе риска переобучения
        if overfitting_risk == "high":
            adjustment_factor *= 0.8  # Уменьшаем на 20%
        elif overfitting_risk == "medium":
            adjustment_factor *= 0.9  # Уменьшаем на 10%
        
        # Корректировка на основе сходимости
        if convergence_status == "converged" and quality_status in ["good", "excellent"]:
            adjustment_factor *= 0.9  # Можно остановиться раньше
        
        # Обновляем оценку
        self.optimal_epochs_estimate = int(self.optimal_epochs_estimate * adjustment_factor)
        
        # Обновляем уверенность
        if overfitting_risk == "low" and quality_status in ["good", "excellent"]:
            self.confidence_score = min(1.0, self.confidence_score + 0.1)
        elif overfitting_risk == "high":
            self.confidence_score = max(0.1, self.confidence_score - 0.2)
    
    def _generate_training_recommendations(self, current_epoch: int, progress_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Генерирует рекомендации по продолжению обучения."""
        recommendations = {
            "continue_training": True,
            "suggested_actions": [],
            "estimated_epochs_remaining": None,
            "confidence": self.confidence_score
        }
        
        overfitting_risk = progress_analysis.get("overfitting_risk", {}).get("risk", "low")
        convergence_status = progress_analysis.get("convergence_status", {}).get("status", "not_converged")
        quality_status = progress_analysis.get("quality_trends", {}).get("quality_status", "poor")
        
        # Рекомендации на основе анализа
        if overfitting_risk == "high":
            recommendations["suggested_actions"].append("Высокий риск переобучения - рассмотрите остановку")
            if quality_status in ["good", "excellent"]:
                recommendations["continue_training"] = False
                recommendations["suggested_actions"].append("Качество достаточное - рекомендуется остановка")
        
        if convergence_status == "converged":
            if quality_status in ["good", "excellent"]:
                recommendations["continue_training"] = False
                recommendations["suggested_actions"].append("Модель сошлась с хорошим качеством")
            else:
                recommendations["suggested_actions"].append("Модель сошлась, но качество низкое - возможно нужна корректировка параметров")
        
        # Оценка оставшихся эпох
        if self.optimal_epochs_estimate and current_epoch < self.optimal_epochs_estimate:
            remaining = self.optimal_epochs_estimate - current_epoch
            recommendations["estimated_epochs_remaining"] = remaining
        
        return recommendations
    
    def save_optimization_result(self, final_metrics: Dict[str, Any], actual_epochs: int, training_time_minutes: float):
        """Сохраняет результат оптимизации в базу данных."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            dataset_size = self.dataset_analysis.get("dataset_size_hours", 0)
            dataset_quality = self.dataset_analysis.get("quality_assessment", {}).get("overall_score", 0)
            voice_complexity = self.dataset_analysis.get("complexity_analysis", {}).get("category", "unknown")
            recommended_epochs = self.optimal_epochs_estimate or 0
            final_quality = final_metrics.get("final_quality_score", 0)
            
            # Оценка успешности (0-5)
            success_rating = self._calculate_success_rating(final_metrics, actual_epochs)
            
            cursor.execute('''
            INSERT INTO optimization_history 
            (dataset_size_hours, dataset_quality_score, voice_complexity, 
             recommended_epochs, actual_epochs, final_quality_score, 
             training_time_minutes, success_rating, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                dataset_size, dataset_quality, voice_complexity,
                recommended_epochs, actual_epochs, final_quality,
                training_time_minutes, success_rating,
                json.dumps(final_metrics)
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Результат оптимизации сохранен: {actual_epochs} эпох, качество: {final_quality:.3f}")
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения результата оптимизации: {e}")
    
    def _calculate_success_rating(self, final_metrics: Dict[str, Any], actual_epochs: int) -> int:
        """Вычисляет рейтинг успешности оптимизации (0-5)."""
        rating = 3  # Базовый рейтинг
        
        # Качество результата
        final_quality = final_metrics.get("final_quality_score", 0.5)
        if final_quality >= 0.9:
            rating += 2
        elif final_quality >= 0.7:
            rating += 1
        elif final_quality < 0.5:
            rating -= 1
        
        # Точность предсказания
        if self.optimal_epochs_estimate:
            error_ratio = abs(actual_epochs - self.optimal_epochs_estimate) / self.optimal_epochs_estimate
            if error_ratio <= 0.1:  # Ошибка менее 10%
                rating += 1
            elif error_ratio > 0.5:  # Ошибка более 50%
                rating -= 1
        
        return max(0, min(5, rating))
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Возвращает сводку оптимизации."""
        return {
            "dataset_analysis": self.dataset_analysis,
            "optimal_epochs_estimate": self.optimal_epochs_estimate,
            "confidence_score": self.confidence_score,
            "current_phase": self.current_phase,
            "training_progress": {
                "epochs_completed": len(self.training_history),
                "last_metrics": self.training_history[-1]["metrics"] if self.training_history else None
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Возвращает текущий статус оптимизатора."""
        return {
            "active": True,
            "status": "Оптимизация",
            "current_phase": self.current_phase,
            "optimal_epochs_estimate": self.optimal_epochs_estimate,
            "confidence_score": self.confidence_score,
            "total_decisions": len(self.optimization_decisions)
        }
    
    def get_recommendations(self) -> List[str]:
        """Возвращает текущие рекомендации оптимизатора."""
        if not self.dataset_analysis:
            return ["Анализ датасета не выполнен"]
        
        recommendations = []
        
        # Рекомендации на основе анализа датасета
        size_category = self.dataset_analysis.get("dataset_size_category", "unknown")
        if size_category == "very_small":
            recommendations.append("Очень маленький датасет - требуется больше эпох для обучения")
        elif size_category == "very_large":
            recommendations.append("Большой датасет - можно использовать меньше эпох")
        
        quality_assessment = self.dataset_analysis.get("quality_assessment", {})
        quality_score = quality_assessment.get("overall_score", 0.5)
        if quality_score < 0.5:
            recommendations.append("Низкое качество датасета - увеличить количество эпох")
        
        complexity_analysis = self.dataset_analysis.get("complexity_analysis", {})
        complexity_score = complexity_analysis.get("complexity_score", 0.5)
        if complexity_score > 0.7:
            recommendations.append("Сложный голос - требуется больше эпох для качественного обучения")
        
        # Рекомендации на основе текущего прогресса
        if self.optimal_epochs_estimate:
            recommendations.append(f"Рекомендуемое количество эпох: {self.optimal_epochs_estimate}")
        
        return recommendations[:3]  # До 3 рекомендаций
    
    def track_parameter_change(self, param_name: str, old_value: Any, new_value: Any, reason: str, step: int):
        """Отслеживает изменение параметра."""
        change_info = {
            'param_name': param_name,
            'old_value': old_value,
            'new_value': new_value,
            'reason': reason,
            'step': step,
            'timestamp': datetime.now().isoformat()
        }
        self.parameter_changes.append(change_info)
        
        # Ограничиваем историю изменений
        if len(self.parameter_changes) > 50:
            self.parameter_changes = self.parameter_changes[-50:]
    
    def get_parameter_changes(self) -> List[Dict[str, Any]]:
        """Возвращает последние изменения параметров."""
        return self.parameter_changes[-10:]  # Последние 10 изменений 