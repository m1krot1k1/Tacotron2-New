#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Quality Controller для Smart Tuner TTS
Революционная система контроля качества на основе исследований 2024-2025:
- Very Attentive Tacotron (Google, 2025)
- Llasa: Scaling LLM TTS (2025) 
- Muyan-TTS: Podcast Optimization (2025)
- MonoAlign: Robust LLM TTS (2024)
- Style-BERT-VITS2: Expressive TTS (2025)
"""

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import sqlite3
from datetime import datetime

class AdvancedQualityController:
    """
    Продвинутый контроллер качества TTS с возможностями:
    1. Диагностика quality в реальном времени
    2. Автоматическое исправление проблем качества
    3. Адаптивные пороги на основе фазы обучения
    4. Интеллектуальное управление guided attention
    5. Предотвращение артефактов и галлюцинаций
    """
    
    def __init__(self, config_path: str = "smart_tuner/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.logger = logging.getLogger(__name__)
        
        # Настройки контроллера качества
        self.quality_config = self.config.get('quality_monitoring', {})
        self.guided_attention_config = self.config.get('guided_attention', {})
        self.tts_metrics_config = self.config.get('tts_metrics', {})
        
        # История качества обучения
        self.quality_history = []
        self.attention_history = []
        self.gate_history = []
        self.mel_quality_history = []
        
        # Адаптивные пороги
        self.adaptive_thresholds = {
            'attention_diagonality': 0.7,
            'gate_accuracy': 0.75,
            'mel_clarity': 0.6,
            'training_stability': 0.8
        }
        
        # Состояние контроллера
        self.current_phase = "pre_alignment"
        self.quality_interventions = []
        self.last_intervention_step = 0
        
        # База данных качества
        self.db_path = "smart_tuner/quality_control_history.db"
        self._init_quality_database()
        
    def _init_quality_database(self):
        """Инициализирует базу данных для хранения истории качества."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                epoch INTEGER,
                training_phase TEXT,
                attention_diagonality REAL,
                attention_monotonicity REAL,
                attention_focus REAL,
                gate_accuracy REAL,
                gate_precision REAL,
                gate_recall REAL,
                mel_spectral_quality REAL,
                mel_temporal_consistency REAL,
                overall_quality_score REAL,
                quality_issues TEXT,
                interventions_applied TEXT
            )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info(f"База данных контроля качества инициализирована: {self.db_path}")
        except Exception as e:
            self.logger.error(f"Ошибка инициализации базы данных качества: {e}")
    
    def analyze_training_quality(self, epoch: int, metrics: Dict[str, Any], 
                                attention_weights: Optional[torch.Tensor] = None,
                                gate_outputs: Optional[torch.Tensor] = None,
                                mel_outputs: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Комплексный анализ качества обучения TTS.
        
        Args:
            epoch: Номер эпохи
            metrics: Метрики обучения
            attention_weights: Веса attention (B, T_out, T_in)
            gate_outputs: Выходы gate (B, T_out)
            mel_outputs: Mel спектрограммы (B, n_mels, T_out)
            
        Returns:
            Словарь с анализом качества и рекомендациями
        """
        quality_analysis = {
            'epoch': epoch,
            'phase': self._determine_training_phase(epoch, metrics),
            'attention_quality': {},
            'gate_quality': {},
            'mel_quality': {},
            'overall_quality_score': 0.0,
            'quality_issues': [],
            'recommended_interventions': [],
            'quality_trend': 'stable'
        }
        
        # Анализ качества attention
        if attention_weights is not None:
            quality_analysis['attention_quality'] = self._analyze_attention_quality(attention_weights)
        
        # Анализ качества gate
        if gate_outputs is not None:
            quality_analysis['gate_quality'] = self._analyze_gate_quality(gate_outputs)
        
        # Анализ качества mel
        if mel_outputs is not None:
            quality_analysis['mel_quality'] = self._analyze_mel_quality(mel_outputs)
        
        # Обнаружение проблем качества
        quality_issues = self._detect_quality_issues(quality_analysis)
        quality_analysis['quality_issues'] = quality_issues
        
        # Рекомендации по вмешательству
        interventions = self._recommend_quality_interventions(quality_analysis)
        quality_analysis['recommended_interventions'] = interventions
        
        # Общий скор качества
        overall_score = self._calculate_overall_quality_score(quality_analysis)
        quality_analysis['overall_quality_score'] = overall_score
        
        # Тренд качества
        quality_trend = self._analyze_quality_trend()
        quality_analysis['quality_trend'] = quality_trend
        
        # Сохранение в историю
        self.quality_history.append(quality_analysis)
        self._save_quality_to_database(quality_analysis)
        
        # Обновление адаптивных порогов
        self._update_adaptive_thresholds(quality_analysis)
        
        return quality_analysis
    
    def _analyze_attention_quality(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """
        Анализирует качество attention matrix на основе современных исследований.
        """
        # attention_weights: (B, T_out, T_in)
        B, T_out, T_in = attention_weights.shape
        
        attention_quality = {
            'diagonality_score': 0.0,
            'monotonicity_score': 0.0,
            'focus_score': 0.0,
            'entropy_score': 0.0,
            'alignment_consistency': 0.0,
            'attention_drift': 0.0
        }
        
        # Переводим в numpy для анализа
        attention_np = attention_weights.detach().cpu().numpy()
        
        for b in range(B):
            att_matrix = attention_np[b]  # (T_out, T_in)
            
            # 1. Диагональность (из Very Attentive Tacotron)
            diagonality = self._calculate_attention_diagonality(att_matrix)
            attention_quality['diagonality_score'] += diagonality
            
            # 2. Монотонность (из MonoAlign исследований)
            monotonicity = self._calculate_attention_monotonicity(att_matrix)
            attention_quality['monotonicity_score'] += monotonicity
            
            # 3. Фокусировка (концентрация attention)
            focus = self._calculate_attention_focus(att_matrix)
            attention_quality['focus_score'] += focus
            
            # 4. Энтропия (мера неопределенности)
            entropy = self._calculate_attention_entropy(att_matrix)
            attention_quality['entropy_score'] += entropy
            
            # 5. Согласованность alignment
            consistency = self._calculate_alignment_consistency(att_matrix)
            attention_quality['alignment_consistency'] += consistency
            
            # 6. Дрейф attention
            drift = self._calculate_attention_drift(att_matrix)
            attention_quality['attention_drift'] += drift
        
        # Усредняем по batch
        for key in attention_quality:
            attention_quality[key] /= B
        
        return attention_quality
    
    def _calculate_attention_diagonality(self, attention_matrix: np.ndarray) -> float:
        """
        Вычисляет диагональность attention matrix.
        Основано на Very Attentive Tacotron (2025).
        """
        T_out, T_in = attention_matrix.shape
        
        # Создаем идеальную диагональную матрицу
        ideal_diagonal = np.zeros((T_out, T_in))
        
        # Заполняем диагональ
        for i in range(T_out):
            # Идеальная позиция на диагонали
            ideal_pos = int(i * T_in / T_out)
            if ideal_pos < T_in:
                ideal_diagonal[i, ideal_pos] = 1.0
        
        # Вычисляем корреляцию с идеальной диагональю
        attention_flat = attention_matrix.flatten()
        diagonal_flat = ideal_diagonal.flatten()
        
        correlation = np.corrcoef(attention_flat, diagonal_flat)[0, 1]
        
        # Обрабатываем NaN (может возникнуть при нулевой дисперсии)
        if np.isnan(correlation):
            correlation = 0.0
        
        return max(0.0, correlation)
    
    def _calculate_attention_monotonicity(self, attention_matrix: np.ndarray) -> float:
        """
        Вычисляет монотонность attention.
        Основано на MonoAlign исследованиях (2024).
        """
        T_out, T_in = attention_matrix.shape
        
        # Находим пик attention для каждого выходного шага
        peak_positions = np.argmax(attention_matrix, axis=1)
        
        # Вычисляем монотонность как долю последовательных увеличений
        monotonic_steps = 0
        total_steps = len(peak_positions) - 1
        
        if total_steps == 0:
            return 1.0
        
        for i in range(total_steps):
            if peak_positions[i+1] >= peak_positions[i]:
                monotonic_steps += 1
        
        return monotonic_steps / total_steps
    
    def _calculate_attention_focus(self, attention_matrix: np.ndarray) -> float:
        """
        Вычисляет фокусировку attention (обратное к размытости).
        """
        # Вычисляем среднюю остроту attention по всем выходным шагам
        focus_scores = []
        
        for t_out in range(attention_matrix.shape[0]):
            att_weights = attention_matrix[t_out]
            
            # Вычисляем концентрацию как обратное к энтропии
            # Убираем нули для избежания log(0)
            att_weights_clean = att_weights + 1e-10
            entropy = -np.sum(att_weights_clean * np.log(att_weights_clean))
            
            # Нормализуем энтропию
            max_entropy = np.log(len(att_weights))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # Focus = 1 - normalized_entropy
            focus = 1.0 - normalized_entropy
            focus_scores.append(focus)
        
        return np.mean(focus_scores)
    
    def _calculate_attention_entropy(self, attention_matrix: np.ndarray) -> float:
        """
        Вычисляет среднюю энтропию attention.
        """
        entropies = []
        
        for t_out in range(attention_matrix.shape[0]):
            att_weights = attention_matrix[t_out]
            
            # Убираем нули
            att_weights_clean = att_weights + 1e-10
            entropy = -np.sum(att_weights_clean * np.log(att_weights_clean))
            
            # Нормализуем
            max_entropy = np.log(len(att_weights))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            entropies.append(normalized_entropy)
        
        return np.mean(entropies)
    
    def _calculate_alignment_consistency(self, attention_matrix: np.ndarray) -> float:
        """
        Вычисляет согласованность alignment между соседними шагами.
        """
        T_out = attention_matrix.shape[0]
        
        if T_out < 2:
            return 1.0
        
        consistencies = []
        
        for i in range(T_out - 1):
            att_curr = attention_matrix[i]
            att_next = attention_matrix[i + 1]
            
            # Вычисляем косинусное сходство
            dot_product = np.dot(att_curr, att_next)
            norm_curr = np.linalg.norm(att_curr)
            norm_next = np.linalg.norm(att_next)
            
            if norm_curr > 0 and norm_next > 0:
                similarity = dot_product / (norm_curr * norm_next)
                consistencies.append(max(0.0, similarity))
        
        return np.mean(consistencies) if consistencies else 0.0
    
    def _calculate_attention_drift(self, attention_matrix: np.ndarray) -> float:
        """
        Вычисляет дрейф attention (нежелательные скачки).
        """
        peak_positions = np.argmax(attention_matrix, axis=1)
        
        if len(peak_positions) < 2:
            return 0.0
        
        # Вычисляем среднее отклонение от ожидаемого прогресса
        expected_step = len(peak_positions[0]) / len(peak_positions)
        drifts = []
        
        for i in range(1, len(peak_positions)):
            expected_pos = peak_positions[0] + i * expected_step
            actual_pos = peak_positions[i]
            drift = abs(actual_pos - expected_pos) / len(peak_positions[0])
            drifts.append(drift)
        
        return np.mean(drifts)
    
    def _analyze_gate_quality(self, gate_outputs: torch.Tensor) -> Dict[str, Any]:
        """
        Анализирует качество gate outputs.
        """
        # gate_outputs: (B, T_out)
        gate_np = gate_outputs.detach().cpu().numpy()
        
        gate_quality = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'premature_stop_rate': 0.0,
            'gate_stability': 0.0
        }
        
        B, T_out = gate_np.shape
        
        for b in range(B):
            gates = gate_np[b]
            
            # Предполагаем, что gate должен активироваться ближе к концу
            # Идеальная позиция - последние 10% последовательности
            ideal_start = int(T_out * 0.9)
            
            # Создаем идеальную gate последовательность
            ideal_gates = np.zeros(T_out)
            ideal_gates[ideal_start:] = 1.0
            
            # Бинаризуем предсказания gate
            binary_gates = (gates > 0.5).astype(int)
            
            # Вычисляем метрики
            tp = np.sum((binary_gates == 1) & (ideal_gates == 1))
            fp = np.sum((binary_gates == 1) & (ideal_gates == 0))
            fn = np.sum((binary_gates == 0) & (ideal_gates == 1))
            tn = np.sum((binary_gates == 0) & (ideal_gates == 0))
            
            accuracy = (tp + tn) / T_out if T_out > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            gate_quality['accuracy'] += accuracy
            gate_quality['precision'] += precision
            gate_quality['recall'] += recall
            gate_quality['f1_score'] += f1
            
            # Проверяем преждевременную остановку
            first_gate_activation = np.argmax(binary_gates)
            premature_rate = 1.0 - (first_gate_activation / T_out) if T_out > 0 else 0
            gate_quality['premature_stop_rate'] += premature_rate
            
            # Стабильность gate (низкая вариация)
            gate_variance = np.var(gates)
            stability = 1.0 / (1.0 + gate_variance)  # Меньшая вариация = больше стабильности
            gate_quality['gate_stability'] += stability
        
        # Усредняем по batch
        for key in gate_quality:
            gate_quality[key] /= B
        
        return gate_quality
    
    def _analyze_mel_quality(self, mel_outputs: torch.Tensor) -> Dict[str, Any]:
        """
        Анализирует качество mel спектрограмм.
        """
        # mel_outputs: (B, n_mels, T_out)
        mel_np = mel_outputs.detach().cpu().numpy()
        
        mel_quality = {
            'spectral_quality': 0.0,
            'temporal_consistency': 0.0,
            'harmonic_clarity': 0.0,
            'noise_level': 0.0,
            'dynamic_range': 0.0
        }
        
        B, n_mels, T_out = mel_np.shape
        
        for b in range(B):
            mel_spec = mel_np[b]  # (n_mels, T_out)
            
            # 1. Спектральное качество (энергия в важных частотах)
            low_freq_energy = np.mean(mel_spec[:n_mels//3])  # Низкие частоты
            mid_freq_energy = np.mean(mel_spec[n_mels//3:2*n_mels//3])  # Средние частоты
            high_freq_energy = np.mean(mel_spec[2*n_mels//3:])  # Высокие частоты
            
            # Хорошее качество - баланс энергий
            spectral_balance = 1.0 - np.std([low_freq_energy, mid_freq_energy, high_freq_energy])
            mel_quality['spectral_quality'] += max(0.0, spectral_balance)
            
            # 2. Временная согласованность
            temporal_diffs = np.diff(mel_spec, axis=1)
            temporal_variance = np.mean(np.var(temporal_diffs, axis=1))
            temporal_consistency = 1.0 / (1.0 + temporal_variance)
            mel_quality['temporal_consistency'] += temporal_consistency
            
            # 3. Гармоническая четкость (пики в спектре)
            mel_fft = np.fft.fft(mel_spec, axis=0)
            harmonic_peaks = np.sum(np.abs(mel_fft) > np.mean(np.abs(mel_fft)) * 2)
            harmonic_clarity = harmonic_peaks / n_mels
            mel_quality['harmonic_clarity'] += harmonic_clarity
            
            # 4. Уровень шума (высокочастотная энергия)
            noise_level = np.mean(mel_spec[-n_mels//4:])  # Самые высокие частоты
            mel_quality['noise_level'] += noise_level
            
            # 5. Динамический диапазон
            dynamic_range = np.max(mel_spec) - np.min(mel_spec)
            mel_quality['dynamic_range'] += dynamic_range
        
        # Усредняем по batch
        for key in mel_quality:
            mel_quality[key] /= B
        
        return mel_quality
    
    def _detect_quality_issues(self, quality_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Обнаруживает проблемы качества на основе анализа.
        """
        issues = []
        
        attention_quality = quality_analysis.get('attention_quality', {})
        gate_quality = quality_analysis.get('gate_quality', {})
        mel_quality = quality_analysis.get('mel_quality', {})
        
        # Проблемы attention
        if attention_quality.get('diagonality_score', 0) < self.adaptive_thresholds['attention_diagonality']:
            issues.append({
                'type': 'attention_misalignment',
                'severity': 'high',
                'description': f"Низкая диагональность attention: {attention_quality.get('diagonality_score', 0):.3f}",
                'recommended_fix': 'increase_guided_attention_weight'
            })
        
        if attention_quality.get('monotonicity_score', 0) < 0.7:
            issues.append({
                'type': 'attention_non_monotonic',
                'severity': 'medium',
                'description': f"Нарушение монотонности: {attention_quality.get('monotonicity_score', 0):.3f}",
                'recommended_fix': 'apply_monotonic_loss'
            })
        
        if attention_quality.get('focus_score', 0) < 0.5:
            issues.append({
                'type': 'attention_blurry',
                'severity': 'medium',
                'description': f"Размытое attention: {attention_quality.get('focus_score', 0):.3f}",
                'recommended_fix': 'reduce_attention_dropout'
            })
        
        # Проблемы gate
        if gate_quality.get('accuracy', 0) < self.adaptive_thresholds['gate_accuracy']:
            issues.append({
                'type': 'gate_inaccuracy',
                'severity': 'high',
                'description': f"Низкая точность gate: {gate_quality.get('accuracy', 0):.3f}",
                'recommended_fix': 'increase_gate_loss_weight'
            })
        
        if gate_quality.get('premature_stop_rate', 0) > 0.3:
            issues.append({
                'type': 'premature_stopping',
                'severity': 'high',
                'description': f"Преждевременная остановка: {gate_quality.get('premature_stop_rate', 0):.3f}",
                'recommended_fix': 'adjust_gate_threshold'
            })
        
        # Проблемы mel
        if mel_quality.get('spectral_quality', 0) < self.adaptive_thresholds['mel_clarity']:
            issues.append({
                'type': 'poor_spectral_quality',
                'severity': 'medium',
                'description': f"Низкое спектральное качество: {mel_quality.get('spectral_quality', 0):.3f}",
                'recommended_fix': 'add_spectral_loss'
            })
        
        if mel_quality.get('noise_level', 0) > 0.3:
            issues.append({
                'type': 'high_noise_level',
                'severity': 'medium',
                'description': f"Высокий уровень шума: {mel_quality.get('noise_level', 0):.3f}",
                'recommended_fix': 'improve_audio_preprocessing'
            })
        
        return issues
    
    def _recommend_quality_interventions(self, quality_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Рекомендует вмешательства для улучшения качества.
        """
        interventions = []
        issues = quality_analysis.get('quality_issues', [])
        
        for issue in issues:
            recommended_fix = issue.get('recommended_fix')
            
            if recommended_fix == 'increase_guided_attention_weight':
                interventions.append({
                    'type': 'guided_attention_boost',
                    'params': {
                        'guide_loss_weight_multiplier': 1.5,
                        'guide_decay_slower': 0.9995
                    },
                    'description': 'Усиление guided attention для лучшего alignment'
                })
            
            elif recommended_fix == 'apply_monotonic_loss':
                interventions.append({
                    'type': 'monotonic_loss_addition',
                    'params': {
                        'monotonic_loss_weight': 0.1
                    },
                    'description': 'Добавление monotonic loss для принуждения к монотонности'
                })
            
            elif recommended_fix == 'reduce_attention_dropout':
                interventions.append({
                    'type': 'attention_dropout_reduction',
                    'params': {
                        'attention_dropout_multiplier': 0.5
                    },
                    'description': 'Снижение attention dropout для большей фокусировки'
                })
            
            elif recommended_fix == 'increase_gate_loss_weight':
                interventions.append({
                    'type': 'gate_loss_boost',
                    'params': {
                        'gate_loss_weight_multiplier': 1.3
                    },
                    'description': 'Усиление gate loss для лучшей точности'
                })
            
            elif recommended_fix == 'adjust_gate_threshold':
                interventions.append({
                    'type': 'gate_threshold_adjustment',
                    'params': {
                        'gate_threshold_adjustment': -0.1
                    },
                    'description': 'Снижение порога gate для предотвращения преждевременной остановки'
                })
            
            elif recommended_fix == 'add_spectral_loss':
                interventions.append({
                    'type': 'spectral_loss_addition',
                    'params': {
                        'spectral_loss_weight': 0.2
                    },
                    'description': 'Добавление spectral loss для улучшения качества mel'
                })
        
        return interventions
    
    def _calculate_overall_quality_score(self, quality_analysis: Dict[str, Any]) -> float:
        """
        Вычисляет общий скор качества на основе всех метрик.
        """
        attention_quality = quality_analysis.get('attention_quality', {})
        gate_quality = quality_analysis.get('gate_quality', {})
        mel_quality = quality_analysis.get('mel_quality', {})
        
        # Веса для разных аспектов качества
        weights = {
            'attention': 0.4,
            'gate': 0.35,
            'mel': 0.25
        }
        
        # Attention score
        attention_score = (
            attention_quality.get('diagonality_score', 0) * 0.3 +
            attention_quality.get('monotonicity_score', 0) * 0.25 +
            attention_quality.get('focus_score', 0) * 0.25 +
            attention_quality.get('alignment_consistency', 0) * 0.2
        )
        
        # Gate score
        gate_score = (
            gate_quality.get('accuracy', 0) * 0.4 +
            gate_quality.get('f1_score', 0) * 0.3 +
            (1.0 - gate_quality.get('premature_stop_rate', 1)) * 0.3
        )
        
        # Mel score
        mel_score = (
            mel_quality.get('spectral_quality', 0) * 0.4 +
            mel_quality.get('temporal_consistency', 0) * 0.3 +
            mel_quality.get('harmonic_clarity', 0) * 0.3
        )
        
        # Общий скор
        overall_score = (
            attention_score * weights['attention'] +
            gate_score * weights['gate'] +
            mel_score * weights['mel']
        )
        
        return overall_score
    
    def _analyze_quality_trend(self) -> str:
        """
        Анализирует тренд качества за последние несколько эпох.
        """
        if len(self.quality_history) < 3:
            return 'insufficient_data'
        
        recent_scores = [q['overall_quality_score'] for q in self.quality_history[-5:]]
        
        # Линейная регрессия для определения тренда
        x = np.arange(len(recent_scores))
        coeffs = np.polyfit(x, recent_scores, 1)
        slope = coeffs[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'degrading'
        else:
            return 'stable'
    
    def _update_adaptive_thresholds(self, quality_analysis: Dict[str, Any]):
        """
        Обновляет адаптивные пороги на основе фазы обучения и прогресса.
        """
        phase = quality_analysis['phase']
        
        # Обновляем пороги в зависимости от фазы
        if phase == 'pre_alignment':
            self.adaptive_thresholds['attention_diagonality'] = 0.5
            self.adaptive_thresholds['gate_accuracy'] = 0.6
        elif phase == 'alignment_learning':
            self.adaptive_thresholds['attention_diagonality'] = 0.7
            self.adaptive_thresholds['gate_accuracy'] = 0.75
        elif phase == 'quality_optimization':
            self.adaptive_thresholds['attention_diagonality'] = 0.85
            self.adaptive_thresholds['gate_accuracy'] = 0.8
        elif phase == 'fine_tuning':
            self.adaptive_thresholds['attention_diagonality'] = 0.9
            self.adaptive_thresholds['gate_accuracy'] = 0.85
    
    def _determine_training_phase(self, epoch: int, metrics: Dict[str, Any]) -> str:
        """
        Определяет текущую фазу обучения.
        """
        # Используем эпохи и качество метрик для определения фазы
        attention_score = metrics.get('attention_alignment_score', 0)
        gate_accuracy = metrics.get('gate_accuracy', 0)
        
        if epoch < 500 or attention_score < 0.5:
            return 'pre_alignment'
        elif epoch < 2000 and attention_score < 0.8:
            return 'alignment_learning'
        elif epoch < 3000 and attention_score >= 0.8:
            return 'quality_optimization'
        else:
            return 'fine_tuning'
    
    def _save_quality_to_database(self, quality_analysis: Dict[str, Any]):
        """
        Сохраняет анализ качества в базу данных.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            attention_quality = quality_analysis.get('attention_quality', {})
            gate_quality = quality_analysis.get('gate_quality', {})
            mel_quality = quality_analysis.get('mel_quality', {})
            
            cursor.execute('''
            INSERT INTO quality_history (
                epoch, training_phase, attention_diagonality, attention_monotonicity,
                attention_focus, gate_accuracy, gate_precision, gate_recall,
                mel_spectral_quality, mel_temporal_consistency, overall_quality_score,
                quality_issues, interventions_applied
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                quality_analysis['epoch'],
                quality_analysis['phase'],
                attention_quality.get('diagonality_score', 0),
                attention_quality.get('monotonicity_score', 0),
                attention_quality.get('focus_score', 0),
                gate_quality.get('accuracy', 0),
                gate_quality.get('precision', 0),
                gate_quality.get('recall', 0),
                mel_quality.get('spectral_quality', 0),
                mel_quality.get('temporal_consistency', 0),
                quality_analysis['overall_quality_score'],
                json.dumps(quality_analysis['quality_issues']),
                json.dumps(quality_analysis['recommended_interventions'])
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Ошибка сохранения данных качества: {e}")
    
    def apply_quality_intervention(self, intervention: Dict[str, Any], 
                                  current_hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Применяет вмешательство для улучшения качества.
        
        Args:
            intervention: Описание вмешательства
            current_hyperparams: Текущие гиперпараметры
            
        Returns:
            Обновленные гиперпараметры
        """
        new_hyperparams = current_hyperparams.copy()
        intervention_type = intervention['type']
        params = intervention.get('params', {})
        
        if intervention_type == 'guided_attention_boost':
            multiplier = params.get('guide_loss_weight_multiplier', 1.5)
            new_hyperparams['guide_loss_weight'] = (
                current_hyperparams.get('guide_loss_weight', 1.0) * multiplier
            )
            
            slower_decay = params.get('guide_decay_slower', 0.9995)
            new_hyperparams['guide_decay'] = slower_decay
        
        elif intervention_type == 'attention_dropout_reduction':
            multiplier = params.get('attention_dropout_multiplier', 0.5)
            new_hyperparams['p_attention_dropout'] = (
                current_hyperparams.get('p_attention_dropout', 0.1) * multiplier
            )
        
        elif intervention_type == 'gate_loss_boost':
            multiplier = params.get('gate_loss_weight_multiplier', 1.3)
            new_hyperparams['gate_loss_weight'] = (
                current_hyperparams.get('gate_loss_weight', 1.0) * multiplier
            )
        
        elif intervention_type == 'gate_threshold_adjustment':
            adjustment = params.get('gate_threshold_adjustment', -0.1)
            new_hyperparams['gate_threshold'] = (
                current_hyperparams.get('gate_threshold', 0.5) + adjustment
            )
        
        # Записываем вмешательство
        self.quality_interventions.append({
            'epoch': len(self.quality_history),
            'intervention': intervention,
            'timestamp': datetime.now().isoformat()
        })
        
        self.last_intervention_step = len(self.quality_history)
        
        return new_hyperparams
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """
        Возвращает сводку по качеству обучения.
        """
        if not self.quality_history:
            return {'status': 'no_data'}
        
        latest_quality = self.quality_history[-1]
        
        summary = {
            'current_phase': latest_quality['phase'],
            'overall_quality_score': latest_quality['overall_quality_score'],
            'quality_trend': latest_quality['quality_trend'],
            'active_issues': len(latest_quality['quality_issues']),
            'interventions_applied': len(self.quality_interventions),
            'training_epochs_analyzed': len(self.quality_history),
            'quality_breakdown': {
                'attention_diagonality': latest_quality['attention_quality'].get('diagonality_score', 0),
                'gate_accuracy': latest_quality['gate_quality'].get('accuracy', 0),
                'mel_quality': latest_quality['mel_quality'].get('spectral_quality', 0)
            },
            'recommendations': latest_quality['recommended_interventions'][:3]  # Топ-3 рекомендации
        }
        
        return summary 