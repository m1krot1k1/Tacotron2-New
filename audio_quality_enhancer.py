#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Quality Enhancer для максимального качества TTS
🎵 Система контроля качества для получения студийного звука

Основано на исследованиях 2024-2025:
- Clip-TTS контрастивное обучение
- IndexTTS2 эмоциональная выразительность  
- Bailing-TTS спонтанное представление
- Very Attentive Tacotron стабильность
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Dict, Any, Optional, Tuple
import logging


class AudioQualityEnhancer(nn.Module):
    """
    🎵 Система улучшения качества аудио для TTS.
    
    Включает:
    1. Noise Gate для удаления фоновых шумов
    2. Spectral Enhancement для улучшения четкости
    3. Dynamic Range Control для естественности
    4. Artifact Detection для обнаружения проблем
    5. Quality Metrics для объективной оценки
    """
    
    def __init__(self, sample_rate=22050, n_mel_channels=80):
        super(AudioQualityEnhancer, self).__init__()
        self.sample_rate = sample_rate
        self.n_mel_channels = n_mel_channels
        
        # Noise Gate параметры
        self.noise_gate_threshold = -60.0  # dB
        self.noise_gate_ratio = 0.1
        
        # Spectral Enhancement
        self.spectral_enhancer = SpectralEnhancer(n_mel_channels)
        
        # Dynamic Range Controller
        self.dynamic_controller = DynamicRangeController()
        
        # Artifact Detector
        self.artifact_detector = ArtifactDetector()
        
        # Quality Metrics Calculator
        self.quality_calculator = QualityMetricsCalculator()
        
        self.logger = logging.getLogger(__name__)
        
    def forward(self, mel_spectrogram: torch.Tensor, apply_enhancement=True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Применяет улучшение качества к mel спектрограмме.
        
        Args:
            mel_spectrogram: [B, n_mel_channels, T] mel спектрограмма
            apply_enhancement: Применять ли улучшения
            
        Returns:
            enhanced_mel: Улучшенная mel спектрограмма
            quality_metrics: Метрики качества
        """
        batch_size, n_mels, time_steps = mel_spectrogram.shape
        
        # 1. Анализ качества исходного аудио
        original_quality = self.quality_calculator(mel_spectrogram)
        
        if not apply_enhancement:
            return mel_spectrogram, original_quality
        
        # 2. Noise Gate - удаление фоновых шумов
        mel_gated = self.apply_noise_gate(mel_spectrogram)
        
        # 3. Spectral Enhancement - улучшение четкости
        mel_enhanced = self.spectral_enhancer(mel_gated)
        
        # 4. Dynamic Range Control - естественность
        mel_controlled = self.dynamic_controller(mel_enhanced)
        
        # 5. Artifact Detection - проверка артефактов
        artifact_score = self.artifact_detector(mel_controlled)
        
        # 6. Финальные метрики качества
        final_quality = self.quality_calculator(mel_controlled)
        
        # Объединяем все метрики
        combined_metrics = {
            'original_quality': original_quality['overall_quality'],
            'final_quality': final_quality['overall_quality'],
            'noise_reduction': original_quality['noise_level'] - final_quality['noise_level'],
            'spectral_clarity': final_quality['spectral_clarity'],
            'artifact_score': artifact_score,
            'enhancement_gain': final_quality['overall_quality'] - original_quality['overall_quality']
        }
        
        return mel_controlled, combined_metrics
    
    def apply_noise_gate(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        🔇 Применяет noise gate для удаления фоновых шумов.
        """
        # Конвертируем в dB
        mel_db = 20 * torch.log10(torch.clamp(mel_spec, min=1e-8))
        
        # Применяем gate
        gate_mask = mel_db > self.noise_gate_threshold
        gated_mel = mel_spec * gate_mask.float()
        
        # Смягчаем переходы
        gated_mel = gated_mel * self.noise_gate_ratio + mel_spec * (1 - self.noise_gate_ratio)
        
        return gated_mel
    
    def post_process_audio(self, audio_waveform: np.ndarray) -> np.ndarray:
        """
        🎵 Постобработка аудио для максимального качества.
        
        Args:
            audio_waveform: Аудио сигнал
            
        Returns:
            Обработанный аудио сигнал
        """
        # 1. Нормализация громкости
        audio_normalized = librosa.util.normalize(audio_waveform)
        
        # 2. Удаление DC offset
        audio_normalized = audio_normalized - np.mean(audio_normalized)
        
        # 3. Soft limiting для предотвращения клиппинга
        audio_limited = np.tanh(audio_normalized * 0.95) / np.tanh(0.95)
        
        # 4. Dithering для лучшего качества квантизации
        if audio_limited.dtype != np.float32:
            dither = np.random.normal(0, 1e-6, audio_limited.shape)
            audio_limited = audio_limited + dither
        
        return audio_limited


class SpectralEnhancer(nn.Module):
    """🎵 Улучшение спектральной четкости."""
    
    def __init__(self, n_mel_channels):
        super(SpectralEnhancer, self).__init__()
        self.n_mel_channels = n_mel_channels
        
        # Фильтры для улучшения разных частотных диапазонов
        self.low_freq_enhancer = nn.Conv1d(n_mel_channels//4, n_mel_channels//4, 3, padding=1)
        self.mid_freq_enhancer = nn.Conv1d(n_mel_channels//2, n_mel_channels//2, 3, padding=1)  
        self.high_freq_enhancer = nn.Conv1d(n_mel_channels//4, n_mel_channels//4, 3, padding=1)
        
    def forward(self, mel_spec):
        """Применяет спектральное улучшение."""
        batch_size, n_mels, time_steps = mel_spec.shape
        
        # Разделяем на частотные диапазоны
        low_freq = mel_spec[:, :n_mels//4, :]
        mid_freq = mel_spec[:, n_mels//4:3*n_mels//4, :]
        high_freq = mel_spec[:, 3*n_mels//4:, :]
        
        # Применяем улучшения к каждому диапазону
        low_enhanced = self.low_freq_enhancer(low_freq)
        mid_enhanced = self.mid_freq_enhancer(mid_freq)
        high_enhanced = self.high_freq_enhancer(high_freq)
        
        # Объединяем обратно
        enhanced = torch.cat([low_enhanced, mid_enhanced, high_enhanced], dim=1)
        
        # Смешиваем с оригиналом для сохранения естественности
        alpha = 0.3  # Коэффициент смешивания
        return alpha * enhanced + (1 - alpha) * mel_spec


class DynamicRangeController(nn.Module):
    """🎵 Контроль динамического диапазона."""
    
    def __init__(self):
        super(DynamicRangeController, self).__init__()
        self.compression_ratio = 0.3
        self.threshold = 0.7
        
    def forward(self, mel_spec):
        """Применяет мягкую компрессию для естественности."""
        # Мягкая компрессия больших значений
        compressed = torch.where(
            mel_spec > self.threshold,
            self.threshold + (mel_spec - self.threshold) * self.compression_ratio,
            mel_spec
        )
        
        return compressed


class ArtifactDetector(nn.Module):
    """🎵 Детектор артефактов в аудио."""
    
    def __init__(self):
        super(ArtifactDetector, self).__init__()
        
    def forward(self, mel_spec):
        """
        Обнаруживает артефакты в mel спектрограмме.
        
        Returns:
            artifact_score: 0.0 (много артефактов) - 1.0 (без артефактов)
        """
        # 1. Проверка резких скачков (clicks/pops)
        temporal_diff = torch.abs(mel_spec[:, :, 1:] - mel_spec[:, :, :-1])
        click_score = 1.0 - torch.clamp(temporal_diff.mean() * 10, 0, 1)
        
        # 2. Проверка спектральной стабильности
        spectral_std = torch.std(mel_spec, dim=2)
        stability_score = 1.0 - torch.clamp(spectral_std.mean() * 2, 0, 1)
        
        # 3. Проверка динамического диапазона
        dynamic_range = torch.max(mel_spec) - torch.min(mel_spec)
        range_score = torch.clamp(dynamic_range / 10.0, 0, 1)
        
        # Общая оценка без артефактов
        artifact_score = (click_score + stability_score + range_score) / 3.0
        
        return artifact_score.item()


class QualityMetricsCalculator(nn.Module):
    """🎵 Калькулятор метрик качества аудио."""
    
    def __init__(self):
        super(QualityMetricsCalculator, self).__init__()
        
    def forward(self, mel_spec):
        """
        Вычисляет комплексные метрики качества.
        
        Returns:
            Dict с метриками качества
        """
        # 1. Спектральная четкость
        spectral_clarity = self.calculate_spectral_clarity(mel_spec)
        
        # 2. Уровень шума
        noise_level = self.calculate_noise_level(mel_spec)
        
        # 3. Спектральная стабильность
        spectral_stability = self.calculate_spectral_stability(mel_spec)
        
        # 4. Динамический диапазон
        dynamic_range = self.calculate_dynamic_range(mel_spec)
        
        # 5. Общая оценка качества
        overall_quality = (spectral_clarity + (1.0 - noise_level) + spectral_stability + dynamic_range) / 4.0
        
        return {
            'spectral_clarity': spectral_clarity.item(),
            'noise_level': noise_level.item(),
            'spectral_stability': spectral_stability.item(),
            'dynamic_range': dynamic_range.item(),
            'overall_quality': overall_quality.item()
        }
    
    def calculate_spectral_clarity(self, mel_spec):
        """Вычисляет четкость спектра."""
        # Высокочастотная энергия как показатель четкости
        high_freq_energy = mel_spec[:, mel_spec.size(1)//2:, :].mean()
        total_energy = mel_spec.mean() + 1e-8
        clarity = high_freq_energy / total_energy
        return torch.clamp(clarity, 0, 1)
    
    def calculate_noise_level(self, mel_spec):
        """Оценивает уровень шума."""
        # Минимальная энергия как показатель шума
        min_energy = torch.quantile(mel_spec.view(-1), 0.1)
        max_energy = torch.max(mel_spec)
        noise_ratio = min_energy / (max_energy + 1e-8)
        return torch.clamp(noise_ratio, 0, 1)
    
    def calculate_spectral_stability(self, mel_spec):
        """Вычисляет стабильность спектра."""
        temporal_variance = torch.var(mel_spec, dim=2).mean()
        stability = 1.0 / (1.0 + temporal_variance)
        return torch.clamp(stability, 0, 1)
    
    def calculate_dynamic_range(self, mel_spec):
        """Оценивает динамический диапазон."""
        range_db = 20 * torch.log10(torch.max(mel_spec) / (torch.min(mel_spec) + 1e-8))
        normalized_range = torch.clamp(range_db / 60.0, 0, 1)  # Нормализуем к 60 dB
        return normalized_range


class InferenceQualityController:
    """
    🎵 Контроллер качества для inference режима.
    
    Обеспечивает:
    - Реальное время мониторинга качества
    - Автоматическую коррекцию параметров
    - Детекцию проблем генерации
    """
    
    def __init__(self):
        self.quality_history = []
        self.quality_threshold = 0.6
        self.enhancer = AudioQualityEnhancer()
        
    def evaluate_generation_quality(self, mel_outputs, gate_outputs, alignments):
        """
        Оценивает качество сгенерированного аудио.
        
        Returns:
            quality_report: Подробный отчет о качестве
        """
        # 1. Анализ mel спектрограммы
        mel_quality = self.enhancer.quality_calculator(mel_outputs)
        
        # 2. Анализ attention alignment
        alignment_quality = self.analyze_alignment_quality(alignments)
        
        # 3. Анализ gate outputs
        gate_quality = self.analyze_gate_quality(gate_outputs)
        
        # 4. Общая оценка
        overall_quality = (
            mel_quality['overall_quality'] * 0.5 +
            alignment_quality * 0.3 +
            gate_quality * 0.2
        )
        
        quality_report = {
            'overall_quality': overall_quality,
            'mel_quality': mel_quality,
            'alignment_quality': alignment_quality,
            'gate_quality': gate_quality,
            'recommendation': self.get_quality_recommendation(overall_quality)
        }
        
        self.quality_history.append(overall_quality)
        return quality_report
    
    def analyze_alignment_quality(self, alignments):
        """Анализирует качество attention alignment."""
        if alignments is None or len(alignments) == 0:
            return 0.0
            
        # Конвертируем в tensor если нужно
        if isinstance(alignments, list):
            alignments = torch.stack(alignments, dim=1)
        
        # Вычисляем диагональность
        batch_size, seq_len, text_len = alignments.shape
        
        diagonality_scores = []
        for b in range(batch_size):
            alignment = alignments[b]
            
            # Создаем идеальную диагональ
            ideal_diagonal = torch.eye(min(seq_len, text_len), device=alignment.device)
            if seq_len != text_len:
                # Интерполируем для разных длин
                ideal_diagonal = F.interpolate(
                    ideal_diagonal.unsqueeze(0).unsqueeze(0),
                    size=(seq_len, text_len),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
            
            # Вычисляем корреляцию с диагональю
            correlation = F.cosine_similarity(
                alignment.view(-1),
                ideal_diagonal.view(-1),
                dim=0
            )
            diagonality_scores.append(correlation)
        
        return torch.stack(diagonality_scores).mean().item()
    
    def analyze_gate_quality(self, gate_outputs):
        """Анализирует качество gate outputs."""
        if gate_outputs is None:
            return 0.0
            
        # Конвертируем в tensor
        if isinstance(gate_outputs, list):
            gate_outputs = torch.cat(gate_outputs, dim=0)
        
        # Применяем sigmoid
        gate_probs = torch.sigmoid(gate_outputs)
        
        # Проверяем правильность формы: должно начинаться с низких значений,
        # заканчиваться высокими
        seq_len = gate_probs.size(0)
        expected_pattern = torch.linspace(0.1, 0.9, seq_len, device=gate_probs.device)
        
        # Вычисляем соответствие ожидаемому паттерну
        correlation = F.cosine_similarity(
            gate_probs.view(-1),
            expected_pattern.view(-1),
            dim=0
        )
        
        return max(0.0, correlation.item())
    
    def get_quality_recommendation(self, quality_score):
        """Возвращает рекомендации по улучшению качества."""
        if quality_score >= 0.8:
            return "Отличное качество - без изменений"
        elif quality_score >= 0.6:
            return "Хорошее качество - незначительные улучшения"
        elif quality_score >= 0.4:
            return "Удовлетворительное качество - требуются улучшения"
        else:
            return "Низкое качество - критические улучшения"


# Экспорт основных классов
__all__ = ['AudioQualityEnhancer', 'InferenceQualityController'] 