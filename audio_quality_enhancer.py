#!/usr/bin/env python3
"""
🎧 Модуль для улучшения качества аудио данных в TTS pipeline
Автор: AI Assistant для проекта Intelligent TTS Training Pipeline

Этот модуль предоставляет комплексную обработку аудио данных для получения
максимально качественного человеческого голоса без артефактов.
"""

import torch
import torch.nn.functional as F
import numpy as np
import librosa
from scipy import signal
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioQualityEnhancer:
    """Класс для улучшения качества аудио данных"""
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 n_mel_channels: int = 80,
                 quality_threshold: float = 0.7):
        self.sample_rate = sample_rate
        self.n_mel_channels = n_mel_channels
        self.quality_threshold = quality_threshold
        
        # Параметры для улучшения качества
        self.noise_gate_threshold = -40  # dB
        self.dynamic_range_target = 60   # dB
        
    def enhance_mel_spectrogram(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        🎵 Улучшает качество мел-спектрограммы
        """
        # 1. Нормализация динамического диапазона
        mel_spec = self._normalize_dynamic_range(mel_spec)
        
        # 2. Подавление шума в тихих участках
        mel_spec = self._apply_noise_gate(mel_spec)
        
        # 3. Сглаживание резких переходов
        mel_spec = self._smooth_transitions(mel_spec)
        
        # 4. Улучшение спектральной четкости
        mel_spec = self._enhance_spectral_clarity(mel_spec)
        
        return mel_spec
    
    def _normalize_dynamic_range(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Нормализует динамический диапазон"""
        # Вычисляем текущий динамический диапазон
        mel_min = torch.min(mel_spec)
        mel_max = torch.max(mel_spec)
        current_range = mel_max - mel_min
        
        if current_range > 0:
            # Целевой диапазон для лучшего качества
            target_min = -8.0
            target_max = 2.0
            target_range = target_max - target_min
            
            # Нормализация
            mel_spec = (mel_spec - mel_min) / current_range * target_range + target_min
        
        return mel_spec
    
    def _apply_noise_gate(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Применяет noise gate для подавления фонового шума"""
        # Определяем порог как процентиль от общей энергии
        energy = torch.sum(mel_spec, dim=1, keepdim=True)
        threshold = torch.quantile(energy, 0.1)  # 10-й процентиль
        
        # Создаем маску для noise gate
        gate_mask = (energy > threshold).float()
        
        # Плавный переход вместо резкого отсечения
        smooth_mask = torch.clamp(gate_mask + 0.1, 0, 1)
        
        return mel_spec * smooth_mask
    
    def _smooth_transitions(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Сглаживает резкие переходы во времени"""
        # Применяем легкое сглаживание по временной оси
        kernel_size = 3
        kernel = torch.ones(1, 1, kernel_size) / kernel_size
        kernel = kernel.to(mel_spec.device)
        
        # Паддинг для сохранения размера
        padded = F.pad(mel_spec.unsqueeze(0), (kernel_size//2, kernel_size//2), mode='reflect')
        
        smoothed = F.conv1d(padded, kernel)
        
        return smoothed.squeeze(0)
    
    def _enhance_spectral_clarity(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Улучшает спектральную четкость"""
        # Небольшое усиление контраста для лучшей четкости
        enhanced = mel_spec * 1.02
        
        # Ограничиваем диапазон чтобы избежать клиппинга
        enhanced = torch.clamp(enhanced, -10.0, 4.0)
        
        return enhanced
    
    def assess_mel_quality(self, mel_spec: torch.Tensor) -> Dict[str, float]:
        """
        📊 Оценивает качество мел-спектрограммы
        """
        quality_metrics = {}
        
        # 1. Динамический диапазон
        mel_min = torch.min(mel_spec)
        mel_max = torch.max(mel_spec)
        dynamic_range = (mel_max - mel_min).item()
        quality_metrics['dynamic_range'] = min(1.0, dynamic_range / 12.0)
        
        # 2. Спектральная плотность
        energy_per_frame = torch.sum(mel_spec, dim=0)
        energy_variance = torch.var(energy_per_frame).item()
        quality_metrics['spectral_density'] = min(1.0, energy_variance / 100.0)
        
        # 3. Отношение сигнал/шум (приблизительно)
        signal_energy = torch.mean(energy_per_frame[energy_per_frame > torch.quantile(energy_per_frame, 0.5)])
        noise_energy = torch.mean(energy_per_frame[energy_per_frame <= torch.quantile(energy_per_frame, 0.1)])
        snr = (signal_energy / (noise_energy + 1e-8)).item()
        quality_metrics['snr_estimate'] = min(1.0, snr / 20.0)
        
        # 4. Спектральная стабильность
        frame_diff = torch.diff(energy_per_frame)
        stability = 1.0 / (1.0 + torch.std(frame_diff).item())
        quality_metrics['spectral_stability'] = stability
        
        # Общая оценка качества
        overall_quality = np.mean(list(quality_metrics.values()))
        quality_metrics['overall_quality'] = overall_quality
        
        return quality_metrics


class InferenceQualityController:
    """Контроллер качества для инференса"""
    
    def __init__(self):
        self.quality_enhancer = AudioQualityEnhancer()
        
    def enhance_inference_output(self, 
                                mel_outputs: torch.Tensor,
                                gate_outputs: torch.Tensor,
                                alignments: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        🎯 Улучшает качество выходных данных инференса
        """
        # 1. Улучшаем качество мел-спектрограммы
        enhanced_mel = self.quality_enhancer.enhance_mel_spectrogram(mel_outputs)
        
        # 2. Оцениваем качество
        quality_metrics = self.quality_enhancer.assess_mel_quality(enhanced_mel)
        
        # 3. Добавляем метрики attention quality
        attention_quality = self._assess_attention_quality(alignments)
        quality_metrics.update(attention_quality)
        
        # 4. Добавляем метрики gate quality
        gate_quality = self._assess_gate_quality(gate_outputs)
        quality_metrics.update(gate_quality)
        
        return enhanced_mel, quality_metrics
    
    def _assess_attention_quality(self, alignments: torch.Tensor) -> Dict[str, float]:
        """Оценивает качество attention alignment"""
        if alignments.dim() != 3:
            return {'attention_quality': 0.0}
            
        # Берем первый пример из батча
        alignment = alignments[0]  # [encoder_len, decoder_len]
        
        # Проверяем диагональность
        diagonality = self._calculate_diagonality(alignment)
        
        # Проверяем фокусировку (не слишком размытая)
        focus = self._calculate_focus(alignment)
        
        # Проверяем монотонность
        monotonicity = self._calculate_monotonicity(alignment)
        
        attention_quality = (diagonality + focus + monotonicity) / 3.0
        
        return {
            'attention_quality': attention_quality,
            'attention_diagonality': diagonality,
            'attention_focus': focus,
            'attention_monotonicity': monotonicity
        }
    
    def _calculate_diagonality(self, alignment: torch.Tensor) -> float:
        """Вычисляет диагональность alignment матрицы"""
        encoder_len, decoder_len = alignment.shape
        
        # Создаем идеальную диагональную матрицу
        ideal_diag = torch.zeros_like(alignment)
        for i in range(decoder_len):
            encoder_pos = int(i * encoder_len / decoder_len)
            if encoder_pos < encoder_len:
                ideal_diag[encoder_pos, i] = 1.0
        
        # Вычисляем корреляцию с идеальной диагональю
        correlation = F.cosine_similarity(
            alignment.flatten().unsqueeze(0),
            ideal_diag.flatten().unsqueeze(0)
        )
        
        return max(0.0, correlation.item())
    
    def _calculate_focus(self, alignment: torch.Tensor) -> float:
        """Вычисляет фокусировку attention"""
        # Энтропия каждого столбца (decoder шага)
        entropies = []
        for i in range(alignment.shape[1]):
            col = alignment[:, i]
            col_normalized = col / (col.sum() + 1e-8)
            # Вычисляем энтропию
            entropy = -torch.sum(col_normalized * torch.log(col_normalized + 1e-8))
            entropies.append(entropy.item())
        
        # Нормализуем энтропию (0 = полная фокусировка, log(len) = равномерное распределение)
        max_entropy = np.log(alignment.shape[0])
        avg_entropy = np.mean(entropies)
        focus = max(0.0, 1.0 - avg_entropy / max_entropy)
        
        return focus
    
    def _calculate_monotonicity(self, alignment: torch.Tensor) -> float:
        """Вычисляет монотонность alignment"""
        # Находим пик в каждом столбце
        peaks = torch.argmax(alignment, dim=0)
        
        # Проверяем монотонность последовательности пиков
        monotonic_increases = 0
        total_transitions = len(peaks) - 1
        
        if total_transitions == 0:
            return 1.0
            
        for i in range(total_transitions):
            if peaks[i+1] >= peaks[i]:
                monotonic_increases += 1
        
        monotonicity = monotonic_increases / total_transitions
        return monotonicity
    
    def _assess_gate_quality(self, gate_outputs: torch.Tensor) -> Dict[str, float]:
        """Оценивает качество gate outputs"""
        if gate_outputs.dim() == 0:
            return {'gate_quality': 0.0}
            
        # Применяем sigmoid к gate outputs
        gate_probs = torch.sigmoid(gate_outputs.squeeze())
        
        # Проверяем наличие четкого сигнала остановки
        max_gate = torch.max(gate_probs).item()
        
        # Проверяем градиент (должен резко возрастать к концу)
        if len(gate_probs) > 10:
            early_mean = torch.mean(gate_probs[:len(gate_probs)//2]).item()
            late_mean = torch.mean(gate_probs[len(gate_probs)//2:]).item()
            gate_gradient = late_mean - early_mean
        else:
            gate_gradient = 0.0
        
        # Общая оценка качества gate
        gate_quality = min(1.0, (max_gate + max(0.0, gate_gradient)) / 2.0)
        
        return {
            'gate_quality': gate_quality,
            'gate_max_prob': max_gate,
            'gate_gradient': gate_gradient
        }


# Экспорт основных классов
__all__ = ['AudioQualityEnhancer', 'InferenceQualityController'] 