#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Telegram Monitor для Smart Tuner TTS
Система мониторинга обучения с отправкой изображений alignment каждые 1000 шагов
Интеграция с современными техниками качества из исследований 2024-2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Для серверной среды без GUI
import torch
import requests
import yaml
import logging
import io
import base64
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import seaborn as sns

class AdvancedTelegramMonitor:
    """
    Продвинутый Telegram монитор для TTS обучения с attachment изображениями.
    
    Возможности:
    1. Отправка alignment изображений каждые 1000 шагов  
    2. Анализ качества обучения в реальном времени
    3. Уведомления о проблемах качества на понятном языке
    4. Статистика по фазам обучения
    5. Предсказания времени завершения
    """
    
    def __init__(self, config_path: str = "smart_tuner/config.yaml"):
        """Инициализация Telegram монитора."""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.logger = logging.getLogger(__name__)
        
        # Настройки Telegram из конфига
        telegram_config = self.config.get('telegram', {})
        self.bot_token = telegram_config.get('bot_token')
        self.chat_id = telegram_config.get('chat_id')
        self.enabled = telegram_config.get('enabled', False)
        
        # Параметры мониторинга
        self.notification_interval = 1000  # Каждые 1000 шагов
        self.last_notification_step = 0
        
        # История для анализа
        self.training_history = []
        self.quality_history = []
        self.phase_history = []
        
        # Настройки изображений
        plt.style.use('seaborn-v0_8-darkgrid')
        self.figure_size = (12, 8)
        self.dpi = 150
        
        self.logger.info(f"🚀 Advanced Telegram Monitor инициализирован")
        
    def should_send_notification(self, current_step: int) -> bool:
        """Проверяет, нужно ли отправлять уведомление."""
        if not self.enabled:
            return False
            
        return (current_step - self.last_notification_step) >= self.notification_interval
    
    def analyze_training_progress(self, step: int, metrics: Dict[str, Any],
                                attention_weights: Optional[torch.Tensor] = None,
                                gate_outputs: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Анализирует прогресс обучения и готовит отчет.
        """
        analysis = {
            'step': step,
            'timestamp': datetime.now(),
            'metrics': metrics,
            'quality_score': 0.0,
            'phase': self._determine_training_phase(step, metrics),
            'issues': [],
            'improvements': [],
            'time_estimate': None
        }
        
        # Анализ attention качества
        if attention_weights is not None:
            attention_analysis = self._analyze_attention_quality(attention_weights)
            analysis['attention_quality'] = attention_analysis
            analysis['quality_score'] += attention_analysis.get('overall_score', 0) * 0.4
        
        # Анализ gate качества
        if gate_outputs is not None:
            gate_analysis = self._analyze_gate_quality(gate_outputs)
            analysis['gate_quality'] = gate_analysis  
            analysis['quality_score'] += gate_analysis.get('accuracy', 0) * 0.3
        
        # Анализ loss тренда
        loss_analysis = self._analyze_loss_trend(metrics)
        analysis['loss_trend'] = loss_analysis
        analysis['quality_score'] += (1.0 - min(loss_analysis.get('instability', 1.0), 1.0)) * 0.3
        
        # Обнаружение проблем
        issues = self._detect_training_issues(analysis)
        analysis['issues'] = issues
        
        # Рекомендации по улучшению
        improvements = self._suggest_improvements(analysis)
        analysis['improvements'] = improvements
        
        # Оценка времени завершения
        time_estimate = self._estimate_completion_time(step, metrics)
        analysis['time_estimate'] = time_estimate
        
        # Сохранение в историю
        self.training_history.append(analysis)
        
        return analysis
    
    def send_training_update(self, step: int, analysis: Dict[str, Any],
                           attention_weights: Optional[torch.Tensor] = None,
                           mel_outputs: Optional[torch.Tensor] = None) -> bool:
        """
        Отправляет обновление обучения в Telegram с изображениями.
        """
        if not self.should_send_notification(step):
            return False
        
        try:
            # Создание и отправка текстового сообщения
            message = self._create_training_message(analysis)
            self._send_text_message(message)
            
            # Создание и отправка изображений
            if attention_weights is not None:
                attention_image = self._create_attention_plot(attention_weights, step)
                if attention_image:
                    self._send_image(attention_image, f"attention_step_{step}.png",
                                   caption=f"🎯 Attention Matrix - Шаг {step}")
            
            # График метрик обучения
            metrics_image = self._create_metrics_plot(step)
            if metrics_image:
                self._send_image(metrics_image, f"metrics_step_{step}.png",
                               caption=f"📊 Метрики обучения - Шаг {step}")
            
            # График качества по фазам
            if len(self.training_history) > 10:
                quality_image = self._create_quality_trend_plot(step)
                if quality_image:
                    self._send_image(quality_image, f"quality_step_{step}.png",
                                   caption=f"📈 Тренд качества - Шаг {step}")
            
            self.last_notification_step = step
            self.logger.info(f"✅ Telegram уведомление отправлено для шага {step}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка отправки Telegram уведомления: {e}")
            return False
    
    def _create_training_message(self, analysis: Dict[str, Any]) -> str:
        """Создает текстовое сообщение о состоянии обучения."""
        step = analysis['step']
        phase = analysis['phase']
        quality_score = analysis['quality_score']
        
        # Emoji для качества
        quality_emoji = "🔥" if quality_score > 0.8 else "✅" if quality_score > 0.6 else "⚠️" if quality_score > 0.4 else "❌"
        
        message = f"🧠 *Smart Tuner V2 - Отчет о Обучении*\n\n"
        message += f"📍 **Шаг:** `{step:,}`\n"
        message += f"🎭 **Фаза:** `{phase}`\n"
        message += f"{quality_emoji} **Качество:** `{quality_score:.1%}`\n\n"
        
        # Основные метрики
        metrics = analysis.get('metrics', {})
        if 'loss' in metrics:
            message += f"📉 **Loss:** `{metrics['loss']:.4f}`\n"
        if 'attention_loss' in metrics:
            message += f"🎯 **Attention Loss:** `{metrics['attention_loss']:.4f}`\n"
        if 'gate_loss' in metrics:
            message += f"🚪 **Gate Loss:** `{metrics['gate_loss']:.4f}`\n"
        
        # Анализ attention
        attention_quality = analysis.get('attention_quality', {})
        if attention_quality:
            diagonality = attention_quality.get('diagonality_score', 0)
            message += f"\n🔍 **Анализ Attention:**\n"
            message += f"  • Диагональность: `{diagonality:.1%}`\n"
            if diagonality < 0.3:
                message += f"  ⚠️ *Низкая диагональность! Нужно улучшение guided attention*\n"
            elif diagonality > 0.7:
                message += f"  ✅ *Отличная диагональность!*\n"
        
        # Проблемы и рекомендации
        issues = analysis.get('issues', [])
        if issues:
            message += f"\n⚠️ **Обнаруженные проблемы:**\n"
            for issue in issues[:3]:  # Топ 3 проблемы
                message += f"  • {issue}\n"
        
        improvements = analysis.get('improvements', [])
        if improvements:
            message += f"\n💡 **Рекомендации:**\n"
            for improvement in improvements[:2]:  # Топ 2 рекомендации
                message += f"  • {improvement}\n"
        
        # Оценка времени
        time_estimate = analysis.get('time_estimate')
        if time_estimate:
            message += f"\n⏰ **Осталось:** `~{time_estimate}`\n"
        
        message += f"\n🕐 {datetime.now().strftime('%H:%M:%S')}"
        
        return message
    
    def _create_attention_plot(self, attention_weights: torch.Tensor, step: int) -> Optional[bytes]:
        """Создает изображение attention matrix."""
        try:
            # Берем первый элемент из батча
            if attention_weights.dim() == 4:  # [B, heads, T_out, T_in]
                attention = attention_weights[0, 0].detach().cpu().numpy()
            elif attention_weights.dim() == 3:  # [B, T_out, T_in]
                attention = attention_weights[0].detach().cpu().numpy()
            else:
                attention = attention_weights.detach().cpu().numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Основной attention plot
            im1 = axes[0].imshow(attention.T, aspect='auto', origin='lower', 
                               cmap='Blues', interpolation='nearest')
            axes[0].set_title(f'Attention Matrix - Шаг {step}', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Decoder Steps')
            axes[0].set_ylabel('Encoder Steps') 
            plt.colorbar(im1, ax=axes[0])
            
            # Диагональная линия для референса
            min_dim = min(attention.shape[0], attention.shape[1])
            diag_x = np.linspace(0, attention.shape[1]-1, min_dim)
            diag_y = np.linspace(0, attention.shape[0]-1, min_dim)
            axes[0].plot(diag_x, diag_y, 'r--', alpha=0.7, linewidth=2, label='Ideal Diagonal')
            axes[0].legend()
            
            # Анализ качества
            diagonality = self._calculate_attention_diagonality(attention)
            monotonicity = self._calculate_attention_monotonicity(attention)
            
            # Гистограмма attention values
            axes[1].hist(attention.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1].set_title(f'Attention Values Distribution', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Attention Weight')
            axes[1].set_ylabel('Frequency')
            
            # Добавляем метрики качества
            textstr = f'Диагональность: {diagonality:.1%}\nМонотонность: {monotonicity:.1%}'
            axes[1].text(0.05, 0.95, textstr, transform=axes[1].transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Сохранение в байты
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
            buffer.seek(0)
            image_data = buffer.getvalue()
            plt.close(fig)
            
            return image_data
            
        except Exception as e:
            self.logger.error(f"Ошибка создания attention plot: {e}")
            return None
    
    def _create_metrics_plot(self, step: int) -> Optional[bytes]:
        """Создает график метрик обучения."""
        try:
            if len(self.training_history) < 5:
                return None
            
            # Извлекаем данные
            steps = [h['step'] for h in self.training_history[-20:]]  # Последние 20 точек
            losses = [h['metrics'].get('loss', 0) for h in self.training_history[-20:]]
            quality_scores = [h['quality_score'] for h in self.training_history[-20:]]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, sharex=True)
            
            # График loss
            ax1.plot(steps, losses, 'b-', linewidth=2, marker='o', markersize=4, label='Training Loss')
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title(f'Прогресс Обучения - Шаг {step}', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Тренд loss
            if len(losses) > 3:
                z = np.polyfit(range(len(losses)), losses, 1)
                p = np.poly1d(z)
                ax1.plot(steps, p(range(len(losses))), "r--", alpha=0.8, label='Trend')
                ax1.legend()
            
            # График качества
            ax2.plot(steps, quality_scores, 'g-', linewidth=2, marker='s', markersize=4, label='Quality Score')
            ax2.set_ylabel('Quality Score', fontsize=12)
            ax2.set_xlabel('Training Step', fontsize=12)
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Цветовая зона качества
            ax2.axhspan(0.8, 1.0, alpha=0.2, color='green', label='Excellent')
            ax2.axhspan(0.6, 0.8, alpha=0.2, color='yellow', label='Good')
            ax2.axhspan(0.4, 0.6, alpha=0.2, color='orange', label='Fair')
            ax2.axhspan(0.0, 0.4, alpha=0.2, color='red', label='Poor')
            
            plt.tight_layout()
            
            # Сохранение в байты
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
            buffer.seek(0)
            image_data = buffer.getvalue()
            plt.close(fig)
            
            return image_data
            
        except Exception as e:
            self.logger.error(f"Ошибка создания metrics plot: {e}")
            return None
    
    def _create_quality_trend_plot(self, step: int) -> Optional[bytes]:
        """Создает график тренда качества по фазам."""
        try:
            if len(self.training_history) < 10:
                return None
            
            # Группировка по фазам
            phases_data = {}
            for h in self.training_history:
                phase = h['phase']
                if phase not in phases_data:
                    phases_data[phase] = {'steps': [], 'quality': [], 'attention': []}
                
                phases_data[phase]['steps'].append(h['step'])
                phases_data[phase]['quality'].append(h['quality_score'])
                
                # Attention quality
                attention_quality = h.get('attention_quality', {})
                diagonality = attention_quality.get('diagonality_score', 0)
                phases_data[phase]['attention'].append(diagonality)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size)
            
            # График качества по фазам
            colors = ['blue', 'green', 'orange', 'red', 'purple']
            for i, (phase, data) in enumerate(phases_data.items()):
                color = colors[i % len(colors)]
                ax1.plot(data['steps'], data['quality'], 
                        color=color, linewidth=2, marker='o', label=f'{phase}')
            
            ax1.set_ylabel('Overall Quality Score', fontsize=12)
            ax1.set_title(f'Качество по Фазам Обучения - Шаг {step}', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_ylim(0, 1)
            
            # График attention quality
            for i, (phase, data) in enumerate(phases_data.items()):
                color = colors[i % len(colors)]
                ax2.plot(data['steps'], data['attention'], 
                        color=color, linewidth=2, marker='s', label=f'{phase}')
            
            ax2.set_ylabel('Attention Diagonality', fontsize=12)
            ax2.set_xlabel('Training Step', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_ylim(0, 1)
            
            # Пороговые линии
            ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Good Threshold')
            ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Poor Threshold')
            
            plt.tight_layout()
            
            # Сохранение в байты
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
            buffer.seek(0)
            image_data = buffer.getvalue()
            plt.close(fig)
            
            return image_data
            
        except Exception as e:
            self.logger.error(f"Ошибка создания quality trend plot: {e}")
            return None
    
    def _send_text_message(self, message: str) -> bool:
        """Отправляет текстовое сообщение в Telegram."""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'Markdown',
            'disable_web_page_preview': True
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            self.logger.error(f"Ошибка отправки текста в Telegram: {e}")
            return False
    
    def _send_image(self, image_data: bytes, filename: str, caption: str = "") -> bool:
        """Отправляет изображение в Telegram."""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        
        files = {
            'photo': (filename, image_data, 'image/png')
        }
        
        data = {
            'chat_id': self.chat_id,
            'caption': caption
        }
        
        try:
            response = requests.post(url, files=files, data=data, timeout=30)
            response.raise_for_status()
            return True
        except Exception as e:
            self.logger.error(f"Ошибка отправки изображения в Telegram: {e}")
            return False
    
    def _analyze_attention_quality(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """Анализирует качество attention."""
        # Берем первый элемент батча
        if attention_weights.dim() == 4:  # [B, heads, T_out, T_in]
            attention = attention_weights[0, 0].detach().cpu().numpy()
        elif attention_weights.dim() == 3:  # [B, T_out, T_in]
            attention = attention_weights[0].detach().cpu().numpy()
        else:
            attention = attention_weights.detach().cpu().numpy()
        
        return {
            'diagonality_score': self._calculate_attention_diagonality(attention),
            'monotonicity_score': self._calculate_attention_monotonicity(attention),
            'focus_score': self._calculate_attention_focus(attention),
            'overall_score': self._calculate_overall_attention_score(attention)
        }
    
    def _calculate_attention_diagonality(self, attention_matrix: np.ndarray) -> float:
        """Вычисляет диагональность attention matrix."""
        try:
            T_out, T_in = attention_matrix.shape
            if T_out == 0 or T_in == 0:
                return 0.0
            
            # Создаем идеальную диагональную матрицу
            ideal_diagonal = np.zeros_like(attention_matrix)
            min_dim = min(T_out, T_in)
            
            for i in range(min_dim):
                # Рассчитываем позицию на диагонали
                diagonal_pos = int(i * T_in / T_out) if T_out > 0 else i
                if diagonal_pos < T_in:
                    ideal_diagonal[i, diagonal_pos] = 1.0
            
            # Вычисляем корреляцию с идеальной диагональю
            attention_flat = attention_matrix.flatten()
            ideal_flat = ideal_diagonal.flatten()
            
            if np.std(attention_flat) == 0 or np.std(ideal_flat) == 0:
                return 0.0
            
            correlation = np.corrcoef(attention_flat, ideal_flat)[0, 1]
            return max(0.0, correlation) if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_attention_monotonicity(self, attention_matrix: np.ndarray) -> float:
        """Вычисляет монотонность attention matrix."""
        try:
            T_out, T_in = attention_matrix.shape
            if T_out <= 1:
                return 1.0
            
            # Находим пик для каждого decoder step
            peak_positions = np.argmax(attention_matrix, axis=1)
            
            # Считаем монотонные переходы
            monotonic_transitions = 0
            for i in range(1, len(peak_positions)):
                if peak_positions[i] >= peak_positions[i-1]:
                    monotonic_transitions += 1
            
            return monotonic_transitions / (len(peak_positions) - 1) if len(peak_positions) > 1 else 1.0
            
        except Exception:
            return 0.0
    
    def _calculate_attention_focus(self, attention_matrix: np.ndarray) -> float:
        """Вычисляет фокусировку attention matrix."""
        try:
            # Средняя энтропия по decoder steps
            entropies = []
            for i in range(attention_matrix.shape[0]):
                attention_step = attention_matrix[i]
                attention_step = attention_step + 1e-8  # Избегаем log(0)
                entropy = -np.sum(attention_step * np.log(attention_step + 1e-8))
                entropies.append(entropy)
            
            # Нормализуем - меньше энтропия = больше фокус
            max_entropy = np.log(attention_matrix.shape[1])
            avg_entropy = np.mean(entropies)
            focus = 1.0 - (avg_entropy / max_entropy)
            
            return max(0.0, min(1.0, focus))
            
        except Exception:
            return 0.0
    
    def _calculate_overall_attention_score(self, attention_matrix: np.ndarray) -> float:
        """Вычисляет общий балл качества attention."""
        diagonality = self._calculate_attention_diagonality(attention_matrix)
        monotonicity = self._calculate_attention_monotonicity(attention_matrix)
        focus = self._calculate_attention_focus(attention_matrix)
        
        # Взвешенная оценка
        return (diagonality * 0.5 + monotonicity * 0.3 + focus * 0.2)
    
    def _analyze_gate_quality(self, gate_outputs: torch.Tensor) -> Dict[str, float]:
        """Анализирует качество gate outputs."""
        try:
            # Берем первый элемент батча
            gates = gate_outputs[0].detach().cpu().numpy()
            
            # Бинаризуем gate outputs
            gate_binary = (gates > 0.5).astype(float)
            
            # Ищем позицию останова
            stop_positions = np.where(gate_binary > 0.5)[0]
            
            if len(stop_positions) > 0:
                stop_position = stop_positions[0]
                # Проверяем, что gate остается активным после первого stop
                false_stops = np.sum(gate_binary[stop_position+1:] < 0.5) if stop_position < len(gates)-1 else 0
                accuracy = 1.0 - (false_stops / max(1, len(gates) - stop_position - 1))
            else:
                accuracy = 0.0  # Нет сигнала останова
            
            return {
                'accuracy': accuracy,
                'stop_position': stop_positions[0] if len(stop_positions) > 0 else len(gates),
                'stability': 1.0 - np.std(gates)  # Меньше вариации = лучше
            }
            
        except Exception:
            return {'accuracy': 0.0, 'stop_position': 0, 'stability': 0.0}
    
    def _analyze_loss_trend(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Анализирует тренд loss."""
        if len(self.training_history) < 5:
            return {'trend': 0.0, 'instability': 0.0}
        
        recent_losses = [h['metrics'].get('loss', 0) for h in self.training_history[-10:]]
        
        # Тренд (отрицательный = убывание)
        if len(recent_losses) > 1:
            trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        else:
            trend = 0.0
        
        # Нестабильность (стандартное отклонение)
        instability = np.std(recent_losses) if len(recent_losses) > 1 else 0.0
        
        return {
            'trend': trend,
            'instability': instability,
            'current_loss': recent_losses[-1] if recent_losses else 0.0
        }
    
    def _determine_training_phase(self, step: int, metrics: Dict[str, Any]) -> str:
        """Определяет текущую фазу обучения."""
        if step < 500:
            return "pre_alignment"
        elif step < 2000:
            return "alignment_learning"
        elif step < 3000:
            return "quality_optimization"
        else:
            return "fine_tuning"
    
    def _detect_training_issues(self, analysis: Dict[str, Any]) -> List[str]:
        """Обнаруживает проблемы в обучении."""
        issues = []
        
        # Проблемы с attention
        attention_quality = analysis.get('attention_quality', {})
        diagonality = attention_quality.get('diagonality_score', 0)
        
        if diagonality < 0.3:
            issues.append("Очень низкая диагональность attention - модель плохо выравнивает текст и аудио")
        elif diagonality < 0.5:
            issues.append("Низкая диагональность attention - нужно улучшить guided attention")
        
        # Проблемы с gate
        gate_quality = analysis.get('gate_quality', {})
        gate_accuracy = gate_quality.get('accuracy', 0)
        
        if gate_accuracy < 0.5:
            issues.append("Низкая точность gate - модель не умеет определять конец предложения")
        
        # Проблемы с loss
        loss_trend = analysis.get('loss_trend', {})
        if loss_trend.get('trend', 0) > 0:
            issues.append("Loss растет - возможно переобучение или слишком высокий learning rate")
        
        if loss_trend.get('instability', 0) > 0.1:
            issues.append("Нестабильный loss - рассмотрите снижение learning rate")
        
        # Общие проблемы качества
        if analysis['quality_score'] < 0.4:
            issues.append("Общее качество низкое - нужны срочные исправления")
        
        return issues
    
    def _suggest_improvements(self, analysis: Dict[str, Any]) -> List[str]:
        """Предлагает улучшения."""
        improvements = []
        
        phase = analysis.get('phase', '')
        attention_quality = analysis.get('attention_quality', {})
        diagonality = attention_quality.get('diagonality_score', 0)
        
        if diagonality < 0.5:
            if phase == "pre_alignment":
                improvements.append("Увеличить вес guided attention loss до 10.0")
            else:
                improvements.append("Снизить learning rate и усилить guided attention")
        
        gate_quality = analysis.get('gate_quality', {})
        if gate_quality.get('accuracy', 0) < 0.7:
            improvements.append("Настроить адаптивный gate threshold (0.3→0.8)")
        
        loss_trend = analysis.get('loss_trend', {})
        if loss_trend.get('instability', 0) > 0.1:
            improvements.append("Снизить learning rate или увеличить batch size")
        
        if analysis['quality_score'] < 0.6:
            improvements.append("Включить curriculum learning для teacher forcing")
        
        return improvements
    
    def _estimate_completion_time(self, current_step: int, metrics: Dict[str, Any]) -> Optional[str]:
        """Оценивает время до завершения обучения."""
        try:
            if len(self.training_history) < 3:
                return None
            
            # Простая оценка на основе прогресса
            target_steps = 10000  # Примерное количество шагов для хорошего качества
            remaining_steps = max(0, target_steps - current_step)
            
            # Скорость обучения (шагов в час)
            time_deltas = []
            for i in range(1, min(len(self.training_history), 10)):
                prev_time = self.training_history[-i-1]['timestamp']
                curr_time = self.training_history[-i]['timestamp']
                time_delta = (curr_time - prev_time).total_seconds() / 3600  # часы
                time_deltas.append(time_delta)
            
            if time_deltas:
                avg_time_per_interval = np.mean(time_deltas)
                steps_per_interval = self.notification_interval
                steps_per_hour = steps_per_interval / avg_time_per_interval if avg_time_per_interval > 0 else 0
                
                if steps_per_hour > 0:
                    hours_remaining = remaining_steps / steps_per_hour
                    if hours_remaining < 24:
                        return f"{hours_remaining:.1f} часов"
                    else:
                        return f"{hours_remaining/24:.1f} дней"
            
            return None
            
        except Exception:
            return None

    def send_training_complete_summary(self, final_analysis: Dict[str, Any]) -> bool:
        """Отправляет итоговый отчет о завершении обучения."""
        try:
            step = final_analysis['step']
            quality_score = final_analysis['quality_score']
            
            message = f"🎉 *Обучение Завершено!*\n\n"
            message += f"📍 **Финальный шаг:** `{step:,}`\n"
            message += f"🏆 **Итоговое качество:** `{quality_score:.1%}`\n\n"
            
            # Итоговая статистика
            if len(self.training_history) > 0:
                total_time = (self.training_history[-1]['timestamp'] - self.training_history[0]['timestamp']).total_seconds() / 3600
                message += f"⏱️ **Общее время:** `{total_time:.1f} часов`\n"
                
                # Лучшее качество
                best_quality = max(h['quality_score'] for h in self.training_history)
                message += f"🎯 **Лучшее качество:** `{best_quality:.1%}`\n"
            
            # Финальные рекомендации
            message += f"\n💡 **Следующие шаги:**\n"
            if quality_score > 0.8:
                message += f"  ✅ Модель готова к продакшену!\n"
                message += f"  🎤 Проведите финальное тестирование\n"
            elif quality_score > 0.6:
                message += f"  📈 Хорошее качество, можно дообучить\n"
                message += f"  🔧 Рассмотрите fine-tuning\n"
            else:
                message += f"  ⚠️ Требуется дополнительное обучение\n"
                message += f"  🔧 Проверьте гиперпараметры\n"
            
            return self._send_text_message(message)
            
        except Exception as e:
            self.logger.error(f"Ошибка отправки итогового отчета: {e}")
            return False 