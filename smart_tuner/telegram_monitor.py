#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram Monitor для Smart Tuner TTS
Отправка изображений alignment каждые 1000 шагов с детальным анализом
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch
import requests
import yaml
import logging
import io
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

class TelegramMonitor:
    """
    Telegram монитор для отправки attachment изображений каждые 1000 шагов.
    """
    
    def __init__(self, config_path: str = "smart_tuner/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.logger = logging.getLogger(__name__)
        
        # Настройки Telegram
        telegram_config = self.config.get('telegram', {})
        self.bot_token = telegram_config.get('bot_token')
        self.chat_id = telegram_config.get('chat_id') 
        self.enabled = telegram_config.get('enabled', False)
        
        # Параметры мониторинга
        self.notification_interval = 100  # Чаще для отладки
        self.last_notification_step = 0
        self.training_history = []
        
        plt.style.use('default')
        self.figure_size = (12, 8)
        self.dpi = 150
        
        self.logger.info("📱 Telegram Monitor инициализирован")
        
    def should_send_notification(self, current_step: int) -> bool:
        """Проверяет нужность отправки уведомления."""
        if not self.enabled:
            return False
            
        return (current_step - self.last_notification_step) >= self.notification_interval
    
    def send_training_update(self, step: int, metrics: Dict[str, Any],
                           attention_weights: Optional[torch.Tensor] = None,
                           gate_outputs: Optional[torch.Tensor] = None) -> bool:
        """Отправляет обновление с изображениями."""
        if not self.should_send_notification(step):
            return False
        
        try:
            # Анализ данных
            analysis = self._analyze_step(step, metrics, attention_weights, gate_outputs)
            
            # Отправка текстового сообщения
            message = self._create_message(analysis)
            self._send_text_message(message)
            
            # Отправка изображений
            if attention_weights is not None:
                attention_image = self._create_attention_plot(attention_weights, step)
                if attention_image:
                    self._send_image(attention_image, f"attention_{step}.png",
                                   f"🎯 Attention Matrix - Шаг {step}")
            
            # График метрик
            metrics_image = self._create_metrics_plot(step)
            if metrics_image:
                self._send_image(metrics_image, f"metrics_{step}.png",
                               f"📊 Метрики - Шаг {step}")
            
            self.last_notification_step = step
            self.logger.info(f"✅ Telegram уведомление отправлено для шага {step}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка Telegram уведомления: {e}")
            return False
    
    def _analyze_step(self, step: int, metrics: Dict[str, Any],
                     attention_weights: Optional[torch.Tensor] = None,
                     gate_outputs: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Анализирует данные шага."""
        analysis = {
            'step': step,
            'metrics': metrics,
            'quality_score': 0.0,
            'phase': self._get_phase(step),
            'issues': [],
            'recommendations': []
        }
        
        # Анализ attention
        if attention_weights is not None:
            attention_analysis = self._analyze_attention(attention_weights)
            analysis['attention'] = attention_analysis
            analysis['quality_score'] += attention_analysis.get('diagonality', 0) * 0.5
        
        # Анализ gate
        if gate_outputs is not None:
            gate_analysis = self._analyze_gate(gate_outputs)
            analysis['gate'] = gate_analysis
            analysis['quality_score'] += gate_analysis.get('accuracy', 0) * 0.3
        
        # Анализ loss
        loss_value = metrics.get('loss', 0)
        analysis['quality_score'] += max(0, 1 - loss_value) * 0.2
        
        # Обнаружение проблем
        analysis['issues'] = self._detect_issues(analysis)
        analysis['recommendations'] = self._get_recommendations(analysis)
        
        # Сохранение в историю
        self.training_history.append(analysis)
        
        return analysis
    
    def _create_message(self, analysis: Dict[str, Any]) -> str:
        """Создает сообщение о состоянии обучения."""
        step = analysis['step']
        phase = analysis['phase']
        quality = analysis['quality_score']
        
        # Emoji качества
        quality_emoji = "🔥" if quality > 0.8 else "✅" if quality > 0.6 else "⚠️" if quality > 0.4 else "❌"
        
        message = f"🧠 *Smart Tuner - Отчет Обучения*\n\n"
        message += f"📍 **Шаг:** `{step:,}`\n"
        message += f"🎭 **Фаза:** `{phase}`\n"
        message += f"{quality_emoji} **Качество:** `{quality:.1%}`\n\n"
        
        # Метрики
        metrics = analysis['metrics']
        if 'loss' in metrics:
            message += f"📉 **Loss:** `{metrics['loss']:.4f}`\n"
        
        # Анализ attention
        attention = analysis.get('attention', {})
        if attention:
            diag = attention.get('diagonality', 0)
            message += f"🎯 **Attention Диагональность:** `{diag:.1%}`\n"
            
            if diag < 0.3:
                message += f"  ⚠️ *Критично низкая! Проблемы с alignment*\n"
            elif diag > 0.7:
                message += f"  ✅ *Отличная диагональность!*\n"
        
        # Проблемы
        issues = analysis.get('issues', [])
        if issues:
            message += f"\n⚠️ **Проблемы:**\n"
            for issue in issues[:2]:
                message += f"  • {issue}\n"
        
        # Рекомендации
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            message += f"\n💡 **Что делать:**\n"
            for rec in recommendations[:2]:
                message += f"  • {rec}\n"
        
        message += f"\n🕐 {datetime.now().strftime('%H:%M:%S')}"
        
        return message
    
    def _create_attention_plot(self, attention_weights: torch.Tensor, step: int) -> Optional[bytes]:
        """Создает изображение attention matrix."""
        try:
            # Берем первый элемент батча
            if attention_weights.dim() == 4:
                attention = attention_weights[0, 0].detach().cpu().numpy()
            elif attention_weights.dim() == 3:
                attention = attention_weights[0].detach().cpu().numpy()
            else:
                attention = attention_weights.detach().cpu().numpy()
            
            # 🔥 ИСПРАВЛЕНИЕ: Проверяем размеры attention матрицы
            if attention.shape[0] < 2 or attention.shape[1] < 2:
                self.logger.warning(f"Attention матрица слишком маленькая: {attention.shape}")
                return self._create_fallback_attention_plot(attention, step)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Основной attention plot
            im1 = ax1.imshow(attention.T, aspect='auto', origin='lower',
                           cmap='Blues', interpolation='nearest')
            ax1.set_title(f'Attention Matrix - Шаг {step}', fontweight='bold')
            ax1.set_xlabel('Decoder Steps')
            ax1.set_ylabel('Encoder Steps')
            plt.colorbar(im1, ax=ax1)
            
            # Диагональная линия
            min_dim = min(attention.shape)
            diag_x = np.linspace(0, attention.shape[1]-1, min_dim)
            diag_y = np.linspace(0, attention.shape[0]-1, min_dim)
            ax1.plot(diag_x, diag_y, 'r--', alpha=0.7, linewidth=2, label='Идеальная диагональ')
            ax1.legend()
            
            # Анализ качества
            diagonality = self._calculate_diagonality(attention)
            monotonicity = self._calculate_monotonicity(attention)
            
            # Гистограмма значений
            ax2.hist(attention.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_title('Распределение Attention Values')
            ax2.set_xlabel('Attention Weight')
            ax2.set_ylabel('Частота')
            
            # Метрики качества
            textstr = f'Диагональность: {diagonality:.1%}\nМонотонность: {monotonicity:.1%}'
            ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=12,
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
            return self._create_fallback_attention_plot(None, step)
    
    def _create_fallback_attention_plot(self, attention: Optional[np.ndarray], step: int) -> Optional[bytes]:
        """Создает fallback изображение для проблемных attention матриц."""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Информационное сообщение
            message = f"Attention Matrix - Шаг {step}\n\n"
            if attention is not None:
                message += f"Размер матрицы: {attention.shape}\n"
                message += f"Мин значение: {attention.min():.4f}\n"
                message += f"Макс значение: {attention.max():.4f}\n"
                message += f"Среднее: {attention.mean():.4f}\n\n"
                
                if attention.size > 0:
                    message += "Матрица слишком маленькая для визуализации\n"
                    message += "Обучение в ранней стадии"
                else:
                    message += "Пустая attention матрица"
            else:
                message += "Ошибка обработки attention данных"
            
            ax.text(0.5, 0.5, message, ha='center', va='center',
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax.set_title(f'Attention Status - Шаг {step}', fontweight='bold')
            ax.axis('off')
            
            plt.tight_layout()
            
            # Сохранение в байты
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
            buffer.seek(0)
            image_data = buffer.getvalue()
            plt.close(fig)
            
            return image_data
            
        except Exception as e:
            self.logger.error(f"Ошибка создания fallback attention plot: {e}")
            return None
    
    def _create_metrics_plot(self, step: int) -> Optional[bytes]:
        """Создает график метрик."""
        try:
            if len(self.training_history) < 3:
                return None
            
            # Данные для графиков
            steps = [h['step'] for h in self.training_history[-15:]]
            losses = [h['metrics'].get('loss', 0) for h in self.training_history[-15:]]
            quality_scores = [h['quality_score'] for h in self.training_history[-15:]]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, sharex=True)
            
            # График loss
            ax1.plot(steps, losses, 'b-', linewidth=2, marker='o', markersize=4, label='Training Loss')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'Прогресс Обучения - Шаг {step}', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Тренд loss
            if len(losses) > 3:
                z = np.polyfit(range(len(losses)), losses, 1)
                p = np.poly1d(z)
                ax1.plot(steps, p(range(len(losses))), "r--", alpha=0.8, label='Тренд')
                ax1.legend()
            
            # График качества
            ax2.plot(steps, quality_scores, 'g-', linewidth=2, marker='s', markersize=4, label='Quality Score')
            ax2.set_ylabel('Quality Score')
            ax2.set_xlabel('Training Step')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Зоны качества
            ax2.axhspan(0.8, 1.0, alpha=0.2, color='green', label='Отлично')
            ax2.axhspan(0.6, 0.8, alpha=0.2, color='yellow', label='Хорошо')
            ax2.axhspan(0.4, 0.6, alpha=0.2, color='orange', label='Средне')
            ax2.axhspan(0.0, 0.4, alpha=0.2, color='red', label='Плохо')
            
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
    
    def _send_text_message(self, message: str) -> bool:
        """Отправляет текстовое сообщение."""
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
            self.logger.error(f"Ошибка отправки текста: {e}")
            return False
    
    def _send_image(self, image_data: bytes, filename: str, caption: str = "") -> bool:
        """Отправляет изображение."""
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
            self.logger.error(f"Ошибка отправки изображения: {e}")
            return False
    
    def _analyze_attention(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """Анализирует attention качество."""
        if attention_weights.dim() == 4:
            attention = attention_weights[0, 0].detach().cpu().numpy()
        elif attention_weights.dim() == 3:
            attention = attention_weights[0].detach().cpu().numpy()
        else:
            attention = attention_weights.detach().cpu().numpy()
        
        return {
            'diagonality': self._calculate_diagonality(attention),
            'monotonicity': self._calculate_monotonicity(attention),
            'focus': self._calculate_focus(attention)
        }
    
    def _calculate_diagonality(self, attention_matrix: np.ndarray) -> float:
        """Вычисляет диагональность."""
        try:
            T_out, T_in = attention_matrix.shape
            if T_out == 0 or T_in == 0:
                return 0.0
            
            # Создаем идеальную диагональ
            ideal_diagonal = np.zeros_like(attention_matrix)
            min_dim = min(T_out, T_in)
            
            for i in range(min_dim):
                diagonal_pos = int(i * T_in / T_out) if T_out > 0 else i
                if diagonal_pos < T_in:
                    ideal_diagonal[i, diagonal_pos] = 1.0
            
            # Корреляция с идеальной диагональю
            attention_flat = attention_matrix.flatten()
            ideal_flat = ideal_diagonal.flatten()
            
            if np.std(attention_flat) == 0 or np.std(ideal_flat) == 0:
                return 0.0
            
            correlation = np.corrcoef(attention_flat, ideal_flat)[0, 1]
            return max(0.0, correlation) if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_monotonicity(self, attention_matrix: np.ndarray) -> float:
        """Вычисляет монотонность."""
        try:
            T_out, T_in = attention_matrix.shape
            if T_out <= 1:
                return 1.0
            
            # Находим пики attention
            peak_positions = np.argmax(attention_matrix, axis=1)
            
            # Считаем монотонные переходы
            monotonic_transitions = 0
            for i in range(1, len(peak_positions)):
                if peak_positions[i] >= peak_positions[i-1]:
                    monotonic_transitions += 1
            
            return monotonic_transitions / (len(peak_positions) - 1) if len(peak_positions) > 1 else 1.0
            
        except Exception:
            return 0.0
    
    def _calculate_focus(self, attention_matrix: np.ndarray) -> float:
        """Вычисляет фокусировку."""
        try:
            # Энтропия по decoder steps
            entropies = []
            for i in range(attention_matrix.shape[0]):
                attention_step = attention_matrix[i] + 1e-8
                entropy = -np.sum(attention_step * np.log(attention_step + 1e-8))
                entropies.append(entropy)
            
            # Нормализация
            max_entropy = np.log(attention_matrix.shape[1])
            avg_entropy = np.mean(entropies)
            focus = 1.0 - (avg_entropy / max_entropy)
            
            return max(0.0, min(1.0, focus))
            
        except Exception:
            return 0.0
    
    def _analyze_gate(self, gate_outputs: torch.Tensor) -> Dict[str, float]:
        """Анализирует gate качество."""
        try:
            gates = gate_outputs[0].detach().cpu().numpy()
            gate_binary = (gates > 0.5).astype(float)
            
            stop_positions = np.where(gate_binary > 0.5)[0]
            
            if len(stop_positions) > 0:
                stop_position = stop_positions[0]
                false_stops = np.sum(gate_binary[stop_position+1:] < 0.5) if stop_position < len(gates)-1 else 0
                accuracy = 1.0 - (false_stops / max(1, len(gates) - stop_position - 1))
            else:
                accuracy = 0.0
            
            return {
                'accuracy': accuracy,
                'stop_position': stop_positions[0] if len(stop_positions) > 0 else len(gates)
            }
            
        except Exception:
            return {'accuracy': 0.0, 'stop_position': 0}
    
    def _get_phase(self, step: int) -> str:
        """Определяет фазу обучения."""
        if step < 500:
            return "pre_alignment"
        elif step < 2000:
            return "alignment_learning"
        elif step < 3000:
            return "quality_optimization"
        else:
            return "fine_tuning"
    
    def _detect_issues(self, analysis: Dict[str, Any]) -> List[str]:
        """Обнаруживает проблемы."""
        issues = []
        
        attention = analysis.get('attention', {})
        diagonality = attention.get('diagonality', 0)
        
        if diagonality < 0.3:
            issues.append("Крайне низкая диагональность attention - модель не выравнивает")
        elif diagonality < 0.5:
            issues.append("Низкая диагональность - нужно усилить guided attention")
        
        gate = analysis.get('gate', {})
        if gate.get('accuracy', 0) < 0.5:
            issues.append("Плохая работа gate - модель не определяет конец")
        
        if analysis['quality_score'] < 0.4:
            issues.append("Общее качество критично низкое")
        
        return issues
    
    def _get_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Дает рекомендации по улучшению."""
        recommendations = []
        
        phase = analysis.get('phase', '')
        attention = analysis.get('attention', {})
        diagonality = attention.get('diagonality', 0)
        
        if diagonality < 0.5:
            if phase == "pre_alignment":
                recommendations.append("Увеличить вес guided attention до 10.0")
            else:
                recommendations.append("Снизить learning rate и усилить guided attention")
        
        gate = analysis.get('gate', {})
        if gate.get('accuracy', 0) < 0.7:
            recommendations.append("Настроить адаптивный gate threshold")
        
        if analysis['quality_score'] < 0.6:
            recommendations.append("Включить curriculum learning")
        
        return recommendations 