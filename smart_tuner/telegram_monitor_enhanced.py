#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенная система Telegram мониторинга для Smart Tuner TTS

🎵 Система реального времени для мониторинга качества обучения:
- Attention alignment изображения с анализом диагональности
- Графики потерь и метрик качества  
- Детальный анализ проблем и рекомендации
- Russian language support для всех уведомлений
"""

import asyncio
import io
import logging
import matplotlib
matplotlib.use('Agg')  # Используем backend без GUI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import telegram
from telegram import Bot
from typing import Dict, List, Optional, Any, Tuple
import time
from datetime import datetime
import os
import traceback

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TelegramMonitorEnhanced:
    """
    🎵 Улучшенная система Telegram мониторинга для TTS.
    
    Возможности:
    - Детальный анализ attention alignment с изображениями
    - Графики обучения с зонами качества
    - Автоматическое обнаружение проблем
    - Рекомендации по улучшению на русском языке
    - Уведомления каждые 1000 шагов
    """
    
    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.bot = None
        
        # История метрик для графиков
        self.metrics_history = {
            'steps': [],
            'train_loss': [],
            'val_loss': [],
            'attention_diagonality': [],
            'gate_accuracy': [],
            'quality_score': [],
            'timestamps': []
        }
        
        # Настройки графиков
        self.figure_size = (12, 8)
        self.dpi = 100
        
        self.logger = logging.getLogger(__name__)
        
        if self.enabled and self.bot_token and self.chat_id:
            try:
                self.bot = Bot(token=self.bot_token)
                self.logger.info("✅ Telegram мониторинг инициализирован")
            except Exception as e:
                self.logger.error(f"❌ Ошибка инициализации Telegram: {e}")
                self.enabled = False
    
    def send_training_update(self, step: int, metrics: Dict[str, Any], 
                           alignments: Optional[torch.Tensor] = None) -> None:
        """
        🎵 Отправляет подробное обновление о тренировке каждые 1000 шагов.
        
        Args:
            step: Номер шага обучения
            metrics: Словарь с метриками
            alignments: Attention alignments для анализа
        """
        if not self.enabled or step % 1000 != 0:
            return
            
        try:
            # 1. Обновляем историю метрик
            self._update_metrics_history(step, metrics)
            
            # 2. Создаем и отправляем attention изображение
            if alignments is not None:
                self._send_attention_analysis(step, alignments)
            
            # 3. Создаем и отправляем графики метрик
            self._send_metrics_plots(step)
            
            # 4. Отправляем текстовый анализ
            self._send_detailed_analysis(step, metrics)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка отправки Telegram обновления: {e}")
            self.logger.error(traceback.format_exc())
    
    def _update_metrics_history(self, step: int, metrics: Dict[str, Any]) -> None:
        """Обновляет историю метрик."""
        self.metrics_history['steps'].append(step)
        self.metrics_history['train_loss'].append(metrics.get('train_loss', 0))
        self.metrics_history['val_loss'].append(metrics.get('val_loss', 0))
        self.metrics_history['attention_diagonality'].append(metrics.get('attention_diagonality', 0))
        self.metrics_history['gate_accuracy'].append(metrics.get('gate_accuracy', 0))
        self.metrics_history['quality_score'].append(metrics.get('quality_score', 0))
        self.metrics_history['timestamps'].append(datetime.now())
        
        # Ограничиваем историю последними 100 точками
        max_history = 100
        for key in self.metrics_history:
            if len(self.metrics_history[key]) > max_history:
                self.metrics_history[key] = self.metrics_history[key][-max_history:]
    
    def _send_attention_analysis(self, step: int, alignments: torch.Tensor) -> None:
        """
        🎯 Создает и отправляет детальный анализ attention alignment.
        """
        try:
            # Берем первый пример из батча для анализа
            if alignments.dim() == 3:
                alignment = alignments[0].detach().cpu().numpy()
            else:
                alignment = alignments.detach().cpu().numpy()
            
            # Создаем фигуру с несколькими subplot'ами
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'🎵 Анализ Attention Alignment - Шаг {step}', fontsize=16, fontweight='bold')
            
            # 1. Основная attention матрица
            ax1 = axes[0, 0]
            im1 = ax1.imshow(alignment, aspect='auto', origin='lower', cmap='Blues')
            ax1.set_title('Attention Matrix', fontweight='bold')
            ax1.set_xlabel('Encoder Steps')
            ax1.set_ylabel('Decoder Steps')
            plt.colorbar(im1, ax=ax1)
            
            # Добавляем идеальную диагональ для сравнения
            diagonal_line = np.linspace(0, alignment.shape[0]-1, alignment.shape[1])
            ax1.plot(diagonal_line, range(alignment.shape[1]), 'r--', alpha=0.7, linewidth=2, label='Идеальная диагональ')
            ax1.legend()
            
            # 2. Диагональность анализ
            ax2 = axes[0, 1]
            diagonality_score = self._calculate_diagonality(alignment)
            
            # Создаем тепловую карту отклонения от диагонали
            ideal_diagonal = self._create_ideal_diagonal(alignment.shape)
            deviation = np.abs(alignment - ideal_diagonal)
            im2 = ax2.imshow(deviation, aspect='auto', origin='lower', cmap='Reds')
            ax2.set_title(f'Отклонение от диагонали\nДиагональность: {diagonality_score:.3f}', fontweight='bold')
            ax2.set_xlabel('Encoder Steps')
            ax2.set_ylabel('Decoder Steps')
            plt.colorbar(im2, ax=ax2)
            
            # 3. Фокусировка attention
            ax3 = axes[1, 0]
            attention_focus = np.max(alignment, axis=0)  # Максимальная attention на каждом decoder шаге
            ax3.plot(attention_focus, 'g-', linewidth=2, marker='o', markersize=4)
            ax3.set_title('Фокусировка Attention по времени', fontweight='bold')
            ax3.set_xlabel('Decoder Steps')
            ax3.set_ylabel('Максимальная Attention')
            ax3.grid(True, alpha=0.3)
            
            # Добавляем зоны качества
            ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Норма (0.5)')
            ax3.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Хорошо (0.7)')
            ax3.legend()
            
            # 4. Монотонность
            ax4 = axes[1, 1]
            attention_peaks = np.argmax(alignment, axis=0)  # Позиция пика на каждом decoder шаге
            ax4.plot(attention_peaks, 'b-', linewidth=2, marker='s', markersize=4)
            ax4.set_title('Монотонность Attention', fontweight='bold')
            ax4.set_xlabel('Decoder Steps')
            ax4.set_ylabel('Позиция пика Attention')
            ax4.grid(True, alpha=0.3)
            
            # Идеальная монотонная линия
            ideal_monotonic = np.linspace(0, len(attention_peaks)-1, len(attention_peaks))
            ax4.plot(ideal_monotonic, 'r--', alpha=0.7, linewidth=2, label='Идеальная монотонность')
            ax4.legend()
            
            plt.tight_layout()
            
            # Сохраняем в буфер
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
            buf.seek(0)
            
            # Отправляем изображение
            asyncio.create_task(self._send_photo_async(
                buf, 
                f"🎯 **Анализ Attention - Шаг {step}**\n\n"
                f"📊 **Диагональность:** {diagonality_score:.3f}\n"
                f"🎯 **Средняя фокусировка:** {np.mean(attention_focus):.3f}\n"
                f"📈 **Монотонность:** {self._calculate_monotonicity(attention_peaks):.3f}\n\n"
                f"{self._get_attention_quality_text(diagonality_score)}"
            ))
            
            plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания attention анализа: {e}")
    
    def _send_metrics_plots(self, step: int) -> None:
        """
        📊 Создает и отправляет графики метрик обучения.
        """
        try:
            if len(self.metrics_history['steps']) < 2:
                return
                
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'📊 Метрики обучения - Шаг {step}', fontsize=16, fontweight='bold')
            
            steps = self.metrics_history['steps']
            
            # 1. Losses
            ax1 = axes[0, 0]
            ax1.plot(steps, self.metrics_history['train_loss'], 'b-', linewidth=2, label='Train Loss', marker='o', markersize=3)
            ax1.plot(steps, self.metrics_history['val_loss'], 'r-', linewidth=2, label='Val Loss', marker='s', markersize=3)
            ax1.set_title('Loss функции', fontweight='bold')
            ax1.set_xlabel('Шаги')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            # 2. Attention Quality
            ax2 = axes[0, 1]
            ax2.plot(steps, self.metrics_history['attention_diagonality'], 'g-', linewidth=2, label='Диагональность', marker='^', markersize=3)
            ax2.set_title('Качество Attention', fontweight='bold')
            ax2.set_xlabel('Шаги')
            ax2.set_ylabel('Диагональность')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Зоны качества
            ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Плохо')
            ax2.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Норма')  
            ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Отлично')
            
            # 3. Gate Accuracy
            ax3 = axes[1, 0]
            ax3.plot(steps, self.metrics_history['gate_accuracy'], 'm-', linewidth=2, label='Gate Accuracy', marker='d', markersize=3)
            ax3.set_title('Gate Accuracy', fontweight='bold')
            ax3.set_xlabel('Шаги')
            ax3.set_ylabel('Accuracy')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Overall Quality
            ax4 = axes[1, 1]
            ax4.plot(steps, self.metrics_history['quality_score'], 'orange', linewidth=3, label='Quality Score', marker='*', markersize=5)
            ax4.set_title('Общее качество', fontweight='bold')
            ax4.set_xlabel('Шаги')
            ax4.set_ylabel('Quality Score')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Зона целевого качества
            ax4.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Цель')
            
            plt.tight_layout()
            
            # Сохраняем в буфер
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
            buf.seek(0)
            
            # Отправляем изображение
            asyncio.create_task(self._send_photo_async(
                buf,
                f"📊 **Графики метрик - Шаг {step}**\n\n"
                f"📈 **Тренд качества:** {self._analyze_quality_trend()}\n"
                f"🎯 **Статус обучения:** {self._get_training_status()}\n"
                f"⚡ **Рекомендации:** {self._get_training_recommendations()}"
            ))
            
            plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания графиков метрик: {e}")
    
    def _send_detailed_analysis(self, step: int, metrics: Dict[str, Any]) -> None:
        """
        📝 Отправляет детальный текстовый анализ качества.
        """
        try:
            # Формируем подробное сообщение
            message = f"🎵 **ДЕТАЛЬНЫЙ АНАЛИЗ - ШАГ {step}**\n\n"
            
            # Основные метрики
            message += f"📊 **ОСНОВНЫЕ МЕТРИКИ:**\n"
            message += f"• Train Loss: `{metrics.get('train_loss', 0):.4f}`\n"
            message += f"• Val Loss: `{metrics.get('val_loss', 0):.4f}`\n"
            message += f"• Quality Score: `{metrics.get('quality_score', 0):.1%}`\n\n"
            
            # Attention анализ
            diag = metrics.get('attention_diagonality', 0)
            message += f"🎯 **ATTENTION КАЧЕСТВО:**\n"
            message += f"• Диагональность: `{diag:.3f}`\n"
            message += f"• Статус: {self._get_attention_status_emoji(diag)} {self._get_attention_quality_text(diag)}\n"
            message += f"• Gate Accuracy: `{metrics.get('gate_accuracy', 0):.1%}`\n\n"
            
            # Прогресс обучения
            message += f"📈 **ПРОГРЕСС ОБУЧЕНИЯ:**\n"
            message += f"• Фаза: `{metrics.get('phase', 'Неизвестно')}`\n"
            message += f"• Эпоха: `{metrics.get('epoch', 'N/A')}`\n"
            message += f"• Learning Rate: `{metrics.get('learning_rate', 0):.2e}`\n\n"
            
            # Проблемы и рекомендации
            issues = self._analyze_potential_issues(metrics)
            if issues:
                message += f"⚠️ **ОБНАРУЖЕННЫЕ ПРОБЛЕМЫ:**\n"
                for issue in issues:
                    message += f"• {issue}\n"
                message += "\n"
            
            recommendations = self._get_specific_recommendations(metrics)
            message += f"💡 **РЕКОМЕНДАЦИИ:**\n"
            for rec in recommendations:
                message += f"• {rec}\n"
            
            # Время следующего обновления
            message += f"\n⏰ **Следующее обновление через 1000 шагов**"
            
            asyncio.create_task(self._send_message_async(message))
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка отправки детального анализа: {e}")
    
    def _calculate_diagonality(self, alignment: np.ndarray) -> float:
        """Вычисляет диагональность attention матрицы."""
        try:
            rows, cols = alignment.shape
            ideal_diagonal = self._create_ideal_diagonal((rows, cols))
            
            # Нормализуем обе матрицы
            alignment_norm = alignment / (np.sum(alignment) + 1e-8)
            ideal_norm = ideal_diagonal / (np.sum(ideal_diagonal) + 1e-8)
            
            # Вычисляем корреляцию
            correlation = np.corrcoef(alignment_norm.flatten(), ideal_norm.flatten())[0, 1]
            return max(0.0, correlation) if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _create_ideal_diagonal(self, shape: Tuple[int, int]) -> np.ndarray:
        """Создает идеальную диагональную матрицу."""
        rows, cols = shape
        diagonal = np.zeros((rows, cols))
        
        for j in range(cols):
            i = int(j * rows / cols)
            if i < rows:
                # Создаем гауссово распределение вокруг диагонали
                for r in range(rows):
                    diagonal[r, j] = np.exp(-0.5 * ((r - i) / (rows * 0.05)) ** 2)
        
        return diagonal
    
    def _calculate_monotonicity(self, peaks: np.ndarray) -> float:
        """Вычисляет монотонность последовательности пиков."""
        if len(peaks) < 2:
            return 1.0
            
        monotonic_count = 0
        for i in range(len(peaks) - 1):
            if peaks[i + 1] >= peaks[i]:
                monotonic_count += 1
        
        return monotonic_count / (len(peaks) - 1)
    
    def _get_attention_quality_text(self, diagonality: float) -> str:
        """Возвращает текстовое описание качества attention."""
        if diagonality >= 0.85:
            return "🟢 Отличная диагональность! Модель идеально выравнивает текст и аудио."
        elif diagonality >= 0.7:
            return "🟡 Хорошая диагональность. Небольшие улучшения возможны."
        elif diagonality >= 0.5:
            return "🟠 Удовлетворительная диагональность. Требуются улучшения."
        elif diagonality >= 0.3:
            return "🔴 Слабая диагональность. Серьезные проблемы с alignment."
        else:
            return "🚨 КРИТИЧЕСКИ низкая диагональность! Горизонтальная полоса вместо диагонали."
    
    def _get_attention_status_emoji(self, diagonality: float) -> str:
        """Возвращает emoji статуса attention."""
        if diagonality >= 0.85:
            return "🎯"
        elif diagonality >= 0.7:
            return "✅"
        elif diagonality >= 0.5:
            return "⚠️"
        elif diagonality >= 0.3:
            return "❌"
        else:
            return "🚨"
    
    def _analyze_quality_trend(self) -> str:
        """Анализирует тренд качества."""
        if len(self.metrics_history['quality_score']) < 5:
            return "Недостаточно данных"
            
        recent_scores = self.metrics_history['quality_score'][-5:]
        if recent_scores[-1] > recent_scores[0]:
            return "📈 Улучшается"
        elif recent_scores[-1] < recent_scores[0]:
            return "📉 Ухудшается"
        else:
            return "📊 Стабильно"
    
    def _get_training_status(self) -> str:
        """Возвращает общий статус обучения."""
        if not self.metrics_history['quality_score']:
            return "Инициализация"
            
        latest_quality = self.metrics_history['quality_score'][-1]
        if latest_quality >= 0.8:
            return "🟢 Отлично"
        elif latest_quality >= 0.6:
            return "🟡 Хорошо"
        elif latest_quality >= 0.4:
            return "🟠 Удовлетворительно"
        else:
            return "🔴 Требует внимания"
    
    def _get_training_recommendations(self) -> str:
        """Возвращает краткие рекомендации."""
        if not self.metrics_history['attention_diagonality']:
            return "Мониторинг качества"
            
        latest_diag = self.metrics_history['attention_diagonality'][-1]
        if latest_diag < 0.3:
            return "🔥 Проверить guided attention loss!"
        elif latest_diag < 0.6:
            return "⚡ Снизить learning rate"
        else:
            return "✅ Продолжить текущие настройки"
    
    def _analyze_potential_issues(self, metrics: Dict[str, Any]) -> List[str]:
        """Анализирует потенциальные проблемы."""
        issues = []
        
        # Проверка диагональности
        diag = metrics.get('attention_diagonality', 0)
        if diag < 0.3:
            issues.append("🚨 Горизонтальная полоса вместо диагонали - критическая проблема alignment!")
        elif diag < 0.5:
            issues.append("⚠️ Слабая диагональность attention")
            
        # Проверка gate accuracy
        gate_acc = metrics.get('gate_accuracy', 0)
        if gate_acc < 0.5:
            issues.append("❌ Низкая точность gate - проблемы с остановкой")
            
        # Проверка loss
        train_loss = metrics.get('train_loss', 0)
        val_loss = metrics.get('val_loss', 0)
        if val_loss > train_loss * 1.5:
            issues.append("📈 Признаки переобучения")
            
        return issues
    
    def _get_specific_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Возвращает специфические рекомендации."""
        recommendations = []
        
        diag = metrics.get('attention_diagonality', 0)
        if diag < 0.3:
            recommendations.extend([
                "🔥 Увеличить guide_loss_weight до 15.0",
                "⚡ Снизить learning_rate до 1e-5",
                "🎯 Проверить guided attention реализацию"
            ])
        elif diag < 0.6:
            recommendations.extend([
                "📊 Мониторить прогресс alignment",
                "⚙️ Рассмотреть адаптивные параметры"
            ])
        else:
            recommendations.append("✅ Отличная работа! Продолжать обучение")
            
        return recommendations
    
    async def _send_message_async(self, message: str) -> None:
        """Асинхронно отправляет текстовое сообщение."""
        try:
            if self.bot:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
        except Exception as e:
            self.logger.error(f"❌ Ошибка отправки сообщения: {e}")
    
    async def _send_photo_async(self, photo_buffer: io.BytesIO, caption: str) -> None:
        """Асинхронно отправляет изображение с подписью."""
        try:
            if self.bot:
                await self.bot.send_photo(
                    chat_id=self.chat_id,
                    photo=photo_buffer,
                    caption=caption,
                    parse_mode='Markdown'
                )
        except Exception as e:
            self.logger.error(f"❌ Ошибка отправки изображения: {e}")
    
    def send_training_start(self, config: Dict[str, Any]) -> None:
        """Отправляет уведомление о старте обучения."""
        if not self.enabled:
            return
            
        message = f"🚀 **ЗАПУСК ОБУЧЕНИЯ TTS**\n\n"
        message += f"📅 **Время:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"🎯 **Модель:** Smart Tuner TTS v2\n"
        message += f"⚙️ **Learning Rate:** `{config.get('learning_rate', 'N/A')}`\n"
        message += f"📦 **Batch Size:** `{config.get('batch_size', 'N/A')}`\n\n"
        message += f"📊 **Уведомления каждые 1000 шагов**\n"
        message += f"🎵 **Цель: максимально человеческий голос без артефактов**"
        
        asyncio.create_task(self._send_message_async(message))
    
    def send_training_complete(self, final_metrics: Dict[str, Any]) -> None:
        """Отправляет уведомление о завершении обучения."""
        if not self.enabled:
            return
            
        message = f"🏁 **ОБУЧЕНИЕ ЗАВЕРШЕНО!**\n\n"
        message += f"📅 **Время завершения:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"🎯 **Финальная диагональность:** `{final_metrics.get('final_diagonality', 0):.3f}`\n"
        message += f"📊 **Финальное качество:** `{final_metrics.get('final_quality', 0):.1%}`\n"
        message += f"🏆 **Лучший val_loss:** `{final_metrics.get('best_val_loss', 0):.4f}`\n\n"
        
        if final_metrics.get('final_diagonality', 0) >= 0.8:
            message += f"🎉 **ПОЗДРАВЛЯЕМ!** Достигнута отличная диагональность!\n"
            message += f"🎵 Модель готова для генерации качественного человеческого голоса!"
        else:
            message += f"💡 **Рекомендация:** Продолжить обучение для улучшения alignment"
        
        asyncio.create_task(self._send_message_async(message)) 