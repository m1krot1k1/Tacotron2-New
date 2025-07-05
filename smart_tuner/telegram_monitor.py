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
        
    def send_auto_improvement_notification(self, improvement_type: str, 
                                         old_params: Dict[str, Any], 
                                         new_params: Dict[str, Any], 
                                         reason: str,
                                         step: int) -> bool:
        """
        🤖 Отправляет уведомление об автоматическом улучшении системы.
        
        Args:
            improvement_type: Тип улучшения (learning_rate, guided_attention, etc.)
            old_params: Старые параметры
            new_params: Новые параметры  
            reason: Причина изменения
            step: Текущий шаг обучения
        """
        if not self.enabled:
            return False
            
        try:
            message = self._create_improvement_message(
                improvement_type, old_params, new_params, reason, step
            )
            
            result = self._send_text_message(message)
            if result:
                self.logger.info(f"✅ Уведомление об улучшении отправлено: {improvement_type}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка отправки уведомления об улучшении: {e}")
            return False
    
    def send_problem_detection_alert(self, problems: List[Dict[str, Any]], step: int) -> bool:
        """
        🚨 Отправляет критическое уведомление об обнаружении серьезных проблем.
        """
        if not self.enabled:
            return False
            
        try:
            message = self._create_problem_alert_message(problems, step)
            result = self._send_text_message(message)
            
            if result:
                self.logger.info(f"✅ Критическое уведомление о проблемах отправлено")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка отправки критического уведомления: {e}")
            return False
    
    def send_training_phase_notification(self, old_phase: str, new_phase: str, 
                                       step: int, achievements: List[str]) -> bool:
        """
        🎯 Уведомление о переходе между фазами обучения.
        """
        if not self.enabled:
            return False
            
        try:
            message = self._create_phase_transition_message(old_phase, new_phase, step, achievements)
            result = self._send_text_message(message)
            
            if result:
                self.logger.info(f"✅ Уведомление о смене фазы отправлено: {old_phase} → {new_phase}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка отправки уведомления о фазе: {e}")
            return False
    
    def send_critical_alert(self, alert_type: str, details: Dict[str, Any], 
                          recommendations: List[str] = None) -> bool:
        """
        🚨 Отправляет критическое уведомление о серьезных проблемах.
        
        Args:
            alert_type: Тип алерта
            details: Детали проблемы
            recommendations: Список рекомендаций
        """
        if not self.enabled:
            return False
            
        try:
            message = self._create_critical_alert_message(alert_type, details, recommendations)
            result = self._send_text_message(message)
            
            if result:
                self.logger.info(f"✅ Критическое уведомление отправлено: {alert_type}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка отправки критического уведомления: {e}")
            return False
    
    def send_restart_notification(self, reason: str, step: int) -> bool:
        """
        🔄 Отправляет уведомление о перезапуске обучения.
        
        Args:
            reason: Причина перезапуска
            step: Шаг на котором произошел перезапуск
        """
        if not self.enabled:
            return False
            
        try:
            message = self._create_restart_message(reason, step)
            result = self._send_text_message(message)
            
            if result:
                self.logger.info(f"✅ Уведомление о перезапуске отправлено")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка отправки уведомления о перезапуске: {e}")
            return False
    
    def send_detailed_telegram_report(self, step: int, metrics: Dict[str, Any], 
                                    actions_taken: List[str], 
                                    gradient_norm: float = None,
                                    attention_diagonality: float = None) -> bool:
        """
        📱 Отправляет детальный отчет с конкретными действиями и метриками.
        
        Args:
            step: Текущий шаг обучения
            metrics: Словарь с метриками
            actions_taken: Список выполненных действий
            gradient_norm: Норма градиентов
            attention_diagonality: Диагональность attention
        """
        if not self.enabled:
            return False
            
        try:
            message = f"🤖 **Smart Tuner V2 - Детальный отчет**\n\n"
            message += f"📍 **Шаг:** {step:,}\n"
            message += f"🕐 **Время:** {datetime.now().strftime('%H:%M:%S')}\n\n"
            
            # Критические метрики
            if gradient_norm is not None:
                status_emoji = "✅" if gradient_norm < 10.0 else "⚠️" if gradient_norm < 100.0 else "🚨"
                message += f"{status_emoji} **Gradient Norm:** {gradient_norm:.2f}\n"
            
            if attention_diagonality is not None:
                status_emoji = "✅" if attention_diagonality > 0.7 else "⚠️" if attention_diagonality > 0.3 else "🚨"
                message += f"{status_emoji} **Attention Diagonality:** {attention_diagonality:.3f}\n"
            
            # Основные метрики
            if 'loss' in metrics:
                message += f"📉 **Loss:** {metrics['loss']:.4f}\n"
            
            if 'learning_rate' in metrics:
                message += f"📈 **Learning Rate:** {metrics['learning_rate']:.2e}\n"
            
            # Выполненные действия
            if actions_taken:
                message += f"\n🛠️ **Выполненные действия:**\n"
                for i, action in enumerate(actions_taken, 1):
                    message += f"  {i}. {action}\n"
            
            # Рекомендации на основе метрик
            recommendations = []
            if gradient_norm and gradient_norm > 100.0:
                recommendations.append("Снизить learning rate")
                recommendations.append("Усилить gradient clipping")
            
            if attention_diagonality and attention_diagonality < 0.3:
                recommendations.append("Увеличить вес guided attention loss")
                recommendations.append("Проверить alignment diagnostics")
            
            if recommendations:
                message += f"\n💡 **Рекомендации:**\n"
                for rec in recommendations:
                    message += f"  • {rec}\n"
            
            # Статус системы
            message += f"\n🎯 **Статус системы:** "
            if gradient_norm and gradient_norm < 10.0 and attention_diagonality and attention_diagonality > 0.7:
                message += "✅ **СТАБИЛЬНА**"
            elif gradient_norm and gradient_norm > 100.0 or (attention_diagonality and attention_diagonality < 0.1):
                message += "🚨 **КРИТИЧЕСКАЯ**"
            else:
                message += "⚠️ **ТРЕБУЕТ ВНИМАНИЯ**"
            
            result = self._send_text_message(message)
            
            if result:
                self.logger.info(f"✅ Детальный отчет отправлен на шаге {step}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка отправки детального отчета: {e}")
            return False
        
    def should_send_notification(self, current_step: int) -> bool:
        """Проверяет нужность отправки уведомления."""
        if not self.enabled:
            return False
            
        # Разрешаем уведомление на самом первом шаге (0) для первичного отчёта
        if current_step == 0 and self.last_notification_step == 0:
            return True

        return (current_step - self.last_notification_step) >= self.notification_interval
    
    def send_training_update(self, step: int, metrics: Dict[str, Any],
                           attention_weights: Optional[torch.Tensor] = None,
                           gate_outputs: Optional[torch.Tensor] = None,
                           smart_tuner_decisions: Optional[Dict[str, Any]] = None) -> bool:
        """Отправляет обновление с изображениями и решениями умной системы."""
        if not self.should_send_notification(step):
            return False
        
        try:
            # Анализ данных
            analysis = self._analyze_step(step, metrics, attention_weights, gate_outputs)
            
            # Добавляем информацию о решениях Smart Tuner
            if smart_tuner_decisions:
                analysis['smart_tuner_decisions'] = smart_tuner_decisions
            
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
        
        message = f"🧠 Smart Tuner - Отчет Обучения\n\n"
        message += f"📍 Шаг: {step:,}\n"
        message += f"🎭 Фаза: {phase}\n"
        message += f"{quality_emoji} Качество: {quality:.1%}\n\n"
        
        # Метрики
        metrics = analysis['metrics']
        if 'loss' in metrics:
            message += f"📉 Loss: {metrics['loss']:.4f}\n"
        
        # Анализ attention
        attention = analysis.get('attention', {})
        if attention:
            diag = attention.get('diagonality', 0)
            message += f"🎯 Attention Диагональность: {diag:.1%}\n"
            
            if diag < 0.3:
                message += f"  ⚠️ Критично низкая! Проблемы с alignment\n"
            elif diag > 0.7:
                message += f"  ✅ Отличная диагональность!\n"
        
        # 🤖 РЕШЕНИЯ SMART TUNER
        smart_decisions = analysis.get('smart_tuner_decisions', {})
        if smart_decisions:
            message += f"\n🤖 УМНАЯ СИСТЕМА ПРИНЯЛА РЕШЕНИЯ:\n"
            
            # Изменения гиперпараметров
            param_changes = smart_decisions.get('parameter_changes', {})
            if param_changes:
                message += f"⚙️ Изменения параметров:\n"
                for param, change_info in param_changes.items():
                    old_val = change_info.get('old_value', 'N/A')
                    new_val = change_info.get('new_value', 'N/A')
                    reason = change_info.get('reason', 'Автоматическая оптимизация')
                    message += f"  • {param}: {old_val} → {new_val}\n"
                    message += f"    💡 Причина: {reason}\n"
            
            # ПРИМЕНЕННЫЕ РЕКОМЕНДАЦИИ
            applied_recommendations = smart_decisions.get('recent_applied_recommendations', [])
            if applied_recommendations:
                message += f"\n✅ ПРИМЕНЕННЫЕ РЕКОМЕНДАЦИИ:\n"
                for rec in applied_recommendations[-3:]:  # Последние 3
                    status_emoji = "✅" if rec.get('success', False) else "❌"
                    message += f"  {status_emoji} {rec.get('recommendation', 'Неизвестно')}\n"
                    message += f"    🛠️ Действие: {rec.get('action_taken', 'Не указано')}\n"
                    if rec.get('result_description'):
                        message += f"    📊 Результат: {rec.get('result_description')}\n"
            
            # СВОДКА РЕКОМЕНДАЦИЙ
            recommendation_summary = smart_decisions.get('recommendation_summary', {})
            if recommendation_summary:
                total = recommendation_summary.get('total_recommendations', 0)
                successful = recommendation_summary.get('successful_recommendations', 0)
                success_rate = recommendation_summary.get('success_rate', 0)
                if total > 0:
                    message += f"\n📊 СВОДКА РЕКОМЕНДАЦИЙ:\n"
                    message += f"  • Всего применено: {total}\n"
                    message += f"  • Успешных: {successful}\n"
                    message += f"  • Успешность: {success_rate:.1%}\n"
            
            # Рекомендации от контроллеров
            recommendations = smart_decisions.get('recommendations', [])
            if recommendations:
                message += f"\n💡 НОВЫЕ РЕКОМЕНДАЦИИ:\n"
                for rec in recommendations[:3]:  # Показываем до 3 рекомендаций
                    message += f"  • {rec}\n"
            
            # Статус контроллеров
            controller_status = smart_decisions.get('controller_status', {})
            if controller_status:
                message += f"\n🎛️ Статус контроллеров:\n"
                for controller, status in controller_status.items():
                    status_emoji = "✅" if status.get('active', False) else "⏸️"
                    message += f"  {status_emoji} {controller}: {status.get('status', 'Неизвестно')}\n"
            
            # Предупреждения и проблемы
            warnings = smart_decisions.get('warnings', [])
            if warnings:
                message += f"\n⚠️ Предупреждения:\n"
                for warning in warnings[:2]:  # Показываем до 2 предупреждений
                    message += f"  • {warning}\n"
        
        # Проблемы
        issues = analysis.get('issues', [])
        if issues:
            message += f"\n⚠️ **Проблемы:**\n"
            for issue in issues[:2]:
                message += f"  • {issue}\n"
        
        # Рекомендации (если нет решений от Smart Tuner)
        if not smart_decisions:
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
    
    def _create_improvement_message(self, improvement_type: str, 
                                  old_params: Dict[str, Any], 
                                  new_params: Dict[str, Any], 
                                  reason: str, step: int) -> str:
        """Создает сообщение об автоматическом улучшении."""
        
        # Иконки для разных типов улучшений
        type_icons = {
            'learning_rate': '⚡',
            'guided_attention': '🎯', 
            'dropout': '🛡️',
            'batch_size': '📦',
            'gate_threshold': '🚪',
            'curriculum_learning': '🎓',
            'early_stopping': '🛑',
            'gradient_clipping': '✂️'
        }
        
        icon = type_icons.get(improvement_type, '🔧')
        
        message = f"🤖 *Smart Tuner - АВТОМАТИЧЕСКОЕ УЛУЧШЕНИЕ*\n\n"
        message += f"{icon} **Тип:** `{improvement_type.replace('_', ' ').title()}`\n"
        message += f"📍 **Шаг:** `{step:,}`\n"
        message += f"🧠 **Причина:** {reason}\n\n"
        
        message += f"**📊 ИЗМЕНЕНИЯ ПАРАМЕТРОВ:**\n"
        
        # Сравнение старых и новых параметров
        for param_name in set(list(old_params.keys()) + list(new_params.keys())):
            old_val = old_params.get(param_name, 'N/A')
            new_val = new_params.get(param_name, 'N/A')
            
            if old_val != new_val:
                # Определяем направление изменения
                if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    if new_val > old_val:
                        trend = "📈"
                    elif new_val < old_val:
                        trend = "📉"
                    else:
                        trend = "➡️"
                else:
                    trend = "🔄"
                
                message += f"  {trend} `{param_name}`: `{old_val}` → `{new_val}`\n"
        
        message += f"\n💡 **ОЖИДАЕМЫЙ ЭФФЕКТ:**\n"
        
        # Предсказания эффекта на основе типа улучшения
        effects = {
            'learning_rate': ["🎯 Более стабильное обучение", "⚡ Лучшая сходимость"],
            'guided_attention': ["🎵 Улучшение alignment", "🎯 Более четкая дикция"],
            'dropout': ["🛡️ Снижение переобучения", "💪 Лучшая генерализация"],
            'batch_size': ["⚡ Оптимизация скорости", "📊 Лучшие градиенты"],
            'gate_threshold': ["🚪 Точное определение конца", "🎵 Лучшая просодия"],
            'curriculum_learning': ["🎓 Поэтапное усложнение", "🚀 Ускорение обучения"]
        }
        
        expected_effects = effects.get(improvement_type, ["🔧 Общее улучшение качества"])
        for effect in expected_effects:
            message += f"  • {effect}\n"
        
        message += f"\n🕐 {datetime.now().strftime('%H:%M:%S')}"
        message += f"\n🎯 *Система продолжает мониторинг...*"
        
        return message
    
    def _create_problem_alert_message(self, problems: List[Dict[str, Any]], step: int) -> str:
        """Создает критическое сообщение о проблемах."""
        
        message = f"🚨 *КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ*\n\n"
        message += f"📍 **Шаг:** `{step:,}`\n"
        message += f"⚠️ **Обнаружено проблем:** `{len(problems)}`\n\n"
        
        message += f"**🔍 ДЕТАЛЬНЫЙ АНАЛИЗ:**\n"
        
        for i, problem in enumerate(problems[:3], 1):  # Показываем до 3 проблем
            severity = problem.get('severity', 'medium')
            severity_icons = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
            icon = severity_icons.get(severity, '⚠️')
            
            message += f"{icon} **Проблема {i}:** {problem.get('description', 'Неизвестная проблема')}\n"
            message += f"   📊 *Значение:* `{problem.get('value', 'N/A')}`\n"
            message += f"   🎯 *Порог:* `{problem.get('threshold', 'N/A')}`\n"
            
            if 'recommendation' in problem:
                message += f"   💡 *Рекомендация:* {problem['recommendation']}\n"
            
            message += "\n"
        
        if len(problems) > 3:
            message += f"⚠️ *И еще {len(problems) - 3} проблем...*\n\n"
        
        message += f"🤖 **АВТОМАТИЧЕСКИЕ ДЕЙСТВИЯ:**\n"
        message += f"  🔄 Система анализирует варианты исправления\n"
        message += f"  ⚡ Подготовка адаптивных изменений\n"
        message += f"  📊 Мониторинг эффективности\n\n"
        
        message += f"🕐 {datetime.now().strftime('%H:%M:%S')}"
        message += f"\n🎯 *Следите за уведомлениями об улучшениях!*"
        
        return message
    
    def _create_phase_transition_message(self, old_phase: str, new_phase: str, 
                                       step: int, achievements: List[str]) -> str:
        """Создает сообщение о переходе между фазами."""
        
        phase_names = {
            'pre_alignment': '🌱 Предварительное выравнивание',
            'alignment_learning': '🎯 Обучение выравниванию', 
            'quality_optimization': '⭐ Оптимизация качества',
            'fine_tuning': '🎵 Финальная настройка'
        }
        
        phase_descriptions = {
            'pre_alignment': 'Модель учится базовым принципам attention',
            'alignment_learning': 'Отработка точного выравнивания текст-аудио',
            'quality_optimization': 'Улучшение качества и естественности речи',
            'fine_tuning': 'Финальная полировка и устранение артефактов'
        }
        
        old_name = phase_names.get(old_phase, old_phase)
        new_name = phase_names.get(new_phase, new_phase) 
        new_desc = phase_descriptions.get(new_phase, 'Продолжение обучения')
        
        message = f"🎯 *ПЕРЕХОД К НОВОЙ ФАЗЕ ОБУЧЕНИЯ*\n\n"
        message += f"📍 **Шаг:** `{step:,}`\n"
        message += f"🔄 **Переход:** {old_name} → {new_name}\n\n"
        
        message += f"**🎭 НОВАЯ ФАЗА:**\n"
        message += f"🎯 *Фокус:* {new_desc}\n\n"
        
        if achievements:
            message += f"**✅ ДОСТИЖЕНИЯ ПРЕДЫДУЩЕЙ ФАЗЫ:**\n"
            for achievement in achievements:
                message += f"  🏆 {achievement}\n"
            message += "\n"
        
        # Предсказания для новой фазы
        phase_predictions = {
            'alignment_learning': [
                "📈 Ожидается улучшение диагональности attention",
                "🎯 Фокус на монотонности выравнивания"
            ],
            'quality_optimization': [
                "⭐ Улучшение качества mel-спектрограмм", 
                "🎵 Повышение естественности речи"
            ],
            'fine_tuning': [
                "🎵 Устранение последних артефактов",
                "✨ Доведение до совершенства"
            ]
        }
        
        predictions = phase_predictions.get(new_phase, ["🚀 Продолжение улучшения качества"])
        message += f"**🔮 ОЖИДАНИЯ ОТ НОВОЙ ФАЗЫ:**\n"
        for prediction in predictions:
            message += f"  • {prediction}\n"
        
        message += f"\n🕐 {datetime.now().strftime('%H:%M:%S')}"
        message += f"\n🎯 *Система адаптирует параметры для новой фазы*"
        
        return message
    
    def _create_critical_alert_message(self, alert_type: str, details: Dict[str, Any], 
                                     recommendations: List[str] = None) -> str:
        """Создает критическое сообщение об алерте."""
        
        message = f"🚨 *КРИТИЧЕСКИЙ АЛЕРТ: {alert_type}*\n\n"
        
        # Детали проблемы
        if 'description' in details:
            message += f"📋 **Описание:** {details['description']}\n\n"
        
        if 'step' in details:
            message += f"📍 **Шаг:** `{details['step']:,}`\n"
        
        # Метрики если есть
        if 'metrics' in details:
            message += f"\n📊 **Проблемные метрики:**\n"
            for metric, value in details['metrics'].items():
                message += f"• {metric}: `{value}`\n"
        
        # Список проблем
        if 'issues' in details:
            message += f"\n🔥 **Обнаруженные проблемы:**\n"
            for issue in details['issues']:
                message += f"• {issue}\n"
        
        # Рекомендации
        if recommendations:
            message += f"\n💡 **Рекомендации:**\n"
            for rec in recommendations:
                message += f"• {rec}\n"
        
        message += f"\n🛡️ **Автоматические действия активированы!**"
        message += f"\n🕐 {datetime.now().strftime('%H:%M:%S')}"
        
        return message
    
    def _create_restart_message(self, reason: str, step: int) -> str:
        """Создает сообщение о перезапуске."""
        
        message = f"🔄 *АВТОМАТИЧЕСКИЙ ПЕРЕЗАПУСК ОБУЧЕНИЯ*\n\n"
        message += f"🚨 **Причина:** {reason}\n"
        message += f"📍 **Шаг:** `{step:,}`\n"
        message += f"🕐 **Время:** {datetime.now().strftime('%H:%M:%S')}\n\n"
        
        message += f"🛡️ **Система восстановления активирована:**\n"
        message += f"• 🔥 Снижение learning rate\n"
        message += f"• 🎯 Усиление guided attention\n"
        message += f"• 📦 Оптимизация batch size\n"
        message += f"• ✂️ Строгое клипирование градиентов\n\n"
        
        message += f"⏰ **Перезапуск через несколько секунд...**"
        
        return message 