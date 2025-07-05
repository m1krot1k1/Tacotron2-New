#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Telegram Monitor для Smart Tuner V2
Улучшенная система уведомлений с детальной диагностикой и конкретными действиями

Особенности:
1. Детальные отчеты с конкретными метриками
2. Интеллектуальные рекомендации
3. Интеграция с EnhancedTacotronTrainer
4. Автоматические алерты при критических проблемах
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List
import requests
import json

class TelegramMonitorEnhanced:
    """
    Улучшенный Telegram Monitor с детальной диагностикой
    """
    
    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        """
        Инициализация улучшенного Telegram Monitor
        
        Args:
            bot_token: Токен Telegram бота
            chat_id: ID чата для отправки уведомлений
            enabled: Включить/выключить мониторинг
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.logger = logging.getLogger('TelegramMonitorEnhanced')
        
        # История метрик для анализа трендов
        self.metrics_history = []
        self.last_alert_time = 0
        self.alert_cooldown = 300  # 5 минут между алертами
        
        # Пороги для критических алертов
        self.critical_thresholds = {
            'gradient_norm': 100.0,
            'attention_diagonality': 0.1,
            'loss': 50.0,
            'gate_accuracy': 0.3
        }
        
        self.logger.info("📱 Enhanced Telegram Monitor инициализирован")
    
    def send_training_update(self, step: int, metrics: Dict[str, Any], 
                           attention_weights=None, gate_outputs=None) -> bool:
        """
        Отправка детального отчета о тренировке
        
        Args:
            step: Текущий шаг обучения
            metrics: Словарь с метриками
            attention_weights: Веса attention для анализа
            gate_outputs: Выходы gate для анализа
            
        Returns:
            True если отправка успешна
        """
        if not self.enabled:
            return False
        
        try:
            # Анализируем метрики
            analysis = self._analyze_metrics(step, metrics, attention_weights, gate_outputs)
            
            # Формируем детальное сообщение
            message = self._format_detailed_message(step, metrics, analysis)
            
            # Отправляем сообщение
            success = self._send_message(message)
            
            # Сохраняем метрики в историю
            self.metrics_history.append({
                'step': step,
                'metrics': metrics,
                'analysis': analysis,
                'timestamp': time.time()
            })
            
            # Ограничиваем размер истории
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-50:]
            
            return success
            
        except Exception as e:
            self.logger.error(f"Ошибка отправки training update: {e}")
            return False
    
    def send_critical_alert(self, issue_type: str, details: Dict[str, Any]) -> bool:
        """
        Отправка критического алерта
        
        Args:
            issue_type: Тип проблемы
            details: Детали проблемы
            
        Returns:
            True если отправка успешна
        """
        if not self.enabled:
            return False
        
        # Проверяем cooldown
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False
        
        try:
            message = self._format_critical_alert(issue_type, details)
            success = self._send_message(message)
            
            if success:
                self.last_alert_time = current_time
            
            return success
            
        except Exception as e:
            self.logger.error(f"Ошибка отправки критического алерта: {e}")
            return False
    
    def send_quality_report(self, step: int, quality_metrics: Dict[str, Any]) -> bool:
        """
        Отправка отчета о качестве
        
        Args:
            step: Текущий шаг
            quality_metrics: Метрики качества
            
        Returns:
            True если отправка успешна
        """
        if not self.enabled:
            return False
        
        try:
            message = self._format_quality_report(step, quality_metrics)
            return self._send_message(message)
            
        except Exception as e:
            self.logger.error(f"Ошибка отправки quality report: {e}")
            return False
    
    def _analyze_metrics(self, step: int, metrics: Dict[str, Any], 
                        attention_weights=None, gate_outputs=None) -> Dict[str, Any]:
        """
        Анализ метрик и выявление проблем
        
        Args:
            step: Текущий шаг
            metrics: Метрики обучения
            attention_weights: Веса attention
            gate_outputs: Выходы gate
            
        Returns:
            Словарь с анализом
        """
        analysis = {
            'issues': [],
            'recommendations': [],
            'trends': {},
            'status': 'normal'
        }
        
        # Анализ градиентов
        grad_norm = metrics.get('grad_norm', 0.0)
        if grad_norm > self.critical_thresholds['gradient_norm']:
            analysis['issues'].append(f"🚨 Критическая норма градиентов: {grad_norm:.2f}")
            analysis['recommendations'].append("Уменьшите learning rate или увеличьте gradient clipping")
            analysis['status'] = 'critical'
        elif grad_norm > 10.0:
            analysis['issues'].append(f"⚠️ Высокая норма градиентов: {grad_norm:.2f}")
            analysis['recommendations'].append("Рассмотрите уменьшение learning rate")
            analysis['status'] = 'warning'
        
        # Анализ attention alignment
        attention_diagonality = metrics.get('attention_diagonality', 0.0)
        if attention_diagonality < self.critical_thresholds['attention_diagonality']:
            analysis['issues'].append(f"🚨 Плохое attention alignment: {attention_diagonality:.3f}")
            analysis['recommendations'].append("Увеличьте guided attention weight")
            analysis['status'] = 'critical'
        elif attention_diagonality < 0.3:
            analysis['issues'].append(f"⚠️ Слабое attention alignment: {attention_diagonality:.3f}")
            analysis['recommendations'].append("Проверьте guided attention loss")
            analysis['status'] = 'warning'
        
        # Анализ loss
        loss = metrics.get('loss', 0.0)
        if loss > self.critical_thresholds['loss']:
            analysis['issues'].append(f"🚨 Критически высокий loss: {loss:.2f}")
            analysis['recommendations'].append("Проверьте данные и гиперпараметры")
            analysis['status'] = 'critical'
        
        # Анализ gate accuracy
        gate_accuracy = metrics.get('gate_accuracy', 0.0)
        if gate_accuracy < self.critical_thresholds['gate_accuracy']:
            analysis['issues'].append(f"🚨 Низкая gate accuracy: {gate_accuracy:.3f}")
            analysis['recommendations'].append("Проверьте gate loss и данные")
            analysis['status'] = 'critical'
        
        # Анализ трендов
        if len(self.metrics_history) >= 5:
            recent_metrics = self.metrics_history[-5:]
            loss_trend = self._calculate_trend([m['metrics'].get('loss', 0) for m in recent_metrics])
            
            if loss_trend > 0.1:  # Loss растет
                analysis['trends']['loss'] = 'increasing'
                analysis['recommendations'].append("Loss растет - проверьте learning rate")
            elif loss_trend < -0.1:  # Loss падает
                analysis['trends']['loss'] = 'decreasing'
                analysis['recommendations'].append("Loss падает - обучение идет хорошо")
        
        return analysis
    
    def _format_detailed_message(self, step: int, metrics: Dict[str, Any], 
                                analysis: Dict[str, Any]) -> str:
        """
        Форматирование детального сообщения
        
        Args:
            step: Текущий шаг
            metrics: Метрики обучения
            analysis: Результаты анализа
            
        Returns:
            Отформатированное сообщение
        """
        message = f"🤖 **Smart Tuner V2 - Детальный отчет**\n\n"
        message += f"📊 **Шаг:** {step:,}\n"
        message += f"⏰ **Время:** {time.strftime('%H:%M:%S')}\n\n"
        
        # Основные метрики
        message += "📈 **Основные метрики:**\n"
        message += f"  • Loss: {metrics.get('loss', 0):.4f}\n"
        message += f"  • Gradient Norm: {metrics.get('grad_norm', 0):.2f}\n"
        message += f"  • Attention Diagonality: {metrics.get('attention_diagonality', 0):.3f}\n"
        message += f"  • Gate Accuracy: {metrics.get('gate_accuracy', 0):.3f}\n\n"
        
        # Детализированные loss компоненты
        if 'loss_breakdown' in metrics:
            breakdown = metrics['loss_breakdown']
            message += "🔍 **Детализация Loss:**\n"
            message += f"  • Mel Loss: {breakdown.get('mel_loss', 0):.4f}\n"
            message += f"  • Gate Loss: {breakdown.get('gate_loss', 0):.4f}\n"
            message += f"  • Guide Loss: {breakdown.get('guide_loss', 0):.4f}\n"
            message += f"  • Emb Loss: {breakdown.get('emb_loss', 0):.4f}\n\n"
        
        # Проблемы и рекомендации
        if analysis['issues']:
            message += "🚨 **Обнаруженные проблемы:**\n"
            for issue in analysis['issues']:
                message += f"  • {issue}\n"
            message += "\n"
        
        if analysis['recommendations']:
            message += "💡 **Рекомендации:**\n"
            for rec in analysis['recommendations']:
                message += f"  • {rec}\n"
            message += "\n"
        
        # Статус
        status_emoji = {
            'normal': '✅',
            'warning': '⚠️',
            'critical': '🚨'
        }
        message += f"{status_emoji.get(analysis['status'], '❓')} **Статус:** {analysis['status'].upper()}\n"
        
        return message
    
    def _format_critical_alert(self, issue_type: str, details: Dict[str, Any]) -> str:
        """
        Форматирование критического алерта
        
        Args:
            issue_type: Тип проблемы
            details: Детали проблемы
            
        Returns:
            Отформатированное сообщение
        """
        message = f"🚨 **КРИТИЧЕСКИЙ АЛЕРТ**\n\n"
        message += f"🔴 **Тип проблемы:** {issue_type}\n"
        message += f"⏰ **Время:** {time.strftime('%H:%M:%S')}\n\n"
        
        message += "📋 **Детали:**\n"
        for key, value in details.items():
            if isinstance(value, float):
                message += f"  • {key}: {value:.4f}\n"
            else:
                message += f"  • {key}: {value}\n"
        
        message += "\n🛠️ **Требуется немедленное вмешательство!**"
        
        return message
    
    def _format_quality_report(self, step: int, quality_metrics: Dict[str, Any]) -> str:
        """
        Форматирование отчета о качестве
        
        Args:
            step: Текущий шаг
            quality_metrics: Метрики качества
            
        Returns:
            Отформатированное сообщение
        """
        message = f"🎯 **Отчет о качестве TTS**\n\n"
        message += f"📊 **Шаг:** {step:,}\n"
        message += f"⏰ **Время:** {time.strftime('%H:%M:%S')}\n\n"
        
        message += "📈 **Метрики качества:**\n"
        for key, value in quality_metrics.items():
            if isinstance(value, float):
                message += f"  • {key}: {value:.4f}\n"
            else:
                message += f"  • {key}: {value}\n"
        
        return message
    
    def _send_message(self, message: str) -> bool:
        """
        Отправка сообщения в Telegram
        
        Args:
            message: Текст сообщения
            
        Returns:
            True если отправка успешна
        """
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                return True
            else:
                self.logger.error(f"Ошибка отправки Telegram: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка отправки Telegram: {e}")
            return False
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Вычисление тренда значений
        
        Args:
            values: Список значений
            
        Returns:
            Коэффициент тренда (положительный = рост, отрицательный = падение)
        """
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Линейная регрессия
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """
        Получение истории метрик
        
        Returns:
            Список метрик
        """
        return self.metrics_history.copy()
    
    def clear_history(self):
        """Очистка истории метрик"""
        self.metrics_history.clear()
        self.logger.info("История метрик очищена") 