"""
Alert Manager для Smart Tuner V2
Система уведомлений через Telegram
"""

import yaml
import logging
import requests
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import threading
from pathlib import Path

class AlertManager:
    """
    Менеджер уведомлений для отправки алертов через Telegram
    Поддерживает различные типы уведомлений и приоритеты
    """
    
    def __init__(self, config_or_path = "smart_tuner/config.yaml"):
        """
        Инициализация менеджера уведомлений
        
        Args:
            config_or_path: Путь к файлу конфигурации или готовый config dict
        """
        if isinstance(config_or_path, dict):
            # Получен готовый config
            self.config = config_or_path
            self.config_path = None
        else:
            # Получен путь к файлу
            self.config_path = config_or_path
            self.config = self._load_config()
            
        self.logger = self._setup_logger()
        
        # Настройки Telegram
        telegram_config = self.config.get('telegram', {})
        self.bot_token = telegram_config.get('bot_token')
        self.chat_id = telegram_config.get('chat_id')
        self.enabled = telegram_config.get('enabled', False)
        self.parse_mode = telegram_config.get('parse_mode', 'Markdown')
        self.notifications = telegram_config.get('notifications', {})
        
        # Настройки по умолчанию для уведомлений
        default_notifications = {
            'training_start': True,
            'training_complete': True,
            'early_stop': True,
            'error_alerts': True,
            'optimization_updates': True,
            'metrics_summary': False
        }
        
        # Объединяем с пользовательскими настройками
        for key, default_value in default_notifications.items():
            if key not in self.notifications:
                self.notifications[key] = default_value
        
        # Буфер сообщений для batch отправки
        self.message_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Проверка настроек
        if self.enabled and (not self.bot_token or not self.chat_id):
            self.logger.warning("Telegram настройки неполные, уведомления отключены")
            self.enabled = False
            
        if self.enabled:
            self.logger.info("Telegram уведомления активированы")
        else:
            self.logger.info("Telegram уведомления отключены")
            
    def _load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации из YAML файла"""
        if self.config_path is None:
            raise ValueError("config_path не установлен")
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Файл конфигурации {self.config_path} не найден")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Ошибка парсинга YAML: {e}")
            raise
            
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger('AlertManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def send_message(self, message: str, priority: str = 'info', 
                    parse_mode: str = 'Markdown') -> bool:
        """
        Отправка сообщения в Telegram
        
        Args:
            message: Текст сообщения
            priority: Приоритет сообщения (info, warning, error, critical)
            parse_mode: Режим парсинга (Markdown, HTML)
            
        Returns:
            True, если сообщение отправлено успешно
        """
        if not self.enabled:
            self.logger.debug(f"Уведомления отключены: {message}")
            return False
            
        # Добавление эмодзи в зависимости от приоритета
        emoji_map = {
            'info': '📊',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨',
            'success': '✅'
        }
        
        emoji = emoji_map.get(priority, '📋')
        formatted_message = f"{emoji} *Smart Tuner V2*\n\n{message}"
        
        # Добавление временной метки
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message += f"\n\n⏰ {timestamp}"
        
        return self._send_telegram_message(formatted_message, self.parse_mode)
        
    def _send_telegram_message(self, message: str, parse_mode: str = 'Markdown') -> bool:
        """
        Отправка сообщения через Telegram API
        
        Args:
            message: Текст сообщения
            parse_mode: Режим парсинга
            
        Returns:
            True, если сообщение отправлено успешно
        """
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': parse_mode,
            'disable_web_page_preview': True
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.debug("Сообщение отправлено в Telegram")
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ошибка отправки в Telegram: {e}")
            return False
            
    def send_training_started(self, config: Dict[str, Any]):
        """
        Уведомление о начале обучения
        
        Args:
            config: Конфигурация обучения
        """
        if not self.notifications.get('training_start', True):
            return
            
        message = "🚀 *Обучение началось*\n\n"
        message += f"📁 Эксперимент: `{config.get('experiment_name', 'Unknown')}`\n"
        message += f"🎯 Модель: `{config.get('model_name', 'Tacotron2')}`\n"
        message += f"📊 Датасет: `{config.get('dataset_path', 'Unknown')}`\n"
        
        # Основные гиперпараметры
        if 'hyperparameters' in config:
            params = config['hyperparameters']
            message += f"\n*Гиперпараметры:*\n"
            message += f"• Learning Rate: `{params.get('learning_rate', 'N/A')}`\n"
            message += f"• Batch Size: `{params.get('batch_size', 'N/A')}`\n"
            message += f"• Epochs: `{params.get('epochs', 'N/A')}`\n"
            
        self.send_message(message, priority='info')
        
    def send_training_completed(self, results: Dict[str, Any]):
        """
        Уведомление о завершении обучения
        
        Args:
            results: Результаты обучения
        """
        if not self.notifications.get('training_complete', True):
            return
            
        message = "🎉 *Обучение завершено*\n\n"
        
        # Финальные метрики
        if 'final_metrics' in results:
            metrics = results['final_metrics']
            message += "*Финальные метрики:*\n"
            for metric_name, value in metrics.items():
                message += f"• {metric_name}: `{value:.4f}`\n"
                
        # Время обучения
        if 'training_time' in results:
            message += f"\n⏱️ Время обучения: `{results['training_time']}`\n"
            
        # Лучшая эпоха
        if 'best_epoch' in results:
            message += f"🏆 Лучшая эпоха: `{results['best_epoch']}`\n"
            
        # Путь к модели
        if 'model_path' in results:
            message += f"💾 Модель сохранена: `{results['model_path']}`\n"
            
        self.send_message(message, priority='success')
        
    def send_training_stopped(self, reason: str, step: int, metrics: Dict[str, float]):
        """
        Уведомление о досрочном останове обучения
        
        Args:
            reason: Причина останова
            step: Шаг, на котором произошел останов
            metrics: Текущие метрики
        """
        message = "⏹️ *Обучение остановлено*\n\n"
        message += f"🛑 Причина: `{reason}`\n"
        message += f"📍 Шаг: `{step}`\n\n"
        
        message += "*Текущие метрики:*\n"
        for metric_name, value in metrics.items():
            message += f"• {metric_name}: `{value:.4f}`\n"
            
        self.send_message(message, priority='warning')
        
    def send_training_restart(self, restart_reason: str, restart_number: int, 
                            current_metrics: Dict[str, float], 
                            improvement_plan: Dict[str, Any]):
        """
        🔄 Уведомление о перезапуске обучения
        
        Args:
            restart_reason: Причина перезапуска
            restart_number: Номер перезапуска
            current_metrics: Текущие метрики
            improvement_plan: План улучшений
        """
        if not self.notifications.get('training_restart', True):
            return
            
        # Определяем критичность ситуации
        is_critical = 'NaN' in restart_reason or 'Inf' in restart_reason or restart_number >= 3
        
        if is_critical:
            message = "🚨 **КРИТИЧЕСКИЙ ПЕРЕЗАПУСК ОБУЧЕНИЯ!**\n\n"
            message += f"⚠️ **СЕРЬЕЗНАЯ ПРОБЛЕМА:** `{restart_reason}`\n"
        else:
            message = "🔄 **Перезапуск обучения**\n\n"
            message += f"📋 **Причина:** `{restart_reason}`\n"
        
        message += f"🔢 **Попытка:** `{restart_number}`\n"
        message += f"📍 **Время:** `{datetime.now().strftime('%H:%M:%S')}`\n\n"
        
        # Текущие метрики
        if current_metrics:
            message += "📊 **Проблемные метрики:**\n"
            for metric, value in current_metrics.items():
                if isinstance(value, float):
                    if 'loss' in metric.lower():
                        message += f"• {metric}: `{value:.6f}`\n"
                    elif 'accuracy' in metric.lower() or 'quality' in metric.lower():
                        message += f"• {metric}: `{value:.1%}`\n"
                    else:
                        message += f"• {metric}: `{value:.4f}`\n"
                else:
                    message += f"• {metric}: `{value}`\n"
        
        # План улучшений
        if improvement_plan:
            message += f"\n🛠️ **ПЛАН ВОССТАНОВЛЕНИЯ:**\n"
            if 'parameter_changes' in improvement_plan:
                for param, change in improvement_plan['parameter_changes'].items():
                    old_val = change.get('old_value', 'N/A')
                    new_val = change.get('new_value', 'N/A')
                    message += f"• {param}: `{old_val}` → `{new_val}`\n"
            
            if 'actions' in improvement_plan:
                message += f"\n🎯 **Действия:**\n"
                for action in improvement_plan['actions']:
                    message += f"• {action}\n"
        
        # Рекомендации
        message += f"\n💡 **АВТОМАТИЧЕСКИЕ УЛУЧШЕНИЯ:**\n"
        if is_critical:
            message += f"• 🔥 Радикальное снижение learning rate\n"
            message += f"• 🎯 Принудительная активация guided attention\n"
            message += f"• 📦 Уменьшение batch size\n"
            message += f"• ✂️ Строгое клипирование градиентов\n"
            message += f"• 🛡️ Отключение нестабильных функций\n"
        else:
            message += f"• ⚙️ Адаптация гиперпараметров\n"
            message += f"• 📊 Улучшение стабильности\n"
        
        # Прогноз
        message += f"\n🔮 **ОЖИДАЕМЫЙ РЕЗУЛЬТАТ:**\n"
        if restart_number == 1:
            message += f"• 📈 Стабилизация loss\n"
            message += f"• 🎯 Восстановление attention alignment\n"
        elif restart_number <= 3:
            message += f"• 🛡️ Максимальная стабильность\n"
            message += f"• 🎯 Постепенное улучшение качества\n"
        else:
            message += f"• ⚠️ Требуется ручная диагностика\n"
            message += f"• 🔧 Возможно, нужно пересмотреть данные\n"
        
        message += f"\n🚀 **Обучение перезапущено автоматически!**"
        
        priority = 'critical' if is_critical else 'warning'
        self.send_message(message, priority=priority)
        
    def send_quality_intervention(self, intervention_type: str, 
                                problem_detected: str,
                                action_taken: Dict[str, Any],
                                expected_outcome: str,
                                step: int):
        """
        🤖 Уведомление об умном вмешательстве системы
        
        Args:
            intervention_type: Тип вмешательства
            problem_detected: Обнаруженная проблема
            action_taken: Принятые меры
            expected_outcome: Ожидаемый результат
            step: Текущий шаг
        """
        if not self.notifications.get('quality_interventions', True):
            return
            
        # Иконки для разных типов вмешательств
        type_icons = {
            'learning_rate_adjustment': '⚡',
            'attention_correction': '🎯',
            'dropout_optimization': '🛡️',
            'batch_size_tuning': '📦',
            'guided_attention_boost': '🧭',
            'early_stop_prevention': '🚫',
            'phase_transition': '🔄',
            'quality_enhancement': '✨'
        }
        
        icon = type_icons.get(intervention_type, '🤖')
        
        message = f"{icon} *УМНОЕ ВМЕШАТЕЛЬСТВО СИСТЕМЫ*\n\n"
        message += f"📍 **Шаг:** `{step:,}`\n"
        message += f"🔍 **Проблема:** {problem_detected}\n"
        message += f"🎯 **Тип:** {intervention_type.replace('_', ' ').title()}\n\n"
        
        message += f"⚙️ **Принятые меры:**\n"
        for key, value in action_taken.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float) and abs(value) < 1:
                    message += f"• {key}: `{value:.6f}`\n"
                else:
                    message += f"• {key}: `{value}`\n"
            else:
                message += f"• {key}: `{value}`\n"
                
        message += f"\n🎯 **Ожидаемый результат:** {expected_outcome}\n\n"
        message += f"🧠 **Smart Tuner автоматически адаптирует обучение!**"
        
        self.send_message(message, priority='info')
        
    def send_info_notification(self, message: str):
        """Отправка информационного уведомления"""
        self.send_message(message, priority='info')
        
    def send_success_notification(self, message: str):
        """Отправка уведомления об успехе"""
        self.send_message(message, priority='success')
        
    def send_error_notification(self, message: str):
        """Отправка уведомления об ошибке"""
        if self.notifications.get('error_alerts', True):
            self.send_message(message, priority='error')
        
    def send_optimization_update(self, trial_number: int, trial_value: float, 
                               best_value: float, params: Dict[str, Any]):
        """
        Уведомление об обновлении оптимизации
        
        Args:
            trial_number: Номер trial
            trial_value: Значение целевой функции
            best_value: Лучшее значение
            params: Параметры trial
        """
        message = f"🔬 *Оптимизация - Trial {trial_number}*\n\n"
        message += f"📈 Значение: `{trial_value:.4f}`\n"
        message += f"🏆 Лучшее: `{best_value:.4f}`\n\n"
        
        message += "*Параметры:*\n"
        for param_name, value in params.items():
            if isinstance(value, float):
                message += f"• {param_name}: `{value:.6f}`\n"
            else:
                message += f"• {param_name}: `{value}`\n"
                
        priority = 'success' if trial_value <= best_value else 'info'
        self.send_message(message, priority=priority)
        
    def send_error_alert(self, error_message: str, traceback_info: str = None):
        """
        Уведомление об ошибке
        
        Args:
            error_message: Сообщение об ошибке
            traceback_info: Информация о трассировке
        """
        message = "💥 *Ошибка в обучении*\n\n"
        message += f"❌ Ошибка: `{error_message}`\n"
        
        if traceback_info:
            # Ограничиваем длину traceback
            if len(traceback_info) > 500:
                traceback_info = traceback_info[:500] + "..."
            message += f"\n```\n{traceback_info}\n```"
            
        self.send_message(message, priority='error')
        
    def send_metrics_summary(self, metrics_data: Dict[str, List[float]], 
                           window_size: int = 10):
        """
        Отправка сводки по метрикам
        
        Args:
            metrics_data: Данные метрик
            window_size: Размер окна для усреднения
        """
        message = "📊 *Сводка метрик*\n\n"
        
        for metric_name, values in metrics_data.items():
            if len(values) >= window_size:
                recent_avg = sum(values[-window_size:]) / window_size
                overall_avg = sum(values) / len(values)
                
                message += f"*{metric_name}:*\n"
                message += f"• Последние {window_size}: `{recent_avg:.4f}`\n"
                message += f"• Общее среднее: `{overall_avg:.4f}`\n"
                message += f"• Всего точек: `{len(values)}`\n\n"
                
        self.send_message(message, priority='info')
        
    def send_custom_alert(self, title: str, content: str, priority: str = 'info'):
        """
        Отправка пользовательского уведомления
        
        Args:
            title: Заголовок
            content: Содержание
            priority: Приоритет
        """
        message = f"*{title}*\n\n{content}"
        self.send_message(message, priority=priority)
        
    def send_batch_messages(self, messages: List[Dict[str, Any]]):
        """
        Отправка пакета сообщений
        
        Args:
            messages: Список сообщений с параметрами
        """
        for msg_data in messages:
            message = msg_data.get('message', '')
            priority = msg_data.get('priority', 'info')
            parse_mode = msg_data.get('parse_mode', 'Markdown')
            
            self.send_message(message, priority, parse_mode)
            
    def test_connection(self) -> bool:
        """
        Тестирование соединения с Telegram
        
        Returns:
            True, если соединение работает
        """
        if not self.enabled:
            self.logger.info("Уведомления отключены")
            return False
            
        test_message = "🧪 *Тест соединения*\n\nSmart Tuner V2 успешно подключен!"
        result = self.send_message(test_message, priority='info')
        
        if result:
            self.logger.info("Тест соединения с Telegram прошел успешно")
        else:
            self.logger.error("Тест соединения с Telegram провалился")
            
        return result
        
    def get_bot_info(self) -> Optional[Dict[str, Any]]:
        """
        Получение информации о боте
        
        Returns:
            Информация о боте или None
        """
        if not self.enabled:
            return None
            
        url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('ok'):
                return data.get('result')
            else:
                self.logger.error(f"Ошибка API Telegram: {data}")
                return None
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ошибка получения информации о боте: {e}")
            return None
            
    def set_webhook(self, webhook_url: str) -> bool:
        """
        Установка webhook для бота
        
        Args:
            webhook_url: URL для webhook
            
        Returns:
            True, если webhook установлен успешно
        """
        if not self.enabled:
            return False
            
        url = f"https://api.telegram.org/bot{self.bot_token}/setWebhook"
        payload = {'url': webhook_url}
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('ok'):
                self.logger.info(f"Webhook установлен: {webhook_url}")
                return True
            else:
                self.logger.error(f"Ошибка установки webhook: {data}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ошибка установки webhook: {e}")
            return False
        
    def send_critical_nan_alert(self, step: int, problematic_components: List[str], 
                              hyperparams: Dict[str, Any]):
        """
        🚨 Специальное уведомление о критической ошибке NaN/Inf
        
        Args:
            step: Шаг обучения где произошла ошибка
            problematic_components: Список проблемных компонентов loss
            hyperparams: Текущие гиперпараметры
        """
        if not self.notifications.get('critical_alerts', True):
            return
            
        message = "🚨 **КРИТИЧЕСКАЯ ОШИБКА: NaN/Inf В LOSS!**\n\n"
        message += f"💥 **Шаг:** `{step}`\n"
        message += f"🕐 **Время:** `{datetime.now().strftime('%H:%M:%S')}`\n\n"
        
        message += f"🔥 **Проблемные компоненты:**\n"
        for component in problematic_components:
            message += f"• {component}\n"
        
        message += f"\n⚙️ **Текущие параметры:**\n"
        key_params = ['learning_rate', 'batch_size', 'grad_clip_thresh']
        for param in key_params:
            if param in hyperparams:
                value = hyperparams[param]
                message += f"• {param}: `{value}`\n"
        
        message += f"\n🛡️ **СИСТЕМА САМОЗАЩИТЫ АКТИВИРОВАНА:**\n"
        message += f"• 🔄 Автоматический перезапуск\n"
        message += f"• 📉 Радикальное снижение learning rate\n"
        message += f"• 🎯 Принудительная guided attention\n"
        message += f"• ✂️ Агрессивное клипирование градиентов\n"
        
        message += f"\n⏰ **Перезапуск через несколько секунд...**"
        
        self.send_message(message, priority='critical') 