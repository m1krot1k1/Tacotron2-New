#!/usr/bin/env python3
"""
Умная система ведения логов обучения TTS

Автор: AI Assistant
Назначение: Создание markdown логов с историей обучения и изменений параметров
"""

import os
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict

@dataclass
class TrainingEvent:
    """Событие в процессе обучения"""
    timestamp: str
    event_type: str  # "start", "metric", "param_change", "warning", "stop"
    data: Dict[str, Any]
    description: str
    
class SmartTrainingLogger:
    """
    Умная система логирования процесса обучения TTS
    """
    
    def __init__(self, logs_dir="smart_logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Создаем подпапки
        (self.logs_dir / "training_sessions").mkdir(exist_ok=True)
        (self.logs_dir / "plots").mkdir(exist_ok=True)
        (self.logs_dir / "param_changes").mkdir(exist_ok=True)
        
        self.current_session = None
        self.session_file = None
        self.events_log = []
        
        print(f"📝 Smart Training Logger инициализирован в {self.logs_dir}")
    
    def start_training_session(self, run_id: str, training_params: Dict):
        """
        Начать новую сессию обучения
        
        Args:
            run_id: MLflow run ID
            training_params: параметры обучения
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"training_session_{run_id[:8]}_{timestamp}"
        
        self.current_session = {
            "session_id": session_name,
            "run_id": run_id,
            "start_time": datetime.now().isoformat(),
            "training_params": training_params,
            "events": [],
            "metrics_history": {},
            "param_changes": []
        }
        
        # Создаем файл сессии
        self.session_file = self.logs_dir / "training_sessions" / f"{session_name}.md"
        
        # Записываем заголовок
        self._write_session_header()
        
        # Логируем начало
        self.log_event(
            event_type="start",
            data=training_params,
            description=f"🚀 Начало обучения TTS модели"
        )
        
        print(f"📊 Начата сессия логирования: {session_name}")
        return session_name
    
    def log_event(self, event_type: str, data: Dict, description: str):
        """
        Добавить событие в лог
        
        Args:
            event_type: тип события
            data: данные события
            description: описание события
        """
        if not self.current_session:
            print("⚠️ Сессия не начата. Используйте start_training_session()")
            return
        
        event = TrainingEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            data=data,
            description=description
        )
        
        self.current_session["events"].append(asdict(event))
        self.events_log.append(event)
        
        # Обновляем файл лога
        self._update_session_log(event)
    
    def log_metrics(self, step: int, metrics: Dict[str, float]):
        """
        Логирование метрик обучения
        
        Args:
            step: шаг обучения
            metrics: словарь с метриками
        """
        # Обновляем историю метрик
        for metric_name, value in metrics.items():
            if metric_name not in self.current_session["metrics_history"]:
                self.current_session["metrics_history"][metric_name] = []
            
            self.current_session["metrics_history"][metric_name].append({
                "step": step,
                "value": value,
                "timestamp": datetime.now().isoformat()
            })
        
        # Логируем как событие (каждые 100 шагов)
        if step % 100 == 0:
            key_metrics = {
                k: v for k, v in metrics.items() 
                if any(x in k.lower() for x in ["loss", "grad", "learning"])
            }
            
            self.log_event(
                event_type="metric",
                data={"step": step, "metrics": key_metrics},
                description=f"📊 Метрики на шаге {step}"
            )
    
    def log_parameter_change(self, param_name: str, old_value: Any, new_value: Any, reason: str):
        """
        Логирование изменения параметра умной системой
        
        Args:
            param_name: имя параметра
            old_value: старое значение
            new_value: новое значение  
            reason: причина изменения
        """
        change_data = {
            "param_name": param_name,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        self.current_session["param_changes"].append(change_data)
        
        self.log_event(
            event_type="param_change",
            data=change_data,
            description=f"⚙️ Изменен параметр {param_name}: {old_value} → {new_value}"
        )
        
        # Сохраняем детальную информацию об изменении
        self._save_param_change_details(change_data)
    
    def log_warning(self, warning_type: str, message: str, data: Dict = None):
        """
        Логирование предупреждения
        
        Args:
            warning_type: тип предупреждения
            message: сообщение
            data: дополнительные данные
        """
        self.log_event(
            event_type="warning",
            data={"warning_type": warning_type, "data": data or {}},
            description=f"⚠️ {warning_type}: {message}"
        )
    
    def end_training_session(self, final_metrics: Dict = None, status: str = "completed"):
        """
        Завершить сессию обучения
        
        Args:
            final_metrics: финальные метрики
            status: статус завершения
        """
        if not self.current_session:
            return
        
        self.current_session["end_time"] = datetime.now().isoformat()
        self.current_session["status"] = status
        self.current_session["final_metrics"] = final_metrics or {}
        
        self.log_event(
            event_type="stop",
            data={"status": status, "final_metrics": final_metrics or {}},
            description=f"🏁 Завершение обучения: {status}"
        )
        
        # Создаем итоговый отчет
        self._create_final_report()
        
        # Генерируем графики
        self._generate_plots()
        
        print(f"✅ Сессия логирования завершена: {self.current_session['session_id']}")
    
    def _write_session_header(self):
        """Записывает заголовок сессии"""
        with open(self.session_file, 'w', encoding='utf-8') as f:
            f.write(f"# 🎯 Сессия обучения TTS\n\n")
            f.write(f"**Session ID:** `{self.current_session['session_id']}`\n")
            f.write(f"**MLflow Run ID:** `{self.current_session['run_id']}`\n")
            f.write(f"**Время начала:** {self.current_session['start_time']}\n")
            f.write(f"**Статус:** 🔄 В процессе...\n\n")
            
            # Параметры обучения
            f.write("## ⚙️ Параметры обучения\n\n")
            f.write("```yaml\n")
            f.write(yaml.dump(self.current_session['training_params'], default_flow_style=False, allow_unicode=True))
            f.write("```\n\n")
            
            # Начинаем секцию событий
            f.write("## 📊 История обучения\n\n")
    
    def _update_session_log(self, event: TrainingEvent):
        """Обновляет лог сессии новым событием"""
        with open(self.session_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.fromisoformat(event.timestamp).strftime("%H:%M:%S")
            
            if event.event_type == "start":
                f.write(f"### 🚀 {timestamp} - Начало обучения\n\n")
                
            elif event.event_type == "metric":
                step = event.data.get("step", 0)
                metrics = event.data.get("metrics", {})
                f.write(f"### 📊 {timestamp} - Шаг {step}\n\n")
                
                # Форматируем ключевые метрики в таблицу
                if metrics:
                    f.write("| Метрика | Значение |\n")
                    f.write("|---------|----------|\n")
                    for name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"| `{name}` | {value:.6f} |\n")
                        else:
                            f.write(f"| `{name}` | {value} |\n")
                    f.write("\n")
                
            elif event.event_type == "param_change":
                param_name = event.data.get("param_name", "")
                old_value = event.data.get("old_value", "")
                new_value = event.data.get("new_value", "")
                reason = event.data.get("reason", "")
                
                f.write(f"### ⚙️ {timestamp} - Изменение параметра\n\n")
                f.write(f"**Параметр:** `{param_name}`\n")
                f.write(f"**Было:** `{old_value}`\n")
                f.write(f"**Стало:** `{new_value}`\n")
                f.write(f"**Причина:** {reason}\n\n")
                
            elif event.event_type == "warning":
                warning_type = event.data.get("warning_type", "")
                f.write(f"### ⚠️ {timestamp} - Предупреждение: {warning_type}\n\n")
                f.write(f"{event.description}\n\n")
                
            elif event.event_type == "stop":
                status = event.data.get("status", "unknown")
                f.write(f"### 🏁 {timestamp} - Завершение обучения\n\n")
                f.write(f"**Статус:** {status}\n\n")
    
    def _save_param_change_details(self, change_data: Dict):
        """Сохраняем детальную информацию об изменении параметра"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"param_change_{change_data['param_name']}_{timestamp}.json"
        filepath = self.logs_dir / "param_changes" / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(change_data, f, indent=2, ensure_ascii=False, default=str)
    
    def _create_final_report(self):
        """Создает итоговый отчет сессии"""
        with open(self.session_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("## 📋 Итоговый отчет\n\n")
            
            # Основная статистика
            start_time = datetime.fromisoformat(self.current_session['start_time'])
            end_time = datetime.fromisoformat(self.current_session['end_time'])
            duration = end_time - start_time
            
            f.write(f"**Длительность обучения:** {duration}\n")
            f.write(f"**Статус завершения:** {self.current_session['status']}\n")
            f.write(f"**Всего событий:** {len(self.current_session['events'])}\n")
            f.write(f"**Изменений параметров:** {len(self.current_session['param_changes'])}\n\n")
            
            # Финальные метрики
            if self.current_session.get('final_metrics'):
                f.write("### 📊 Финальные метрики\n\n")
                f.write("| Метрика | Значение |\n")
                f.write("|---------|----------|\n")
                for name, value in self.current_session['final_metrics'].items():
                    if isinstance(value, (int, float)):
                        f.write(f"| `{name}` | {value:.6f} |\n")
                    else:
                        f.write(f"| `{name}` | {value} |\n")
                f.write("\n")
            
            # Сводка изменений параметров
            if self.current_session['param_changes']:
                f.write("### ⚙️ Сводка изменений параметров\n\n")
                f.write("| Время | Параметр | Было | Стало | Причина |\n")
                f.write("|-------|----------|------|-------|----------|\n")
                
                for change in self.current_session['param_changes']:
                    time_str = datetime.fromisoformat(change['timestamp']).strftime("%H:%M:%S")
                    f.write(f"| {time_str} | `{change['param_name']}` | `{change['old_value']}` | `{change['new_value']}` | {change['reason']} |\n")
                f.write("\n")
            
            # Рекомендации
            f.write("### 💡 Рекомендации для следующего обучения\n\n")
            recommendations = self._generate_recommendations()
            for rec in recommendations:
                f.write(f"- {rec}\n")
            f.write("\n")
            
            f.write(f"---\n")
            f.write(f"*Отчет создан: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    def _generate_recommendations(self):
        """Генерирует рекомендации на основе истории обучения"""
        recommendations = []
        
        # Анализ длительности
        if self.current_session.get('end_time') and self.current_session.get('start_time'):
            start_time = datetime.fromisoformat(self.current_session['start_time'])
            end_time = datetime.fromisoformat(self.current_session['end_time'])
            duration_hours = (end_time - start_time).total_seconds() / 3600
            
            if duration_hours < 3:
                recommendations.append("⏰ Обучение было коротким. Рассмотрите увеличение patience для early stopping")
            elif duration_hours > 24:
                recommendations.append("⏰ Обучение было очень долгим. Возможно, стоит пересмотреть learning rate schedule")
        
        # Анализ изменений параметров
        param_changes_count = len(self.current_session['param_changes'])
        if param_changes_count > 10:
            recommendations.append("⚙️ Много автоматических изменений параметров. Рассмотрите более консервативные настройки")
        elif param_changes_count == 0:
            recommendations.append("⚙️ Параметры не изменялись. Возможно, стоит активировать адаптивные механизмы")
        
        # Анализ предупреждений
        warnings = [e for e in self.current_session['events'] if e['event_type'] == 'warning']
        if len(warnings) > 5:
            recommendations.append("⚠️ Много предупреждений. Проверьте стабильность обучения")
        
        if not recommendations:
            recommendations.append("✅ Обучение прошло стабильно. Параметры можно оставить без изменений")
        
        return recommendations
    
    def _generate_plots(self):
        """Генерирует графики метрик"""
        if not self.current_session.get('metrics_history'):
            return
        
        try:
            # Настройка стиля
            plt.style.use('default')
            sns.set_palette("husl")
            
            # График потерь
            self._plot_losses()
            
            # График градиентов
            self._plot_gradients()
            
            # График learning rate
            self._plot_learning_rate()
            
            print(f"📊 Графики сохранены в {self.logs_dir / 'plots'}")
            
        except Exception as e:
            print(f"❌ Ошибка создания графиков: {e}")
    
    def _plot_losses(self):
        """График потерь"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('📊 История потерь обучения', fontsize=16)
        
        loss_metrics = {
            k: v for k, v in self.current_session['metrics_history'].items()
            if 'loss' in k.lower()
        }
        
        if not loss_metrics:
            return
        
        # Основные потери
        ax = axes[0, 0]
        for name, history in loss_metrics.items():
            if 'training' in name or 'validation' in name:
                steps = [h['step'] for h in history]
                values = [h['value'] for h in history]
                ax.plot(steps, values, label=name, linewidth=2)
        
        ax.set_title('Основные потери')
        ax.set_xlabel('Шаг')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Tacotron потери
        ax = axes[0, 1]
        for name, history in loss_metrics.items():
            if 'taco' in name.lower() or 'gate' in name.lower():
                steps = [h['step'] for h in history]
                values = [h['value'] for h in history]
                ax.plot(steps, values, label=name, linewidth=2)
        
        ax.set_title('Tacotron потери')
        ax.set_xlabel('Шаг')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Логарифмический масштаб
        ax = axes[1, 0]
        for name, history in loss_metrics.items():
            if 'training' in name or 'validation' in name:
                steps = [h['step'] for h in history]
                values = [h['value'] for h in history]
                ax.semilogy(steps, values, label=name, linewidth=2)
        
        ax.set_title('Основные потери (лог. масштаб)')
        ax.set_xlabel('Шаг')
        ax.set_ylabel('Log Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Сглаженные потери
        ax = axes[1, 1]
        try:
            from scipy.ndimage import uniform_filter1d
            for name, history in loss_metrics.items():
                if 'training' in name or 'validation' in name:
                    steps = [h['step'] for h in history]
                    values = [h['value'] for h in history]
                    if len(values) > 10:
                        smoothed = uniform_filter1d(values, size=min(50, len(values)//10))
                        ax.plot(steps, smoothed, label=f'{name} (сглаженный)', linewidth=2)
        except ImportError:
            ax.text(0.5, 0.5, 'Требуется scipy для сглаживания', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Сглаженные потери')
        ax.set_xlabel('Шаг')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохраняем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"losses_{self.current_session['session_id']}_{timestamp}.png"
        filepath = self.logs_dir / "plots" / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Добавляем ссылку в markdown
        with open(self.session_file, 'a', encoding='utf-8') as f:
            f.write(f"\n### 📊 График потерь\n\n")
            f.write(f"![Потери](plots/{filename})\n\n")
    
    def _plot_gradients(self):
        """График градиентов"""
        grad_metrics = {
            k: v for k, v in self.current_session['metrics_history'].items()
            if 'grad' in k.lower()
        }
        
        if not grad_metrics:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('📊 История градиентов', fontsize=16)
        
        # Норма градиентов
        ax = axes[0]
        for name, history in grad_metrics.items():
            steps = [h['step'] for h in history]
            values = [h['value'] for h in history]
            ax.plot(steps, values, label=name, linewidth=2)
        
        ax.set_title('Норма градиентов')
        ax.set_xlabel('Шаг')
        ax.set_ylabel('Gradient Norm')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Логарифмический масштаб
        ax = axes[1]
        for name, history in grad_metrics.items():
            steps = [h['step'] for h in history]
            values = [h['value'] for h in history]
            ax.semilogy(steps, values, label=name, linewidth=2)
        
        ax.set_title('Норма градиентов (лог. масштаб)')
        ax.set_xlabel('Шаг')
        ax.set_ylabel('Log Gradient Norm')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохраняем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gradients_{self.current_session['session_id']}_{timestamp}.png"
        filepath = self.logs_dir / "plots" / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Добавляем ссылку в markdown
        with open(self.session_file, 'a', encoding='utf-8') as f:
            f.write(f"\n### 📊 График градиентов\n\n")
            f.write(f"![Градиенты](plots/{filename})\n\n")
    
    def _plot_learning_rate(self):
        """График learning rate"""
        lr_metrics = {
            k: v for k, v in self.current_session['metrics_history'].items()
            if 'learning' in k.lower() and 'rate' in k.lower()
        }
        
        if not lr_metrics:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('📊 История Learning Rate', fontsize=16)
        
        for name, history in lr_metrics.items():
            steps = [h['step'] for h in history]
            values = [h['value'] for h in history]
            ax.plot(steps, values, label=name, linewidth=2)
        
        ax.set_title('Learning Rate')
        ax.set_xlabel('Шаг')
        ax.set_ylabel('Learning Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Сохраняем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"learning_rate_{self.current_session['session_id']}_{timestamp}.png"
        filepath = self.logs_dir / "plots" / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Добавляем ссылку в markdown
        with open(self.session_file, 'a', encoding='utf-8') as f:
            f.write(f"\n### 📊 График Learning Rate\n\n")
            f.write(f"![Learning Rate](plots/{filename})\n\n")

# Глобальный экземпляр логгера
_global_logger = None

def get_training_logger():
    """Получить глобальный экземпляр логгера"""
    global _global_logger
    if _global_logger is None:
        _global_logger = SmartTrainingLogger()
    return _global_logger

def log_training_start(run_id: str, params: Dict):
    """Быстрая функция для начала логирования"""
    logger = get_training_logger()
    return logger.start_training_session(run_id, params)

def log_training_metrics(step: int, metrics: Dict):
    """Быстрая функция для логирования метрик"""
    logger = get_training_logger()
    logger.log_metrics(step, metrics)

def log_param_change(param_name: str, old_value: Any, new_value: Any, reason: str):
    """Быстрая функция для логирования изменения параметра"""
    logger = get_training_logger()
    logger.log_parameter_change(param_name, old_value, new_value, reason)

def log_training_warning(warning_type: str, message: str, data: Dict = None):
    """Быстрая функция для логирования предупреждения"""
    logger = get_training_logger()
    logger.log_warning(warning_type, message, data)

def log_training_end(final_metrics: Dict = None, status: str = "completed"):
    """Быстрая функция для завершения логирования"""
    logger = get_training_logger()
    logger.end_training_session(final_metrics, status)

if __name__ == "__main__":
    print("🚀 Тестирование Smart Training Logger")
    
    # Пример использования
    logger = SmartTrainingLogger()
    
    # Начинаем сессию
    session_id = logger.start_training_session(
        run_id="test_run_12345",
        training_params={
            "learning_rate": 0.001,
            "batch_size": 32,
            "model": "Tacotron2"
        }
    )
    
    # Примеры логирования
    logger.log_metrics(100, {
        "training.loss": 2.5,
        "validation.loss": 2.8,
        "grad_norm": 5.2,
        "learning_rate": 0.001
    })
    
    logger.log_parameter_change(
        param_name="learning_rate",
        old_value=0.001,
        new_value=0.0008,
        reason="Градиенты стали нестабильными"
    )
    
    logger.log_warning(
        warning_type="GradientWarning",
        message="Высокая норма градиентов",
        data={"grad_norm": 50.0}
    )
    
    # Завершаем сессию
    logger.end_training_session(
        final_metrics={
            "final_loss": 1.2,
            "validation_loss": 1.5
        },
        status="completed"
    )
    
    print("✅ Тест завершен") 