#!/usr/bin/env python3
"""
Система экспорта данных обучения для AI анализа

Автор: AI Assistant
Назначение: Экспорт истории обучения в удобном текстовом формате
"""

import os
import json
import mlflow
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional

class TrainingExportSystem:
    """
    Система экспорта данных обучения для AI анализа
    """
    
    def __init__(self, export_dir="training_exports"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
        
        # Создаем подпапки
        (self.export_dir / "text_reports").mkdir(exist_ok=True)
        (self.export_dir / "csv_data").mkdir(exist_ok=True)
        (self.export_dir / "json_raw").mkdir(exist_ok=True)
        
        print(f"📁 Training Export System инициализирован в {self.export_dir}")
    
    def export_current_training(self, run_id: str = None, format_type: str = "all"):
        """
        Экспорт текущего обучения в различных форматах
        
        Args:
            run_id: ID MLflow run (если None, берется последний)
            format_type: тип экспорта ("text", "csv", "json", "all")
        """
        if run_id is None:
            run_id = self._get_latest_run_id()
        
        if not run_id:
            print("❌ Не найден активный run для экспорта")
            return None
        
        print(f"📊 Начинаю экспорт run: {run_id}")
        
        # Получаем данные из MLflow
        training_data = self._extract_mlflow_data(run_id)
        
        if not training_data:
            print("❌ Не удалось получить данные обучения")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        exported_files = {}
        
        # Экспорт в текстовом формате (для AI)
        if format_type in ["text", "all"]:
            text_file = self._export_text_format(training_data, run_id, timestamp)
            exported_files["text"] = text_file
        
        # Экспорт в CSV формате
        if format_type in ["csv", "all"]:
            csv_file = self._export_csv_format(training_data, run_id, timestamp)
            exported_files["csv"] = csv_file
        
        # Экспорт в JSON формате
        if format_type in ["json", "all"]:
            json_file = self._export_json_format(training_data, run_id, timestamp)
            exported_files["json"] = json_file
        
        # Создаем итоговый отчет
        summary_file = self._create_export_summary(exported_files, training_data, timestamp)
        
        print("✅ Экспорт завершен!")
        print(f"📄 Файлы сохранены в: {self.export_dir}")
        
        return {
            "exported_files": exported_files,
            "summary": summary_file,
            "run_id": run_id
        }
    
    def _get_latest_run_id(self):
        """Получает ID последнего активного run"""
        try:
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            
            if not experiments:
                return None
            
            # Ищем последний run в активном эксперименте
            latest_run = None
            for experiment in experiments:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    max_results=1,
                    order_by=["start_time DESC"]
                )
                if runs and (latest_run is None or runs[0].info.start_time > latest_run.info.start_time):
                    latest_run = runs[0]
            
            return latest_run.info.run_id if latest_run else None
            
        except Exception as e:
            print(f"❌ Ошибка получения последнего run: {e}")
            return None
    
    def _extract_mlflow_data(self, run_id: str):
        """Извлекает данные из MLflow"""
        try:
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            
            # Базовая информация
            run_info = {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "duration_ms": run.info.end_time - run.info.start_time if run.info.end_time else None
            }
            
            # Параметры
            params = dict(run.data.params)
            
            # Метрики (текущие значения)
            metrics = dict(run.data.metrics)
            
            # История метрик
            metrics_history = {}
            for metric_name in metrics.keys():
                history = client.get_metric_history(run_id, metric_name)
                metrics_history[metric_name] = [
                    {
                        "timestamp": metric.timestamp,
                        "step": metric.step,
                        "value": metric.value
                    }
                    for metric in history
                ]
            
            # Теги
            tags = dict(run.data.tags)
            
            return {
                "info": run_info,
                "params": params,
                "metrics": metrics,
                "metrics_history": metrics_history,
                "tags": tags
            }
            
        except Exception as e:
            print(f"❌ Ошибка извлечения данных MLflow: {e}")
            return None
    
    def _export_text_format(self, training_data: Dict, run_id: str, timestamp: str):
        """Экспорт в текстовом формате для AI анализа"""
        try:
            filename = f"training_report_{run_id[:8]}_{timestamp}.txt"
            filepath = self.export_dir / "text_reports" / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("🤖 ОТЧЕТ ОБУЧЕНИЯ ДЛЯ AI АНАЛИЗА\n")
                f.write("=" * 80 + "\n\n")
                
                # Основная информация
                info = training_data["info"]
                f.write("📊 ОСНОВНАЯ ИНФОРМАЦИЯ\n")
                f.write("-" * 40 + "\n")
                f.write(f"Run ID: {info['run_id']}\n")
                f.write(f"Статус: {info['status']}\n")
                f.write(f"Время старта: {datetime.fromtimestamp(info['start_time']/1000)}\n")
                if info['end_time']:
                    f.write(f"Время окончания: {datetime.fromtimestamp(info['end_time']/1000)}\n")
                if info['duration_ms']:
                    hours = info['duration_ms'] / (1000 * 60 * 60)
                    f.write(f"Длительность: {hours:.2f} часов\n")
                f.write("\n")
                
                # Параметры обучения
                params = training_data["params"]
                f.write("⚙️ ПАРАМЕТРЫ ОБУЧЕНИЯ\n")
                f.write("-" * 40 + "\n")
                for key, value in params.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                # Финальные метрики
                metrics = training_data["metrics"]
                f.write("📈 ФИНАЛЬНЫЕ МЕТРИКИ\n")
                f.write("-" * 40 + "\n")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{key}: {value:.6f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
                
                # История ключевых метрик (сжато)
                f.write("📊 ИСТОРИЯ КЛЮЧЕВЫХ МЕТРИК (ПОСЛЕДНИЕ 20 ЗНАЧЕНИЙ)\n")
                f.write("-" * 40 + "\n")
                
                key_metrics = [
                    "training.loss", "validation.loss", "grad_norm", 
                    "learning_rate", "training.gate_loss", "training.taco_loss"
                ]
                
                for metric_name in key_metrics:
                    if metric_name in training_data["metrics_history"]:
                        history = training_data["metrics_history"][metric_name]
                        if history:
                            # Берем последние 20 значений
                            recent_history = history[-20:]
                            f.write(f"\n{metric_name} (последние {len(recent_history)} значений):\n")
                            
                            for i, entry in enumerate(recent_history):
                                step = entry['step']
                                value = entry['value']
                                if i < 5 or i >= len(recent_history) - 5:
                                    # Показываем первые 5 и последние 5
                                    f.write(f"  Шаг {step:6d}: {value:.6f}\n")
                                elif i == 5:
                                    f.write("  ...\n")
                f.write("\n")
                
                # Анализ трендов
                f.write("📈 АНАЛИЗ ТРЕНДОВ\n")
                f.write("-" * 40 + "\n")
                
                for metric_name in key_metrics:
                    if metric_name in training_data["metrics_history"]:
                        history = training_data["metrics_history"][metric_name]
                        if len(history) > 10:
                            values = [h["value"] for h in history]
                            
                            # Простой анализ тренда
                            start_avg = sum(values[:5]) / 5 if len(values) >= 5 else values[0]
                            end_avg = sum(values[-5:]) / 5 if len(values) >= 5 else values[-1]
                            
                            change = end_avg - start_avg
                            change_percent = (change / start_avg) * 100 if start_avg != 0 else 0
                            
                            trend = "улучшается" if change < 0 else "ухудшается" if change > 0 else "стабилен"
                            if "loss" not in metric_name.lower():
                                trend = "растет" if change > 0 else "падает" if change < 0 else "стабилен"
                            
                            f.write(f"{metric_name}: {trend} ({change_percent:+.2f}%)\n")
                f.write("\n")
                
                # Проблемы и рекомендации (упрощенные)
                f.write("🔍 КРАТКИЙ АНАЛИЗ ПРОБЛЕМ\n")
                f.write("-" * 40 + "\n")
                
                issues = []
                
                # Проверка длительности
                if info['duration_ms']:
                    hours = info['duration_ms'] / (1000 * 60 * 60)
                    if hours < 3:
                        issues.append(f"⏰ Короткое обучение: {hours:.1f} часов")
                
                # Проверка validation loss
                if 'validation.loss' in metrics:
                    val_loss = metrics['validation.loss']
                    if val_loss > 20:
                        issues.append(f"📈 Высокий validation loss: {val_loss:.3f}")
                
                # Проверка градиентов
                if 'grad_norm' in metrics:
                    grad_norm = metrics['grad_norm']
                    if grad_norm > 100:
                        issues.append(f"💥 Высокая норма градиентов: {grad_norm:.3f}")
                
                if issues:
                    for issue in issues:
                        f.write(f"• {issue}\n")
                else:
                    f.write("✅ Критических проблем не обнаружено\n")
                
                f.write("\n")
                f.write("=" * 80 + "\n")
                f.write(f"Отчет создан: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Для отправки AI Assistant скопируйте весь текст выше\n")
                f.write("=" * 80 + "\n")
            
            print(f"📄 Текстовый отчет: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Ошибка создания текстового отчета: {e}")
            return None
    
    def _export_csv_format(self, training_data: Dict, run_id: str, timestamp: str):
        """Экспорт в CSV формате для анализа в Excel/Python"""
        try:
            filename = f"training_metrics_{run_id[:8]}_{timestamp}.csv"
            filepath = self.export_dir / "csv_data" / filename
            
            # Создаем DataFrame из истории метрик
            all_metrics = []
            
            for metric_name, history in training_data["metrics_history"].items():
                for entry in history:
                    all_metrics.append({
                        "metric_name": metric_name,
                        "step": entry["step"],
                        "value": entry["value"],
                        "timestamp": entry["timestamp"]
                    })
            
            if all_metrics:
                df = pd.DataFrame(all_metrics)
                df.to_csv(filepath, index=False, encoding='utf-8')
                print(f"📊 CSV файл: {filepath}")
                return filepath
            else:
                print("⚠️ Нет данных для CSV экспорта")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка создания CSV: {e}")
            return None
    
    def _export_json_format(self, training_data: Dict, run_id: str, timestamp: str):
        """Экспорт в JSON формате для программного анализа"""
        try:
            filename = f"training_data_{run_id[:8]}_{timestamp}.json"
            filepath = self.export_dir / "json_raw" / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"📦 JSON файл: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Ошибка создания JSON: {e}")
            return None
    
    def _create_export_summary(self, exported_files: Dict, training_data: Dict, timestamp: str):
        """Создает итоговую сводку экспорта"""
        try:
            filename = f"export_summary_{timestamp}.md"
            filepath = self.export_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# 📊 Сводка экспорта обучения\n\n")
                f.write(f"**Дата экспорта:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Информация об обучении
                info = training_data["info"]
                f.write("## 🎯 Информация об обучении\n\n")
                f.write(f"- **Run ID:** `{info['run_id']}`\n")
                f.write(f"- **Статус:** {info['status']}\n")
                
                if info['duration_ms']:
                    hours = info['duration_ms'] / (1000 * 60 * 60)
                    f.write(f"- **Длительность:** {hours:.2f} часов\n")
                
                # Экспортированные файлы
                f.write("\n## 📁 Экспортированные файлы\n\n")
                for format_type, filepath in exported_files.items():
                    if filepath:
                        f.write(f"- **{format_type.upper()}:** `{filepath.name}`\n")
                
                # Быстрые метрики
                f.write("\n## 📈 Быстрые метрики\n\n")
                metrics = training_data["metrics"]
                quick_metrics = [
                    "training.loss", "validation.loss", "grad_norm", 
                    "learning_rate", "training.gate_loss"
                ]
                
                for metric in quick_metrics:
                    if metric in metrics:
                        value = metrics[metric]
                        if isinstance(value, (int, float)):
                            f.write(f"- **{metric}:** {value:.6f}\n")
                
                f.write("\n## 🚀 Как использовать\n\n")
                f.write("1. **Для отправки AI Assistant:** используйте файл из папки `text_reports/`\n")
                f.write("2. **Для анализа в Excel:** используйте файл из папки `csv_data/`\n") 
                f.write("3. **Для программного анализа:** используйте файл из папки `json_raw/`\n")
            
            print(f"📋 Сводка экспорта: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Ошибка создания сводки: {e}")
            return None

def export_training_for_ai(run_id: str = None):
    """
    Быстрая функция для экспорта обучения для AI анализа
    
    Args:
        run_id: ID MLflow run (если None, берется последний)
    
    Returns:
        Путь к текстовому файлу для отправки AI
    """
    exporter = TrainingExportSystem()
    result = exporter.export_current_training(run_id, format_type="text")
    
    if result and "text" in result["exported_files"]:
        text_file = result["exported_files"]["text"]
        print(f"\n📤 ГОТОВО ДЛЯ ОТПРАВКИ AI:")
        print(f"   Файл: {text_file}")
        print(f"   📋 Скопируйте содержимое файла и отправьте AI Assistant")
        return text_file
    else:
        print("❌ Не удалось создать экспорт для AI")
        return None

if __name__ == "__main__":
    print("🚀 Запуск системы экспорта обучения")
    
    # Быстрый экспорт для AI
    export_training_for_ai() 