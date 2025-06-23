#!/usr/bin/env python3
"""
Скрипт для экспорта и анализа данных из MLflow

Автор: Smart Tuner System
Назначение: Экспорт данных обучения для анализа причин остановки
"""

import mlflow
import pandas as pd
import json
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

class MLFlowDataExporter:
    """
    Экспортер данных из MLflow для анализа причин остановки обучения
    """
    
    def __init__(self, tracking_uri="mlruns", experiment_name="tacotron2_production"):
        """
        Инициализация экспортера
        
        Args:
            tracking_uri: URI для MLflow
            experiment_name: Имя эксперимента
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        
        # Создаем директорию для экспорта
        self.export_dir = Path("mlflow_export")
        self.export_dir.mkdir(exist_ok=True)
        
        print(f"🔍 MLFlow Data Exporter инициализирован")
        print(f"📁 URI: {tracking_uri}")
        print(f"🎯 Эксперимент: {experiment_name}")
        print(f"💾 Директория экспорта: {self.export_dir}")
    
    def get_experiment_runs(self):
        """Получение всех runs из эксперимента"""
        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                print(f"❌ Эксперимент {self.experiment_name} не найден")
                return []
            
            runs = self.client.search_runs(experiment_ids=[experiment.experiment_id])
            print(f"📊 Найдено {len(runs)} runs в эксперименте")
            return runs
        except Exception as e:
            print(f"❌ Ошибка получения runs: {e}")
            return []
    
    def export_run_data(self, run_id):
        """
        Экспорт данных конкретного run
        
        Args:
            run_id: ID run для экспорта
            
        Returns:
            dict: Экспортированные данные
        """
        try:
            run = self.client.get_run(run_id)
            
            # Базовая информация о run
            run_info = {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "duration_ms": run.info.end_time - run.info.start_time if run.info.end_time else None,
                "lifecycle_stage": run.info.lifecycle_stage,
                "artifact_uri": run.info.artifact_uri
            }
            
            # Параметры
            params = dict(run.data.params)
            
            # Метрики (текущие значения)
            metrics = dict(run.data.metrics)
            
            # История метрик
            metrics_history = {}
            for metric_name in metrics.keys():
                history = self.client.get_metric_history(run_id, metric_name)
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
            print(f"❌ Ошибка экспорта run {run_id}: {e}")
            return None
    
    def analyze_training_failure(self, run_data):
        """
        Анализ причин остановки обучения
        
        Args:
            run_data: Данные run из export_run_data
            
        Returns:
            dict: Результаты анализа
        """
        if not run_data:
            return {"error": "Нет данных для анализа"}
        
        analysis = {
            "summary": {},
            "potential_issues": [],
            "recommendations": [],
            "metrics_analysis": {}
        }
        
        info = run_data["info"]
        metrics_history = run_data["metrics_history"]
        
        # Анализ длительности
        if info["duration_ms"]:
            duration_hours = info["duration_ms"] / (1000 * 60 * 60)
            analysis["summary"]["duration_hours"] = round(duration_hours, 2)
            
            if duration_hours < 3:
                analysis["potential_issues"].append(
                    f"⏰ Слишком короткое обучение: {duration_hours:.1f} часов"
                )
        
        # Анализ статуса
        if info["status"] == "FINISHED":
            analysis["potential_issues"].append(
                "🔴 Обучение завершилось преждевременно (статус: FINISHED)"
            )
        elif info["status"] == "FAILED":
            analysis["potential_issues"].append(
                "💥 Обучение завершилось с ошибкой (статус: FAILED)"
            )
        
        # Анализ метрик
        for metric_name, history in metrics_history.items():
            if not history:
                continue
                
            values = [h["value"] for h in history]
            steps = [h["step"] for h in history]
            
            metric_analysis = {
                "total_steps": len(steps),
                "final_value": values[-1] if values else None,
                "min_value": min(values) if values else None,
                "max_value": max(values) if values else None,
                "trend": "неопределен"
            }
            
            # Анализ тренда (последние 20% значений)
            if len(values) > 10:
                recent_count = max(5, len(values) // 5)
                recent_values = values[-recent_count:]
                early_values = values[:recent_count]
                
                if len(recent_values) > 1 and len(early_values) > 1:
                    recent_avg = np.mean(recent_values)
                    early_avg = np.mean(early_values)
                    
                    if "loss" in metric_name.lower():
                        if recent_avg < early_avg * 0.95:
                            metric_analysis["trend"] = "улучшается"
                        elif recent_avg > early_avg * 1.05:
                            metric_analysis["trend"] = "ухудшается"
                        else:
                            metric_analysis["trend"] = "стабилен"
                    else:
                        if recent_avg > early_avg * 1.05:
                            metric_analysis["trend"] = "растет"
                        elif recent_avg < early_avg * 0.95:
                            metric_analysis["trend"] = "падает"
                        else:
                            metric_analysis["trend"] = "стабилен"
            
            # Проверка на зависание (одинаковые значения подряд)
            if len(values) > 20:
                last_values = values[-20:]
                if len(set(np.round(last_values, 6))) == 1:
                    analysis["potential_issues"].append(
                        f"🔒 Метрика {metric_name} зависла на значении {last_values[0]:.6f}"
                    )
            
            # Проверка на взрывной градиент
            if "grad_norm" in metric_name:
                if values and max(values) > 100:
                    analysis["potential_issues"].append(
                        f"💥 Взрывной градиент! Максимальная норма: {max(values):.2f}"
                    )
            
            analysis["metrics_analysis"][metric_name] = metric_analysis
        
        # Генерация рекомендаций
        self._generate_recommendations(analysis, run_data)
        
        return analysis
    
    def _generate_recommendations(self, analysis, run_data):
        """Генерация рекомендаций по улучшению обучения"""
        recommendations = []
        
        # Рекомендации по длительности
        if analysis["summary"].get("duration_hours", 0) < 5:
            recommendations.append(
                "⏳ Увеличить максимальное время обучения (сейчас слишком короткое)"
            )
        
        # Рекомендации по градиентам
        grad_issues = [issue for issue in analysis["potential_issues"] if "градиент" in issue.lower()]
        if grad_issues:
            recommendations.append(
                "📉 Уменьшить learning rate или добавить градиентный клиппинг"
            )
        
        # Рекомендации по зависанию метрик
        stuck_metrics = [issue for issue in analysis["potential_issues"] if "зависла" in issue]
        if stuck_metrics:
            recommendations.append(
                "🔄 Добавить более агрессивные условия early stopping"
            )
            recommendations.append(
                "📊 Проверить разнообразие данных и перемешивание"
            )
        
        # Проверка validation loss
        val_loss_history = run_data["metrics_history"].get("validation.loss", [])
        if val_loss_history:
            val_losses = [h["value"] for h in val_loss_history]
            if len(val_losses) > 5:
                recent_losses = val_losses[-5:]
                if all(loss > 10 for loss in recent_losses):  # Высокие значения validation loss
                    recommendations.append(
                        "📈 Validation loss слишком высокий - проверить переобучение или качество данных"
                    )
        
        analysis["recommendations"] = recommendations
    
    def create_visualization(self, run_data, save_path):
        """
        Создание визуализации метрик
        
        Args:
            run_data: Данные run
            save_path: Путь для сохранения графиков
        """
        if not run_data or not run_data["metrics_history"]:
            print("❌ Нет данных для визуализации")
            return
        
        # Настройка стиля
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Создаем фигуру с подграфиками
        metrics_count = len(run_data["metrics_history"])
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Анализ метрик обучения\nRun ID: {run_data["info"]["run_id"]}', fontsize=16)
        
        axes = axes.flatten()
        
        # Ключевые метрики для отображения
        key_metrics = [
            "training.loss",
            "validation.loss", 
            "grad_norm",
            "learning_rate"
        ]
        
        for i, metric_name in enumerate(key_metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if metric_name in run_data["metrics_history"]:
                history = run_data["metrics_history"][metric_name]
                steps = [h["step"] for h in history]
                values = [h["value"] for h in history]
                
                ax.plot(steps, values, linewidth=2, marker='o', markersize=3, alpha=0.7)
                ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Step')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                
                # Добавляем аннотации для экстремальных значений
                if values:
                    min_idx = values.index(min(values))
                    max_idx = values.index(max(values))
                    
                    ax.annotate(f'Min: {min(values):.4f}', 
                               xy=(steps[min_idx], values[min_idx]),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                               fontsize=8)
                    ax.annotate(f'Max: {max(values):.4f}', 
                               xy=(steps[max_idx], values[max_idx]),
                               xytext=(10, -15), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                               fontsize=8)
            else:
                ax.text(0.5, 0.5, f'Нет данных для\n{metric_name}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, alpha=0.7)
                ax.set_title(f'{metric_name} (отсутствует)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Визуализация сохранена: {save_path}")
    
    def export_specific_run(self, run_id):
        """
        Экспорт конкретного run по ID
        
        Args:
            run_id: ID run для экспорта
        """
        print(f"\n🔍 Экспорт данных для Run ID: {run_id}")
        
        # Экспортируем данные
        run_data = self.export_run_data(run_id)
        if not run_data:
            print("❌ Не удалось экспортировать данные")
            return None
        
        # Анализируем причины остановки
        analysis = self.analyze_training_failure(run_data)
        
        # Создаем отчет
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.export_dir / f"training_analysis_{run_id}_{timestamp}.json"
        
        # Объединяем все данные
        full_report = {
            "export_info": {
                "timestamp": timestamp,
                "run_id": run_id,
                "exporter_version": "1.0"
            },
            "run_data": run_data,
            "analysis": analysis
        }
        
        # Сохраняем JSON отчет
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Отчет сохранен: {report_path}")
        
        # Создаем визуализацию
        viz_path = self.export_dir / f"metrics_visualization_{run_id}_{timestamp}.png"
        self.create_visualization(run_data, viz_path)
        
        # Создаем текстовый отчет для удобного чтения
        text_report_path = self.export_dir / f"analysis_report_{run_id}_{timestamp}.txt"
        self.create_text_report(full_report, text_report_path)
        
        return full_report
    
    def create_text_report(self, full_report, save_path):
        """Создание текстового отчета для удобного чтения"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("🤖 АНАЛИЗ ОСТАНОВКИ ОБУЧЕНИЯ TACOTRON2\n")
            f.write("=" * 80 + "\n\n")
            
            # Основная информация
            info = full_report["run_data"]["info"]
            f.write(f"📊 ОСНОВНАЯ ИНФОРМАЦИЯ\n")
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
            params = full_report["run_data"]["params"]
            f.write(f"⚙️ ПАРАМЕТРЫ ОБУЧЕНИЯ\n")
            f.write("-" * 40 + "\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Финальные метрики
            metrics = full_report["run_data"]["metrics"]
            f.write(f"📈 ФИНАЛЬНЫЕ МЕТРИКИ\n")
            f.write("-" * 40 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.6f}\n")
            f.write("\n")
            
            # Анализ проблем
            analysis = full_report["analysis"]
            f.write(f"🔍 ОБНАРУЖЕННЫЕ ПРОБЛЕМЫ\n")
            f.write("-" * 40 + "\n")
            if analysis["potential_issues"]:
                for issue in analysis["potential_issues"]:
                    f.write(f"• {issue}\n")
            else:
                f.write("Критических проблем не обнаружено\n")
            f.write("\n")
            
            # Рекомендации
            f.write(f"💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ\n")
            f.write("-" * 40 + "\n")
            if analysis["recommendations"]:
                for rec in analysis["recommendations"]:
                    f.write(f"• {rec}\n")
            else:
                f.write("Специальных рекомендаций нет\n")
            f.write("\n")
            
            # Детальный анализ метрик
            f.write(f"📊 ДЕТАЛЬНЫЙ АНАЛИЗ МЕТРИК\n")
            f.write("-" * 40 + "\n")
            for metric_name, metric_analysis in analysis["metrics_analysis"].items():
                f.write(f"\n{metric_name}:\n")
                f.write(f"  • Шагов: {metric_analysis['total_steps']}\n")
                f.write(f"  • Финальное значение: {metric_analysis['final_value']:.6f}\n")
                f.write(f"  • Минимум: {metric_analysis['min_value']:.6f}\n")
                f.write(f"  • Максимум: {metric_analysis['max_value']:.6f}\n")
                f.write(f"  • Тренд: {metric_analysis['trend']}\n")
        
        print(f"📄 Текстовый отчет сохранен: {save_path}")

def main():
    """Основная функция"""
    print("🚀 Запуск MLflow Data Exporter")
    print("=" * 50)
    
    # Создаем экспортер
    exporter = MLFlowDataExporter()
    
    # ID run из вашего сообщения
    target_run_id = "4f9a0a2937fc49a09b0c1233de968601"
    
    # Экспортируем данные для анализа
    result = exporter.export_specific_run(target_run_id)
    
    if result:
        print("\n✅ ЭКСПОРТ ЗАВЕРШЕН УСПЕШНО!")
        print(f"📁 Проверьте директорию: {exporter.export_dir}")
        
        # Выводим краткий анализ в консоль
        analysis = result["analysis"]
        print(f"\n🔍 КРАТКИЙ АНАЛИЗ:")
        print(f"   Длительность: {analysis['summary'].get('duration_hours', 'неизвестно')} часов")
        print(f"   Проблем обнаружено: {len(analysis['potential_issues'])}")
        print(f"   Рекомендаций: {len(analysis['recommendations'])}")
        
        if analysis["potential_issues"]:
            print(f"\n⚠️ ГЛАВНЫЕ ПРОБЛЕМЫ:")
            for issue in analysis["potential_issues"][:3]:  # Показываем первые 3
                print(f"   • {issue}")
    else:
        print("❌ Ошибка при экспорте данных")

if __name__ == "__main__":
    main()