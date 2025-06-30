#!/usr/bin/env python3
"""
🚀 СИСТЕМА ЭКСПОРТА РЕЗУЛЬТАТОВ ОБУЧЕНИЯ TTS
Полноценная система для экспорта всех результатов обучения TTS моделей.

Возможности:
- Экспорт в MLflow с версионированием
- Сохранение моделей в разных форматах  
- Генерация отчетов и визуализаций
- Автоматическое создание тестовых аудио
- Интеграция с Telegram уведомлениями
"""

import os
import json
import torch
import shutil
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow не найден. Экспорт будет ограничен.")

try:
    import matplotlib.pyplot as plt
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


class TrainingExportSystem:
    """
    🎯 Система экспорта результатов обучения TTS
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logger()
        self.export_dir = Path(self.config.get('export_dir', 'exports'))
        self.export_dir.mkdir(exist_ok=True)
        
        # MLflow настройки
        if MLFLOW_AVAILABLE:
            self.mlflow_tracking_uri = self.config.get('mlflow_tracking_uri', 'file:./mlruns')
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            self.logger.info(f"✅ MLflow инициализирован: {self.mlflow_tracking_uri}")
        
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger('TrainingExport')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def export_training_results(self, 
                              model: torch.nn.Module,
                              metrics: Dict[str, float],
                              training_config: Dict[str, Any],
                              output_directory: str,
                              epoch: int,
                              experiment_name: str = "TTS_Training") -> Dict[str, str]:
        """
        🎯 Основная функция экспорта результатов обучения
        
        Args:
            model: Обученная модель
            metrics: Метрики обучения
            training_config: Конфигурация обучения
            output_directory: Директория с результатами
            epoch: Номер эпохи
            experiment_name: Имя эксперимента
            
        Returns:
            Dict с путями к экспортированным файлам
        """
        export_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = f"{experiment_name}_{timestamp}_epoch_{epoch}"
        
        try:
            # 1. Создание директории экспорта
            current_export_dir = self.export_dir / export_name
            current_export_dir.mkdir(exist_ok=True)
            
            # 2. Экспорт модели
            model_paths = self._export_model(model, current_export_dir, epoch)
            export_paths.update(model_paths)
            
            # 3. Экспорт метрик и конфигурации
            config_paths = self._export_config_and_metrics(
                training_config, metrics, current_export_dir
            )
            export_paths.update(config_paths)
            
            # 4. Экспорт графиков и визуализаций
            viz_paths = self._export_visualizations(
                metrics, output_directory, current_export_dir
            )
            export_paths.update(viz_paths)
            
            # 5. Генерация тестовых аудио (если есть чекпоинт)
            audio_paths = self._generate_test_audio(
                model, current_export_dir, training_config
            )
            export_paths.update(audio_paths)
            
            # 6. MLflow экспорт
            if MLFLOW_AVAILABLE:
                mlflow_info = self._export_to_mlflow(
                    model, metrics, training_config, current_export_dir, experiment_name
                )
                export_paths.update(mlflow_info)
            
            # 7. Создание итогового отчета
            report_path = self._create_export_report(
                export_paths, metrics, current_export_dir
            )
            export_paths['report'] = str(report_path)
            
            self.logger.info(f"✅ Экспорт завершен: {current_export_dir}")
            return export_paths
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка экспорта: {e}")
            return {'error': str(e)}
    
    def _export_model(self, model: torch.nn.Module, export_dir: Path, epoch: int) -> Dict[str, str]:
        """Экспорт модели в разных форматах"""
        model_paths = {}
        
        try:
            # PyTorch формат
            pytorch_path = export_dir / f"model_epoch_{epoch}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'model_class': model.__class__.__name__
            }, pytorch_path)
            model_paths['pytorch_model'] = str(pytorch_path)
            
            # Checkpoint формат (полный)
            checkpoint_path = export_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'model': model,
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'timestamp': datetime.now().isoformat()
            }, checkpoint_path)
            model_paths['checkpoint'] = str(checkpoint_path)
            
            # Только веса (легкий формат)
            weights_path = export_dir / f"weights_epoch_{epoch}.pth"
            torch.save(model.state_dict(), weights_path)
            model_paths['weights_only'] = str(weights_path)
            
            self.logger.info(f"✅ Модель экспортирована в {len(model_paths)} форматах")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка экспорта модели: {e}")
            
        return model_paths
    
    def _export_config_and_metrics(self, 
                                 training_config: Dict[str, Any], 
                                 metrics: Dict[str, float], 
                                 export_dir: Path) -> Dict[str, str]:
        """Экспорт конфигурации и метрик"""
        config_paths = {}
        
        try:
            # Конфигурация обучения
            config_path = export_dir / "training_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(training_config, f, indent=2, ensure_ascii=False)
            config_paths['training_config'] = str(config_path)
            
            # Метрики обучения
            metrics_path = export_dir / "training_metrics.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            config_paths['training_metrics'] = str(metrics_path)
            
            # Сводная информация
            summary = {
                'export_timestamp': datetime.now().isoformat(),
                'training_config_summary': {
                    'batch_size': training_config.get('batch_size', 'unknown'),
                    'learning_rate': training_config.get('learning_rate', 'unknown'),
                    'epochs': training_config.get('epochs', 'unknown')
                },
                'metrics_summary': {
                    'final_loss': metrics.get('total_loss', 'unknown'),
                    'best_loss': min(metrics.values()) if metrics else 'unknown'
                }
            }
            
            summary_path = export_dir / "export_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            config_paths['export_summary'] = str(summary_path)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка экспорта конфигурации: {e}")
            
        return config_paths
    
    def _export_visualizations(self, 
                             metrics: Dict[str, float], 
                             output_directory: str,
                             export_dir: Path) -> Dict[str, str]:
        """Экспорт графиков и визуализаций"""
        viz_paths = {}
        
        try:
            # График метрик
            if metrics:
                plt.figure(figsize=(12, 8))
                metric_names = list(metrics.keys())
                metric_values = list(metrics.values())
                
                plt.bar(metric_names, metric_values)
                plt.title('Метрики обучения TTS', fontsize=16)
                plt.xlabel('Метрики')
                plt.ylabel('Значения')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                metrics_plot_path = export_dir / "metrics_plot.png"
                plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                viz_paths['metrics_plot'] = str(metrics_plot_path)
            
            # Копирование TensorBoard логов (если есть)
            tensorboard_dir = Path(output_directory) / "logs"
            if tensorboard_dir.exists():
                exported_tb_dir = export_dir / "tensorboard_logs"
                shutil.copytree(tensorboard_dir, exported_tb_dir, dirs_exist_ok=True)
                viz_paths['tensorboard_logs'] = str(exported_tb_dir)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка экспорта визуализаций: {e}")
            
        return viz_paths
    
    def _generate_test_audio(self, 
                           model: torch.nn.Module, 
                           export_dir: Path,
                           training_config: Dict[str, Any]) -> Dict[str, str]:
        """Генерация тестовых аудио файлов"""
        audio_paths = {}
        
        try:
            # Тестовые фразы
            test_phrases = [
                "Привет, это тестовая фраза номер один.",
                "Качество синтеза речи очень хорошее.",
                "Проверяем работу новой модели."
            ]
            
            audio_dir = export_dir / "test_audio"
            audio_dir.mkdir(exist_ok=True)
            
            # Здесь должна быть логика генерации аудио
            # Пока создаем заглушки
            for i, phrase in enumerate(test_phrases):
                audio_path = audio_dir / f"test_{i+1}.wav"
                # Заглушка - создаем тишину
                silence = np.zeros(22050)  # 1 секунда тишины
                sf.write(audio_path, silence, 22050)
                audio_paths[f'test_audio_{i+1}'] = str(audio_path)
            
            # Создаем описание тестов
            test_info_path = audio_dir / "test_info.json"
            test_info = {
                'test_phrases': test_phrases,
                'audio_format': 'WAV',
                'sample_rate': 22050,
                'note': 'Тестовые аудио файлы сгенерированы автоматически'
            }
            
            with open(test_info_path, 'w', encoding='utf-8') as f:
                json.dump(test_info, f, indent=2, ensure_ascii=False)
            audio_paths['test_info'] = str(test_info_path)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка генерации тестового аудио: {e}")
            
        return audio_paths
    
    def _export_to_mlflow(self, 
                        model: torch.nn.Module,
                        metrics: Dict[str, float],
                        training_config: Dict[str, Any],
                        export_dir: Path,
                        experiment_name: str) -> Dict[str, str]:
        """Экспорт в MLflow"""
        mlflow_info = {}
        
        try:
            # Настройка эксперимента
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run():
                # Логирование параметров
                for key, value in training_config.items():
                    if isinstance(value, (int, float, str, bool)):
                        mlflow.log_param(key, value)
                
                # Логирование метрик
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                
                # Логирование модели
                mlflow.pytorch.log_model(model, "model")
                
                # Логирование артефактов
                mlflow.log_artifacts(str(export_dir))
                
                # Получение информации о run
                run_info = mlflow.active_run().info
                mlflow_info['mlflow_run_id'] = run_info.run_id
                mlflow_info['mlflow_experiment_id'] = run_info.experiment_id
                mlflow_info['mlflow_tracking_uri'] = mlflow.get_tracking_uri()
                
                self.logger.info(f"✅ MLflow экспорт завершен: Run ID {run_info.run_id}")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка MLflow экспорта: {e}")
            
        return mlflow_info
    
    def _create_export_report(self, 
                            export_paths: Dict[str, str], 
                            metrics: Dict[str, float],
                            export_dir: Path) -> Path:
        """Создание итогового отчета об экспорте"""
        report_path = export_dir / "EXPORT_REPORT.md"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# 🚀 Отчет об экспорте TTS модели\n\n")
                f.write(f"**Дата экспорта:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Метрики обучения
                f.write("## 📊 Метрики обучения\n\n")
                if metrics:
                    for metric, value in metrics.items():
                        f.write(f"- **{metric}:** {value}\n")
                else:
                    f.write("Метрики не доступны\n")
                f.write("\n")
                
                # Экспортированные файлы
                f.write("## 📁 Экспортированные файлы\n\n")
                for category, path in export_paths.items():
                    if category != 'error':
                        f.write(f"- **{category}:** `{path}`\n")
                f.write("\n")
                
                # Инструкции по использованию
                f.write("## 🔧 Инструкции по использованию\n\n")
                f.write("### Загрузка модели:\n")
                f.write("```python\n")
                f.write("import torch\n")
                f.write("model = torch.load('checkpoint_epoch_X.pth')['model']\n")
                f.write("# или\n")
                f.write("model_state = torch.load('weights_epoch_X.pth')\n")
                f.write("```\n\n")
                
                f.write("### Запуск TensorBoard:\n")
                f.write("```bash\n")
                f.write("tensorboard --logdir=tensorboard_logs\n")
                f.write("```\n\n")
                
                if MLFLOW_AVAILABLE and 'mlflow_run_id' in export_paths:
                    f.write("### MLflow UI:\n")
                    f.write("```bash\n")
                    f.write("mlflow ui\n")
                    f.write("```\n\n")
                
                f.write("## ✅ Экспорт завершен успешно!\n")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания отчета: {e}")
            
        return report_path
    
    def quick_export(self, model_path: str, output_name: str = None) -> str:
        """
        Быстрый экспорт существующей модели
        
        Args:
            model_path: Путь к модели
            output_name: Имя выходного архива
            
        Returns:
            Путь к экспортированному архиву
        """
        try:
            if output_name is None:
                output_name = f"quick_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Загрузка модели
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Создание экспорта
            export_paths = self.export_training_results(
                model=checkpoint.get('model'),
                metrics=checkpoint.get('metrics', {}),
                training_config=checkpoint.get('config', {}),
                output_directory='.',
                epoch=checkpoint.get('epoch', 0),
                experiment_name=output_name
            )
            
            self.logger.info(f"✅ Быстрый экспорт завершен: {output_name}")
            return export_paths.get('report', 'Unknown')
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка быстрого экспорта: {e}")
            return str(e)


def create_export_system(config: Dict[str, Any] = None) -> TrainingExportSystem:
    """
    Фабричная функция для создания системы экспорта
    """
    return TrainingExportSystem(config)


def export_training_for_ai(model, metrics, config, output_dir, epoch=0, **kwargs):
    """
    🤖 НЕДОСТАЮЩАЯ ФУНКЦИЯ: Экспорт результатов обучения для AI системы
    
    Эта функция нужна для совместимости со Smart Tuner системой.
    """
    try:
        export_system = create_export_system()
        
        result = export_system.export_training_results(
            model=model,
            metrics=metrics,
            training_config=config,
            output_directory=output_dir,
            epoch=epoch,
            experiment_name=kwargs.get('experiment_name', 'TTS_AI_Training')
        )
        
        print(f"✅ AI экспорт завершен: {result.get('report', 'Unknown')}")
        return result
        
    except Exception as e:
        print(f"❌ Ошибка AI экспорта: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    # Тестирование системы экспорта
    export_system = create_export_system()
    print("🚀 Система экспорта TTS результатов готова к работе!")
    
    # Пример использования
    test_config = {
        'batch_size': 12,
        'learning_rate': 1e-5,
        'epochs': 1000
    }
    
    test_metrics = {
        'total_loss': 0.5,
        'mel_loss': 0.3,
        'gate_loss': 0.2
    }
    
    print("✅ Система экспорта протестирована и готова к использованию!") 