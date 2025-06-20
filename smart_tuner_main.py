#!/usr/bin/env python3
"""Smart Tuner V2 Main"""

import os
import sys
import logging
import argparse
import yaml
import threading
import time
import subprocess
import requests
from smart_tuner.trainer_wrapper import TrainerWrapper
from smart_tuner.optimization_engine import OptimizationEngine
from smart_tuner.alert_manager import AlertManager
from smart_tuner.web_interfaces import WebInterfaceManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - (%(name)s) - %(message)s',
    handlers=[
        logging.FileHandler('smart_tuner_main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SmartTunerMain:
    def __init__(self, config_path="smart_tuner/config.yaml"):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Проверяем и запускаем MLflow если нужно
        self.ensure_mlflow_running()
        
        self.trainer = TrainerWrapper(self.config)
        self.optimizer = OptimizationEngine(config_path)
        self.alert_manager = AlertManager(self.config)
        
        # Инициализация веб-интерфейсов
        self.web_manager = WebInterfaceManager(config_path)
        self.web_manager.create_interfaces()
        
        logger.info("SmartTunerMain инициализирован")
        
        # Запуск веб-интерфейсов в фоновом режиме
        self.start_web_interfaces()
        
    def is_mlflow_running(self, port=5000):
        """Проверяет, запущен ли MLflow на указанном порту"""
        try:
            response = requests.get(f"http://localhost:{port}", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def ensure_mlflow_running(self):
        """Убеждается, что MLflow сервер запущен"""
        if not self.is_mlflow_running():
            logger.info("MLflow не запущен. Запускаем...")
            try:
                # Создаем директорию для логов если её нет
                os.makedirs("mlruns", exist_ok=True)
                
                # Запускаем MLflow в фоновом режиме
                subprocess.Popen([
                    sys.executable, "-m", "mlflow", "ui",
                    "--host", "0.0.0.0",
                    "--port", "5000",
                    "--backend-store-uri", f"file://{os.path.abspath('mlruns')}"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Ждем запуска
                for _ in range(10):
                    time.sleep(1)
                    if self.is_mlflow_running():
                        logger.info("✅ MLflow успешно запущен на порту 5000")
                        return
                
                logger.warning("⚠️ MLflow запущен, но проверка подключения не прошла")
                
            except Exception as e:
                logger.error(f"❌ Ошибка запуска MLflow: {e}")
        else:
            logger.info("✅ MLflow уже запущен")
        
    def start_web_interfaces(self):
        """Запуск всех веб-интерфейсов в фоновом режиме"""
        try:
            logger.info("🚀 Запуск веб-интерфейсов Smart Tuner V2...")
            
            # Запуск в отдельном потоке
            web_thread = threading.Thread(
                target=self.web_manager.start_all,
                name="WebInterfaceManager",
                daemon=True
            )
            web_thread.start()
            
            # Небольшая задержка для инициализации
            time.sleep(2)
            
            # Отображение дашборда
            self.web_manager.print_dashboard()
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска веб-интерфейсов: {e}")
        
    def run_single_training(self, hyperparams=None):
        """Запуск полноценного обучения с опциональными гиперпараметрами"""
        logger.info("Запуск полноценного продакшн обучения")
        
        if hyperparams:
            logger.info(f"Используются кастомные гиперпараметры: {hyperparams}")
            
        # Отправляем уведомление о начале обучения
        self.alert_manager.send_info_notification(
            "🚀 Smart Tuner V2 - ПОЛНОЦЕННОЕ ОБУЧЕНИЕ\n\n"
            "🎯 Режим: Продакшн обучение\n"
            f"⚙️ Параметры: {'Кастомные' if hyperparams else 'По умолчанию'}\n"
            "🔄 Автопродолжение с чекпоинтов: ВКЛ\n"
            "💾 Полное сохранение моделей: ВКЛ\n\n"
            "📊 Мониторинг доступен через веб-интерфейсы!"
        )
        
        try:
            # Формируем гиперпараметры для полноценного обучения
            if hyperparams:
                # Конвертируем строку параметров в словарь
                hparams_str = ",".join([f"{k}={v}" for k, v in hyperparams.items()])
            else:
                hparams_str = None
                
            # Запускаем полноценное обучение
            process = self.trainer.start_training(hparams_override=hparams_str)
            
            if process:
                logger.info("Процесс обучения запущен, ожидаем завершения...")
                return_code = process.wait()
                
                if return_code == 0:
                    metrics = {"status": "completed", "return_code": return_code}
                    logger.info("Полноценное обучение завершено успешно")
                else:
                    metrics = {"status": "failed", "return_code": return_code}
                    logger.error(f"Обучение завершилось с ошибкой, код: {return_code}")
            else:
                metrics = {"status": "failed", "error": "Не удалось запустить процесс"}
                logger.error("Не удалось запустить процесс обучения")
            
            # Отправляем уведомление о завершении
            if metrics["status"] == "completed":
                self.alert_manager.send_success_notification(
                    "🎉 ПОЛНОЦЕННОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО!\n\n"
                    "✅ Статус: Успешно\n"
                    "📁 Модели сохранены в output/\n"
                    "📊 Логи доступны в MLflow UI\n"
                    "🌐 Все метрики в веб-интерфейсах\n"
                    "🏆 Готово к использованию!"
                )
            else:
                self.alert_manager.send_error_notification(
                    "❌ Обучение завершилось с ошибкой\n\n"
                    f"🔴 Статус: {metrics.get('error', 'Неизвестная ошибка')}\n"
                    "📋 Проверьте логи для диагностики\n"
                    "🌐 Детали в веб-интерфейсах"
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Критическая ошибка при обучении: {e}")
            self.alert_manager.send_error_notification(
                f"💥 Критическая ошибка обучения\n\n"
                f"🔴 Ошибка: {str(e)}\n"
                "🔧 Требуется ручное вмешательство\n"
                "🌐 Проверьте веб-интерфейсы для диагностики"
            )
            raise
        
    def run_optimization(self, n_trials=10):
        """Запуск оптимизации гиперпараметров"""
        logger.info(f"Запуск оптимизации с {n_trials} trials")
        
        # Отправляем уведомление о начале оптимизации
        self.alert_manager.send_info_notification(
            "🤖 Smart Tuner V2 - Старт оптимизации\n\n"
            f"🎯 Количество trials: {n_trials}\n"
            f"⏰ Ожидаемое время: ~{n_trials * 15} минут\n\n"
            "📊 Следите за прогрессом в веб-интерфейсах:\n"
            "• MLflow UI для экспериментов\n"
            "• Optimization Engine для Optuna\n"
            "• Metrics Store для статистики"
        )
        
        try:
            best_params = self.optimizer.optimize(self.objective, n_trials)
            
            logger.info(f"Оптимизация завершена! Лучшие параметры: {best_params}")
            
            # Отправляем уведомление о завершении оптимизации
            params_text = "\n".join([f"• {k}: {v}" for k, v in best_params.items()])
            self.alert_manager.send_success_notification(
                "🎉 Оптимизация завершена!\n\n"
                f"🏆 Лучшие параметры:\n{params_text}\n\n"
                "📁 Лучшая модель сохранена в smart_tuner/models/\n"
                "🌐 Все результаты доступны в веб-интерфейсах"
            )
            
            return best_params
            
        except Exception as e:
            logger.error(f"Ошибка при оптимизации: {e}")
            self.alert_manager.send_error_notification(
                f"❌ Ошибка при оптимизации\n\n"
                f"🔴 Ошибка: {str(e)}\n"
                "🌐 Проверьте веб-интерфейсы для диагностики"
            )
            raise
    
    def objective(self, trial):
        """
        Целевая функция для оптимизации Optuna
        
        Args:
            trial: Объект trial от Optuna
            
        Returns:
            Значение целевой функции для минимизации
        """
        try:
            # Получаем предложенные гиперпараметры
            hyperparams = self.optimizer.suggest_hyperparameters(trial)
            logger.info(f"Trial {trial.number}: Тестируем параметры {hyperparams}")
            
            # Отправляем уведомление о начале trial
            params_text = "\n".join([f"• {k}: {v}" for k, v in hyperparams.items()])
            self.alert_manager.send_info_notification(
                f"🧪 Trial #{trial.number} начат\n\n"
                f"🔧 Гиперпараметры:\n{params_text}\n\n"
                f"⏱️ Ожидаемое время: ~15 минут\n"
                "🌐 Прогресс в веб-интерфейсах"
            )
            
            # Запускаем обучение с этими параметрами
            metrics = self.trainer.train_with_params(hyperparams)
            
            if not metrics:
                logger.warning(f"Trial {trial.number}: Получены пустые метрики")
                return float('inf')  # Плохой результат
            
            # Вычисляем целевую функцию
            objective_value = self.optimizer.calculate_objective_value(metrics)
            
            logger.info(f"Trial {trial.number}: Результат = {objective_value}")
            
            # Отправляем уведомление о завершении trial
            best_value = trial.study.best_value if hasattr(trial.study, 'best_value') and trial.study.best_value else float('inf')
            self.alert_manager.send_info_notification(
                f"✅ Trial #{trial.number} завершен\n\n"
                f"📈 Результат: {objective_value:.4f}\n"
                f"🏆 Лучший результат: {best_value:.4f}\n\n"
                f"⏭️ Переходим к следующему trial...\n"
                "🌐 Детали в веб-интерфейсах"
            )
            
            return objective_value
            
        except Exception as e:
            logger.error(f"Trial {trial.number}: Ошибка - {e}")
            
            # Отправляем уведомление об ошибке
            self.alert_manager.send_error_notification(
                f"❌ Trial #{trial.number} завершен с ошибкой\n\n"
                f"🔴 Ошибка: {str(e)}\n\n"
                f"🔄 Автоматически переходим к следующему trial\n"
                "🌐 Логи в веб-интерфейсах"
            )
            
            # Возвращаем плохое значение при ошибке
            return float('inf')

def main():
    parser = argparse.ArgumentParser(description='Smart Tuner V2 - Автоматическая оптимизация гиперпараметров')
    parser.add_argument('--mode', choices=['train', 'optimize'], default='train',
                        help='Режим работы: train (обучение) или optimize (оптимизация)')
    parser.add_argument('--trials', type=int, default=10,
                        help='Количество trials для оптимизации (по умолчанию: 10)')
    parser.add_argument('--hyperparams', type=str, default=None,
                        help='Кастомные гиперпараметры в формате key1=value1,key2=value2')
    args = parser.parse_args()
    
    try:
        tuner = SmartTunerMain()
        
        if args.mode == 'train':
            # Парсим кастомные гиперпараметры если есть
            hyperparams = None
            if args.hyperparams:
                hyperparams = {}
                for param in args.hyperparams.split(','):
                    key, value = param.split('=')
                    # Пытаемся преобразовать в число
                    try:
                        if '.' in value:
                            hyperparams[key.strip()] = float(value.strip())
                        else:
                            hyperparams[key.strip()] = int(value.strip())
                    except ValueError:
                        hyperparams[key.strip()] = value.strip()
            
            # Запуск обучения
            result = tuner.run_single_training(hyperparams)
            print(f"Результат обучения: {result}")
            
        elif args.mode == 'optimize':
            # Запуск оптимизации
            best_params = tuner.run_optimization(args.trials)
            print(f"Лучшие параметры: {best_params}")
        
        # Держим программу активной для веб-интерфейсов
        print("\n🌐 Веб-интерфейсы активны. Нажмите Ctrl+C для завершения...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Завершение работы Smart Tuner V2...")
            
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
