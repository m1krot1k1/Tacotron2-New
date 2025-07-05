#!/usr/bin/env python3
"""
Скрипт для очистки старых логов TensorBoard и MLflow
"""

import os
import shutil
import glob
import argparse

def clean_tensorboard_logs(log_directory):
    """Очищает старые логи TensorBoard"""
    if not os.path.exists(log_directory):
        print(f"📁 Директория {log_directory} не существует")
        return
    
    print(f"🧹 Очищаем логи TensorBoard в {log_directory}")
    
    # Удаляем event файлы
    event_files = glob.glob(os.path.join(log_directory, "events.out.tfevents*"))
    for file in event_files:
        try:
            os.remove(file)
            print(f"🗑️ Удален: {os.path.basename(file)}")
        except Exception as e:
            print(f"⚠️ Ошибка удаления {file}: {e}")
    
    # Удаляем другие временные файлы TensorBoard
    temp_files = glob.glob(os.path.join(log_directory, "*.tmp"))
    for file in temp_files:
        try:
            os.remove(file)
            print(f"🗑️ Удален временный файл: {os.path.basename(file)}")
        except Exception as e:
            print(f"⚠️ Ошибка удаления {file}: {e}")
    
    print(f"✅ Очистка TensorBoard логов завершена")

def clean_mlflow_logs(mlflow_dir=None):
    """Очищает старые логи MLflow"""
    if mlflow_dir is None:
        mlflow_dir = os.path.expanduser("~/.mlflow")
    
    if not os.path.exists(mlflow_dir):
        print(f"📁 MLflow директория {mlflow_dir} не существует")
        return
    
    print(f"🧹 Очищаем логи MLflow в {mlflow_dir}")
    
    # Удаляем старые runs (старше 7 дней)
    import time
    current_time = time.time()
    week_ago = current_time - (7 * 24 * 60 * 60)  # 7 дней
    
    runs_dir = os.path.join(mlflow_dir, "mlruns")
    if os.path.exists(runs_dir):
        for experiment_dir in os.listdir(runs_dir):
            experiment_path = os.path.join(runs_dir, experiment_dir)
            if os.path.isdir(experiment_path):
                for run_dir in os.listdir(experiment_path):
                    run_path = os.path.join(experiment_path, run_dir)
                    if os.path.isdir(run_path):
                        # Проверяем время создания
                        try:
                            creation_time = os.path.getctime(run_path)
                            if creation_time < week_ago:
                                shutil.rmtree(run_path)
                                print(f"🗑️ Удален старый run: {run_dir}")
                        except Exception as e:
                            print(f"⚠️ Ошибка проверки {run_path}: {e}")
    
    print(f"✅ Очистка MLflow логов завершена")

def main():
    parser = argparse.ArgumentParser(description="Очистка старых логов")
    parser.add_argument("--log-dir", type=str, default="logs", 
                       help="Директория с логами TensorBoard")
    parser.add_argument("--mlflow-dir", type=str, default=None,
                       help="Директория MLflow (по умолчанию ~/.mlflow)")
    parser.add_argument("--all", action="store_true",
                       help="Очистить все логи")
    
    args = parser.parse_args()
    
    if args.all or args.log_dir:
        clean_tensorboard_logs(args.log_dir)
    
    if args.all or args.mlflow_dir:
        clean_mlflow_logs(args.mlflow_dir)
    
    print("🎉 Очистка завершена!")

if __name__ == "__main__":
    main() 