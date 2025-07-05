#!/usr/bin/env python3
"""
Скрипт для проверки метрик в MLflow
"""

import mlflow
import os

def check_mlflow_metrics():
    """Проверяет метрики в MLflow"""
    
    try:
        # Получаем список экспериментов
        experiments = mlflow.search_experiments()
        
        print(f"📊 Найдено {len(experiments)} экспериментов в MLflow:")
        
        for exp in experiments:
            print(f"\n🔬 Эксперимент: {exp.name}")
            print(f"   ID: {exp.experiment_id}")
            
            # Получаем runs для этого эксперимента
            runs = mlflow.search_runs(exp.experiment_id, max_results=5)
            
            if not runs.empty:
                print(f"   ✅ Найдено {len(runs)} runs:")
                
                for idx, run in runs.iterrows():
                    run_id = run['run_id']
                    status = run['status']
                    start_time = run['start_time']
                    
                    print(f"\n      🏃 Run {run_id[:8]}...")
                    print(f"         Статус: {status}")
                    print(f"         Время начала: {start_time}")
                    
                    # Получаем метрики для этого run
                    try:
                        client = mlflow.tracking.MlflowClient()
                        metrics = client.get_run(run_id).data.metrics
                        
                        if metrics:
                            print(f"         📈 Метрики ({len(metrics)}):")
                            for metric_name, metric_value in metrics.items():
                                print(f"            • {metric_name}: {metric_value:.4f}")
                        else:
                            print(f"         ⚠️ Метрики не найдены")
                            
                    except Exception as e:
                        print(f"         ❌ Ошибка получения метрик: {e}")
            else:
                print(f"   ⚠️ Runs не найдены")
                
    except Exception as e:
        print(f"❌ Ошибка подключения к MLflow: {e}")

if __name__ == "__main__":
    print("🔍 Проверка MLflow метрик...")
    check_mlflow_metrics() 