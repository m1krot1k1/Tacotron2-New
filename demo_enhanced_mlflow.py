#!/usr/bin/env python3
"""
Демонстрация улучшенного MLflow логирования
"""

import mlflow
import time
from mlflow_metrics_enhancer import log_enhanced_training_metrics, log_system_metrics

def demo_enhanced_logging():
    """Демонстрирует новые возможности логирования"""
    
    print("🚀 Демонстрация улучшенного MLflow логирования")
    
    # Запускаем MLflow эксперимент
    mlflow.set_experiment("demo_enhanced_logging")
    
    with mlflow.start_run(run_name="enhanced_demo"):
        print("📊 Логируем демо-метрики...")
        
        # Симулируем обучение
        for step in range(1, 21):
            # Симулируем метрики обучения
            demo_metrics = {
                "training.loss": 1.0 - (step * 0.03),
                "training.taco_loss": 0.8 - (step * 0.025),
                "training.mi_loss": 0.1 - (step * 0.002),
                "training.guide_loss": 0.05 - (step * 0.001),
                "training.gate_loss": 0.03 - (step * 0.0005),
                "training.emb_loss": 0.02 - (step * 0.0003),
                "grad_norm": 10.0 - (step * 0.2),
                "learning_rate": 0.001 * (0.95 ** step),
                "duration": 1.2 + (step % 3) * 0.1,
                "guide_loss_weight": 0.95
            }
            
            # Используем улучшенное логирование
            log_enhanced_training_metrics(demo_metrics, step)
            
            # Validation loss каждые 5 шагов
            if step % 5 == 0:
                val_metrics = {
                    "validation.loss": 1.2 - (step * 0.02),
                    "validation.step": step
                }
                log_enhanced_training_metrics(val_metrics, step)
            
            print(f"  ✅ Шаг {step}: loss={demo_metrics['training.loss']:.3f}")
            time.sleep(0.5)  # Пауза для демонстрации
        
        print("\n🎯 Демо завершено!")
        print("\n📈 Проверьте MLflow UI:")
        print("   http://localhost:5000")
        print("\n💡 Новые возможности:")
        print("   • Группировка метрик по категориям")
        print("   • Системные метрики (CPU, RAM, GPU)")
        print("   • Улучшенная визуализация")
        print("   • Автоматический мониторинг производительности")

if __name__ == "__main__":
    demo_enhanced_logging()
