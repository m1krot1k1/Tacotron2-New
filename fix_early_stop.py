#!/usr/bin/env python3
"""
Патч для исправления преждевременной остановки обучения
Основан на анализе MLflow данных
"""

import yaml
import shutil
from pathlib import Path

def apply_fixes():
    """Применяет критические исправления к Smart Tuner"""
    
    print("🚀 Применение исправлений Smart Tuner...")
    
    # 1. Создаем резервную копию конфигурации
    config_path = Path("smart_tuner/config.yaml")
    backup_path = Path("smart_tuner/config_backup.yaml")
    
    if config_path.exists():
        shutil.copy2(config_path, backup_path)
        print(f"✅ Резервная копия создана: {backup_path}")
    
    # 2. Читаем текущую конфигурацию
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 3. Применяем исправления
    print("📝 Применение исправлений...")
    
    # Отключаем агрессивный советник
    if 'adaptive_advisor' not in config:
        config['adaptive_advisor'] = {}
    
    config['adaptive_advisor']['enabled'] = False
    config['adaptive_advisor']['min_history_for_decision'] = 100  # Увеличено
    config['adaptive_advisor']['evaluation_window'] = 50  # Увеличено
    
    # Увеличиваем пороги для TTS
    if 'diagnostics' not in config['adaptive_advisor']:
        config['adaptive_advisor']['diagnostics'] = {}
    
    diag = config['adaptive_advisor']['diagnostics']
    
    if 'stagnation' not in diag:
        diag['stagnation'] = {}
    diag['stagnation']['window_size'] = 100  # Было 20
    diag['stagnation']['min_delta'] = 0.001  # Было 0.005
    
    if 'instability' not in diag:
        diag['instability'] = {}
    diag['instability']['grad_norm_threshold'] = 200.0  # Было 50.0
    
    if 'overfitting' not in diag:
        diag['overfitting'] = {}
    diag['overfitting']['threshold'] = 5.0  # Было 0.1
    diag['overfitting']['window_size'] = 30  # Было 10
    
    # Добавляем настройки безопасности
    config['training_safety'] = {
        'enabled': True,
        'min_training_hours': 4.0,
        'max_training_hours': 15.0,
        'min_training_steps': 5000,
        'max_validation_loss': 100.0
    }
    
    # Улучшаем early stopping
    config['early_stopping'] = {
        'enabled': True,
        'patience': 100,  # Увеличено с 20
        'min_delta': 0.001,  # Уменьшено с 0.01
        'monitor': 'validation.loss',
        'mode': 'min'
    }
    
    # 4. Сохраняем исправленную конфигурацию
    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=2, default_flow_style=False)
    
    print("✅ Конфигурация исправлена!")
    
    # 5. Создаем отчет об изменениях
    with open("CHANGES_APPLIED.md", 'w') as f:
        f.write("""# Примененные исправления Smart Tuner

## Изменения:
1. ❌ Отключен агрессивный Adaptive Advisor
2. 📈 Увеличены пороги остановки для TTS
3. ⏰ Добавлена защита от слишком ранней остановки
4. 🛡️ Добавлены настройки безопасности обучения

## Новые настройки:
- Минимальное время обучения: 4 часа
- Максимальный validation loss: 100.0
- Терпение early stopping: 100 эпох
- Порог нестабильности: 200.0

## Следующие шаги:
1. Запустить новое обучение: `python smart_tuner_main.py`
2. Мониторить через MLflow: http://localhost:5000
3. Проверить через 6 часов

""")
    
    print("📄 Отчет создан: CHANGES_APPLIED.md")
    print("🎯 Готово! Теперь можно запускать обучение.")

if __name__ == "__main__":
    apply_fixes()
