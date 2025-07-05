# Создание анализа проблем обучения на основе логов Telegram
import pandas as pd
import numpy as np

# Создание таблицы с проблемами из Telegram логов
training_issues = pd.DataFrame({
    'Время': ['20:38:41', '20:38:42', '20:38:57', '20:39:14', '20:39:30', '20:41:21', '20:42:28'],
    'Шаг': [0, 0, 0, 0, 0, 100, 0],
    'Взрыв_градиентов': [476894.70, 476894.70, 563978.33, 544413.67, 596312.02, 222373.06, 487421.58],
    'Loss': [32.5662, 32.5662, 36.1027, 31.6153, 33.5660, 200.6304, 34.3313],
    'Качество_%': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Фаза': ['prealignment'] * 7,
    'Перезапуск': [True, True, True, True, True, True, True]
})

print("АНАЛИЗ ПРОБЛЕМ ОБУЧЕНИЯ ИЗ TELEGRAM ЛОГОВ")
print("=" * 60)
print(training_issues.to_string(index=False))

# Статистика проблем
print("\n" + "=" * 60)
print("СТАТИСТИКА КРИТИЧЕСКИХ ПРОБЛЕМ:")
print(f"Среднее значение взрыва градиентов: {training_issues['Взрыв_градиентов'].mean():.2f}")
print(f"Максимальное значение взрыва градиентов: {training_issues['Взрыв_градиентов'].max():.2f}")
print(f"Минимальное значение взрыва градиентов: {training_issues['Взрыв_градиентов'].min():.2f}")
print(f"Среднее значение Loss: {training_issues['Loss'].mean():.2f}")
print(f"Количество перезапусков на шаге 0: {(training_issues['Шаг'] == 0).sum()}")
print(f"Качество обучения: {training_issues['Качество_%'].max()}% (критическое)")

# Создание DataFrame с улучшениями в репозитории
improvements_status = pd.DataFrame({
    'Компонент': [
        'Smart Tuner v2',
        'Gradient Stability Monitor', 
        'Enhanced MLFlow Logger',
        'Emergency Recovery System',
        'Alignment Diagnostics',
        'Audio Quality Enhancer',
        'Smart Training Logger',
        'Gradient Adaptive Factor',
        'Loss Scaler',
        'Smart Segmenter',
        'MLFlow Data Exporter',
        'Training Integration',
        'Debug Reporter'
    ],
    'Статус_интеграции': [
        'Частично интегрирован',
        'Интегрирован',
        'Интегрирован', 
        'Интегрирован',
        'Требует доработки',
        'Интегрирован',
        'Интегрирован',
        'Требует доработки',
        'Интегрирован',
        'Не интегрирован',
        'Интегрирован',
        'Требует доработки',
        'Интегрирован'
    ],
    'Критичность': [
        'Высокая',
        'Критическая',
        'Средняя',
        'Высокая', 
        'Критическая',
        'Низкая',
        'Средняя',
        'Критическая',
        'Высокая',
        'Средняя',
        'Низкая',
        'Высокая',
        'Низкая'
    ],
    'Влияние_на_проблему': [
        'Прямое - основная система',
        'Прямое - не работает правильно',
        'Косвенное',
        'Косвенное - срабатывает но не решает',
        'Прямое - нужна для диагностики',
        'Нет влияния',
        'Косвенное',
        'Прямое - критично для градиентов',
        'Прямое - может помочь с loss',
        'Нет влияния',
        'Нет влияния',
        'Косвенное',
        'Косвенное'
    ]
})

print("\n" + "=" * 60)
print("СТАТУС ИНТЕГРАЦИИ УЛУЧШЕНИЙ В SMART TUNER V2")
print("=" * 60)
print(improvements_status.to_string(index=False))

# Подсчет статистики интеграции
integration_stats = improvements_status['Статус_интеграции'].value_counts()
print(f"\n\nСТАТИСТИКА ИНТЕГРАЦИИ:")
for status, count in integration_stats.items():
    print(f"- {status}: {count} компонентов")

print("\nКРИТИЧЕСКИЕ КОМПОНЕНТЫ С ПРОБЛЕМАМИ:")
critical_issues = improvements_status[
    (improvements_status['Критичность'] == 'Критическая') & 
    (improvements_status['Статус_интеграции'] != 'Интегрирован')
]
for _, row in critical_issues.iterrows():
    print(f"- {row['Компонент']}: {row['Статус_интеграции']}")