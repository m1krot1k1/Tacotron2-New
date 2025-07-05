# Проанализируем основные проблемы из логов и создадим структурированные данные для последующей визуализации
import pandas as pd
import json

# Основные проблемы, выявленные из логов и исследования
problems_data = {
    "Проблема": [
        "AttributeError: 'tuple' object has no attribute 'device'",
        "ModuleNotFoundError: No module named 'cmaes'", 
        "MLflow parameter overwrite error",
        "SQLite database locked error",
        "Tensor shape mismatch в forward pass",
        "CmaEsSampler независимое семплирование",
        "Утечки памяти в долгих trial",
        "Неправильная обработка DataLoader collate_fn"
    ],
    "Локация": [
        "train.py:897",
        "optimization_engine.py:591-595",
        "smart_tuner_main.py:logging",
        "optimization_engine.py:76-82",
        "tacotron2/model.py:forward",
        "optuna CmaEsSampler",
        "smart_tuner_main.py:268",
        "data_utils.py:DataLoader"
    ],
    "Критичность": [
        "Критическая",
        "Критическая", 
        "Высокая",
        "Критическая",
        "Критическая",
        "Средняя",
        "Высокая",
        "Критическая"
    ],
    "Статус_исправления": [
        "Требует исправления",
        "Требует установки",
        "Требует рефакторинга",
        "Требует конфигурации",
        "Требует отладки",
        "Требует конфигурации",
        "Требует оптимизации",
        "Требует исправления"
    ],
    "Влияние_на_обучение": [
        "Полная остановка",
        "Полная остановка",
        "Перезапуск trials",
        "Блокировка БД",
        "Полная остановка", 
        "Снижение производительности",
        "Деградация памяти",
        "Полная остановка"
    ]
}

problems_df = pd.DataFrame(problems_data)
print("Основные проблемы выявленные в системе:")
print(problems_df.to_string(index=False))
print()

# Рекомендации по решению проблем
solutions_data = {
    "Проблема": [
        "tuple object has no attribute device",
        "cmaes module missing",
        "MLflow parameter overwrite", 
        "SQLite database locked",
        "Tensor shape mismatch",
        "CmaEsSampler warnings",
        "Memory leaks in trials",
        "DataLoader collate_fn errors"
    ],
    "Приоритет": [1, 1, 2, 1, 1, 3, 2, 1],
    "Решение": [
        "Исправить DataLoader collate_fn для правильной обработки device transfer",
        "Установить cmaes: pip install cmaes",
        "Использовать nested runs в MLflow для каждого trial",
        "Настроить SQLite WAL режим и timeout",
        "Проверить размерности тензоров в forward pass",
        "Настроить warn_independent_sampling=False",
        "Добавить принудительную очистку памяти gc.collect()",
        "Обновить collate_fn для правильной обработки tuple"
    ],
    "Время_реализации": [
        "2-4 часа",
        "10 минут",
        "1-2 часа",
        "30 минут",
        "2-6 часов",
        "15 минут", 
        "1 час",
        "1-3 часа"
    ]
}

solutions_df = pd.DataFrame(solutions_data)
print("Приоритизированные решения:")
print(solutions_df.to_string(index=False))

# Сохраняем данные для диаграмм
problems_df.to_csv('problems_analysis.csv', index=False)
solutions_df.to_csv('solutions_analysis.csv', index=False)