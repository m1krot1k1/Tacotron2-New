# Smart Tuner V2 для Tacotron2

Интеллектуальная система автоматического тюнинга гиперпараметров для модели Tacotron2.

## 🚀 Возможности

- **Автоматическая оптимизация гиперпараметров** с использованием Optuna
- **Интеллектуальный ранний останов** с множественными критериями
- **Динамическое планирование параметров** во время обучения
- **Мониторинг в реальном времени** через MLflow
- **Уведомления через Telegram** о ходе обучения
- **Реестр моделей** для управления лучшими результатами
- **Детекция переобучения** и автоматическая коррекция

## 📁 Структура проекта

```
smart_tuner/
├── __init__.py                 # Инициализация пакета
├── config.yaml                 # Главная конфигурация
├── trainer_wrapper.py          # Управление процессом обучения
├── metrics_store.py            # Хранение и анализ метрик
├── log_watcher.py              # Мониторинг MLflow логов
├── optimization_engine.py      # Движок оптимизации Optuna
├── param_scheduler.py          # Планировщик параметров
├── early_stop_controller.py    # Контроллер раннего останова
├── alert_manager.py            # Менеджер уведомлений
├── model_registry.py           # Реестр моделей
└── README.md                   # Документация
```

## 🔧 Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Настройте конфигурацию в `smart_tuner/config.yaml`

3. (Опционально) Настройте Telegram бота для уведомлений

## ⚙️ Конфигурация

### Основные настройки

```yaml
# Эксперимент
experiment_name: "tacotron2_smart_tuning"
dataset_path: "data/dataset"
checkpoint_dir: "data/checkpoint"

# Пространство поиска гиперпараметров
hyperparameter_search_space:
  learning_rate:
    type: "float"
    min: 0.0001
    max: 0.01
    log: true
    
  batch_size:
    type: "categorical"
    choices: [16, 32, 64]
```

### Настройки оптимизации

```yaml
optimization:
  direction: "minimize"
  objective_metric: "val_loss"
  n_trials: 20
  overfitting_penalty: 0.1
```

### Ранний останов

```yaml
early_stopping:
  patience_criterion:
    enabled: true
    type: "patience"
    metric: "val_loss"
    patience: 10
    
  overfitting_criterion:
    enabled: true
    type: "overfitting"
    overfitting_threshold: 0.2
```

### Telegram уведомления

```yaml
telegram:
  enabled: true
  bot_token: "YOUR_BOT_TOKEN"
  chat_id: "YOUR_CHAT_ID"
```

## 🚀 Использование

### Запуск оптимизации

```bash
# Полная оптимизация с 20 trials
python smart_tuner.py

# Оптимизация с кастомным количеством trials
python smart_tuner.py --trials 50

# Использование кастомной конфигурации
python smart_tuner.py --config my_config.yaml
```

### Тестирование компонентов

```bash
# Тест всех компонентов
python smart_tuner.py --test

# Демонстрация возможностей
python smart_tuner.py --demo

# Статус системы
python smart_tuner.py --status
```

### Программное использование

```python
from smart_tuner import SmartTunerV2

# Инициализация
tuner = SmartTunerV2("smart_tuner/config.yaml")

# Запуск оптимизации
results = tuner.run_optimization(n_trials=20)

# Получение лучших параметров
best_params = results['best_parameters']
print(f"Лучшие параметры: {best_params}")
```

## 🧩 Компоненты системы

### 1. TrainerWrapper
Управляет жизненным циклом процесса обучения:
- Запуск/останов train.py
- Поиск лучших чекпоинтов
- Формирование команд с гиперпараметрами

### 2. OptimizationEngine
Автоматический подбор гиперпараметров:
- Использует Optuna TPE sampler
- Поддерживает pruning слабых trials
- Штрафы за переобучение

### 3. EarlyStopController
Интеллектуальный ранний останов:
- Критерий терпения (patience)
- Детекция переобучения
- Контроль расхождения loss
- Анализ плато метрик

### 4. ParamScheduler
Динамическое изменение параметров:
- Linear, exponential, cosine планировщики
- Warmup стратегии
- Адаптация к плато

### 5. AlertManager
Система уведомлений:
- Telegram интеграция
- Уведомления о начале/завершении
- Алерты об ошибках
- Сводки по метрикам

### 6. ModelRegistry
Управление моделями:
- Автоматическое сохранение лучших моделей
- Версионирование
- Метаданные экспериментов
- Экспорт моделей

### 7. LogWatcher
Мониторинг обучения:
- Интеграция с MLflow
- Отслеживание метрик в реальном времени
- Поддержка нескольких экспериментов

### 8. MetricsStore
Хранение и анализ метрик:
- Временные ряды метрик
- Статистический анализ
- Детекция аномалий

## 📊 Примеры результатов

### Результат оптимизации
```
=== Результаты оптимизации ===
Лучшие параметры: {
  'learning_rate': 0.0023,
  'batch_size': 32,
  'epochs': 150,
  'warmup_steps': 1200
}
Лучшее значение: 1.234
Количество trials: 20
```

### Уведомления в Telegram
- 🚀 **Обучение началось** - информация об эксперименте
- 🔬 **Trial завершен** - результаты каждого trial
- ⏹️ **Ранний останов** - причины досрочного завершения
- 🎉 **Обучение завершено** - финальные результаты
- ❌ **Ошибка** - уведомления об ошибках

## 🔍 Мониторинг

### MLflow Dashboard
```bash
mlflow ui --backend-store-uri mlruns
```

### Логи системы
```
smart_tuner/logs/smart_tuner.log
```

### Состояние компонентов
```bash
python smart_tuner.py --status
```

## 🛠️ Расширение функциональности

### Добавление нового критерия останова

```python
# В early_stop_controller.py
def _check_custom_criterion(self, config, metrics):
    # Ваша логика
    return should_stop
```

### Кастомный планировщик параметров

```python
# В param_scheduler.py
def _create_custom_scheduler(self, config):
    def scheduler(step):
        # Ваша логика
        return value
    return scheduler
```

### Новый тип уведомлений

```python
# В alert_manager.py
def send_custom_notification(self, data):
    # Интеграция с другими сервисами
    pass
```

## 🐛 Устранение неполадок

### Проблемы с Optuna
- Проверьте права на запись в `smart_tuner/optuna_studies.db`
- Убедитесь в корректности пространства поиска

### Ошибки MLflow
- Проверьте путь к `mlruns`
- Убедитесь что эксперимент существует

### Проблемы с Telegram
- Проверьте токен бота и chat_id
- Убедитесь в доступности api.telegram.org

### Ошибки обучения
- Проверьте путь к train.py
- Убедитесь в корректности аргументов командной строки

## 📈 Производительность

### Рекомендации по настройке
- Используйте pruning для ускорения оптимизации
- Настройте разумные границы поиска
- Включите раннй останов для экономии времени
- Ограничьте количество сохраняемых моделей

### Мониторинг ресурсов
```bash
# Использование GPU
nvidia-smi

# Использование диска
du -sh smart_tuner/models/
```

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch
3. Внесите изменения
4. Добавьте тесты
5. Создайте Pull Request

## 📄 Лицензия

MIT License - см. файл LICENSE

## 🆘 Поддержка

- GitHub Issues для багов и предложений
- Telegram чат для быстрых вопросов
- Email для коммерческой поддержки

---

**Smart Tuner V2** - делает обучение Tacotron2 умнее и эффективнее! 🎯 

## 📱 Настройка Telegram уведомлений

Smart Tuner V2 может отправлять уведомления о ходе обучения прямо в ваш Telegram. Вот пошаговая инструкция:

### Шаг 1: Создание Telegram бота

1. **Откройте Telegram** и найдите бота `@BotFather`
2. **Отправьте команду** `/newbot`
3. **Введите имя бота** (например: `My TTS Training Bot`)
4. **Введите username бота** (например: `my_tts_bot` - должен заканчиваться на `bot`)
5. **Скопируйте токен** (выглядит как `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`)

### Шаг 2: Получение вашего Chat ID

1. **Найдите бота** `@userinfobot` в Telegram
2. **Отправьте ему** `/start`
3. **Скопируйте ваш ID** (число вида `123456789`)

### Шаг 3: Настройка конфигурации

Откройте файл `smart_tuner/config.yaml` и заполните раздел `telegram`:

```yaml
telegram:
  enabled: true                           # ✅ Включаем уведомления
  bot_token: "1234567890:ABCdefGHI..."    # 🤖 Токен от @BotFather
  chat_id: "123456789"                    # 👤 Ваш Chat ID от @userinfobot
```

### Шаг 4: Тестирование

Запустите тест подключения:

```bash
python -c "
from smart_tuner.alert_manager import AlertManager
alert = AlertManager('smart_tuner/config.yaml')
if alert.test_connection():
    print('✅ Telegram настроен правильно!')
    alert.send_message('🎉 Smart Tuner V2 подключен к Telegram!')
else:
    print('❌ Проблема с настройками Telegram')
"
```

### Типы уведомлений

Smart Tuner отправляет следующие уведомления:

- 🚀 **Начало обучения** - с основными параметрами
- ✅ **Успешное завершение** - с финальными метриками  
- ⚠️ **Ранняя остановка** - с причиной и рекомендациями
- ❌ **Ошибки** - с подробностями для диагностики
- 🔄 **Перезапуски** - когда система адаптирует параметры
- 📊 **Optuna обновления** - прогресс оптимизации

### Настройка фильтров

Вы можете выборочно включать/выключать типы уведомлений:

```yaml
telegram:
  enabled: true
  bot_token: "your_token"
  chat_id: "your_chat_id"
  
  notifications:
    training_start: true        # Уведомления о начале
    training_complete: true     # Уведомления о завершении
    early_stop: true           # Ранние остановки
    error_alerts: true         # Ошибки
    optimization_updates: true  # Optuna прогресс
    metrics_summary: false     # Периодические сводки (может быть много)
```

### Устранение проблем

**Бот не отвечает:**
- Убедитесь, что токен правильный
- Проверьте, что бот не заблокирован

**Сообщения не приходят:**
- Проверьте Chat ID (должен быть числом)
- Убедитесь, что вы начали диалог с ботом (отправьте `/start`)

**Тест подключения:**
```bash
# Проверка токена бота
curl "https://api.telegram.org/bot<YOUR_TOKEN>/getMe"

# Проверка отправки сообщения
curl -X POST "https://api.telegram.org/bot<YOUR_TOKEN>/sendMessage" \
  -d "chat_id=<YOUR_CHAT_ID>&text=Test message"
``` 