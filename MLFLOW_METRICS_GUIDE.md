# 📊 Руководство по метрикам MLflow в Smart Tuner TTS

## 🎯 Обзор

Система Smart Tuner TTS теперь логирует **полный набор метрик** в MLflow для детального мониторинга процесса обучения.

## 📈 Доступные метрики

### 🏋️ **Training Metrics (каждая итерация):**
- `training.loss` - общий loss обучения
- `training.taco_loss` - loss Tacotron2 модели
- `training.atten_loss` - loss механизма внимания
- `training.mi_loss` - MMI loss (если включен)
- `training.guide_loss` - Guided Attention loss (если включен)
- `training.gate_loss` - loss gate выходов
- `training.emb_loss` - loss эмбеддингов
- `grad.norm` - норма градиентов
- `learning.rate` - текущий learning rate
- `duration` - время на итерацию
- `epoch` - текущая эпоха
- `iteration` - номер итерации

### 🔍 **Validation Metrics (каждые N итераций):**
- `validation.loss` - validation loss
- `validation.alignment_score` - качество alignment матрицы
- `validation.attention_entropy` - энтропия attention
- `validation.attention_focus` - фокусировка attention
- `validation.gate_mean` - среднее значение gate выходов
- `validation.gate_std` - стандартное отклонение gate выходов
- `validation.step` - шаг валидации

### ⚙️ **Model Parameters (один раз):**
- `model.total_params` - общее количество параметров модели
- `model.trainable_params` - количество обучаемых параметров
- `hparams.batch_size` - размер батча
- `hparams.learning_rate` - learning rate
- `hparams.epochs` - количество эпох
- `hparams.grad_clip_thresh` - порог обрезки градиентов
- `hparams.fp16_run` - использование FP16
- `hparams.use_mmi` - использование MMI loss
- `hparams.use_guided_attn` - использование Guided Attention
- `dataset.train_size` - размер training датасета
- `dataset.val_size` - размер validation датасета

## 🖥️ **Как просматривать метрики:**

### 1. **MLflow UI:**
```bash
# Запустить MLflow UI (если не запущен)
mlflow ui --host 0.0.0.0 --port 5000

# Открыть в браузере
http://localhost:5000
```

### 2. **Навигация в MLflow UI:**
- **Experiments** → **Default** - список всех экспериментов
- **Runs** - список всех запусков обучения
- **Metrics** - графики всех метрик
- **Parameters** - параметры модели и обучения
- **Artifacts** - сохраненные файлы

### 3. **Сравнение экспериментов:**
- Выберите несколько runs в списке
- Нажмите **Compare** для сравнения метрик
- Используйте **Parallel Coordinates** для многомерного анализа

## 📊 **Ключевые метрики для мониторинга:**

### 🎯 **Основные индикаторы качества:**
1. **`validation.loss`** - главная метрика качества
2. **`training.loss`** - прогресс обучения
3. **`validation.alignment_score`** - качество alignment
4. **`validation.attention_focus`** - фокусировка attention

### ⚠️ **Индикаторы проблем:**
1. **`grad.norm`** - слишком большие градиенты (>10)
2. **`validation.gate_mean`** - должно быть близко к 0.5
3. **`training.guide_loss`** - должно уменьшаться
4. **`duration`** - время на итерацию

## 🔧 **Настройка логирования:**

### Включение расширенного логирования:
```python
# В train.py автоматически определяется наличие mlflow_metrics_enhancer
ENHANCED_LOGGING = True  # если доступен enhanced logger
```

### Частота логирования:
```python
# В hparams.py
validation_freq = 50  # валидация каждые 50 итераций
```

## 📈 **Интерпретация метрик:**

### ✅ **Хорошие показатели:**
- `validation.loss` стабильно уменьшается
- `validation.alignment_score` > 0.5
- `validation.attention_focus` > 0.8
- `grad.norm` < 5.0
- `validation.gate_mean` ≈ 0.5

### ❌ **Проблемные показатели:**
- `validation.loss` растет или стагнирует
- `validation.alignment_score` < 0.3
- `grad.norm` > 10.0 (взрывающиеся градиенты)
- `validation.gate_mean` < 0.1 или > 0.9

## 🎯 **Рекомендации по использованию:**

1. **Мониторинг в реальном времени:**
   - Держите MLflow UI открытым во время обучения
   - Обновляйте страницу для новых данных

2. **Анализ трендов:**
   - Используйте сглаживание графиков
   - Сравнивайте несколько экспериментов

3. **Раннее обнаружение проблем:**
   - Следите за `grad.norm` и `validation.loss`
   - Остановите обучение при аномальных значениях

4. **Оптимизация гиперпараметров:**
   - Сравнивайте разные `learning_rate`
   - Анализируйте влияние `batch_size`

## 🚀 **Быстрый старт:**

```bash
# 1. Запустить обучение
python3 smart_tuner_main.py --mode train --trials 1

# 2. Открыть MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# 3. Перейти в браузере
http://localhost:5000

# 4. Выбрать эксперимент → Run → Metrics
```

**🎉 Теперь у вас есть полный контроль над процессом обучения TTS модели!** 