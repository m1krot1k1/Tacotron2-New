# 📚 Документация гиперпараметров Tacotron2-New

## 🎯 Новые гиперпараметры Smart Tuner V2

### **Smart Tuner параметры**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `use_bucket_batching` | bool | `True` | Использовать bucket batching для оптимизации памяти |
| `force_model_reinit` | bool | `False` | Принудительная реинициализация модели при проблемах |
| `xavier_init` | bool | `True` | Использовать Xavier инициализацию весов |
| `use_audio_quality_enhancement` | bool | `True` | Включить улучшение качества аудио |
| `use_mmi` | bool | `False` | Использовать MMI loss |
| `use_guided_attn` | bool | `True` | Использовать guided attention |
| `guide_loss_weight` | float | `1.0` | Вес guided attention loss |
| `guide_loss_initial_weight` | float | `1.0` | Начальный вес guided attention |
| `use_ddc_loss` | bool | `False` | Использовать DDC loss |
| `ddc_loss_weight` | float | `0.1` | Вес DDC loss |

### **Параметры безопасности**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `grad_clip_thresh` | float | `1.0` | Порог обрезания градиентов |
| `dynamic_loss_scaling` | bool | `True` | Динамическое масштабирование loss |
| `fp16_run` | bool | `False` | Использовать FP16 обучение |
| `emergency_restart_enabled` | bool | `True` | Включить экстренный перезапуск |
| `max_restart_attempts` | int | `3` | Максимальное количество перезапусков |

### **Параметры мониторинга**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `telegram_monitoring` | bool | `True` | Включить Telegram мониторинг |
| `debug_reporting` | bool | `True` | Включить детальную диагностику |
| `mlflow_logging` | bool | `True` | Включить логирование в MLflow |
| `tensorboard_logging` | bool | `True` | Включить логирование в TensorBoard |

## 🚀 Примеры использования

### **Базовое обучение**
```bash
python train.py \
    -o output \
    -l logs \
    --hparams "learning_rate=1e-3,batch_size=32,use_guided_attn=True"
```

### **Обучение с оптимизацией гиперпараметров**
```bash
python train.py \
    -o output \
    -l logs \
    --optimize-hyperparams \
    --n-trials 20 \
    --optimization-timeout 3600
```

### **Distributed обучение**
```bash
python multiproc.py \
    train.py \
    -o output \
    -l logs \
    --n_gpus 4 \
    --hparams "distributed_run=True,batch_size=16"
```

### **Экстренное восстановление**
```bash
python emergency_recovery.py
```

## 🔧 Настройка через CLI

### **Установка параметров через --hparams**
```bash
# Базовые параметры
--hparams "learning_rate=1e-4,batch_size=16"

# Smart Tuner параметры
--hparams "use_guided_attn=True,guide_loss_weight=2.0,use_mmi=True"

# Параметры безопасности
--hparams "grad_clip_thresh=0.5,fp16_run=True,emergency_restart_enabled=True"

# Полная конфигурация
--hparams "learning_rate=1e-4,batch_size=16,use_guided_attn=True,guide_loss_weight=2.0,use_mmi=True,grad_clip_thresh=0.5,fp16_run=True"
```

### **Оптимизация гиперпараметров**
```bash
# Базовая оптимизация (10 trials)
python train.py --optimize-hyperparams

# Расширенная оптимизация
python train.py \
    --optimize-hyperparams \
    --n-trials 50 \
    --optimization-timeout 7200 \
    -o output/optimization \
    -l logs/optimization

# Оптимизация с ограничениями
python train.py \
    --optimize-hyperparams \
    --n-trials 30 \
    --hparams "max_restart_attempts=5,emergency_restart_enabled=True"
```

## 📊 Мониторинг и диагностика

### **Telegram уведомления**
```bash
# Включить Telegram мониторинг
--hparams "telegram_monitoring=True"

# Настроить интервалы уведомлений
--hparams "telegram_notification_interval=1000"
```

### **Debug отчеты**
```bash
# Включить детальную диагностику
--hparams "debug_reporting=True"

# Настроить интервалы отчетов
--hparams "debug_report_interval=500"
```

### **MLflow логирование**
```bash
# Включить MLflow
--hparams "mlflow_logging=True"

# Настроить эксперимент
--hparams "mlflow_experiment_name=tacotron2_experiment"
```

## 🛡️ Параметры безопасности

### **Автоматическое восстановление**
```bash
# Включить экстренный перезапуск
--hparams "emergency_restart_enabled=True,max_restart_attempts=3"

# Настроить пороги для перезапуска
--hparams "nan_detection_threshold=1e-6,gradient_explosion_threshold=1000"
```

### **Стабилизация обучения**
```bash
# Строгое обрезание градиентов
--hparams "grad_clip_thresh=0.1"

# Динамическое масштабирование loss
--hparams "dynamic_loss_scaling=True"

# Отключение FP16 для стабильности
--hparams "fp16_run=False"
```

## 🎯 Рекомендуемые конфигурации

### **Для быстрого прототипирования**
```bash
--hparams "learning_rate=1e-3,batch_size=8,use_guided_attn=True,guide_loss_weight=1.0,grad_clip_thresh=1.0,fp16_run=False"
```

### **Для стабильного обучения**
```bash
--hparams "learning_rate=1e-4,batch_size=16,use_guided_attn=True,guide_loss_weight=2.0,use_mmi=True,grad_clip_thresh=0.5,fp16_run=True,emergency_restart_enabled=True"
```

### **Для максимального качества**
```bash
--hparams "learning_rate=5e-5,batch_size=32,use_guided_attn=True,guide_loss_weight=3.0,use_mmi=True,use_ddc_loss=True,ddc_loss_weight=0.1,grad_clip_thresh=0.3,fp16_run=True,emergency_restart_enabled=True"
```

### **Для distributed обучения**
```bash
--hparams "distributed_run=True,batch_size=8,learning_rate=1e-4,use_guided_attn=True,grad_clip_thresh=0.5,fp16_run=True"
```

## 🔍 Диагностика проблем

### **Проблемы с attention alignment**
```bash
# Увеличить guided attention weight
--hparams "use_guided_attn=True,guide_loss_weight=5.0"

# Снизить learning rate
--hparams "learning_rate=1e-5"

# Увеличить batch size
--hparams "batch_size=32"
```

### **Проблемы с градиентами**
```bash
# Строгое обрезание градиентов
--hparams "grad_clip_thresh=0.1"

# Отключить FP16
--hparams "fp16_run=False"

# Включить экстренный перезапуск
--hparams "emergency_restart_enabled=True"
```

### **Проблемы с памятью**
```bash
# Уменьшить batch size
--hparams "batch_size=4"

# Включить bucket batching
--hparams "use_bucket_batching=True"

# Отключить FP16
--hparams "fp16_run=False"
```

## 📈 Оптимизация производительности

### **Автоматическая оптимизация**
```bash
# Запустить Optuna HPO
python train.py --optimize-hyperparams --n-trials 50

# Оптимизация с ограничениями времени
python train.py --optimize-hyperparams --n-trials 20 --optimization-timeout 3600
```

### **Ручная настройка**
```bash
# Эксперимент с разными learning rates
for lr in 1e-3 1e-4 5e-5; do
    python train.py --hparams "learning_rate=$lr,batch_size=16"
done

# Эксперимент с guided attention
for weight in 1.0 2.0 5.0; do
    python train.py --hparams "guide_loss_weight=$weight,use_guided_attn=True"
done
```

## 🎉 Заключение

Эта документация покрывает все новые гиперпараметры Smart Tuner V2. Для получения дополнительной информации обратитесь к:

- `smart_tuner/config.yaml` - основной конфигурационный файл
- `smart_tuner/README.md` - документация Smart Tuner
- `FINAL_ALL_FIXES_REPORT.md` - отчет о всех исправлениях 