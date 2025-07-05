# 🔧 Руководство по устранению неполадок

## Проблемы с логированием

### 1. TensorBoard: "No image data was found"

**Причина:** Старые логи TensorBoard не удаляются автоматически.

**Решение:**
```bash
# Очистить логи TensorBoard
python clean_logs.py --log-dir logs

# Или очистить все логи
python clean_logs.py --all
```

### 2. MLflow: Нет метрик в интерфейсе

**Причина:** MLflow не инициализируется корректно или run не завершается.

**Решение:**
- Проверьте, что MLflow установлен: `pip install mlflow`
- Убедитесь, что директория `~/.mlflow` существует
- Очистите старые runs: `python clean_logs.py --all`

### 3. Ошибка: "No module named 'utils.dynamic_padding'"

**Причина:** Отсутствуют файлы utils или неправильная структура пакетов.

**Решение:**
```bash
# Создать структуру utils
mkdir -p utils
touch utils/__init__.py

# Проверить наличие файлов
ls -la utils/
```

## Проблемы с обучением

### 4. Критические перезапуски обучения

**Причина:** NaN/Inf в loss, нестабильные градиенты.

**Решение:**
- Уменьшите learning rate: `--hparams "learning_rate=0.001"`
- Включите guided attention: `--hparams "use_guided_attn=True"`
- Уменьшите batch size: `--hparams "batch_size=8"`

### 5. Отсутствие изображений в TensorBoard

**Причина:** Проблемы с forward pass или размерностями тензоров.

**Решение:**
- Проверьте размеры входных данных
- Убедитесь, что модель возвращает корректные выходы
- Проверьте логи на наличие ошибок валидации

## Команды для диагностики

### Проверка установки
```bash
# Проверить TensorBoard
tensorboard --version

# Проверить MLflow
mlflow --version

# Проверить PyTorch
python -c "import torch; print(torch.__version__)"
```

### Очистка и перезапуск
```bash
# Очистить все логи
python clean_logs.py --all

# Перезапустить обучение с безопасными параметрами
python train.py \
  --output_directory output \
  --log_directory logs \
  --hparams "learning_rate=0.001,batch_size=8,use_guided_attn=True,grad_clip_thresh=1.0"
```

### Мониторинг в реальном времени
```bash
# TensorBoard
tensorboard --logdir logs --port 6006

# MLflow UI
mlflow ui --port 5000
```

## Структура проекта

```
Tacotron2-New/
├── train.py              # Основной скрипт обучения
├── clean_logs.py         # Скрипт очистки логов
├── utils/
│   ├── __init__.py       # Инициализация пакета
│   ├── dynamic_padding.py # Динамический паддинг
│   └── bucket_batching.py # Батчинг по группам
├── logs/                 # Логи TensorBoard
├── output/               # Чекпоинты модели
└── smart_tuner/          # Компоненты Smart Tuner
```

## Частые ошибки и решения

### Ошибка: "CUDA out of memory"
```bash
# Уменьшить batch size
--hparams "batch_size=4"

# Включить gradient checkpointing
--hparams "use_gradient_checkpointing=True"
```

### Ошибка: "RuntimeError: size mismatch"
```bash
# Проверить размеры данных
python -c "from data_utils import TextMelLoader; print('Data loaded successfully')"
```

### Ошибка: "ImportError: No module named 'smart_tuner'"
```bash
# Установить зависимости
pip install -r requirements.txt

# Или отключить Smart Tuner
--hparams "use_smart_tuner=False"
```

## Контакты

При возникновении проблем:
1. Проверьте логи в консоли
2. Запустите диагностику: `python clean_logs.py --all`
3. Попробуйте безопасные параметры
4. Создайте issue с подробным описанием ошибки 