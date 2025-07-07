# 🚀 Enhanced Tacotron2 Training System

## Обзор

**EnhancedTacotronTrainer** — современная система обучения Tacotron2 с полной интеграцией Smart Tuner V2, автоматическим контролем качества и интеллектуальным управлением процессом обучения.

## 🎯 Основные возможности

### ✅ **Интегрированные компоненты:**
- **EnhancedTacotronTrainer** — основной движок обучения
- **Smart Tuner V2** — автоматическая оптимизация гиперпараметров
- **Telegram Monitor** — уведомления о прогрессе и проблемах
- **MLflow** — логирование экспериментов и метрик
- **TensorBoard** — визуализация обучения
- **Alignment Diagnostics** — мониторинг attention alignment

### 🧠 **Интеллектуальные функции:**
- **Автоматическая оптимизация** гиперпараметров на основе качества
- **Фазовое обучение** — адаптация стратегии по мере прогресса
- **Раннее обнаружение проблем** — автоматические исправления
- **Guided Attention** — улучшенное выравнивание attention
- **Quality Control** — постоянный мониторинг качества TTS

## 🚀 Быстрый старт

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Запуск обучения
```bash
# Основной entrypoint для обучения
python train_enhanced.py --mode train --epochs 100

# Тестовый запуск
python train_enhanced.py --mode test

# Валидация модели
python train_enhanced.py --mode validate
```

### Параметры запуска
```bash
python train_enhanced.py \
    --config smart_tuner/config.yaml \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --mode train \
    --output output/
```

## 📊 Мониторинг обучения

### Telegram уведомления
Система автоматически отправляет уведомления о:
- Прогрессе обучения
- Критических проблемах
- Рекомендациях по улучшению
- Завершении обучения

### MLflow эксперименты
Все эксперименты логируются в MLflow:
- Гиперпараметры
- Метрики качества
- Чекпоинты моделей
- Анализ результатов

### TensorBoard
Визуализация в реальном времени:
- Loss функции
- Attention alignment
- Gradient norms
- Learning rate

## 🔧 Конфигурация

### Smart Tuner конфигурация
```yaml
# smart_tuner/config.yaml
telegram:
  enabled: true
  bot_token: "YOUR_BOT_TOKEN"
  chat_id: "YOUR_CHAT_ID"

optimization:
  n_trials: 20
  objective_metric: "composite_tts_score"
  
quality_control:
  attention_diagonality_threshold: 0.3
  gate_accuracy_threshold: 0.7
  gradient_norm_threshold: 10.0
```

### Гиперпараметры
Основные гиперпараметры настраиваются в `hparams.py`:
- `learning_rate: 1e-4` — консервативный learning rate
- `guide_loss_weight: 2.5` — вес guided attention loss
- `grad_clip_thresh: 1.0` — gradient clipping
- `batch_size: 32` — размер батча

## 📈 Архитектура системы

### EnhancedTacotronTrainer
```python
class EnhancedTacotronTrainer:
    def __init__(self, hparams, dataset_info):
        # Инициализация всех компонентов
        self.smart_tuner = SmartTunerIntegration()
        self.telegram_monitor = TelegramMonitorEnhanced()
        self.debug_reporter = DebugReporter()
    
    def train_step(self, batch):
        # Один шаг обучения с анализом качества
        # Автоматические исправления проблем
    
    def train(self, train_loader, val_loader):
        # Полный цикл обучения с мониторингом
```

### Фазы обучения
1. **Pre-alignment** (0-500 эпох) — формирование attention
2. **Alignment learning** (500-2000 эпох) — стабилизация attention
3. **Quality optimization** (2000-3000 эпох) — улучшение качества
4. **Fine-tuning** (3000+ эпох) — финальная полировка

## 🛠 Устранение неполадок

### Проблемы с attention alignment
```bash
# Увеличить guided attention weight
python train_enhanced.py --hparams "guide_loss_weight=5.0"
```

### Проблемы с градиентами
```bash
# Уменьшить learning rate
python train_enhanced.py --learning_rate 5e-5
```

### Проблемы с памятью
```bash
# Уменьшить batch size
python train_enhanced.py --batch_size 8
```

## 📊 Метрики качества

### Ключевые метрики
- **Attention Diagonality** — качество выравнивания (>0.7)
- **Gate Accuracy** — точность gate (>0.8)
- **Gradient Norm** — стабильность градиентов (<10.0)
- **Loss** — общий loss (<1.0)

### Автоматические исправления
Система автоматически:
- Увеличивает guided attention weight при плохом alignment
- Снижает learning rate при высоких градиентах
- Адаптирует gate threshold при преждевременной остановке
- Перезапускает обучение при критических проблемах

## 🔄 Миграция с legacy train.py

### Рекомендуемый переход
1. **Используйте `train_enhanced.py`** для новых экспериментов
2. **Legacy `train.py`** — только для отладки и совместимости
3. **Постепенно мигрируйте** существующие пайплайны

### Совместимость
- Все существующие чекпоинты совместимы
- Гиперпараметры из `hparams.py` поддерживаются
- DataLoader'ы работают без изменений

## 🎯 Лучшие практики

### Для стабильного обучения
1. Начните с консервативных гиперпараметров
2. Используйте guided attention с высоким весом
3. Мониторьте метрики через Telegram/MLflow
4. Дайте системе время на автоматическую оптимизацию

### Для быстрого прототипирования
1. Используйте режим `--mode test` для проверки
2. Начните с малого количества эпох
3. Включите Telegram уведомления
4. Анализируйте MLflow эксперименты

## 📞 Поддержка

При возникновении проблем:
1. Проверьте логи в `logs/`
2. Анализируйте MLflow эксперименты
3. Проверьте Telegram уведомления
4. Используйте `--mode test` для диагностики

---

**EnhancedTacotronTrainer** — современное решение для обучения TTS с максимальным качеством и минимальным вмешательством пользователя! 🎉 