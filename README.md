# 🚀 Tacotron2-New с EnhancedTacotronTrainer

Современная система обучения TTS с полной интеграцией Smart Tuner V2, автоматическим контролем качества и интеллектуальным управлением процессом обучения.

## 🎯 Быстрый старт

### 🚀 **Рекомендуемый способ запуска (EnhancedTacotronTrainer):**
```bash
# Основной entrypoint для обучения
python train_enhanced.py --mode train --epochs 100

# Тестовый запуск
python train_enhanced.py --mode test

# Валидация модели
python train_enhanced.py --mode validate
```

### 📚 **Legacy способ (только для отладки):**
```bash
python train.py --hparams "learning_rate=1e-4,guide_loss_weight=2.5,grad_clip_thresh=1.0"
```

## 🆕 Что нового

### ✅ **EnhancedTacotronTrainer** — основной движок обучения
- **Автоматическая оптимизация** гиперпараметров через Smart Tuner
- **Интеллектуальный контроль качества** в реальном времени
- **Telegram уведомления** о прогрессе и проблемах
- **MLflow логирование** экспериментов и метрик
- **Фазовое обучение** — адаптация стратегии по мере прогресса

### 🧠 **Smart Tuner V2** — автоматизация обучения
- **Оптимизация гиперпараметров** на основе качества TTS
- **Раннее обнаружение проблем** с автоматическими исправлениями
- **Guided Attention** — улучшенное выравнивание attention
- **Quality Control** — постоянный мониторинг качества

### 📊 **Мониторинг и диагностика**
- **Telegram бот** — уведомления о прогрессе
- **MLflow** — логирование экспериментов
- **TensorBoard** — визуализация в реальном времени
- **Alignment Diagnostics** — мониторинг attention alignment

## 🔧 Установка

```bash
# Клонирование репозитория
git clone <repository-url>
cd Tacotron2-New

# Установка зависимостей
pip install -r requirements.txt

# Настройка конфигурации
cp smart_tuner/config.yaml.example smart_tuner/config.yaml
# Отредактируйте config.yaml с вашими настройками
```

## 📖 Документация

- **[EnhancedTacotronTrainer Guide](README_ENHANCED_TRAINING.md)** — подробное руководство по новому движку обучения
- **[Smart Tuner Documentation](smart_tuner/README.md)** — документация по Smart Tuner V2
- **[Technical Specification](TECHNICAL_SPECIFICATION_FULL_INTEGRATION.md)** — техническое задание проекта

## 🎯 Основные возможности

### 🚀 **Современное обучение**
- **EnhancedTacotronTrainer** — основной движок с автоматизацией
- **Smart Tuner V2** — интеллектуальная оптимизация
- **Quality Control** — автоматический контроль качества
- **Phase-based Training** — адаптивное обучение по фазам

### 📊 **Мониторинг и логирование**
- **Telegram уведомления** — в реальном времени
- **MLflow эксперименты** — полное логирование
- **TensorBoard** — визуализация метрик
- **Alignment Diagnostics** — мониторинг attention

### 🔧 **Автоматические исправления**
- **Gradient Clipping** — стабилизация градиентов
- **Guided Attention** — улучшение alignment
- **Learning Rate Adaptation** — адаптивная скорость обучения
- **Emergency Recovery** — автоматическое восстановление

## 📈 Метрики качества

### Ключевые показатели
- **Attention Diagonality** > 0.7 (качество выравнивания)
- **Gate Accuracy** > 0.8 (точность gate)
- **Gradient Norm** < 10.0 (стабильность градиентов)
- **Loss** < 1.0 (общий loss)

### Автоматические исправления
Система автоматически:
- Увеличивает guided attention weight при плохом alignment
- Снижает learning rate при высоких градиентах
- Адаптирует gate threshold при преждевременной остановке
- Перезапускает обучение при критических проблемах

## 🛠 Конфигурация

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
```python
# hparams.py
learning_rate = 1e-4          # Консервативный learning rate
guide_loss_weight = 2.5       # Вес guided attention loss
grad_clip_thresh = 1.0        # Gradient clipping
batch_size = 32               # Размер батча
```

## 🔄 Миграция

### Рекомендуемый переход
1. **Используйте `train_enhanced.py`** для новых экспериментов
2. **Legacy `train.py`** — только для отладки
3. **Постепенно мигрируйте** существующие пайплайны

### Совместимость
- ✅ Все существующие чекпоинты совместимы
- ✅ Гиперпараметры из `hparams.py` поддерживаются
- ✅ DataLoader'ы работают без изменений

## 🎯 Лучшие практики

### Для стабильного обучения
1. Начните с `train_enhanced.py --mode test`
2. Используйте консервативные гиперпараметры
3. Включите Telegram уведомления
4. Мониторьте MLflow эксперименты

### Для быстрого прототипирования
1. Используйте режим `--mode test`
2. Начните с малого количества эпох
3. Анализируйте качество через Smart Tuner
4. Используйте автоматические исправления

## 📞 Поддержка

При возникновении проблем:
1. Проверьте логи в `logs/`
2. Анализируйте MLflow эксперименты
3. Проверьте Telegram уведомления
4. Используйте `--mode test` для диагностики

## 🚀 Статус проекта

### ✅ **Завершено:**
- EnhancedTacotronTrainer — основной движок обучения
- Smart Tuner V2 — автоматическая оптимизация
- Telegram мониторинг — уведомления в реальном времени
- MLflow интеграция — логирование экспериментов
- Критические исправления — стабильность обучения

### 🔄 **В разработке:**
- Optimization Engine — полная интеграция
- Архитектурная оптимизация — консолидация компонентов
- Безопасность конфигурации — переменные окружения
- Документация — полное руководство пользователя

---

**Tacotron2-New** — современное решение для обучения TTS с максимальным качеством и минимальным вмешательством пользователя! 🎉 