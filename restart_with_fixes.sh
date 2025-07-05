#!/bin/bash
# 🚨 СКРИПТ ПЕРЕЗАПУСКА С КРИТИЧЕСКИМИ ИСПРАВЛЕНИЯМИ

echo "🚨 ПЕРЕЗАПУСК ОБУЧЕНИЯ С КРИТИЧЕСКИМИ ИСПРАВЛЕНИЯМИ"
echo "=================================================="

# Останавливаем текущее обучение если запущено
pkill -f "python train.py" || true
sleep 2

# Очищаем кэш CUDA
python -c "import torch; torch.cuda.empty_cache()" || true

# Запускаем обучение с новыми параметрами
echo "🎯 Запуск обучения с критическими исправлениями..."
python train.py \
    --output_directory=outdir \
    --log_directory=logdir \
    --hparams="learning_rate=1e-4,guided_attention_weight=5.0" \
    --n_gpus=1 \
    --batch_size=8 \
    --epochs=1000

echo "✅ Обучение перезапущено с критическими исправлениями"
