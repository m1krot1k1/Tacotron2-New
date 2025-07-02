#!/usr/bin/env python3
"""
🚨 ЭКСТРЕННОЕ ВОССТАНОВЛЕНИЕ ОБУЧЕНИЯ
Скрипт для критического восстановления после NaN/Inf и других проблем

Использование:
python emergency_recovery.py
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hparams import create_hparams
from smart_tuner_main import SmartTunerMain
from smart_tuner.alert_manager import AlertManager


def create_emergency_hparams():
    """
    🛡️ Создает УЛЬТРА-безопасные параметры для критического восстановления
    """
    hparams = create_hparams()
    
    # 🔥 РАДИКАЛЬНОЕ снижение параметров
    hparams.learning_rate = 1e-6          # Минимальный learning rate
    hparams.batch_size = 2                # Минимальный batch size
    hparams.grad_clip_thresh = 0.01       # Максимально строгое клипирование
    
    # 🎯 ПРИНУДИТЕЛЬНАЯ guided attention
    hparams.use_guided_attn = True
    hparams.guide_loss_weight = 50.0      # Максимальный вес guided attention
    hparams.guide_loss_initial_weight = 50.0
    
    # 🛡️ Максимальная стабильность
    hparams.p_attention_dropout = 0.001   # Минимальный dropout
    hparams.p_decoder_dropout = 0.001
    hparams.gate_threshold = 0.3          # Строгий gate threshold
    
    # 🚫 Отключение нестабильных функций
    hparams.use_mmi = False
    hparams.use_audio_quality_enhancement = False
    hparams.fp16_run = False              # Отключаем fp16 для стабильности
    
    # 📊 Увеличенные интервалы проверки
    hparams.iters_per_checkpoint = 100    # Частые чекпоинты
    hparams.validation_interval = 50      # Частая валидация
    
    print("🛡️ УЛЬТРА-БЕЗОПАСНЫЕ ПАРАМЕТРЫ СОЗДАНЫ:")
    print(f"  🔥 learning_rate: {hparams.learning_rate}")
    print(f"  📦 batch_size: {hparams.batch_size}")
    print(f"  🎯 guide_loss_weight: {hparams.guide_loss_weight}")
    print(f"  ✂️ grad_clip_thresh: {hparams.grad_clip_thresh}")
    print(f"  🛡️ use_guided_attn: {hparams.use_guided_attn}")
    print(f"  🚫 fp16_run: {hparams.fp16_run}")
    
    return hparams


def send_recovery_notification():
    """
    📱 Отправляет уведомление о начале экстренного восстановления
    """
    try:
        alert_manager = AlertManager()
        
        message = "🚨 **ЭКСТРЕННОЕ ВОССТАНОВЛЕНИЕ ЗАПУЩЕНО!**\n\n"
        message += "🛡️ **Активированы УЛЬТРА-безопасные параметры:**\n"
        message += "• 🔥 Learning rate: 1e-6 (минимальный)\n"
        message += "• 📦 Batch size: 2 (минимальный)\n"
        message += "• 🎯 Guided attention: 50.0 (максимальный)\n"
        message += "• ✂️ Grad clipping: 0.01 (строгий)\n"
        message += "• 🚫 FP16: отключен\n"
        message += "• 🛡️ Все нестабильные функции отключены\n\n"
        message += "🎯 **Цель:** Восстановить диагональность attention и стабилизировать loss\n"
        message += "⏰ **Время:** Может потребоваться больше времени для восстановления\n"
        message += "🚀 **Обучение стартует в безопасном режиме!**"
        
        alert_manager.send_message(message, priority='critical')
        print("📱 Уведомление о восстановлении отправлено")
        
    except Exception as e:
        print(f"⚠️ Не удалось отправить уведомление: {e}")


def main():
    """
    🚨 Основная функция экстренного восстановления
    """
    parser = argparse.ArgumentParser(description='Экстренное восстановление обучения')
    parser.add_argument('--checkpoint', type=str, help='Путь к последнему чекпоинту')
    parser.add_argument('--notify', action='store_true', help='Отправить Telegram уведомление')
    
    args = parser.parse_args()
    
    print("🚨" + "="*60)
    print("🚨 ЭКСТРЕННОЕ ВОССТАНОВЛЕНИЕ ОБУЧЕНИЯ TACOTRON2")
    print("🚨" + "="*60)
    print("🛡️ Активирую УЛЬТРА-безопасные параметры...")
    print("🎯 Цель: Восстановить attention диагональность и стабилизировать loss")
    print("⏰ Процесс может занять больше времени, но будет максимально стабильным")
    print("="*60)
    
    # Отправляем уведомление если запрошено
    if args.notify:
        send_recovery_notification()
    
    # Создаем ультра-безопасные параметры
    hparams = create_emergency_hparams()
    
    # Настраиваем чекпоинт если указан
    checkpoint_path = args.checkpoint
    if checkpoint_path and not os.path.exists(checkpoint_path):
        print(f"⚠️ Чекпоинт {checkpoint_path} не найден. Начинаю с нуля.")
        checkpoint_path = None
    
    try:
        # Инициализируем Smart Tuner
        smart_tuner = SmartTunerMain()
        smart_tuner.initialize_components()
        
        print("\n🚀 Запускаю экстренное восстановление...")
        print("💡 Система будет автоматически:")
        print("   - Мониторить диагональность attention")
        print("   - Адаптировать guided attention вес")
        print("   - Перезапускать при проблемах")
        print("   - Отправлять уведомления о прогрессе")
        
        # Запускаем обучение с ультра-безопасными параметрами
        results = smart_tuner.run_single_training(
            hyperparams=hparams.__dict__
        )
        
        print("\n✅ Экстренное восстановление завершено!")
        print(f"📊 Результаты: {results}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Экстренное восстановление остановлено пользователем")
        
    except Exception as e:
        print(f"\n❌ Ошибка при экстренном восстановлении: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        
        # Отправляем уведомление об ошибке
        try:
            alert_manager = AlertManager()
            error_message = f"❌ **ОШИБКА ЭКСТРЕННОГО ВОССТАНОВЛЕНИЯ**\n\n"
            error_message += f"🔥 **Ошибка:** {str(e)}\n"
            error_message += f"🛠️ **Рекомендация:** Проверить данные и настройки\n"
            error_message += f"📞 **Требуется:** Ручное вмешательство"
            
            alert_manager.send_message(error_message, priority='critical')
        except:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    main() 