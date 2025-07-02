#!/usr/bin/env python3
"""
🔍 Тест системы Debug Reporter
Проверяет функциональность сбора и отправки debug отчетов
"""

import numpy as np
import torch
from debug_reporter import initialize_debug_reporter

# Создаем мок telegram_monitor для тестирования
class MockTelegramMonitor:
    def __init__(self):
        self.bot_token = "test_token"
        self.chat_id = "test_chat"
        self.messages_sent = []
        self.files_sent = []
    
    def _send_text_message(self, text, parse_mode="Markdown"):
        self.messages_sent.append(text)
        print(f"📱 Mock Telegram: Отправлено сообщение ({len(text)} символов)")
        return True
    
    def _send_document(self, filename, caption):
        self.files_sent.append((filename, caption))
        print(f"📄 Mock Telegram: Отправлен файл {filename}")
        return True

def test_debug_reporter():
    print("🔍 Тестирование Debug Reporter...")
    
    # Инициализируем mock telegram
    mock_telegram = MockTelegramMonitor()
    
    # Инициализируем debug reporter
    debug_reporter = initialize_debug_reporter(mock_telegram)
    
    print("✅ Debug Reporter инициализирован")
    
    # Создаем тестовые данные
    for step in range(1, 1005):  # Тестируем до отправки отчета
        # Симулируем метрики
        loss = 2.5 + np.random.normal(0, 0.1)  # Нормальный loss
        if step > 500:  # Ухудшение после 500 шагов
            loss += 0.01 * (step - 500)
        
        metrics = {
            'loss': loss,
            'grad_norm': np.random.uniform(0.1, 2.0),
            'learning_rate': 1e-4,
            'batch_size': 32,
            'iteration': step,
            'epoch': step // 100,
            'diagonality': max(0, min(1, 0.8 + np.random.normal(0, 0.1))),
            'quality': max(0, min(1, 0.75 + np.random.normal(0, 0.05)))
        }
        
        # Симулируем y_pred с attention
        mel_len, text_len = 80, 20
        attention_matrix = np.random.rand(mel_len, text_len)
        # Делаем более диагональным
        for i in range(mel_len):
            j = int(i * text_len / mel_len)
            if j < text_len:
                attention_matrix[i, j] += 0.5
        
        # Создаем мок y_pred
        alignments = torch.tensor(attention_matrix).unsqueeze(0)  # Добавляем batch dim
        y_pred = [None, None, None, alignments]  # Mock y_pred
        
        # Loss компоненты
        loss_components = {
            'total_loss': loss,
            'mel_loss': loss * 0.7,
            'gate_loss': loss * 0.2,
            'guided_loss': loss * 0.1
        }
        
        # Mock hparams
        class MockHparams:
            learning_rate = 1e-4
            batch_size = 32
            use_guided_attn = True
            grad_clip_thresh = 1.0
            
        hparams = MockHparams()
        
        # Smart tuner decisions
        smart_tuner_decisions = {
            'quality_controller': {
                'active': True,
                'status': 'Мониторинг качества'
            },
            'recommendations': ['Стабильное обучение'],
            'warnings': ['Небольшое увеличение loss'] if step > 800 else []
        }
        
        # Отправляем данные в debug reporter
        debug_reporter.collect_step_data(
            step=step,
            metrics=metrics,
            model=None,  # Для теста не передаем модель
            y_pred=y_pred,
            loss_components=loss_components,
            hparams=hparams,
            smart_tuner_decisions=smart_tuner_decisions
        )
        
        if step % 200 == 0:
            print(f"🔄 Обработан шаг {step}, loss: {loss:.4f}")
    
    print("\n📊 Результаты тестирования:")
    print(f"  • Обработано шагов: 1004")
    print(f"  • Отправлено сообщений: {len(mock_telegram.messages_sent)}")
    print(f"  • Отправлено файлов: {len(mock_telegram.files_sent)}")
    
    if mock_telegram.files_sent:
        last_file = mock_telegram.files_sent[-1]
        print(f"  • Последний отчет: {last_file[0]}")
        print(f"  • Подпись: {last_file[1][:50]}...")
    
    print("\n✅ Тест завершен успешно!")

if __name__ == "__main__":
    test_debug_reporter() 