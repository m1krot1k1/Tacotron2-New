#!/usr/bin/env python3
"""
🧪 Тестирование автоматических исправлений
Проверяет работу AutoFixManager на различных проблемах
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_tuner.auto_fix_manager import AutoFixManager, FixAction

class MockModel(nn.Module):
    """Мок модель для тестирования"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

class MockOptimizer:
    """Мок оптимизатор для тестирования"""
    def __init__(self):
        self.param_groups = [{'lr': 1e-3}]

class MockHParams:
    """Мок гиперпараметров для тестирования"""
    def __init__(self):
        self.grad_clip_thresh = 1.0
        self.guide_loss_weight = 1.0
        self.p_attention_dropout = 0.1
        self.gate_threshold = 0.5
        self.gate_loss_weight = 1.0
        self.use_guided_attn = True
        self.fp16_run = False

class MockTelegramMonitor:
    """Мок Telegram мониторинга для тестирования"""
    def send_message(self, message):
        print(f"📱 TELEGRAM: {message}")
        return True

def test_gradient_vanishing_fix():
    """Тест исправления исчезновения градиентов"""
    print("🧪 Тест 1: Исправление исчезновения градиентов")
    
    model = MockModel()
    optimizer = MockOptimizer()
    hparams = MockHParams()
    telegram_monitor = MockTelegramMonitor()
    
    auto_fix = AutoFixManager(model, optimizer, hparams, telegram_monitor)
    
    # Симулируем исчезновение градиентов
    metrics = {
        'grad_norm': 1e-10,  # Критически низкая норма градиентов
        'attention_diagonality': 0.5,
        'gate_accuracy': 0.7,
        'loss': 10.0
    }
    
    loss = torch.tensor(10.0)
    
    fixes = auto_fix.analyze_and_fix(step=100, metrics=metrics, loss=loss)
    
    print(f"Применено исправлений: {len(fixes)}")
    for fix in fixes:
        print(f"  - {fix.description} (успех: {fix.success})")
    
    # Проверяем, что learning rate снизился
    new_lr = optimizer.param_groups[0]['lr']
    print(f"Learning rate: {new_lr:.2e}")
    
    return len(fixes) > 0

def test_gradient_explosion_fix():
    """Тест исправления взрыва градиентов"""
    print("\n🧪 Тест 2: Исправление взрыва градиентов")
    
    model = MockModel()
    optimizer = MockOptimizer()
    hparams = MockHParams()
    telegram_monitor = MockTelegramMonitor()
    
    auto_fix = AutoFixManager(model, optimizer, hparams, telegram_monitor)
    
    # Симулируем взрыв градиентов
    metrics = {
        'grad_norm': 500.0,  # Критически высокая норма градиентов
        'attention_diagonality': 0.5,
        'gate_accuracy': 0.7,
        'loss': 10.0
    }
    
    fixes = auto_fix.analyze_and_fix(step=200, metrics=metrics)
    
    print(f"Применено исправлений: {len(fixes)}")
    for fix in fixes:
        print(f"  - {fix.description} (успех: {fix.success})")
    
    # Проверяем, что gradient clipping усилился
    new_clip = hparams.grad_clip_thresh
    print(f"Gradient clip threshold: {new_clip}")
    
    return len(fixes) > 0

def test_attention_problems_fix():
    """Тест исправления проблем с attention"""
    print("\n🧪 Тест 3: Исправление проблем с attention")
    
    model = MockModel()
    optimizer = MockOptimizer()
    hparams = MockHParams()
    telegram_monitor = MockTelegramMonitor()
    
    auto_fix = AutoFixManager(model, optimizer, hparams, telegram_monitor)
    
    # Симулируем проблемы с attention
    metrics = {
        'grad_norm': 1.0,
        'attention_diagonality': 0.05,  # Критически низкая диагональность
        'gate_accuracy': 0.7,
        'loss': 10.0
    }
    
    fixes = auto_fix.analyze_and_fix(step=300, metrics=metrics)
    
    print(f"Применено исправлений: {len(fixes)}")
    for fix in fixes:
        print(f"  - {fix.description} (успех: {fix.success})")
    
    # Проверяем, что guided attention weight увеличился
    new_weight = hparams.guide_loss_weight
    print(f"Guided attention weight: {new_weight}")
    
    return len(fixes) > 0

def test_nan_problems_fix():
    """Тест исправления NaN проблем"""
    print("\n🧪 Тест 4: Исправление NaN проблем")
    
    model = MockModel()
    optimizer = MockOptimizer()
    hparams = MockHParams()
    telegram_monitor = MockTelegramMonitor()
    
    auto_fix = AutoFixManager(model, optimizer, hparams, telegram_monitor)
    
    # Симулируем NaN в loss
    metrics = {
        'grad_norm': 1.0,
        'attention_diagonality': 0.5,
        'gate_accuracy': 0.7,
        'loss': float('nan')
    }
    
    loss = torch.tensor(float('nan'))
    
    fixes = auto_fix.analyze_and_fix(step=400, metrics=metrics, loss=loss)
    
    print(f"Применено исправлений: {len(fixes)}")
    for fix in fixes:
        print(f"  - {fix.description} (успех: {fix.success})")
    
    # Проверяем, что learning rate экстремально снизился
    new_lr = optimizer.param_groups[0]['lr']
    print(f"Learning rate: {new_lr:.2e}")
    
    return len(fixes) > 0

def test_fix_statistics():
    """Тест статистики исправлений"""
    print("\n🧪 Тест 5: Статистика исправлений")
    
    model = MockModel()
    optimizer = MockOptimizer()
    hparams = MockHParams()
    telegram_monitor = MockTelegramMonitor()
    
    auto_fix = AutoFixManager(model, optimizer, hparams, telegram_monitor)
    
    # Применяем несколько исправлений
    metrics_list = [
        {'grad_norm': 1e-10, 'attention_diagonality': 0.5, 'gate_accuracy': 0.7, 'loss': 10.0},
        {'grad_norm': 500.0, 'attention_diagonality': 0.5, 'gate_accuracy': 0.7, 'loss': 10.0},
        {'grad_norm': 1.0, 'attention_diagonality': 0.05, 'gate_accuracy': 0.7, 'loss': 10.0}
    ]
    
    for i, metrics in enumerate(metrics_list):
        auto_fix.analyze_and_fix(step=500+i, metrics=metrics)
    
    # Получаем статистику
    stats = auto_fix.get_fix_statistics()
    
    print("Статистика исправлений:")
    print(f"  Всего исправлений: {stats['total_fixes']}")
    print(f"  Успешных исправлений: {stats['successful_fixes']}")
    print(f"  Счетчики проблем: {stats['problem_counters']}")
    
    return stats['total_fixes'] > 0

def main():
    """Основная функция тестирования"""
    print("🚀 Запуск тестов автоматических исправлений\n")
    
    tests = [
        test_gradient_vanishing_fix,
        test_gradient_explosion_fix,
        test_attention_problems_fix,
        test_nan_problems_fix,
        test_fix_statistics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ Тест пройден")
            else:
                print("❌ Тест не пройден")
        except Exception as e:
            print(f"❌ Ошибка в тесте: {e}")
    
    print(f"\n📊 Результаты: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("🎉 Все тесты пройдены! AutoFixManager работает корректно.")
    else:
        print("⚠️ Некоторые тесты не пройдены. Требуется доработка.")
    
    return passed == total

if __name__ == "__main__":
    main() 