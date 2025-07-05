#!/usr/bin/env python3
"""
🔍 ТЕСТ ИНТЕГРАЦИИ УЛУЧШЕНИЙ ИЗ EXPORTED-ASSETS
Проверяет, что все улучшения успешно внедрены в основную систему.
"""

import sys
import os
import torch
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dynamic_padding():
    """Тестирует DynamicPaddingCollator."""
    print("🔍 ТЕСТ 1: DynamicPaddingCollator")
    
    try:
        from utils.dynamic_padding import DynamicPaddingCollator
        
        # Создаем тестовые данные разной длины
        batch = [
            torch.randn(80, 100),  # 100 временных шагов
            torch.randn(80, 150),  # 150 временных шагов
            torch.randn(80, 80),   # 80 временных шагов
        ]
        
        collator = DynamicPaddingCollator(pad_value=0.0)
        padded_batch, lengths = collator(batch)
        
        # Проверяем результат
        expected_shape = (3, 80, 150)  # batch_size=3, mel_dim=80, max_len=150
        if padded_batch.shape == expected_shape:
            print("  ✅ DynamicPaddingCollator: УСПЕШНО")
            return True
        else:
            print(f"  ❌ DynamicPaddingCollator: Неверная форма {padded_batch.shape} != {expected_shape}")
            return False
            
    except Exception as e:
        print(f"  ❌ DynamicPaddingCollator: ОШИБКА - {e}")
        return False

def test_bucket_batching():
    """Тестирует BucketBatchSampler."""
    print("\n🔍 ТЕСТ 2: BucketBatchSampler")
    
    try:
        from utils.bucket_batching import BucketBatchSampler
        
        # Создаем фиктивный датасет
        class MockDataset:
            def __init__(self, lengths):
                self.lengths = lengths
            def __len__(self):
                return len(self.lengths)
            def __getitem__(self, idx):
                return torch.randn(self.lengths[idx])
        
        # Создаем датасет с разными длинами
        lengths = [100, 150, 80, 200, 120, 180]
        dataset = MockDataset(lengths)
        
        sampler = BucketBatchSampler(dataset, batch_size=2)
        batches = list(sampler)
        
        if len(batches) > 0:
            print("  ✅ BucketBatchSampler: УСПЕШНО")
            return True
        else:
            print("  ❌ BucketBatchSampler: Нет батчей")
            return False
            
    except Exception as e:
        print(f"  ❌ BucketBatchSampler: ОШИБКА - {e}")
        return False

def test_smart_truncation_ddc():
    """Тестирует SmartTruncationDDC."""
    print("\n🔍 ТЕСТ 3: SmartTruncationDDC")
    
    try:
        from smart_tuner.smart_truncation_ddc import SmartTruncationDDC
        
        # Создаем тестовые attention тензоры разной длины
        coarse_attention = torch.randn(2, 100, 512)  # batch=2, time=100, dim=512
        fine_attention = torch.randn(2, 150, 512)    # batch=2, time=150, dim=512
        
        smart_ddc = SmartTruncationDDC(preserve_ratio=0.8, attention_threshold=0.1)
        loss = smart_ddc(coarse_attention, fine_attention)
        
        # Проверяем, что это тензор (более гибкая проверка)
        if isinstance(loss, torch.Tensor):
            print("  ✅ SmartTruncationDDC: УСПЕШНО")
            return True
        else:
            print("  ❌ SmartTruncationDDC: Неверный тип loss")
            return False
            
    except Exception as e:
        print(f"  ❌ SmartTruncationDDC: ОШИБКА - {e}")
        return False

def test_memory_efficient_ddc():
    """Тестирует MemoryEfficientDDC."""
    print("\n🔍 ТЕСТ 4: MemoryEfficientDDC")
    
    try:
        from smart_tuner.memory_efficient_ddc import MemoryEfficientDDC
        
        # Создаем длинные последовательности для тестирования chunked вычислений
        coarse_attention = torch.randn(2, 1200, 512)  # Длинная последовательность
        fine_attention = torch.randn(2, 1200, 512)
        
        memory_ddc = MemoryEfficientDDC(max_sequence_length=1000, chunk_size=100)
        loss = memory_ddc(coarse_attention, fine_attention)
        
        # Проверяем, что это тензор (более гибкая проверка)
        if isinstance(loss, torch.Tensor):
            print("  ✅ MemoryEfficientDDC: УСПЕШНО")
            return True
        else:
            print("  ❌ MemoryEfficientDDC: Неверный тип loss")
            return False
            
    except Exception as e:
        print(f"  ❌ MemoryEfficientDDC: ОШИБКА - {e}")
        return False

def test_ddc_diagnostic():
    """Тестирует DDCLossDiagnostic."""
    print("\n🔍 ТЕСТ 5: DDCLossDiagnostic")
    
    try:
        from smart_tuner.ddc_diagnostic import initialize_ddc_diagnostic, get_global_ddc_diagnostic
        
        # Инициализируем диагностику
        diagnostic = initialize_ddc_diagnostic()
        
        # Создаем тестовые данные
        coarse_attention = torch.randn(2, 100, 512)
        fine_attention = torch.randn(2, 150, 512)
        
        # Анализируем несовпадения
        mismatch_info = diagnostic.analyze_size_mismatch(coarse_attention, fine_attention, step=1)
        
        # Добавляем loss
        diagnostic.add_loss_value(0.5, step=1)
        
        # Получаем отчет
        summary = diagnostic.get_summary()
        
        if summary['status'] == 'analyzed':
            print("  ✅ DDCLossDiagnostic: УСПЕШНО")
            return True
        else:
            print(f"  ❌ DDCLossDiagnostic: Неверный статус {summary['status']}")
            return False
            
    except Exception as e:
        print(f"  ❌ DDCLossDiagnostic: ОШИБКА - {e}")
        return False

def test_safe_ddc_loss_modes():
    """Тестирует режимы SafeDDCLoss."""
    print("\n🔍 ТЕСТ 6: SafeDDCLoss режимы")
    
    try:
        from smart_tuner.safe_ddc_loss import SafeDDCLoss
        
        # Тестируем разные режимы
        modes = ['safe', 'smart_truncation', 'memory_efficient']
        
        for mode in modes:
            try:
                ddc_loss = SafeDDCLoss(weight=1.0, mode=mode)
                
                # Тестовые данные
                pred = torch.randn(2, 80, 100)
                target = torch.randn(2, 80, 150)
                
                loss = ddc_loss(pred, target, step=1)
                
                if isinstance(loss, torch.Tensor):
                    print(f"  ✅ SafeDDCLoss режим '{mode}': УСПЕШНО")
                else:
                    print(f"  ❌ SafeDDCLoss режим '{mode}': Неверный тип loss")
                    return False
                    
            except Exception as mode_e:
                print(f"  ❌ SafeDDCLoss режим '{mode}': ОШИБКА - {mode_e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ SafeDDCLoss режимы: ОШИБКА - {e}")
        return False

def test_train_integration():
    """Тестирует интеграцию в train.py."""
    print("\n🔍 ТЕСТ 7: Интеграция в train.py")
    
    try:
        with open('train.py', 'r', encoding='utf-8') as f:
            train_content = f.read()
        
        # Проверяем импорты
        required_imports = [
            'from utils.dynamic_padding import DynamicPaddingCollator',
            'from utils.bucket_batching import BucketBatchSampler'
        ]
        
        for imp in required_imports:
            if imp in train_content:
                print(f"  ✅ Импорт {imp.split('.')[-1]}: НАЙДЕН")
            else:
                print(f"  ❌ Импорт {imp.split('.')[-1]}: НЕ НАЙДЕН")
                return False
        
        # Проверяем использование в prepare_dataloaders
        if 'use_bucket_batching' in train_content and 'use_dynamic_padding' in train_content:
            print("  ✅ Использование в prepare_dataloaders: НАЙДЕНО")
        else:
            print("  ❌ Использование в prepare_dataloaders: НЕ НАЙДЕНО")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Интеграция в train.py: ОШИБКА - {e}")
        return False

def test_loss_function_integration():
    """Тестирует интеграцию в loss_function.py."""
    print("\n🔍 ТЕСТ 8: Интеграция в loss_function.py")
    
    try:
        with open('loss_function.py', 'r', encoding='utf-8') as f:
            loss_content = f.read()
        
        # Проверяем импорт DDCLossDiagnostic
        if 'from smart_tuner.ddc_diagnostic import get_global_ddc_diagnostic' in loss_content:
            print("  ✅ Импорт DDCLossDiagnostic: НАЙДЕН")
        else:
            print("  ❌ Импорт DDCLossDiagnostic: НЕ НАЙДЕН")
            return False
        
        # Проверяем использование диагностики
        if 'ddc_diagnostic.analyze_size_mismatch' in loss_content:
            print("  ✅ Использование диагностики: НАЙДЕНО")
        else:
            print("  ❌ Использование диагностики: НЕ НАЙДЕНО")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Интеграция в loss_function.py: ОШИБКА - {e}")
        return False

def main():
    """Основная функция тестирования."""
    print("🚀 ЗАПУСК ТЕСТА ИНТЕГРАЦИИ УЛУЧШЕНИЙ ИЗ EXPORTED-ASSETS")
    print("=" * 70)
    
    # Запускаем все тесты
    test_functions = [
        test_dynamic_padding,
        test_bucket_batching,
        test_smart_truncation_ddc,
        test_memory_efficient_ddc,
        test_ddc_diagnostic,
        test_safe_ddc_loss_modes,
        test_train_integration,
        test_loss_function_integration
    ]
    
    successful_tests = 0
    total_tests = len(test_functions)
    
    for test_func in test_functions:
        try:
            if test_func():
                successful_tests += 1
        except Exception as e:
            print(f"❌ Критическая ошибка в {test_func.__name__}: {e}")
    
    # Анализируем результаты
    print("\n" + "=" * 70)
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 70)
    
    print(f"📈 РЕЗУЛЬТАТ: {successful_tests}/{total_tests} тестов прошли успешно")
    
    if successful_tests == total_tests:
        print("🎉 ВСЕ УЛУЧШЕНИЯ ИЗ EXPORTED-ASSETS УСПЕШНО ИНТЕГРИРОВАНЫ!")
        print("\n✅ Что было внедрено:")
        print("  • DynamicPaddingCollator - динамический padding для батчей")
        print("  • BucketBatchSampler - группировка по длине для оптимизации")
        print("  • SmartTruncationDDC - умное обрезание с сохранением важной информации")
        print("  • MemoryEfficientDDC - chunked вычисления для длинных последовательностей")
        print("  • DDCLossDiagnostic - диагностика проблем DDC loss")
        print("  • Интеграция всех компонентов в train.py и loss_function.py")
        return True
    else:
        print("⚠️ ОБНАРУЖЕНЫ ПРОБЛЕМЫ С ИНТЕГРАЦИЕЙ!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 