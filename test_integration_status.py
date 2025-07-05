#!/usr/bin/env python3
"""
🔍 ТЕСТ ИНТЕГРАЦИИ SMART TUNER V2
Проверяет статус всех компонентов и их интеграцию в процесс обучения.
"""

import sys
import os
import torch
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_smart_tuner_imports():
    """Тестирует импорт всех компонентов Smart Tuner."""
    print("🔍 ТЕСТ 1: Импорт компонентов Smart Tuner")
    
    components = {
        'gradient_clipper': 'smart_tuner.gradient_clipper',
        'smart_lr_adapter': 'smart_tuner.smart_lr_adapter', 
        'safe_ddc_loss': 'smart_tuner.safe_ddc_loss',
        'integration_manager': 'smart_tuner.integration_manager'
    }
    
    results = {}
    for name, module in components.items():
        try:
            __import__(module)
            results[name] = "✅ УСПЕШНО"
            print(f"  {name}: ✅ УСПЕШНО")
        except ImportError as e:
            results[name] = f"❌ ОШИБКА: {e}"
            print(f"  {name}: ❌ ОШИБКА: {e}")
    
    return results

def test_component_initialization():
    """Тестирует инициализацию компонентов."""
    print("\n🔍 ТЕСТ 2: Инициализация компонентов")
    
    results = {}
    
    # Тест Gradient Clipper
    try:
        from smart_tuner.gradient_clipper import AdaptiveGradientClipper, get_global_clipper, set_global_clipper
        
        clipper = AdaptiveGradientClipper(max_norm=1.0, adaptive=True)
        set_global_clipper(clipper)
        retrieved_clipper = get_global_clipper()
        
        if retrieved_clipper is not None:
            results['gradient_clipper'] = "✅ УСПЕШНО"
            print("  gradient_clipper: ✅ УСПЕШНО")
        else:
            results['gradient_clipper'] = "❌ ОШИБКА: Не удалось получить clipper"
            print("  gradient_clipper: ❌ ОШИБКА: Не удалось получить clipper")
            
    except Exception as e:
        results['gradient_clipper'] = f"❌ ОШИБКА: {e}"
        print(f"  gradient_clipper: ❌ ОШИБКА: {e}")
    
    # Тест Smart LR Adapter
    try:
        from smart_tuner.smart_lr_adapter import SmartLRAdapter, get_global_lr_adapter, set_global_lr_adapter
        
        # Создаем фиктивный оптимизатор
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        lr_adapter = SmartLRAdapter(optimizer=optimizer, patience=10, factor=0.5)
        set_global_lr_adapter(lr_adapter)
        retrieved_adapter = get_global_lr_adapter()
        
        if retrieved_adapter is not None:
            results['smart_lr_adapter'] = "✅ УСПЕШНО"
            print("  smart_lr_adapter: ✅ УСПЕШНО")
        else:
            results['smart_lr_adapter'] = "❌ ОШИБКА: Не удалось получить adapter"
            print("  smart_lr_adapter: ❌ ОШИБКА: Не удалось получить adapter")
            
    except Exception as e:
        results['smart_lr_adapter'] = f"❌ ОШИБКА: {e}"
        print(f"  smart_lr_adapter: ❌ ОШИБКА: {e}")
    
    # Тест Safe DDC Loss
    try:
        from smart_tuner.safe_ddc_loss import SafeDDCLoss, get_global_ddc_loss, set_global_ddc_loss
        
        ddc_loss = SafeDDCLoss(weight=1.0, use_masking=True)
        set_global_ddc_loss(ddc_loss)
        retrieved_ddc = get_global_ddc_loss()
        
        if retrieved_ddc is not None:
            results['safe_ddc_loss'] = "✅ УСПЕШНО"
            print("  safe_ddc_loss: ✅ УСПЕШНО")
        else:
            results['safe_ddc_loss'] = "❌ ОШИБКА: Не удалось получить DDC loss"
            print("  safe_ddc_loss: ❌ ОШИБКА: Не удалось получить DDC loss")
            
    except Exception as e:
        results['safe_ddc_loss'] = f"❌ ОШИБКА: {e}"
        print(f"  safe_ddc_loss: ❌ ОШИБКА: {e}")
    
    # Тест Integration Manager
    try:
        from smart_tuner.integration_manager import initialize_smart_tuner
        
        manager = initialize_smart_tuner()
        
        if manager is not None:
            results['integration_manager'] = "✅ УСПЕШНО"
            print("  integration_manager: ✅ УСПЕШНО")
        else:
            results['integration_manager'] = "❌ ОШИБКА: Не удалось получить manager"
            print("  integration_manager: ❌ ОШИБКА: Не удалось получить manager")
            
    except Exception as e:
        results['integration_manager'] = f"❌ ОШИБКА: {e}"
        print(f"  integration_manager: ❌ ОШИБКА: {e}")
    
    return results

def test_train_integration():
    """Тестирует интеграцию в train.py."""
    print("\n🔍 ТЕСТ 3: Интеграция в train.py")
    
    results = {}
    
    # Проверяем наличие импортов в train.py
    try:
        with open('train.py', 'r', encoding='utf-8') as f:
            train_content = f.read()
        
        required_imports = [
            'from smart_tuner.integration_manager import initialize_smart_tuner',
            'from smart_tuner.gradient_clipper import get_global_clipper, AdaptiveGradientClipper',
            'from smart_tuner.smart_lr_adapter import get_global_lr_adapter, SmartLRAdapter',
            'from smart_tuner.safe_ddc_loss import get_global_ddc_loss, SafeDDCLoss'
        ]
        
        for imp in required_imports:
            if imp in train_content:
                print(f"  {imp}: ✅ НАЙДЕН")
            else:
                print(f"  {imp}: ❌ НЕ НАЙДЕН")
                results['train_imports'] = "❌ ОШИБКА: Не все импорты найдены"
        
        # Проверяем наличие инициализации IntegrationManager
        if 'integration_manager = initialize_smart_tuner()' in train_content:
            print("  Инициализация IntegrationManager: ✅ НАЙДЕНА")
        else:
            print("  Инициализация IntegrationManager: ❌ НЕ НАЙДЕНА")
            results['train_init'] = "❌ ОШИБКА: Инициализация не найдена"
        
        # Проверяем наличие вызова step
        if 'integration_manager.step(' in train_content:
            print("  Вызов integration_manager.step(): ✅ НАЙДЕН")
        else:
            print("  Вызов integration_manager.step(): ❌ НЕ НАЙДЕН")
            results['train_step'] = "❌ ОШИБКА: Вызов step не найден"
        
        if 'results' not in results:
            results['train_integration'] = "✅ УСПЕШНО"
            
    except Exception as e:
        results['train_integration'] = f"❌ ОШИБКА: {e}"
        print(f"  train_integration: ❌ ОШИБКА: {e}")
    
    return results

def test_loss_function_integration():
    """Тестирует интеграцию в loss_function.py."""
    print("\n🔍 ТЕСТ 4: Интеграция в loss_function.py")
    
    results = {}
    
    try:
        with open('loss_function.py', 'r', encoding='utf-8') as f:
            loss_content = f.read()
        
        # Проверяем наличие импортов SafeDDCLoss
        if 'from smart_tuner.safe_ddc_loss import get_global_ddc_loss, SafeDDCLoss' in loss_content:
            print("  Импорт SafeDDCLoss: ✅ НАЙДЕН")
        else:
            print("  Импорт SafeDDCLoss: ❌ НЕ НАЙДЕН")
            results['loss_imports'] = "❌ ОШИБКА: Импорт SafeDDCLoss не найден"
        
        # Проверяем наличие использования SafeDDCLoss
        if 'get_global_ddc_loss()' in loss_content:
            print("  Использование get_global_ddc_loss(): ✅ НАЙДЕНО")
        else:
            print("  Использование get_global_ddc_loss(): ❌ НЕ НАЙДЕНО")
            results['loss_usage'] = "❌ ОШИБКА: Использование SafeDDCLoss не найдено"
        
        if 'results' not in results:
            results['loss_integration'] = "✅ УСПЕШНО"
            
    except Exception as e:
        results['loss_integration'] = f"❌ ОШИБКА: {e}"
        print(f"  loss_integration: ❌ ОШИБКА: {e}")
    
    return results

def test_component_functionality():
    """Тестирует функциональность компонентов."""
    print("\n🔍 ТЕСТ 5: Функциональность компонентов")
    
    results = {}
    
    # Тест Gradient Clipper
    try:
        from smart_tuner.gradient_clipper import AdaptiveGradientClipper
        
        model = torch.nn.Linear(10, 1)
        clipper = AdaptiveGradientClipper(max_norm=1.0, adaptive=True)
        
        # Создаем фиктивные градиенты
        model.weight.grad = torch.randn_like(model.weight) * 10
        model.bias.grad = torch.randn_like(model.bias) * 10
        
        was_clipped, grad_norm, clip_threshold = clipper.clip_gradients(model, step=1)
        
        if isinstance(was_clipped, bool) and isinstance(grad_norm, float):
            results['gradient_clipper_func'] = "✅ УСПЕШНО"
            print("  gradient_clipper функциональность: ✅ УСПЕШНО")
        else:
            results['gradient_clipper_func'] = "❌ ОШИБКА: Неверный тип возвращаемых значений"
            print("  gradient_clipper функциональность: ❌ ОШИБКА: Неверный тип возвращаемых значений")
            
    except Exception as e:
        results['gradient_clipper_func'] = f"❌ ОШИБКА: {e}"
        print(f"  gradient_clipper функциональность: ❌ ОШИБКА: {e}")
    
    # Тест Smart LR Adapter
    try:
        from smart_tuner.smart_lr_adapter import SmartLRAdapter
        
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        lr_adapter = SmartLRAdapter(optimizer=optimizer, patience=10, factor=0.5)
        
        # Тестируем step
        lr_changed = lr_adapter.step(loss=1.0, grad_norm=5.0, step=1)
        
        if isinstance(lr_changed, bool):
            results['lr_adapter_func'] = "✅ УСПЕШНО"
            print("  lr_adapter функциональность: ✅ УСПЕШНО")
        else:
            results['lr_adapter_func'] = "❌ ОШИБКА: Неверный тип возвращаемого значения"
            print("  lr_adapter функциональность: ❌ ОШИБКА: Неверный тип возвращаемого значения")
            
    except Exception as e:
        results['lr_adapter_func'] = f"❌ ОШИБКА: {e}"
        print(f"  lr_adapter функциональность: ❌ ОШИБКА: {e}")
    
    # Тест Safe DDC Loss
    try:
        from smart_tuner.safe_ddc_loss import SafeDDCLoss
        
        ddc_loss = SafeDDCLoss(weight=1.0, use_masking=True)
        
        # Создаем фиктивные тензоры
        mel1 = torch.randn(2, 80, 100)
        mel2 = torch.randn(2, 80, 100)
        
        loss_value = ddc_loss(mel1, mel2, step=1)
        
        if isinstance(loss_value, torch.Tensor):
            results['ddc_loss_func'] = "✅ УСПЕШНО"
            print("  ddc_loss функциональность: ✅ УСПЕШНО")
        else:
            results['ddc_loss_func'] = "❌ ОШИБКА: Неверный тип возвращаемого значения"
            print("  ddc_loss функциональность: ❌ ОШИБКА: Неверный тип возвращаемого значения")
            
    except Exception as e:
        results['ddc_loss_func'] = f"❌ ОШИБКА: {e}"
        print(f"  ddc_loss функциональность: ❌ ОШИБКА: {e}")
    
    return results

def main():
    """Основная функция тестирования."""
    print("🚀 ЗАПУСК ТЕСТА ИНТЕГРАЦИИ SMART TUNER V2")
    print("=" * 60)
    
    # Запускаем все тесты
    test_results = {}
    
    test_results.update(test_smart_tuner_imports())
    test_results.update(test_component_initialization())
    test_results.update(test_train_integration())
    test_results.update(test_loss_function_integration())
    test_results.update(test_component_functionality())
    
    # Анализируем результаты
    print("\n" + "=" * 60)
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 60)
    
    successful_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        if "✅ УСПЕШНО" in result:
            successful_tests += 1
            print(f"✅ {test_name}: {result}")
        else:
            print(f"❌ {test_name}: {result}")
    
    print(f"\n📈 РЕЗУЛЬТАТ: {successful_tests}/{total_tests} тестов прошли успешно")
    
    if successful_tests == total_tests:
        print("🎉 ВСЕ КОМПОНЕНТЫ SMART TUNER УСПЕШНО ИНТЕГРИРОВАНЫ!")
        return True
    else:
        print("⚠️ ОБНАРУЖЕНЫ ПРОБЛЕМЫ С ИНТЕГРАЦИЕЙ!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 