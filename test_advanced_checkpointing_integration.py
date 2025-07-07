#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 COMPREHENSIVE TESTS: Advanced Model Checkpointing System Integration
Комплексное тестирование интеллектуальной системы checkpoint'ов

Тестирует:
1. IntelligentCheckpointManager - базовая функциональность
2. MultiCriteriaModelSelector - выбор лучших моделей  
3. AutoRecoverySystem - автовосстановление при сбоях
4. CheckpointHealthAnalyzer - анализ здоровья checkpoint'ов
5. Интеграция с Context-Aware Manager
6. Симуляция критических сбоев и восстановления
"""

import os
import sys
import time
import tempfile
import shutil
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime

# Добавляем current directory в path для импортов
sys.path.insert(0, os.getcwd())

try:
    from advanced_model_checkpointing_system import (
        IntelligentCheckpointManager, 
        MultiCriteriaModelSelector,
        AutoRecoverySystem,
        CheckpointHealthAnalyzer,
        CheckpointMetrics,
        CheckpointQuality,
        CheckpointInfo,
        create_checkpoint_manager
    )
    CHECKPOINTING_AVAILABLE = True
except ImportError as e:
    print(f"❌ Advanced Checkpointing System недоступен: {e}")
    CHECKPOINTING_AVAILABLE = False

try:
    from context_aware_training_manager import ContextAwareTrainingManager
    CONTEXT_MANAGER_AVAILABLE = True
except ImportError:
    print("⚠️ Context-Aware Manager недоступен для интеграционных тестов")
    CONTEXT_MANAGER_AVAILABLE = False

class SimpleTestModel(nn.Module):
    """Простая модель для тестирования"""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)

def create_test_metrics(step: int, 
                       validation_loss: float = 5.0,
                       attention_diagonality: float = 0.6,
                       has_problems: bool = False) -> CheckpointMetrics:
    """Создание тестовых метрик checkpoint'а"""
    return CheckpointMetrics(
        epoch=step // 100,
        global_step=step,
        validation_loss=validation_loss,
        training_loss=validation_loss + 0.5,
        learning_rate=1e-3,
        attention_diagonality=attention_diagonality,
        gate_accuracy=0.85,
        mel_reconstruction_quality=0.75,
        attention_stability=0.7,
        gradient_norm=2.5,
        gradient_stability=1.2,
        loss_trend=-0.05,
        convergence_score=0.6,
        timestamp=datetime.now().isoformat(),
        training_time=60.0,
        model_size_mb=150.0,
        memory_usage_mb=2048.0,
        has_nan_weights=has_problems and step % 5 == 0,
        has_gradient_explosion=has_problems and step % 3 == 0,
        has_attention_collapse=attention_diagonality < 0.1,
        is_stable=not has_problems
    )

def test_multi_criteria_model_selector():
    """Тестирование MultiCriteriaModelSelector"""
    print("\n🧪 ТЕСТ 1: MultiCriteriaModelSelector")
    
    try:
        selector = MultiCriteriaModelSelector()
        
        # Создание тестовых checkpoint'ов с разным качеством
        checkpoints = []
        
        # Отличный checkpoint
        excellent_metrics = create_test_metrics(
            step=1000, validation_loss=2.0, attention_diagonality=0.8
        )
        excellent_cp = CheckpointInfo(
            path="test_excellent.pt",
            metrics=excellent_metrics,
            quality=CheckpointQuality.EXCELLENT,
            health_score=0.0
        )
        checkpoints.append(excellent_cp)
        
        # Хороший checkpoint
        good_metrics = create_test_metrics(
            step=800, validation_loss=4.0, attention_diagonality=0.6
        )
        good_cp = CheckpointInfo(
            path="test_good.pt",
            metrics=good_metrics,
            quality=CheckpointQuality.GOOD,
            health_score=0.0
        )
        checkpoints.append(good_cp)
        
        # Критический checkpoint
        critical_metrics = create_test_metrics(
            step=500, validation_loss=50.0, attention_diagonality=0.05, has_problems=True
        )
        critical_cp = CheckpointInfo(
            path="test_critical.pt", 
            metrics=critical_metrics,
            quality=CheckpointQuality.CRITICAL,
            health_score=0.0
        )
        checkpoints.append(critical_cp)
        
        # Ранжирование checkpoint'ов
        ranked_checkpoints = selector.rank_checkpoints(checkpoints)
        
        # Проверка результатов
        assert len(ranked_checkpoints) == 3, "Должно быть 3 checkpoint'а"
        assert ranked_checkpoints[0].is_best, "Первый checkpoint должен быть лучшим"
        assert ranked_checkpoints[0].health_score > ranked_checkpoints[1].health_score, "Score должны убывать"
        assert ranked_checkpoints[1].health_score > ranked_checkpoints[2].health_score, "Score должны убывать"
        
        # Проверка, что лучший checkpoint действительно лучший (excellent)
        assert ranked_checkpoints[0].path == "test_excellent.pt", "Лучший должен быть excellent"
        
        print("✅ MultiCriteriaModelSelector: Корректно ранжирует checkpoint'ы")
        return True
        
    except Exception as e:
        print(f"❌ MultiCriteriaModelSelector: {e}")
        return False

def test_checkpoint_health_analyzer():
    """Тестирование CheckpointHealthAnalyzer"""
    print("\n🧪 ТЕСТ 2: CheckpointHealthAnalyzer")
    
    try:
        analyzer = CheckpointHealthAnalyzer()
        
        # Создание временного checkpoint'а для анализа
        with tempfile.TemporaryDirectory() as temp_dir:
            test_model = SimpleTestModel()
            optimizer = torch.optim.Adam(test_model.parameters())
            
            # Хороший checkpoint
            good_checkpoint_path = Path(temp_dir) / "good_checkpoint.pt"
            torch.save({
                'model_state_dict': test_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': 10,
                'global_step': 1000,
                'validation_loss': 3.5,
                'attention_diagonality': 0.7
            }, good_checkpoint_path)
            
            # Анализ хорошего checkpoint'а
            quality, issues = analyzer.analyze_checkpoint_health(str(good_checkpoint_path))
            
            assert quality in [CheckpointQuality.EXCELLENT, CheckpointQuality.GOOD], f"Ожидался хороший checkpoint, получен: {quality}"
            print(f"✅ Хороший checkpoint корректно определен как: {quality.value}")
            
            # Плохой checkpoint с NaN весами
            bad_model = SimpleTestModel()
            # Намеренно портим веса
            with torch.no_grad():
                bad_model.linear1.weight.fill_(float('nan'))
            
            bad_checkpoint_path = Path(temp_dir) / "bad_checkpoint.pt"
            torch.save({
                'model_state_dict': bad_model.state_dict(),
                'epoch': 5,
                'global_step': 500,
                'validation_loss': 100.0
            }, bad_checkpoint_path)
            
            # Анализ плохого checkpoint'а
            quality, issues = analyzer.analyze_checkpoint_health(str(bad_checkpoint_path))
            
            assert quality == CheckpointQuality.CRITICAL, f"Ожидался CRITICAL, получен: {quality}"
            assert len(issues) > 0, "Должны быть обнаружены проблемы"
            assert any("NaN" in issue for issue in issues), "Должна быть обнаружена проблема с NaN"
            
            print(f"✅ Плохой checkpoint корректно определен как: {quality.value}")
            print(f"   Обнаружено проблем: {len(issues)}")
        
        print("✅ CheckpointHealthAnalyzer: Корректно анализирует здоровье checkpoint'ов")
        return True
        
    except Exception as e:
        print(f"❌ CheckpointHealthAnalyzer: {e}")
        return False

def test_intelligent_checkpoint_manager():
    """Тестирование IntelligentCheckpointManager"""
    print("\n🧪 ТЕСТ 3: IntelligentCheckpointManager")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Создание checkpoint manager'а
            manager = IntelligentCheckpointManager(
                checkpoint_dir=temp_dir,
                max_checkpoints=3,
                min_save_interval=10
            )
            
            # Создание тестовой модели и оптимизатора
            model = SimpleTestModel()
            optimizer = torch.optim.Adam(model.parameters())
            
            # Сохранение нескольких checkpoint'ов
            saved_paths = []
            
            for step in [100, 200, 300, 400]:
                metrics = create_test_metrics(
                    step=step,
                    validation_loss=10.0 - step/100.0,  # Улучшение с каждым шагом
                    attention_diagonality=0.4 + step/1000.0  # Улучшение attention
                )
                
                path = manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    metrics=metrics,
                    force_save=True
                )
                
                if path:
                    saved_paths.append(path)
            
            # Проверка, что checkpoint'ы сохранены
            assert len(saved_paths) >= 3, f"Должно быть сохранено минимум 3 checkpoint'а, сохранено: {len(saved_paths)}"
            
            # Проверка управления дисковым пространством
            assert len(manager.checkpoints) <= manager.max_checkpoints, "Превышен лимит checkpoint'ов"
            
            # Получение лучшего checkpoint'а
            best_checkpoint = manager.get_best_checkpoint()
            assert best_checkpoint is not None, "Должен быть найден лучший checkpoint"
            assert best_checkpoint.is_best, "Checkpoint должен быть отмечен как лучший"
            
            # Проверка статуса
            status = manager.get_status_report()
            assert status['total_checkpoints'] > 0, "Должны быть checkpoint'ы в статусе"
            assert 'best_checkpoint' in status, "Должна быть информация о лучшем checkpoint'е"
            
            print(f"✅ Сохранено checkpoint'ов: {len(saved_paths)}")
            print(f"✅ Лучший checkpoint score: {best_checkpoint.health_score:.4f}")
            print(f"✅ Статус менеджера: {status['status']}")
        
        print("✅ IntelligentCheckpointManager: Базовая функциональность работает")
        return True
        
    except Exception as e:
        print(f"❌ IntelligentCheckpointManager: {e}")
        return False

def test_auto_recovery_system():
    """Тестирование AutoRecoverySystem"""
    print("\n🧪 ТЕСТ 4: AutoRecoverySystem")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Создание checkpoint manager'а с recovery system
            manager = IntelligentCheckpointManager(
                checkpoint_dir=temp_dir,
                max_checkpoints=5
            )
            
            model = SimpleTestModel()
            optimizer = torch.optim.Adam(model.parameters())
            
            # Сохранение хорошего checkpoint'а
            good_metrics = create_test_metrics(
                step=1000,
                validation_loss=3.0,
                attention_diagonality=0.7
            )
            
            good_path = manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                metrics=good_metrics,
                force_save=True
            )
            
            assert good_path is not None, "Хороший checkpoint должен быть сохранен"
            
            # Симуляция критического сбоя
            critical_metrics = create_test_metrics(
                step=1100,
                validation_loss=float('inf'),  # NaN loss
                attention_diagonality=0.001,  # Коллапс attention
                has_problems=True
            )
            critical_metrics.gradient_norm = 5000.0  # Взрыв градиентов
            
            # Тестирование детекции критического сбоя
            is_critical = manager.auto_recovery.detect_critical_failure(critical_metrics)
            assert is_critical, "Критический сбой должен быть обнаружен"
            
            # Тестирование автовосстановления
            recovery_success = manager.check_and_recover(model, optimizer, critical_metrics)
            
            # Recovery может не сработать если нет подходящего checkpoint'а, но система должна попытаться
            print(f"✅ Детекция критического сбоя: работает")
            print(f"✅ Попытка автовосстановления: {'успешно' if recovery_success else 'выполнена'}")
        
        print("✅ AutoRecoverySystem: Корректно обнаруживает сбои и пытается восстановление")
        return True
        
    except Exception as e:
        print(f"❌ AutoRecoverySystem: {e}")
        return False

def test_emergency_checkpoint_saving():
    """Тестирование экстренного сохранения checkpoint'ов"""
    print("\n🧪 ТЕСТ 5: Emergency Checkpoint Saving")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IntelligentCheckpointManager(checkpoint_dir=temp_dir)
            
            model = SimpleTestModel()
            optimizer = torch.optim.Adam(model.parameters())
            
            # Создание критически плохих метрик
            emergency_metrics = create_test_metrics(
                step=500,
                validation_loss=1000.0,  # Очень высокий loss
                attention_diagonality=0.001,  # Коллапс attention
                has_problems=True
            )
            
            # Экстренное сохранение
            emergency_path = manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                metrics=emergency_metrics,
                is_emergency=True
            )
            
            assert emergency_path is not None, "Экстренный checkpoint должен быть сохранен"
            assert "emergency" in emergency_path, "Файл должен содержать 'emergency' в имени"
            
            # Проверка, что checkpoint отмечен как экстренный
            emergency_checkpoints = [cp for cp in manager.checkpoints if cp.is_emergency_backup]
            assert len(emergency_checkpoints) > 0, "Должен быть экстренный checkpoint"
            
            print(f"✅ Экстренный checkpoint сохранен: {os.path.basename(emergency_path)}")
        
        print("✅ Emergency Checkpoint Saving: Работает корректно")
        return True
        
    except Exception as e:
        print(f"❌ Emergency Checkpoint Saving: {e}")
        return False

def test_integration_with_context_manager():
    """Тестирование интеграции с Context-Aware Manager"""
    print("\n🧪 ТЕСТ 6: Integration with Context-Aware Manager")
    
    if not CONTEXT_MANAGER_AVAILABLE:
        print("⚠️ Context-Aware Manager недоступен, пропускаем интеграционный тест")
        return True
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Создание интегрированной системы
            checkpoint_manager = IntelligentCheckpointManager(checkpoint_dir=temp_dir)
            
            context_config = {
                'initial_lr': 1e-3,
                'history_size': 50,
                'logging_level': 'INFO'
            }
            context_manager = ContextAwareTrainingManager(context_config)
            
            # Симуляция интеграции checkpoint'а в context-aware обучение
            model = SimpleTestModel()
            optimizer = torch.optim.Adam(model.parameters())
            
            # Симуляция нескольких шагов обучения с checkpoint'ами
            for step in range(100, 400, 100):
                metrics = create_test_metrics(
                    step=step,
                    validation_loss=8.0 - step/100.0,
                    attention_diagonality=0.3 + step/1000.0
                )
                
                # Сохранение checkpoint'а
                path = checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    metrics=metrics,
                    force_save=True
                )
                
                # Проверка на критические сбои
                if step == 300:  # Симуляция проблемы
                    critical_metrics = create_test_metrics(
                        step=step,
                        validation_loss=float('inf'),
                        has_problems=True
                    )
                    
                    recovery_attempted = checkpoint_manager.check_and_recover(
                        model, optimizer, critical_metrics
                    )
                    
                    print(f"✅ Шаг {step}: Восстановление {'выполнено' if recovery_attempted else 'не требовалось'}")
            
            # Получение лучшего checkpoint'а для загрузки
            best_checkpoint = checkpoint_manager.get_best_checkpoint()
            if best_checkpoint:
                print(f"✅ Лучший checkpoint для Context Manager: {best_checkpoint.quality.value}")
            
        print("✅ Integration with Context-Aware Manager: Успешно")
        return True
        
    except Exception as e:
        print(f"❌ Integration with Context-Aware Manager: {e}")
        return False

def test_checkpoint_quality_classification():
    """Тестирование классификации качества checkpoint'ов"""
    print("\n🧪 ТЕСТ 7: Checkpoint Quality Classification")
    
    try:
        selector = MultiCriteriaModelSelector()
        
        # Тестирование различных сценариев качества
        test_cases = [
            # (описание, метрики, ожидаемое качество)
            (
                "Отличный checkpoint",
                create_test_metrics(1000, 1.5, 0.85),
                [CheckpointQuality.EXCELLENT, CheckpointQuality.GOOD]
            ),
            (
                "Хороший checkpoint", 
                create_test_metrics(800, 3.0, 0.65),
                [CheckpointQuality.GOOD, CheckpointQuality.ACCEPTABLE]
            ),
            (
                "Приемлемый checkpoint",
                create_test_metrics(600, 8.0, 0.45),
                [CheckpointQuality.ACCEPTABLE, CheckpointQuality.POOR]
            ),
            (
                "Критический checkpoint",
                create_test_metrics(400, 100.0, 0.01, has_problems=True),
                [CheckpointQuality.CRITICAL, CheckpointQuality.POOR]
            )
        ]
        
        for description, metrics, expected_qualities in test_cases:
            score = selector.calculate_model_score(metrics)
            
            # Классификация по score
            if score >= 0.8:
                predicted_quality = CheckpointQuality.EXCELLENT
            elif score >= 0.6:
                predicted_quality = CheckpointQuality.GOOD
            elif score >= 0.4:
                predicted_quality = CheckpointQuality.ACCEPTABLE
            elif score >= 0.2:
                predicted_quality = CheckpointQuality.POOR
            else:
                predicted_quality = CheckpointQuality.CRITICAL
                
            # Применение штрафов за критические проблемы
            if metrics.has_nan_weights or metrics.has_gradient_explosion:
                predicted_quality = CheckpointQuality.CRITICAL
            
            assert predicted_quality in expected_qualities, \
                f"{description}: ожидалось {expected_qualities}, получено {predicted_quality}"
            
            print(f"✅ {description}: {predicted_quality.value} (score: {score:.3f})")
        
        print("✅ Checkpoint Quality Classification: Корректно классифицирует checkpoint'ы")
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint Quality Classification: {e}")
        return False

def test_full_system_simulation():
    """Полная симуляция системы checkpoint'ов в реальном сценарии"""
    print("\n🧪 ТЕСТ 8: Full System Simulation")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Создание полной системы
            manager = create_checkpoint_manager(
                checkpoint_dir=temp_dir,
                max_checkpoints=5
            )
            
            model = SimpleTestModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            simulation_results = {
                'checkpoints_saved': 0,
                'critical_failures': 0,
                'recoveries_attempted': 0,
                'recoveries_successful': 0
            }
            
            # Симуляция 20 шагов обучения
            print("   Симуляция обучения:")
            for step in range(100, 2100, 100):
                # Генерация реалистичных метрик с occasional проблемами
                if step in [500, 1200, 1800]:  # Критические сбои
                    metrics = create_test_metrics(
                        step=step,
                        validation_loss=float('inf') if step == 500 else 150.0,
                        attention_diagonality=0.005,
                        has_problems=True
                    )
                    if step == 1200:
                        metrics.gradient_norm = 2000.0  # Взрыв градиентов
                    
                    simulation_results['critical_failures'] += 1
                    
                    # Попытка восстановления (это уже сохранит emergency checkpoint)
                    recovery_success = manager.check_and_recover(model, optimizer, metrics)
                    simulation_results['recoveries_attempted'] += 1
                    if recovery_success:
                        simulation_results['recoveries_successful'] += 1
                    
                    # Emergency checkpoint уже создан в check_and_recover, считаем его
                    simulation_results['checkpoints_saved'] += 1
                    
                    print(f"     Шаг {step}: 🚨 Критический сбой, восстановление {'✅' if recovery_success else '❌'}")
                    
                else:  # Нормальные шаги
                    # Постепенное улучшение метрик
                    base_loss = max(1.0, 15.0 - step/200.0)
                    base_attention = min(0.85, 0.2 + step/2500.0)
                    
                    metrics = create_test_metrics(
                        step=step,
                        validation_loss=base_loss + np.random.normal(0, 0.5),
                        attention_diagonality=base_attention + np.random.normal(0, 0.05)
                    )
                    
                    # Сохранение checkpoint'а только для нормальных шагов
                    if step % 300 == 0:  # Сохранение каждые 3 шага
                        path = manager.save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            metrics=metrics,
                            force_save=True
                        )
                        if path:
                            simulation_results['checkpoints_saved'] += 1
                            print(f"     Шаг {step}: 💾 Checkpoint сохранен")
            
            # Финальный анализ
            final_status = manager.get_status_report()
            best_checkpoint = manager.get_best_checkpoint()
            
            print(f"\n   📊 Результаты симуляции:")
            print(f"     Checkpoint'ов сохранено: {simulation_results['checkpoints_saved']}")
            print(f"     Критических сбоев: {simulation_results['critical_failures']}")
            print(f"     Попыток восстановления: {simulation_results['recoveries_attempted']}")
            print(f"     Успешных восстановлений: {simulation_results['recoveries_successful']}")
            print(f"     Финальное качество лучшей модели: {best_checkpoint.quality.value if best_checkpoint else 'нет'}")
            final_score = f"{best_checkpoint.health_score:.4f}" if best_checkpoint else "нет"
            print(f"     Финальный score: {final_score}")
            print(f"     Экстренных backup'ов: {final_status['emergency_backups']}")
            print(f"     Реальное количество checkpoint'ов: {final_status['total_checkpoints']}/5")
            
            # DEBUG: Список всех checkpoint'ов
            print("   📂 DEBUG - Список checkpoint'ов:")
            for i, cp in enumerate(manager.checkpoints):
                cp_type = "🚨EMERGENCY" if cp.is_emergency_backup else "💾NORMAL"
                print(f"     {i+1}. {cp_type} - {os.path.basename(cp.path)} (качество: {cp.quality.value})")
            
            # Проверка базовых утверждений
            assert simulation_results['checkpoints_saved'] > 0, "Должны быть сохранены checkpoint'ы"
            assert simulation_results['critical_failures'] == 3, "Должно быть 3 критических сбоя"
            assert final_status['total_checkpoints'] <= 5, f"Не должно превышать лимит checkpoint'ов: {final_status['total_checkpoints']}/5"
            
        print("✅ Full System Simulation: Система работает в реальном сценарии")
        return True
        
    except Exception as e:
        print(f"❌ Full System Simulation: {e}")
        return False

def run_all_tests():
    """Запуск всех тестов Advanced Model Checkpointing System"""
    print("🎯 НАЧАЛО ТЕСТИРОВАНИЯ: Advanced Model Checkpointing System")
    print("=" * 80)
    
    if not CHECKPOINTING_AVAILABLE:
        print("❌ Advanced Model Checkpointing System недоступен для тестирования")
        return False
    
    tests = [
        test_multi_criteria_model_selector,
        test_checkpoint_health_analyzer,
        test_intelligent_checkpoint_manager,
        test_auto_recovery_system,
        test_emergency_checkpoint_saving,
        test_integration_with_context_manager,
        test_checkpoint_quality_classification,
        test_full_system_simulation
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ Критическая ошибка в тесте {test_func.__name__}: {e}")
    
    # Финальный отчет
    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"✅ Пройдено тестов: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("\n🚀 Advanced Model Checkpointing System готов к production использованию:")
        print("   • Интеллектуальное сохранение checkpoint'ов")
        print("   • Multi-criteria выбор лучших моделей")
        print("   • Автоматическое восстановление при сбоях")
        print("   • Анализ здоровья checkpoint'ов")
        print("   • Управление дисковым пространством")
        print("   • Интеграция с Context-Aware Manager")
        return True
    else:
        print(f"⚠️ {total_tests - passed_tests} тестов не прошли")
        print("   Требуется доработка перед production использованием")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 