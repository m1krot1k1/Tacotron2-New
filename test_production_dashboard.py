#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 ТЕСТ: Production Real-time Dashboard
Проверка работоспособности dashboard компонентов
"""

import os
import sys
import time
import tempfile
import unittest
from unittest.mock import Mock, patch
import sqlite3
from pathlib import Path

# Добавляем current directory в path для импортов
sys.path.insert(0, os.getcwd())

try:
    from production_realtime_dashboard import (
        MetricsDatabase,
        RealtimeMetricsCollector,
        AlertManager,
        DashboardGraphGenerator,
        ProductionRealtimeDashboard,
        create_dashboard_template
    )
    DASHBOARD_AVAILABLE = True
except ImportError as e:
    print(f"❌ Dashboard недоступен: {e}")
    DASHBOARD_AVAILABLE = False


def test_metrics_database():
    """Тестирование MetricsDatabase"""
    print("\n🧪 ТЕСТ 1: MetricsDatabase")
    
    try:
        # Создание временной базы данных
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            database = MetricsDatabase(db_path)
            
            # Проверка создания таблиц
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                required_tables = ['system_metrics', 'training_metrics', 'alerts', 'optimizations']
                for table in required_tables:
                    assert table in tables, f"Таблица {table} не найдена"
            
            # Тестирование вставки системных метрик
            test_system_metrics = {
                'cpu_usage': 75.5,
                'memory_usage': 60.2,
                'gpu_usage': 80.0,
                'gpu_memory_usage': 45.3,
                'disk_usage': 30.1
            }
            
            database.insert_system_metrics(test_system_metrics)
            
            # Проверка вставки
            recent_metrics = database.get_recent_metrics('system_metrics', 60)
            assert len(recent_metrics) == 1, "Должна быть одна запись системных метрик"
            assert recent_metrics[0]['cpu_usage'] == 75.5, "CPU usage должен совпадать"
            
            # Тестирование вставки метрик обучения
            test_training_metrics = {
                'loss': 3.45,
                'learning_rate': 1e-3,
                'gradient_norm': 2.1,
                'attention_quality': 0.85,
                'epoch': 5,
                'step': 1000,
                'phase': 'training'
            }
            
            database.insert_training_metrics(test_training_metrics)
            
            # Проверка вставки обучения
            training_metrics = database.get_recent_metrics('training_metrics', 60)
            assert len(training_metrics) == 1, "Должна быть одна запись обучения"
            assert training_metrics[0]['loss'] == 3.45, "Loss должен совпадать"
            
            # Тестирование алертов
            database.insert_alert('warning', 'system', 'Тестовый алерт')
            
            alerts = database.get_recent_metrics('alerts', 60)
            assert len(alerts) == 1, "Должен быть один алерт"
            assert alerts[0]['message'] == 'Тестовый алерт', "Сообщение алерта должно совпадать"
            
            print("✅ База данных метрик работает корректно")
            print(f"✅ Создано таблиц: {len(tables)}")
            print(f"✅ Системных метрик: {len(recent_metrics)}")
            print(f"✅ Метрик обучения: {len(training_metrics)}")
            print(f"✅ Алертов: {len(alerts)}")
            
            return True
            
        finally:
            # Удаление временного файла
            if os.path.exists(db_path):
                os.unlink(db_path)
        
    except Exception as e:
        print(f"❌ MetricsDatabase: {e}")
        return False


def test_realtime_metrics_collector():
    """Тестирование RealtimeMetricsCollector"""
    print("\n🧪 ТЕСТ 2: RealtimeMetricsCollector")
    
    try:
        # Создание временной базы данных
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            database = MetricsDatabase(db_path)
            collector = RealtimeMetricsCollector(database)
            
            # Тестирование сбора системных метрик
            metrics = collector.collect_system_metrics()
            
            # Проверка структуры метрик
            required_keys = ['cpu_usage', 'memory_usage', 'disk_usage', 'gpu_usage', 'gpu_memory_usage']
            for key in required_keys:
                assert key in metrics, f"Ключ {key} должен присутствовать в метриках"
                assert isinstance(metrics[key], (int, float)), f"{key} должен быть числом"
                assert 0 <= metrics[key] <= 100, f"{key} должен быть в диапазоне 0-100"
            
            print(f"✅ CPU Usage: {metrics['cpu_usage']:.1f}%")
            print(f"✅ Memory Usage: {metrics['memory_usage']:.1f}%")
            print(f"✅ Disk Usage: {metrics['disk_usage']:.1f}%")
            print(f"✅ GPU Usage: {metrics['gpu_usage']:.1f}%")
            
            # Тестирование кэширования
            collector.metrics_cache['test'] = {'value': 123}
            cached = collector.get_cached_metrics()
            assert 'test' in cached, "Кэш должен содержать тестовые данные"
            assert cached['test']['value'] == 123, "Значение в кэше должно совпадать"
            
            # Тестирование запуска и остановки сбора
            collector.start_collection(interval=0.1)
            assert collector.running == True, "Сбор должен быть запущен"
            
            time.sleep(0.2)  # Дождаться одного цикла
            
            collector.stop_collection()
            assert collector.running == False, "Сбор должен быть остановлен"
            
            print("✅ Real-time сбор метрик работает корректно")
            return True
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
        
    except Exception as e:
        print(f"❌ RealtimeMetricsCollector: {e}")
        return False


def test_alert_manager():
    """Тестирование AlertManager"""
    print("\n🧪 ТЕСТ 3: AlertManager")
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            database = MetricsDatabase(db_path)
            alert_manager = AlertManager(database)
            
            # Тестирование системных алертов
            critical_system_metrics = {
                'cpu_usage': 95.0,   # Критическое значение
                'memory_usage': 92.0,  # Критическое значение
                'gpu_memory_usage': 97.0  # Критическое значение
            }
            
            system_alerts = alert_manager.check_system_alerts(critical_system_metrics)
            
            # Должны быть сгенерированы алерты
            assert len(system_alerts) > 0, "Должны быть созданы системные алерты"
            
            critical_alerts = [a for a in system_alerts if a['level'] == 'critical']
            warning_alerts = [a for a in system_alerts if a['level'] == 'warning']
            
            print(f"✅ Критических алертов: {len(critical_alerts)}")
            print(f"✅ Предупреждений: {len(warning_alerts)}")
            
            # Тестирование алертов обучения
            problematic_training_metrics = {
                'gradient_norm': 15.0,  # Высокая норма градиентов
                'loss': 60.0,           # Критический скачок loss
                'attention_quality': 0.05  # Низкое качество attention
            }
            
            training_alerts = alert_manager.check_training_alerts(problematic_training_metrics)
            assert len(training_alerts) > 0, "Должны быть созданы алерты обучения"
            
            print(f"✅ Алертов обучения: {len(training_alerts)}")
            
            # Проверка недавних алертов
            recent_alerts = alert_manager.get_recent_alerts(10)
            assert len(recent_alerts) > 0, "Должны быть недавние алерты"
            
            # Проверка структуры алертов
            for alert in recent_alerts:
                assert 'level' in alert, "Алерт должен содержать level"
                assert 'component' in alert, "Алерт должен содержать component"
                assert 'message' in alert, "Алерт должен содержать message"
                assert alert['level'] in ['warning', 'critical'], "Уровень должен быть warning или critical"
            
            print("✅ Alert Manager работает корректно")
            return True
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
        
    except Exception as e:
        print(f"❌ AlertManager: {e}")
        return False


def test_dashboard_graph_generator():
    """Тестирование DashboardGraphGenerator"""
    print("\n🧪 ТЕСТ 4: DashboardGraphGenerator")
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            database = MetricsDatabase(db_path)
            generator = DashboardGraphGenerator(database)
            
            # Добавление тестовых данных
            for i in range(10):
                test_metrics = {
                    'cpu_usage': 50.0 + i * 2,
                    'memory_usage': 40.0 + i * 1.5,
                    'gpu_usage': 60.0 + i * 1.8,
                    'gpu_memory_usage': 30.0 + i * 2.2,
                    'disk_usage': 25.0 + i * 0.5
                }
                database.insert_system_metrics(test_metrics)
                
                training_metrics = {
                    'loss': 10.0 - i * 0.8,
                    'learning_rate': 1e-3 - i * 1e-5,
                    'gradient_norm': 2.0 + i * 0.1,
                    'attention_quality': 0.5 + i * 0.04,
                    'epoch': i,
                    'step': i * 100,
                    'phase': 'training'
                }
                database.insert_training_metrics(training_metrics)
            
            # Тестирование создания графика системных метрик
            system_plot = generator.create_system_metrics_plot(30)
            assert system_plot != "{}", "График системных метрик не должен быть пустым"
            assert "CPU Usage" in system_plot, "График должен содержать CPU Usage"
            assert "Memory Usage" in system_plot, "График должен содержать Memory Usage"
            
            print("✅ График системных метрик создан")
            
            # Тестирование создания графика обучения
            training_plot = generator.create_training_progress_plot(60)
            assert training_plot != "{}", "График обучения не должен быть пустым"
            assert "Loss" in training_plot, "График должен содержать Loss"
            assert "Learning Rate" in training_plot, "График должен содержать Learning Rate"
            
            print("✅ График прогресса обучения создан")
            
            # Тестирование создания gauge производительности
            current_metrics = {
                'cpu_usage': 75.0,
                'memory_usage': 60.0,
                'gpu_usage': 80.0
            }
            
            gauge_plot = generator.create_performance_gauge(current_metrics)
            assert gauge_plot != "{}", "Gauge производительности не должен быть пустым"
            assert "Overall Performance" in gauge_plot, "Gauge должен содержать Overall Performance"
            
            print("✅ Gauge производительности создан")
            
            print("✅ Graph Generator работает корректно")
            return True
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
        
    except Exception as e:
        print(f"❌ DashboardGraphGenerator: {e}")
        return False


def test_dashboard_template_creation():
    """Тестирование создания HTML шаблона"""
    print("\n🧪 ТЕСТ 5: Dashboard Template Creation")
    
    try:
        # Создание шаблона
        create_dashboard_template()
        
        # Проверка создания файла
        template_path = Path("templates/dashboard.html")
        assert template_path.exists(), "HTML шаблон должен быть создан"
        
        # Проверка содержимого
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Проверка наличия ключевых компонентов
        required_components = [
            "Tacotron2 Production Dashboard",
            "CPU Usage",
            "Memory Usage", 
            "GPU Usage",
            "Performance",
            "socket.io",
            "plotly",
            "bootstrap"
        ]
        
        for component in required_components:
            assert component in content, f"Компонент '{component}' должен присутствовать в шаблоне"
        
        print(f"✅ HTML шаблон создан: {template_path}")
        print(f"✅ Размер файла: {len(content)} символов")
        print("✅ Все необходимые компоненты присутствуют")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard Template Creation: {e}")
        return False


def run_dashboard_tests():
    """Запуск всех тестов dashboard"""
    print("🚀 НАЧАЛО ТЕСТИРОВАНИЯ: Production Real-time Dashboard")
    print("=" * 80)
    
    if not DASHBOARD_AVAILABLE:
        print("❌ Production Real-time Dashboard недоступен для тестирования")
        return False
    
    tests = [
        test_metrics_database,
        test_realtime_metrics_collector,
        test_alert_manager,
        test_dashboard_graph_generator,
        test_dashboard_template_creation
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
        print("\n🚀 Production Real-time Dashboard готов к запуску:")
        print("   • MetricsDatabase - хранение метрик в SQLite")
        print("   • RealtimeMetricsCollector - real-time сбор системных метрик")
        print("   • AlertManager - интеллектуальная система алертов")
        print("   • DashboardGraphGenerator - интерактивные графики Plotly")
        print("   • HTML Template - responsive веб-интерфейс")
        print("   • WebSocket поддержка для real-time обновлений")
        print("\n📋 Для запуска dashboard:")
        print("   python production_realtime_dashboard.py")
        print("   Откроется на: http://localhost:5001")
        return True
    else:
        print(f"⚠️ {total_tests - passed_tests} тестов не прошли")
        return False


if __name__ == "__main__":
    success = run_dashboard_tests()
    sys.exit(0 if success else 1) 