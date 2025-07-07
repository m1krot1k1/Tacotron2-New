"""
Демонстрация Risk Assessment Module для Enhanced Tacotron2 AI System

Демонстрация возможностей оценки рисков изменений параметров с помощью
Monte Carlo симуляций и bootstrap анализа.
"""

import logging
import json
from datetime import datetime
from risk_assessment_module import (
    RiskAssessmentModule,
    RiskAssessmentConfig,
    quick_risk_assessment,
    quick_system_assessment
)

def setup_logging():
    """Настройка логирования для демонстрации"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('risk_assessment_demo.log')
        ]
    )

def print_separator(title: str):
    """Печать разделителя с заголовком"""
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)

def print_parameter_risk(risk):
    """Красивый вывод риска параметра"""
    print(f"\n📊 Параметр: {risk.parameter_name}")
    print(f"   Текущее значение: {risk.current_value}")
    print(f"   Предлагаемое значение: {risk.proposed_value}")
    print(f"   🎯 Risk Score: {risk.risk_score:.3f}")
    print(f"   📈 Доверительный интервал: [{risk.confidence_interval[0]:.3f}, {risk.confidence_interval[1]:.3f}]")
    print(f"   📐 Метрика стабильности: {risk.stability_metric:.3f}")
    print(f"   ⚠️  Серьезность: {risk.impact_severity.upper()}")
    print(f"   💡 Рекомендация: {risk.recommendation}")

def print_system_assessment(assessment):
    """Красивый вывод системной оценки"""
    print(f"\n🎯 Общий риск системы: {assessment.overall_risk_score:.3f}")
    print(f"✅ Безопасно продолжать: {assessment.is_safe_to_proceed}")
    print(f"📊 Количество оцененных параметров: {len(assessment.parameter_risks)}")
    
    print(f"\n🔒 Ограничения безопасности:")
    constraints = assessment.safety_constraints
    if constraints.get('rejected_changes'):
        print(f"   🚫 Отклоненные изменения: {constraints['rejected_changes']}")
    if constraints.get('gradual_changes'):
        print(f"   🐌 Требуют постепенного изменения: {constraints['gradual_changes']}")
    if constraints.get('monitoring_required'):
        print(f"   👀 Требуют мониторинга: {constraints['monitoring_required']}")
    
    print(f"\n💡 Рекомендации системы:")
    for i, rec in enumerate(assessment.recommendations, 1):
        print(f"   {i}. {rec}")

def demo_single_parameter_assessment():
    """Демонстрация оценки одного параметра"""
    print_separator("ДЕМО: Оценка риска одного параметра")
    
    # Создание модуля с конфигурацией для демо
    config = RiskAssessmentConfig(
        n_samples=2000,
        n_bootstrap=200,
        risk_db_path="demo_risk_assessment.db"
    )
    module = RiskAssessmentModule(config)
    
    # Сценарий 1: Безопасное изменение learning rate
    print("\n🟢 Сценарий 1: Безопасное изменение learning rate")
    risk1 = module.assess_parameter_risk(
        parameter_name="learning_rate",
        current_value=1e-3,
        proposed_value=1.2e-3,  # Увеличение на 20%
        parameter_type="learning_rate"
    )
    print_parameter_risk(risk1)
    
    # Сценарий 2: Рискованное изменение learning rate
    print("\n🟡 Сценарий 2: Рискованное изменение learning rate")
    risk2 = module.assess_parameter_risk(
        parameter_name="learning_rate",
        current_value=1e-3,
        proposed_value=5e-3,  # Увеличение в 5 раз
        parameter_type="learning_rate"
    )
    print_parameter_risk(risk2)
    
    # Сценарий 3: Экстремальное изменение
    print("\n🔴 Сценарий 3: Экстремальное изменение learning rate")
    risk3 = module.assess_parameter_risk(
        parameter_name="learning_rate",
        current_value=1e-4,
        proposed_value=1e-1,  # Увеличение в 1000 раз
        parameter_type="learning_rate"
    )
    print_parameter_risk(risk3)
    
    return module

def demo_system_risk_assessment(module):
    """Демонстрация системной оценки рисков"""
    print_separator("ДЕМО: Системная оценка рисков")
    
    # Сценарий 1: Консервативные изменения
    print("\n🟢 Сценарий 1: Консервативные изменения параметров")
    conservative_changes = {
        "learning_rate": (1e-3, 1.1e-3),      # +10%
        "batch_size": (32, 36),                # +12.5%
        "gradient_clip": (1.0, 1.1),           # +10%
        "weight_decay": (1e-4, 1.2e-4)        # +20%
    }
    
    assessment1 = module.assess_system_risk(conservative_changes)
    print_system_assessment(assessment1)
    
    # Детали по каждому параметру
    print(f"\n📋 Детали по параметрам:")
    for risk in assessment1.parameter_risks:
        print(f"   • {risk.parameter_name}: {risk.risk_score:.3f} ({risk.impact_severity})")
    
    # Сценарий 2: Смешанные изменения
    print("\n🟡 Сценарий 2: Смешанные изменения (консервативные + агрессивные)")
    mixed_changes = {
        "learning_rate": (1e-3, 0.9e-3),      # Уменьшение на 10% (безопасно)
        "batch_size": (32, 128),               # Увеличение в 4 раза (средний риск)
        "gradient_clip": (1.0, 3.0),           # Увеличение в 3 раза (высокий риск)
        "dropout_rate": (0.1, 0.15)           # Небольшое увеличение (безопасно)
    }
    
    assessment2 = module.assess_system_risk(mixed_changes)
    print_system_assessment(assessment2)
    
    print(f"\n📋 Детали по параметрам:")
    for risk in assessment2.parameter_risks:
        print(f"   • {risk.parameter_name}: {risk.risk_score:.3f} ({risk.impact_severity})")
    
    # Сценарий 3: Агрессивные изменения
    print("\n🔴 Сценарий 3: Агрессивные изменения параметров")
    aggressive_changes = {
        "learning_rate": (1e-3, 1e-2),        # Увеличение в 10 раз
        "batch_size": (32, 512),               # Увеличение в 16 раз
        "gradient_clip": (1.0, 10.0),          # Увеличение в 10 раз
    }
    
    assessment3 = module.assess_system_risk(aggressive_changes)
    print_system_assessment(assessment3)
    
    print(f"\n📋 Детали по параметрам:")
    for risk in assessment3.parameter_risks:
        print(f"   • {risk.parameter_name}: {risk.risk_score:.3f} ({risk.impact_severity})")

def demo_quick_functions():
    """Демонстрация быстрых функций"""
    print_separator("ДЕМО: Быстрые функции оценки")
    
    print("\n⚡ Быстрая оценка одного параметра:")
    quick_risk = quick_risk_assessment("batch_size", 32, 64)
    print_parameter_risk(quick_risk)
    
    print("\n⚡ Быстрая системная оценка:")
    quick_changes = {
        "learning_rate": (1e-3, 2e-3),
        "batch_size": (32, 48)
    }
    quick_assessment = quick_system_assessment(quick_changes)
    print_system_assessment(quick_assessment)

def demo_real_world_scenarios():
    """Демонстрация реальных сценариев обучения"""
    print_separator("ДЕМО: Реальные сценарии обучения")
    
    module = RiskAssessmentModule()
    
    # Сценарий 1: Модель переобучается
    print("\n📉 Сценарий: Модель переобучается, нужно добавить регуляризацию")
    overfitting_fix = {
        "learning_rate": (1e-3, 5e-4),        # Уменьшаем LR
        "weight_decay": (1e-4, 1e-3),         # Увеличиваем регуляризацию
        "dropout_rate": (0.1, 0.3),           # Больше dropout
        "gradient_clip": (1.0, 0.5)           # Более жесткий clipping
    }
    
    assessment = module.assess_system_risk(overfitting_fix)
    print_system_assessment(assessment)
    
    # Сценарий 2: Медленное обучение
    print("\n🐌 Сценарий: Обучение идет слишком медленно, нужно ускорить")
    slow_training_fix = {
        "learning_rate": (5e-4, 1.5e-3),      # Увеличиваем LR
        "batch_size": (16, 32),                # Больше batch size
        "gradient_accumulation": (1, 2)        # Аккумуляция градиентов
    }
    
    assessment = module.assess_system_risk(slow_training_fix)
    print_system_assessment(assessment)
    
    # Сценарий 3: Проблемы с вниманием
    print("\n👁️ Сценарий: Проблемы с механизмом внимания")
    attention_fix = {
        "attention_dropout": (0.1, 0.05),     # Меньше dropout для внимания
        "attention_heads": (8, 4),             # Меньше голов внимания
        "attention_dim": (512, 256)            # Меньше размерность
    }
    
    assessment = module.assess_system_risk(attention_fix)
    print_system_assessment(assessment)

def demo_database_features(module):
    """Демонстрация функций базы данных"""
    print_separator("ДЕМО: Работа с базой данных")
    
    # Получение истории оценок
    print("\n📚 История оценок параметра learning_rate:")
    history = module.get_parameter_risk_history("learning_rate", days=1)
    
    if history:
        print(f"   Найдено {len(history)} записей:")
        for record in history[-3:]:  # Показываем последние 3
            print(f"   • {record['timestamp'][:19]}: risk={record['risk_score']:.3f}")
    else:
        print("   История пуста")
    
    # Статус системы
    print("\n🔧 Статус системы оценки рисков:")
    status = module.get_system_status()
    print(f"   • Статус модуля: {status['module_status']}")
    print(f"   • Количество параметров в истории: {len(status['parameter_history_count'])}")
    print(f"   • База данных: {status['database_path']}")
    
    # Конфигурация
    config = status['config']
    print(f"   • Monte Carlo сэмплы: {config['n_samples']}")
    print(f"   • Bootstrap сэмплы: {config['n_bootstrap']}")
    print(f"   • Порог высокого риска: {config['high_risk_threshold']}")

def main():
    """Главная функция демонстрации"""
    setup_logging()
    
    print("🎯 RISK ASSESSMENT MODULE DEMO")
    print("Enhanced Tacotron2 AI System")
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Демонстрация оценки одного параметра
        module = demo_single_parameter_assessment()
        
        # Демонстрация системной оценки
        demo_system_risk_assessment(module)
        
        # Демонстрация быстрых функций
        demo_quick_functions()
        
        # Демонстрация реальных сценариев
        demo_real_world_scenarios()
        
        # Демонстрация функций базы данных
        demo_database_features(module)
        
        print_separator("ЗАВЕРШЕНИЕ ДЕМОНСТРАЦИИ")
        print("\n✅ Демонстрация Risk Assessment Module завершена успешно!")
        print("📊 Все сценарии оценки рисков протестированы")
        print("💾 Результаты сохранены в базу данных: demo_risk_assessment.db")
        print("📝 Логи записаны в файл: risk_assessment_demo.log")
        
        # Финальная статистика
        final_status = module.get_system_status()
        total_assessments = sum(final_status['parameter_history_count'].values())
        print(f"📈 Всего выполнено оценок: {total_assessments}")
        
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении демонстрации: {e}")
        logging.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main() 