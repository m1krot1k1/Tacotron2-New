"""
Тесты для Risk Assessment Module

Комплексное тестирование Monte Carlo симуляций, bootstrap анализа
и системной оценки рисков для Enhanced Tacotron2 AI System.
"""

import unittest
import numpy as np
import tempfile
import shutil
import os
import sqlite3
from unittest.mock import patch, MagicMock
from datetime import datetime
import json

from risk_assessment_module import (
    RiskAssessmentModule,
    RiskAssessmentConfig,
    MonteCarloSimulator,
    BootstrapAnalyzer,
    RiskDatabase,
    ParameterRisk,
    SystemRiskAssessment,
    quick_risk_assessment,
    quick_system_assessment
)

class TestRiskAssessmentConfig(unittest.TestCase):
    """Тесты конфигурации Risk Assessment"""
    
    def test_default_config(self):
        """Тест дефолтной конфигурации"""
        config = RiskAssessmentConfig()
        
        self.assertEqual(config.n_samples, 10000)
        self.assertEqual(config.confidence_level, 0.95)
        self.assertEqual(config.n_bootstrap, 1000)
        self.assertEqual(config.high_risk_threshold, 0.8)
        self.assertEqual(config.critical_risk_threshold, 0.95)
    
    def test_custom_config(self):
        """Тест кастомной конфигурации"""
        config = RiskAssessmentConfig(
            n_samples=5000,
            confidence_level=0.99,
            high_risk_threshold=0.7
        )
        
        self.assertEqual(config.n_samples, 5000)
        self.assertEqual(config.confidence_level, 0.99)
        self.assertEqual(config.high_risk_threshold, 0.7)

class TestMonteCarloSimulator(unittest.TestCase):
    """Тесты Monte Carlo симулятора"""
    
    def setUp(self):
        self.config = RiskAssessmentConfig(n_samples=1000)  # Меньше сэмплов для быстрых тестов
        self.simulator = MonteCarloSimulator(self.config)
    
    def test_simulate_learning_rate_change(self):
        """Тест симуляции изменения learning rate"""
        result = self.simulator.simulate_parameter_change(
            current_value=1e-3,
            proposed_value=5e-3,
            parameter_type="learning_rate"
        )
        
        self.assertIn('risk_score', result)
        self.assertIn('mean_outcome', result)
        self.assertIn('std_outcome', result)
        self.assertIn('percentiles', result)
        self.assertIn('rare_events', result)
        self.assertIn('stability_metric', result)
        
        # Risk score должен быть между 0 и 1
        self.assertGreaterEqual(result['risk_score'], 0)
        self.assertLessEqual(result['risk_score'], 1)
        
        # Percentiles должны быть в правильном порядке
        percentiles = result['percentiles']
        self.assertLessEqual(percentiles['5'], percentiles['25'])
        self.assertLessEqual(percentiles['25'], percentiles['50'])
        self.assertLessEqual(percentiles['50'], percentiles['75'])
        self.assertLessEqual(percentiles['75'], percentiles['95'])
    
    def test_simulate_batch_size_change(self):
        """Тест симуляции изменения batch size"""
        result = self.simulator.simulate_parameter_change(
            current_value=32,
            proposed_value=64,
            parameter_type="batch_size"
        )
        
        self.assertIn('risk_score', result)
        self.assertIsInstance(result['risk_score'], float)
        self.assertGreaterEqual(result['risk_score'], 0)
        self.assertLessEqual(result['risk_score'], 1)
    
    def test_simulate_general_parameter(self):
        """Тест симуляции общего параметра"""
        result = self.simulator.simulate_parameter_change(
            current_value=0.5,
            proposed_value=0.8,
            parameter_type="general"
        )
        
        self.assertIn('risk_score', result)
        self.assertIn('stability_metric', result)
    
    def test_extreme_values(self):
        """Тест симуляции с экстремальными значениями"""
        # Очень большое изменение должно давать высокий риск
        result = self.simulator.simulate_parameter_change(
            current_value=1e-3,
            proposed_value=1e1,  # Увеличение в 10000 раз
            parameter_type="learning_rate"
        )
        
        # Ожидаем высокий риск для такого экстремального изменения
        self.assertGreater(result['risk_score'], 0.5)
    
    def test_small_change(self):
        """Тест симуляции с малым изменением"""
        result = self.simulator.simulate_parameter_change(
            current_value=1e-3,
            proposed_value=1.1e-3,  # Небольшое изменение
            parameter_type="learning_rate"
        )
        
        # Ожидаем низкий риск для небольшого изменения
        self.assertLess(result['risk_score'], 0.7)
    
    def test_rare_events_analysis(self):
        """Тест анализа редких событий"""
        # Создаем симулированные данные с некоторыми редкими событиями
        outcomes = np.concatenate([
            np.random.normal(0.8, 0.1, 950),  # Большинство хороших результатов
            np.random.uniform(0, 0.01, 50)    # Редкие плохие события
        ])
        
        rare_events = self.simulator._importance_sampling_analysis(outcomes)
        
        self.assertIn('rare_event_probability', rare_events)
        self.assertIn('worst_case_outcome', rare_events)
        self.assertGreater(rare_events['rare_event_probability'], 0)
        self.assertLess(rare_events['worst_case_outcome'], 0.1)
    
    def test_stability_metric(self):
        """Тест расчета метрики стабильности"""
        # Стабильные данные
        stable_outcomes = np.random.normal(0.8, 0.01, 1000)
        stable_metric = self.simulator._calculate_stability_metric(stable_outcomes)
        
        # Нестабильные данные
        unstable_outcomes = np.random.uniform(0, 1, 1000)
        unstable_metric = self.simulator._calculate_stability_metric(unstable_outcomes)
        
        # Нестабильные данные должны иметь более высокую метрику нестабильности
        self.assertGreater(unstable_metric, stable_metric)

class TestBootstrapAnalyzer(unittest.TestCase):
    """Тесты Bootstrap анализатора"""
    
    def setUp(self):
        self.config = RiskAssessmentConfig(n_bootstrap=100)  # Меньше bootstrap для быстрых тестов
        self.analyzer = BootstrapAnalyzer(self.config)
    
    def test_confidence_intervals_mean(self):
        """Тест расчета доверительных интервалов для среднего"""
        data = np.random.normal(50, 10, 200)
        
        result = self.analyzer.calculate_confidence_intervals(data, np.mean)
        
        self.assertIn('statistic', result)
        self.assertIn('ci_lower', result)
        self.assertIn('ci_upper', result)
        self.assertIn('ci_width', result)
        
        # CI должны содержать статистику
        self.assertLessEqual(result['ci_lower'], result['statistic'])
        self.assertGreaterEqual(result['ci_upper'], result['statistic'])
        
        # Статистика должна быть близка к истинному среднему
        self.assertAlmostEqual(result['statistic'], 50, delta=3)
    
    def test_confidence_intervals_median(self):
        """Тест расчета доверительных интервалов для медианы"""
        data = np.random.exponential(10, 200)  # Асимметричное распределение
        
        result = self.analyzer.calculate_confidence_intervals(data, np.median)
        
        self.assertIn('statistic', result)
        self.assertIn('ci_lower', result)
        self.assertIn('ci_upper', result)
        
        # Медиана должна быть в доверительном интервале
        self.assertLessEqual(result['ci_lower'], result['statistic'])
        self.assertGreaterEqual(result['ci_upper'], result['statistic'])
    
    def test_uncertainty_estimation(self):
        """Тест полной оценки неопределенности"""
        data = np.random.normal(0, 1, 100)
        
        uncertainties = self.analyzer.uncertainty_estimation(data)
        
        # Проверяем наличие всех необходимых метрик
        required_keys = ['mean', 'median', 'std', 'percentile_5', 'percentile_95']
        for key in required_keys:
            self.assertIn(key, uncertainties)
            self.assertIn('statistic', uncertainties[key])
            self.assertIn('ci_lower', uncertainties[key])
            self.assertIn('ci_upper', uncertainties[key])
    
    def test_empty_data(self):
        """Тест обработки пустых данных"""
        empty_data = np.array([])
        
        result = self.analyzer.calculate_confidence_intervals(empty_data)
            
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'Empty data provided')
    
    def test_single_value_data(self):
        """Тест обработки данных с одним значением"""
        single_data = np.array([42.0])
        
        result = self.analyzer.calculate_confidence_intervals(single_data)
        
        # Должен вернуть ошибку или обработать gracefully
        # В зависимости от реализации

class TestRiskDatabase(unittest.TestCase):
    """Тесты базы данных рисков"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_risk.db")
        self.database = RiskDatabase(self.db_path)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_database_initialization(self):
        """Тест инициализации базы данных"""
        # Проверяем, что файл базы данных создан
        self.assertTrue(os.path.exists(self.db_path))
        
        # Проверяем, что таблицы созданы
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Проверяем таблицу risk_assessments
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='risk_assessments'")
            self.assertEqual(len(cursor.fetchall()), 1)
            
            # Проверяем таблицу system_assessments
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='system_assessments'")
            self.assertEqual(len(cursor.fetchall()), 1)
    
    def test_save_parameter_risk(self):
        """Тест сохранения риска параметра"""
        risk = ParameterRisk(
            parameter_name="learning_rate",
            current_value=1e-3,
            proposed_value=5e-3,
            risk_score=0.7,
            confidence_interval=(0.6, 0.8),
            stability_metric=0.3,
            impact_severity="medium",
            recommendation="Proceed with caution"
        )
        
        self.database.save_parameter_risk(risk)
        
        # Проверяем, что запись сохранена
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM risk_assessments")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
            
            # Проверяем содержимое
            cursor.execute("SELECT * FROM risk_assessments")
            row = cursor.fetchone()
            self.assertEqual(row[2], "learning_rate")  # parameter_name
            self.assertEqual(row[3], 1e-3)  # current_value
            self.assertEqual(row[4], 5e-3)  # proposed_value
            self.assertEqual(row[5], 0.7)   # risk_score
    
    def test_save_system_assessment(self):
        """Тест сохранения системной оценки"""
        assessment = SystemRiskAssessment(
            timestamp=datetime.now().isoformat(),
            overall_risk_score=0.6,
            parameter_risks=[],
            safety_constraints={},
            simulation_results={},
            recommendations=[],
            is_safe_to_proceed=True
        )
        
        self.database.save_system_assessment(assessment)
        
        # Проверяем, что запись сохранена
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM system_assessments")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
    
    def test_get_parameter_history(self):
        """Тест получения истории параметра"""
        # Сохраняем несколько записей
        for i in range(3):
            risk = ParameterRisk(
                parameter_name="learning_rate",
                current_value=1e-3,
                proposed_value=(i+1)*1e-3,
                risk_score=0.1*i,
                confidence_interval=(0.0, 0.1),
                stability_metric=0.1,
                impact_severity="low",
                recommendation="OK"
            )
            self.database.save_parameter_risk(risk)
        
        # Получаем историю
        history = self.database.get_parameter_history("learning_rate", days=7)
        
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]['parameter_name'], "learning_rate")
    
    def test_cleanup_old_records(self):
        """Тест очистки старых записей"""
        # Добавляем тестовую запись
        risk = ParameterRisk(
            parameter_name="test_param",
            current_value=1.0,
            proposed_value=2.0,
            risk_score=0.5,
            confidence_interval=(0.4, 0.6),
            stability_metric=0.2,
            impact_severity="medium",
            recommendation="Test"
        )
        self.database.save_parameter_risk(risk)
        
        # Очищаем записи старше 0 дней (все записи)
        self.database.cleanup_old_records(retention_days=0)
        
        # Проверяем, что записи удалены
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM risk_assessments")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 0)

class TestRiskAssessmentModule(unittest.TestCase):
    """Тесты главного модуля оценки рисков"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        config = RiskAssessmentConfig(
            n_samples=500,  # Меньше сэмплов для быстрых тестов
            n_bootstrap=100,
            risk_db_path=os.path.join(self.temp_dir, "test_risk.db")
        )
        self.module = RiskAssessmentModule(config)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_assess_parameter_risk_learning_rate(self):
        """Тест оценки риска изменения learning rate"""
        risk = self.module.assess_parameter_risk(
            parameter_name="learning_rate",
            current_value=1e-3,
            proposed_value=5e-3,
            parameter_type="learning_rate"
        )
        
        self.assertIsInstance(risk, ParameterRisk)
        self.assertEqual(risk.parameter_name, "learning_rate")
        self.assertEqual(risk.current_value, 1e-3)
        self.assertEqual(risk.proposed_value, 5e-3)
        self.assertGreaterEqual(risk.risk_score, 0)
        self.assertLessEqual(risk.risk_score, 1)
        self.assertIn(risk.impact_severity, ["low", "medium", "high", "critical"])
        self.assertIsInstance(risk.recommendation, str)
    
    def test_assess_parameter_risk_batch_size(self):
        """Тест оценки риска изменения batch size"""
        risk = self.module.assess_parameter_risk(
            parameter_name="batch_size",
            current_value=32,
            proposed_value=64,
            parameter_type="batch_size"
        )
        
        self.assertIsInstance(risk, ParameterRisk)
        self.assertEqual(risk.parameter_name, "batch_size")
        # Небольшое изменение batch size должно иметь низкий риск
        self.assertLess(risk.risk_score, 0.8)
    
    def test_assess_system_risk(self):
        """Тест системной оценки рисков"""
        parameter_changes = {
            "learning_rate": (1e-3, 2e-3),
            "batch_size": (32, 48),
            "gradient_clip": (1.0, 1.5)
        }
        
        assessment = self.module.assess_system_risk(parameter_changes)
        
        self.assertIsInstance(assessment, SystemRiskAssessment)
        self.assertGreaterEqual(assessment.overall_risk_score, 0)
        self.assertLessEqual(assessment.overall_risk_score, 1)
        self.assertEqual(len(assessment.parameter_risks), 3)
        self.assertIsInstance(assessment.is_safe_to_proceed, bool)
        self.assertIsInstance(assessment.recommendations, list)
        self.assertIsInstance(assessment.safety_constraints, dict)
    
    def test_classify_impact_severity(self):
        """Тест классификации серьезности воздействия"""
        # Низкий риск
        severity = self.module._classify_impact_severity(0.3)
        self.assertEqual(severity, "low")
        
        # Средний риск
        severity = self.module._classify_impact_severity(0.6)
        self.assertEqual(severity, "medium")
        
        # Высокий риск
        severity = self.module._classify_impact_severity(0.85)
        self.assertEqual(severity, "high")
        
        # Критический риск
        severity = self.module._classify_impact_severity(0.98)
        self.assertEqual(severity, "critical")
    
    def test_infer_parameter_type(self):
        """Тест определения типа параметра"""
        self.assertEqual(self.module._infer_parameter_type("learning_rate"), "learning_rate")
        self.assertEqual(self.module._infer_parameter_type("lr"), "learning_rate")
        self.assertEqual(self.module._infer_parameter_type("batch_size"), "batch_size")
        self.assertEqual(self.module._infer_parameter_type("gradient_clip"), "gradient_clip")
        self.assertEqual(self.module._infer_parameter_type("weight_decay"), "weight_decay")
        self.assertEqual(self.module._infer_parameter_type("unknown_param"), "general")
    
    def test_calculate_overall_risk(self):
        """Тест расчета общего риска системы"""
        parameter_risks = [
            ParameterRisk("param1", 1.0, 2.0, 0.3, (0.2, 0.4), 0.1, "low", "OK"),
            ParameterRisk("param2", 1.0, 2.0, 0.7, (0.6, 0.8), 0.2, "medium", "Caution"),
            ParameterRisk("param3", 1.0, 2.0, 0.9, (0.8, 1.0), 0.3, "high", "High risk")
        ]
        
        overall_risk = self.module._calculate_overall_risk(parameter_risks)
        
        # Общий риск должен быть между максимальным и средним
        max_risk = 0.9
        avg_risk = (0.3 + 0.7 + 0.9) / 3
        expected_risk = 0.6 * max_risk + 0.4 * avg_risk
        
        self.assertAlmostEqual(overall_risk, expected_risk, places=2)
    
    def test_generate_safety_constraints(self):
        """Тест генерации ограничений безопасности"""
        parameter_risks = [
            ParameterRisk("low_risk", 1.0, 2.0, 0.3, (0.2, 0.4), 0.1, "low", "OK"),
            ParameterRisk("medium_risk", 1.0, 2.0, 0.6, (0.5, 0.7), 0.2, "medium", "Caution"),
            ParameterRisk("high_risk", 1.0, 2.0, 0.85, (0.8, 0.9), 0.3, "high", "High risk"),
            ParameterRisk("critical_risk", 1.0, 2.0, 0.97, (0.95, 0.99), 0.4, "critical", "Critical")
        ]
        
        constraints = self.module._generate_safety_constraints(parameter_risks)
        
        self.assertIn("rejected_changes", constraints)
        self.assertIn("gradual_changes", constraints)
        self.assertIn("monitoring_required", constraints)
        
        self.assertIn("critical_risk", constraints["rejected_changes"])
        self.assertIn("high_risk", constraints["gradual_changes"])
        self.assertIn("medium_risk", constraints["monitoring_required"])
    
    def test_extreme_risk_scenario(self):
        """Тест экстремального сценария с высоким риском"""
        # Очень большое изменение learning rate
        risk = self.module.assess_parameter_risk(
            parameter_name="learning_rate",
            current_value=1e-4,
            proposed_value=1.0,  # Увеличение в 10000 раз
            parameter_type="learning_rate"
        )
        
        # Ожидаем высокий или критический риск
        self.assertGreater(risk.risk_score, 0.7)
        self.assertIn(risk.impact_severity, ["high", "critical"])
        self.assertIn("REJECT", risk.recommendation.upper())
    
    def test_safe_change_scenario(self):
        """Тест безопасного сценария с низким риском"""
        # Небольшое изменение
        risk = self.module.assess_parameter_risk(
            parameter_name="learning_rate",
            current_value=1e-3,
            proposed_value=1.1e-3,  # Увеличение на 10%
            parameter_type="learning_rate"
        )
        
        # Ожидаем низкий или средний риск
        self.assertLess(risk.risk_score, 0.8)
        self.assertIn(risk.impact_severity, ["low", "medium"])
    
    def test_get_system_status(self):
        """Тест получения статуса системы"""
        status = self.module.get_system_status()
        
        self.assertIn('module_status', status)
        self.assertIn('config', status)
        self.assertIn('parameter_history_count', status)
        self.assertIn('database_path', status)
        self.assertEqual(status['module_status'], 'active')

class TestQuickFunctions(unittest.TestCase):
    """Тесты вспомогательных функций"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('risk_assessment_module.RiskAssessmentModule')
    def test_quick_risk_assessment(self, mock_module_class):
        """Тест быстрой оценки риска параметра"""
        mock_module = MagicMock()
        mock_module.assess_parameter_risk.return_value = ParameterRisk(
            "test_param", 1.0, 2.0, 0.5, (0.4, 0.6), 0.2, "medium", "OK"
        )
        mock_module_class.return_value = mock_module
        
        result = quick_risk_assessment("test_param", 1.0, 2.0)
        
        self.assertIsInstance(result, ParameterRisk)
        mock_module.assess_parameter_risk.assert_called_once_with("test_param", 1.0, 2.0)
    
    @patch('risk_assessment_module.RiskAssessmentModule')
    def test_quick_system_assessment(self, mock_module_class):
        """Тест быстрой системной оценки"""
        mock_module = MagicMock()
        mock_assessment = SystemRiskAssessment(
            timestamp=datetime.now().isoformat(),
            overall_risk_score=0.5,
            parameter_risks=[],
            safety_constraints={},
            simulation_results={},
            recommendations=[],
            is_safe_to_proceed=True
        )
        mock_module.assess_system_risk.return_value = mock_assessment
        mock_module_class.return_value = mock_module
        
        changes = {"param1": (1.0, 2.0)}
        result = quick_system_assessment(changes)
        
        self.assertIsInstance(result, SystemRiskAssessment)
        mock_module.assess_system_risk.assert_called_once_with(changes)

class TestIntegrationScenarios(unittest.TestCase):
    """Интеграционные тесты реальных сценариев"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        config = RiskAssessmentConfig(
            n_samples=200,  # Минимальные сэмплы для скорости
            n_bootstrap=50,
            risk_db_path=os.path.join(self.temp_dir, "integration_test.db")
        )
        self.module = RiskAssessmentModule(config)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_training_hyperparameter_adjustment_scenario(self):
        """Тест сценария корректировки гиперпараметров обучения"""
        # Сценарий: модель переобучается, нужно скорректировать параметры
        parameter_changes = {
            "learning_rate": (1e-3, 5e-4),      # Уменьшаем LR
            "weight_decay": (1e-4, 1e-3),        # Увеличиваем регуляризацию
            "gradient_clip": (1.0, 0.5),         # Уменьшаем gradient clipping
            "dropout_rate": (0.1, 0.3)           # Увеличиваем dropout
        }
        
        assessment = self.module.assess_system_risk(parameter_changes)
        
        # Проверяем, что оценка выполнена для всех параметров
        self.assertEqual(len(assessment.parameter_risks), 4)
        
        # Проверяем, что система может принять решение
        self.assertIsInstance(assessment.is_safe_to_proceed, bool)
        
        # Проверяем, что есть рекомендации
        self.assertGreater(len(assessment.recommendations), 0)
        
        # Проверяем наличие ограничений безопасности
        self.assertIsInstance(assessment.safety_constraints, dict)
    
    def test_aggressive_optimization_scenario(self):
        """Тест сценария агрессивной оптимизации"""
        # Сценарий: пытаемся агрессивно ускорить обучение
        parameter_changes = {
            "learning_rate": (1e-3, 1e-2),      # Увеличиваем LR в 10 раз
            "batch_size": (32, 256),             # Увеличиваем batch size в 8 раз
        }
        
        assessment = self.module.assess_system_risk(parameter_changes)
        
        # Агрессивные изменения должны давать высокий риск
        self.assertGreater(assessment.overall_risk_score, 0.5)
        
        # Должны быть ограничения безопасности
        constraints = assessment.safety_constraints
        self.assertTrue(
            len(constraints.get('rejected_changes', [])) > 0 or
            len(constraints.get('gradual_changes', [])) > 0
        )
    
    def test_conservative_adjustment_scenario(self):
        """Тест сценария консервативной корректировки"""
        # Сценарий: небольшие консервативные изменения
        parameter_changes = {
            "learning_rate": (1e-3, 9e-4),      # Уменьшение на 10%
            "batch_size": (32, 36),              # Увеличение на 12.5%
        }
        
        assessment = self.module.assess_system_risk(parameter_changes)
        
        # Консервативные изменения должны иметь низкий риск
        self.assertLess(assessment.overall_risk_score, 0.7)
        
        # Система должна разрешить продолжение
        self.assertTrue(assessment.is_safe_to_proceed or assessment.overall_risk_score < 0.8)
    
    def test_mixed_risk_scenario(self):
        """Тест смешанного сценария с разными уровнями риска"""
        parameter_changes = {
            "learning_rate": (1e-3, 1.05e-3),   # Низкий риск: +5%
            "batch_size": (32, 128),             # Средний риск: x4
            "gradient_clip": (1.0, 10.0),        # Высокий риск: x10
        }
        
        assessment = self.module.assess_system_risk(parameter_changes)
        
        # Проверяем разнообразие рисков
        risk_scores = [risk.risk_score for risk in assessment.parameter_risks]
        
        # Должен быть хотя бы один низкий и один высокий риск
        self.assertTrue(any(score < 0.4 for score in risk_scores))  # Низкий риск
        self.assertTrue(any(score > 0.6 for score in risk_scores))  # Высокий риск
        
        # Общий риск должен отражать наличие высокорисковых изменений
        self.assertGreater(assessment.overall_risk_score, 0.4)
    
    def test_database_persistence_scenario(self):
        """Тест сценария сохранения в базу данных"""
        # Выполняем несколько оценок
        for i in range(3):
            risk = self.module.assess_parameter_risk(
                parameter_name=f"param_{i}",
                current_value=float(i),
                proposed_value=float(i + 1),
                parameter_type="general"
            )
        
        # Проверяем, что данные сохранены
        history = self.module.get_parameter_risk_history("param_0", days=1)
        self.assertGreater(len(history), 0)
        
        # Проверяем статус системы
        status = self.module.get_system_status()
        self.assertEqual(status['module_status'], 'active')

if __name__ == '__main__':
    # Настройка логирования для тестов
    import logging
    logging.basicConfig(level=logging.WARNING)  # Уменьшаем verbose для тестов
    
    # Запуск тестов
    unittest.main(verbosity=2) 