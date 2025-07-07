"""
Risk Assessment Module для Enhanced Tacotron2 AI System

Модуль оценки рисков с Monte Carlo симуляциями для анализа безопасности изменений параметров.
Включает importance sampling для rare events и bootstrap sampling для uncertainty estimation.
"""

import logging
import numpy as np
import torch
import json
import sqlite3
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pathlib import Path
import scipy.stats as stats
from collections import defaultdict
import threading
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logger = logging.getLogger(__name__)

@dataclass
class RiskAssessmentConfig:
    """Конфигурация для системы оценки рисков"""
    # Monte Carlo параметры
    n_samples: int = 10000
    confidence_level: float = 0.95
    rare_event_threshold: float = 0.01
    
    # Bootstrap параметры
    n_bootstrap: int = 1000
    bootstrap_alpha: float = 0.05
    
    # Пороговые значения рисков
    high_risk_threshold: float = 0.8
    medium_risk_threshold: float = 0.5
    critical_risk_threshold: float = 0.95
    
    # Параметры симуляций
    simulation_noise_std: float = 0.1
    max_parameter_change: float = 0.5
    stability_window: int = 100
    
    # База данных рисков
    risk_db_path: str = "risk_assessments.db"
    history_retention_days: int = 30

@dataclass
class ParameterRisk:
    """Информация о риске параметра"""
    parameter_name: str
    current_value: float
    proposed_value: float
    risk_score: float
    confidence_interval: Tuple[float, float]
    stability_metric: float
    impact_severity: str  # 'low', 'medium', 'high', 'critical'
    recommendation: str

@dataclass
class SystemRiskAssessment:
    """Полная оценка рисков системы"""
    timestamp: str
    overall_risk_score: float
    parameter_risks: List[ParameterRisk]
    safety_constraints: Dict[str, Any]
    simulation_results: Dict[str, Any]
    recommendations: List[str]
    is_safe_to_proceed: bool

class MonteCarloSimulator:
    """Monte Carlo симулятор для оценки рисков"""
    
    def __init__(self, config: RiskAssessmentConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.MonteCarloSimulator")
        
    def simulate_parameter_change(self, 
                                current_value: float,
                                proposed_value: float,
                                parameter_type: str = "learning_rate") -> Dict[str, Any]:
        """
        Симуляция изменения параметра с Monte Carlo методом
        
        Args:
            current_value: Текущее значение параметра
            proposed_value: Предлагаемое значение
            parameter_type: Тип параметра для специфичного моделирования
            
        Returns:
            Результаты симуляции
        """
        try:
            results = {}
            
            # Генерация случайных сценариев
            scenarios = self._generate_scenarios(current_value, proposed_value, parameter_type)
            
            # Симуляция каждого сценария
            outcomes = []
            for scenario in scenarios:
                outcome = self._simulate_scenario(scenario, parameter_type)
                outcomes.append(outcome)
            
            outcomes = np.array(outcomes)
            
            # Анализ результатов
            results['mean_outcome'] = np.mean(outcomes)
            results['std_outcome'] = np.std(outcomes)
            results['percentiles'] = {
                '5': np.percentile(outcomes, 5),
                '25': np.percentile(outcomes, 25),
                '50': np.percentile(outcomes, 50),
                '75': np.percentile(outcomes, 75),
                '95': np.percentile(outcomes, 95)
            }
            
            # Оценка риска
            risk_score = self._calculate_risk_score(outcomes, current_value, proposed_value)
            results['risk_score'] = risk_score
            
            # Importance sampling для rare events
            rare_event_analysis = self._importance_sampling_analysis(outcomes)
            results['rare_events'] = rare_event_analysis
            
            # Стабильность изменения
            stability_metric = self._calculate_stability_metric(outcomes)
            results['stability_metric'] = stability_metric
            
            self.logger.debug(f"Monte Carlo simulation completed for {parameter_type}: risk={risk_score:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Monte Carlo simulation failed: {e}")
            return {'error': str(e), 'risk_score': 1.0}
    
    def _generate_scenarios(self, current_value: float, proposed_value: float, 
                          parameter_type: str) -> np.ndarray:
        """Генерация сценариев для симуляции"""
        n_samples = self.config.n_samples
        
        # Базовое распределение изменений
        change_magnitude = abs(proposed_value - current_value)
        
        if parameter_type == "learning_rate":
            # Для learning rate используем log-normal распределение
            scenarios = np.random.lognormal(
                mean=np.log(proposed_value),
                sigma=self.config.simulation_noise_std,
                size=n_samples
            )
        elif parameter_type == "batch_size":
            # Для batch size используем нормальное распределение с округлением
            scenarios = np.random.normal(
                loc=proposed_value,
                scale=change_magnitude * self.config.simulation_noise_std,
                size=n_samples
            )
            scenarios = np.round(scenarios).astype(int)
            scenarios = np.clip(scenarios, 1, 1000)  # Разумные границы
        else:
            # Общий случай - нормальное распределение
            scenarios = np.random.normal(
                loc=proposed_value,
                scale=change_magnitude * self.config.simulation_noise_std,
                size=n_samples
            )
        
        return scenarios
    
    def _simulate_scenario(self, scenario_value: float, parameter_type: str) -> float:
        """
        Симуляция одного сценария
        
        Возвращает метрику производительности (0-1, где 1 - лучше)
        """
        # Простая модель влияния параметров на производительность
        if parameter_type == "learning_rate":
            # Оптимальная learning rate около 1e-3
            optimal_lr = 1e-3
            distance = abs(np.log10(scenario_value) - np.log10(optimal_lr))
            performance = np.exp(-distance * 2)  # Экспоненциальный спад
            
        elif parameter_type == "batch_size":
            # Оптимальный batch size в диапазоне 16-64
            if 16 <= scenario_value <= 64:
                performance = 1.0
            else:
                distance = min(abs(scenario_value - 16), abs(scenario_value - 64))
                performance = np.exp(-distance / 50)
                
        elif parameter_type == "gradient_clip":
            # Оптимальный gradient clipping около 1.0
            optimal_clip = 1.0
            distance = abs(scenario_value - optimal_clip)
            performance = np.exp(-distance)
            
        else:
            # Общий случай - квадратичная функция потерь
            performance = np.exp(-abs(scenario_value) * 0.1)
        
        # Добавляем случайный шум
        noise = np.random.normal(0, 0.05)
        performance = np.clip(performance + noise, 0, 1)
        
        return performance
    
    def _calculate_risk_score(self, outcomes: np.ndarray, 
                            current_value: float, proposed_value: float) -> float:
        """Расчет общего риска изменения параметра"""
        # Базовый риск на основе распределения результатов
        mean_performance = np.mean(outcomes)
        std_performance = np.std(outcomes)
        
        # Риск на основе вариативности
        variability_risk = std_performance
        
        # Риск на основе снижения производительности
        performance_risk = max(0, 1 - mean_performance)
        
        # Риск на основе величины изменения (более агрессивная оценка)
        change_magnitude = abs(proposed_value - current_value) / max(abs(current_value), 1e-6)
        # Используем логарифмическую шкалу для экстремальных изменений
        if change_magnitude > 1.0:
            magnitude_risk = 0.5 + 0.4 * min(np.log10(change_magnitude), 2.0)  # Логарифмическая шкала
        else:
            magnitude_risk = min(change_magnitude / self.config.max_parameter_change, 1.0)
        
        # Риск на основе rare events
        negative_outcomes = outcomes[outcomes < 0.5]
        rare_event_risk = len(negative_outcomes) / len(outcomes)
        
        # Экстремальные изменения получают дополнительный штраф
        extreme_change_penalty = 0.0
        if change_magnitude > 10.0:  # Изменение больше чем в 10 раз
            extreme_change_penalty = 0.3
        elif change_magnitude > 5.0:  # Изменение больше чем в 5 раз
            extreme_change_penalty = 0.2
        elif change_magnitude > 2.0:  # Изменение больше чем в 2 раза
            extreme_change_penalty = 0.1
        
        # Комбинированный риск
        base_risk = (
            0.3 * performance_risk +
            0.25 * variability_risk +
            0.25 * magnitude_risk +
            0.2 * rare_event_risk
        )
        
        final_risk = base_risk + extreme_change_penalty
        
        return float(np.clip(final_risk, 0, 1))
    
    def _importance_sampling_analysis(self, outcomes: np.ndarray) -> Dict[str, Any]:
        """Анализ rare events с importance sampling"""
        threshold = self.config.rare_event_threshold
        
        # Определяем rare events как результаты ниже порога
        rare_events = outcomes[outcomes < threshold]
        
        analysis = {
            'rare_event_probability': len(rare_events) / len(outcomes),
            'rare_event_threshold': threshold,
            'worst_case_outcome': np.min(outcomes) if len(outcomes) > 0 else 0,
            'rare_event_severity': np.mean(rare_events) if len(rare_events) > 0 else 0
        }
        
        # Importance sampling для более точной оценки хвостов распределения
        if len(rare_events) > 10:
            # Используем экспоненциальное importance sampling
            weights = np.exp(-outcomes * 5)  # Больший вес для худших результатов
            weighted_probability = np.sum(weights * (outcomes < threshold)) / np.sum(weights)
            analysis['importance_sampled_probability'] = weighted_probability
        
        return analysis
    
    def _calculate_stability_metric(self, outcomes: np.ndarray) -> float:
        """Расчет метрики стабильности"""
        # Коэффициент вариации
        cv = np.std(outcomes) / (np.mean(outcomes) + 1e-6)
        
        # Устойчивость к выбросам (межквартильный размах)
        q75, q25 = np.percentile(outcomes, [75, 25])
        iqr_stability = (q75 - q25) / (np.median(outcomes) + 1e-6)
        
        # Комбинированная метрика (чем меньше, тем стабильнее)
        stability = (cv + iqr_stability) / 2
        
        return stability

class BootstrapAnalyzer:
    """Bootstrap анализатор для uncertainty estimation"""
    
    def __init__(self, config: RiskAssessmentConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.BootstrapAnalyzer")
    
    def calculate_confidence_intervals(self, data: np.ndarray, 
                                     statistic_func: Callable = np.mean) -> Dict[str, float]:
        """
        Расчет доверительных интервалов с bootstrap методом
        
        Args:
            data: Данные для анализа
            statistic_func: Функция для расчета статистики
            
        Returns:
            Доверительные интервалы
        """
        try:
            # Проверка входных данных
            if len(data) == 0:
                self.logger.error("Bootstrap analysis failed: empty data")
                return {'error': 'Empty data provided'}
            
            if len(data) == 1:
                stat_value = statistic_func(data)
                return {
                    'statistic': stat_value,
                    'ci_lower': stat_value,
                    'ci_upper': stat_value,
                    'ci_width': 0.0,
                    'bootstrap_std': 0.0,
                    'bootstrap_mean': stat_value
                }
            
            n_bootstrap = self.config.n_bootstrap
            alpha = self.config.bootstrap_alpha
            
            # Bootstrap sampling
            bootstrap_stats = []
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                stat = statistic_func(bootstrap_sample)
                bootstrap_stats.append(stat)
            
            bootstrap_stats = np.array(bootstrap_stats)
            
            # Расчет доверительных интервалов
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_stats, lower_percentile)
            ci_upper = np.percentile(bootstrap_stats, upper_percentile)
            
            result = {
                'statistic': float(statistic_func(data)),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'ci_width': float(ci_upper - ci_lower),
                'bootstrap_std': float(np.std(bootstrap_stats)),
                'bootstrap_mean': float(np.mean(bootstrap_stats))
            }
            
            self.logger.debug(f"Bootstrap CI calculated: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Bootstrap analysis failed: {e}")
            return {'error': str(e)}
    
    def uncertainty_estimation(self, outcomes: np.ndarray) -> Dict[str, Any]:
        """Полная оценка неопределенности"""
        uncertainties = {}
        
        # Доверительные интервалы для среднего
        mean_ci = self.calculate_confidence_intervals(outcomes, np.mean)
        uncertainties['mean'] = mean_ci
        
        # Доверительные интервалы для медианы
        median_ci = self.calculate_confidence_intervals(outcomes, np.median)
        uncertainties['median'] = median_ci
        
        # Доверительные интервалы для стандартного отклонения
        std_ci = self.calculate_confidence_intervals(outcomes, np.std)
        uncertainties['std'] = std_ci
        
        # Доверительные интервалы для percentiles
        p5_ci = self.calculate_confidence_intervals(outcomes, lambda x: np.percentile(x, 5))
        p95_ci = self.calculate_confidence_intervals(outcomes, lambda x: np.percentile(x, 95))
        
        uncertainties['percentile_5'] = p5_ci
        uncertainties['percentile_95'] = p95_ci
        
        return uncertainties

class RiskDatabase:
    """База данных для хранения истории оценок рисков"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.RiskDatabase")
        self._init_database()
    
    def _init_database(self):
        """Инициализация базы данных"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Таблица оценок рисков
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS risk_assessments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        parameter_name TEXT NOT NULL,
                        current_value REAL NOT NULL,
                        proposed_value REAL NOT NULL,
                        risk_score REAL NOT NULL,
                        confidence_lower REAL,
                        confidence_upper REAL,
                        stability_metric REAL,
                        simulation_results TEXT,
                        recommendation TEXT
                    )
                ''')
                
                # Таблица системных оценок
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_assessments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        overall_risk_score REAL NOT NULL,
                        is_safe_to_proceed BOOLEAN NOT NULL,
                        assessment_data TEXT NOT NULL
                    )
                ''')
                
                conn.commit()
                self.logger.info("Risk database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def save_parameter_risk(self, risk: ParameterRisk):
        """Сохранение оценки риска параметра"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO risk_assessments 
                    (timestamp, parameter_name, current_value, proposed_value,
                     risk_score, confidence_lower, confidence_upper, stability_metric,
                     simulation_results, recommendation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    risk.parameter_name,
                    risk.current_value,
                    risk.proposed_value,
                    risk.risk_score,
                    risk.confidence_interval[0],
                    risk.confidence_interval[1],
                    risk.stability_metric,
                    "",  # Пока без детальных результатов симуляции
                    risk.recommendation
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save parameter risk: {e}")
    
    def save_system_assessment(self, assessment: SystemRiskAssessment):
        """Сохранение системной оценки рисков"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Конвертация в JSON с обработкой numpy типов
                assessment_dict = asdict(assessment)
                # Конвертация numpy типов в Python типы
                assessment_dict = self._convert_numpy_types(assessment_dict)
                assessment_json = json.dumps(assessment_dict)
                
                cursor.execute('''
                    INSERT INTO system_assessments 
                    (timestamp, overall_risk_score, is_safe_to_proceed, assessment_data)
                    VALUES (?, ?, ?, ?)
                ''', (
                    assessment.timestamp,
                    float(assessment.overall_risk_score),
                    bool(assessment.is_safe_to_proceed),
                    assessment_json
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save system assessment: {e}")
    
    def _convert_numpy_types(self, obj):
        """Конвертация numpy типов в стандартные Python типы для JSON сериализации"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def get_parameter_history(self, parameter_name: str, days: int = 7) -> List[Dict]:
        """Получение истории оценок для параметра"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                since_date = (datetime.now() - timedelta(days=days)).isoformat()
                
                cursor.execute('''
                    SELECT * FROM risk_assessments 
                    WHERE parameter_name = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (parameter_name, since_date))
                
                columns = [desc[0] for desc in cursor.description]
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to get parameter history: {e}")
            return []
    
    def cleanup_old_records(self, retention_days: int):
        """Очистка старых записей"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cutoff_date = (datetime.now() - timedelta(days=retention_days)).isoformat()
                
                cursor.execute('DELETE FROM risk_assessments WHERE timestamp < ?', (cutoff_date,))
                cursor.execute('DELETE FROM system_assessments WHERE timestamp < ?', (cutoff_date,))
                
                conn.commit()
                
                self.logger.info(f"Cleaned up records older than {retention_days} days")
                
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

class RiskAssessmentModule:
    """
    Главный модуль оценки рисков для Enhanced Tacotron2 AI System
    
    Использует Monte Carlo симуляции и bootstrap анализ для оценки рисков
    изменений параметров обучения.
    """
    
    def __init__(self, config: Optional[RiskAssessmentConfig] = None):
        self.config = config or RiskAssessmentConfig()
        self.logger = logging.getLogger(__name__)
        
        # Инициализация компонентов
        self.monte_carlo = MonteCarloSimulator(self.config)
        self.bootstrap = BootstrapAnalyzer(self.config)
        self.database = RiskDatabase(self.config.risk_db_path)
        
        # Кэш для истории параметров
        self.parameter_history = defaultdict(list)
        
        # Блокировка для thread safety
        self._lock = threading.Lock()
        
        self.logger.info("Risk Assessment Module initialized successfully")
    
    def assess_parameter_risk(self, 
                            parameter_name: str,
                            current_value: float,
                            proposed_value: float,
                            parameter_type: str = "general") -> ParameterRisk:
        """
        Оценка риска изменения конкретного параметра
        
        Args:
            parameter_name: Название параметра
            current_value: Текущее значение
            proposed_value: Предлагаемое значение
            parameter_type: Тип параметра для специфичного анализа
            
        Returns:
            Оценка риска параметра
        """
        try:
            self.logger.info(f"Assessing risk for {parameter_name}: {current_value} -> {proposed_value}")
            
            # Monte Carlo симуляция
            simulation_results = self.monte_carlo.simulate_parameter_change(
                current_value, proposed_value, parameter_type
            )
            
            if 'error' in simulation_results:
                risk_score = 1.0
                confidence_interval = (0.0, 1.0)
                stability_metric = 1.0
            else:
                risk_score = simulation_results['risk_score']
                
                # Bootstrap анализ для доверительных интервалов
                # Используем результаты Monte Carlo как базовые данные
                fake_outcomes = np.random.normal(
                    simulation_results['mean_outcome'],
                    simulation_results['std_outcome'],
                    size=100
                )
                
                uncertainty = self.bootstrap.uncertainty_estimation(fake_outcomes)
                
                if 'mean' in uncertainty and 'ci_lower' in uncertainty['mean']:
                    confidence_interval = (
                        uncertainty['mean']['ci_lower'],
                        uncertainty['mean']['ci_upper']
                    )
                else:
                    confidence_interval = (risk_score * 0.8, risk_score * 1.2)
                
                stability_metric = simulation_results.get('stability_metric', 0.5)
            
            # Определение серьезности воздействия
            impact_severity = self._classify_impact_severity(risk_score)
            
            # Генерация рекомендации
            recommendation = self._generate_recommendation(
                parameter_name, current_value, proposed_value, 
                risk_score, impact_severity
            )
            
            # Создание объекта риска
            parameter_risk = ParameterRisk(
                parameter_name=parameter_name,
                current_value=current_value,
                proposed_value=proposed_value,
                risk_score=risk_score,
                confidence_interval=confidence_interval,
                stability_metric=stability_metric,
                impact_severity=impact_severity,
                recommendation=recommendation
            )
            
            # Сохранение в базу данных
            self.database.save_parameter_risk(parameter_risk)
            
            # Обновление истории
            with self._lock:
                self.parameter_history[parameter_name].append({
                    'timestamp': datetime.now(),
                    'risk_score': risk_score,
                    'proposed_change': abs(proposed_value - current_value)
                })
            
            self.logger.info(f"Risk assessment completed: {risk_score:.3f} ({impact_severity})")
            
            return parameter_risk
            
        except Exception as e:
            self.logger.error(f"Parameter risk assessment failed: {e}")
            
            # Возвращаем высокий риск в случае ошибки
            return ParameterRisk(
                parameter_name=parameter_name,
                current_value=current_value,
                proposed_value=proposed_value,
                risk_score=1.0,
                confidence_interval=(0.8, 1.0),
                stability_metric=1.0,
                impact_severity="critical",
                recommendation="REJECT: Assessment failed, change not recommended"
            )
    
    def assess_system_risk(self, parameter_changes: Dict[str, Tuple[float, float]]) -> SystemRiskAssessment:
        """
        Оценка общего риска системы при множественных изменениях параметров
        
        Args:
            parameter_changes: Словарь {parameter_name: (current_value, proposed_value)}
            
        Returns:
            Системная оценка рисков
        """
        try:
            self.logger.info(f"Assessing system risk for {len(parameter_changes)} parameter changes")
            
            parameter_risks = []
            
            # Оценка риска каждого параметра
            for param_name, (current, proposed) in parameter_changes.items():
                param_type = self._infer_parameter_type(param_name)
                risk = self.assess_parameter_risk(param_name, current, proposed, param_type)
                parameter_risks.append(risk)
            
            # Расчет общего риска системы
            overall_risk_score = self._calculate_overall_risk(parameter_risks)
            
            # Генерация ограничений безопасности
            safety_constraints = self._generate_safety_constraints(parameter_risks)
            
            # Анализ результатов симуляций
            simulation_results = self._aggregate_simulation_results(parameter_risks)
            
            # Генерация рекомендаций
            recommendations = self._generate_system_recommendations(parameter_risks, overall_risk_score)
            
            # Определение безопасности для продолжения
            is_safe = bool(overall_risk_score < self.config.high_risk_threshold)
            
            # Создание системной оценки
            system_assessment = SystemRiskAssessment(
                timestamp=datetime.now().isoformat(),
                overall_risk_score=float(overall_risk_score),
                parameter_risks=parameter_risks,
                safety_constraints=safety_constraints,
                simulation_results=simulation_results,
                recommendations=recommendations,
                is_safe_to_proceed=is_safe
            )
            
            # Сохранение в базу данных
            self.database.save_system_assessment(system_assessment)
            
            self.logger.info(f"System risk assessment completed: {overall_risk_score:.3f} (safe: {is_safe})")
            
            return system_assessment
            
        except Exception as e:
            self.logger.error(f"System risk assessment failed: {e}")
            
            # Возвращаем критический риск в случае ошибки
            return SystemRiskAssessment(
                timestamp=datetime.now().isoformat(),
                overall_risk_score=1.0,
                parameter_risks=[],
                safety_constraints={'error': str(e)},
                simulation_results={'error': str(e)},
                recommendations=["CRITICAL: System assessment failed"],
                is_safe_to_proceed=False
            )
    
    def _classify_impact_severity(self, risk_score: float) -> str:
        """Классификация серьезности воздействия"""
        if risk_score >= self.config.critical_risk_threshold:
            return "critical"
        elif risk_score >= self.config.high_risk_threshold:
            return "high"
        elif risk_score >= self.config.medium_risk_threshold:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendation(self, param_name: str, current: float, 
                               proposed: float, risk_score: float, severity: str) -> str:
        """Генерация рекомендации по изменению параметра"""
        change_ratio = abs(proposed - current) / max(abs(current), 1e-6)
        
        if severity == "critical":
            return f"REJECT: Critical risk ({risk_score:.3f}). Change not recommended."
        elif severity == "high":
            return f"CAUTION: High risk ({risk_score:.3f}). Consider smaller change or monitoring."
        elif severity == "medium":
            if change_ratio > 0.2:
                return f"MODERATE: Medium risk ({risk_score:.3f}). Recommend gradual change."
            else:
                return f"ACCEPTABLE: Medium risk ({risk_score:.3f}). Proceed with monitoring."
        else:
            return f"APPROVED: Low risk ({risk_score:.3f}). Safe to proceed."
    
    def _infer_parameter_type(self, parameter_name: str) -> str:
        """Определение типа параметра по его названию"""
        name_lower = parameter_name.lower()
        
        if any(keyword in name_lower for keyword in ['lr', 'learning_rate', 'rate']):
            return "learning_rate"
        elif any(keyword in name_lower for keyword in ['batch', 'size']):
            return "batch_size"
        elif any(keyword in name_lower for keyword in ['clip', 'gradient']):
            return "gradient_clip"
        elif any(keyword in name_lower for keyword in ['weight', 'decay']):
            return "weight_decay"
        else:
            return "general"
    
    def _calculate_overall_risk(self, parameter_risks: List[ParameterRisk]) -> float:
        """Расчет общего риска системы"""
        if not parameter_risks:
            return 0.0
        
        # Простая схема: максимальный риск с весом 0.6, средний риск с весом 0.4
        max_risk = max(risk.risk_score for risk in parameter_risks)
        avg_risk = sum(risk.risk_score for risk in parameter_risks) / len(parameter_risks)
        
        overall_risk = 0.6 * max_risk + 0.4 * avg_risk
        
        return min(overall_risk, 1.0)
    
    def _generate_safety_constraints(self, parameter_risks: List[ParameterRisk]) -> Dict[str, Any]:
        """Генерация ограничений безопасности"""
        constraints = {
            'max_allowed_risk': self.config.high_risk_threshold,
            'monitoring_required': [],
            'rejected_changes': [],
            'gradual_changes': []
        }
        
        for risk in parameter_risks:
            if risk.risk_score >= self.config.critical_risk_threshold:
                constraints['rejected_changes'].append(risk.parameter_name)
            elif risk.risk_score >= self.config.high_risk_threshold:
                constraints['gradual_changes'].append(risk.parameter_name)
            elif risk.risk_score >= self.config.medium_risk_threshold:
                constraints['monitoring_required'].append(risk.parameter_name)
        
        return constraints
    
    def _aggregate_simulation_results(self, parameter_risks: List[ParameterRisk]) -> Dict[str, Any]:
        """Агрегация результатов симуляций"""
        if not parameter_risks:
            return {}
        
        total_confidence_width = 0
        total_stability = 0
        high_risk_count = 0
        
        for risk in parameter_risks:
            total_confidence_width += risk.confidence_interval[1] - risk.confidence_interval[0]
            total_stability += risk.stability_metric
            if risk.risk_score >= self.config.high_risk_threshold:
                high_risk_count += 1
        
        n_params = len(parameter_risks)
        
        return {
            'average_confidence_width': total_confidence_width / n_params,
            'average_stability': total_stability / n_params,
            'high_risk_parameter_count': high_risk_count,
            'total_parameters_assessed': n_params
        }
    
    def _generate_system_recommendations(self, parameter_risks: List[ParameterRisk], 
                                       overall_risk: float) -> List[str]:
        """Генерация системных рекомендаций"""
        recommendations = []
        
        # Общая рекомендация по системе
        if overall_risk >= self.config.critical_risk_threshold:
            recommendations.append("SYSTEM CRITICAL: Stop all parameter changes immediately")
        elif overall_risk >= self.config.high_risk_threshold:
            recommendations.append("SYSTEM HIGH RISK: Implement changes gradually with monitoring")
        elif overall_risk >= self.config.medium_risk_threshold:
            recommendations.append("SYSTEM MODERATE RISK: Proceed with caution and monitoring")
        else:
            recommendations.append("SYSTEM LOW RISK: Safe to proceed with normal monitoring")
        
        # Специфичные рекомендации
        critical_params = [r for r in parameter_risks if r.impact_severity == "critical"]
        if critical_params:
            recommendations.append(f"CRITICAL PARAMETERS: {[p.parameter_name for p in critical_params]}")
        
        high_risk_params = [r for r in parameter_risks if r.impact_severity == "high"]
        if high_risk_params:
            recommendations.append(f"HIGH RISK PARAMETERS: {[p.parameter_name for p in high_risk_params]}")
        
        # Рекомендация по мониторингу
        if overall_risk > self.config.medium_risk_threshold:
            recommendations.append("Increase monitoring frequency for all parameters")
        
        return recommendations
    
    def get_parameter_risk_history(self, parameter_name: str, days: int = 7) -> List[Dict]:
        """Получение истории рисков параметра"""
        return self.database.get_parameter_history(parameter_name, days)
    
    def cleanup_old_assessments(self):
        """Очистка старых оценок"""
        self.database.cleanup_old_records(self.config.history_retention_days)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Получение текущего статуса системы оценки рисков"""
        with self._lock:
            status = {
                'module_status': 'active',
                'config': asdict(self.config),
                'parameter_history_count': {
                    name: len(history) 
                    for name, history in self.parameter_history.items()
                },
                'database_path': self.config.risk_db_path
            }
        
        return status

# Удобные функции для быстрого использования
def quick_risk_assessment(parameter_name: str, current_value: float, 
                         proposed_value: float) -> ParameterRisk:
    """Быстрая оценка риска одного параметра"""
    module = RiskAssessmentModule()
    return module.assess_parameter_risk(parameter_name, current_value, proposed_value)

def quick_system_assessment(parameter_changes: Dict[str, Tuple[float, float]]) -> SystemRiskAssessment:
    """Быстрая оценка риска системы"""
    module = RiskAssessmentModule()
    return module.assess_system_risk(parameter_changes)

if __name__ == "__main__":
    # Демонстрация использования
    logging.basicConfig(level=logging.INFO)
    
    # Создание модуля оценки рисков
    risk_module = RiskAssessmentModule()
    
    # Пример оценки одного параметра
    risk = risk_module.assess_parameter_risk(
        parameter_name="learning_rate",
        current_value=1e-3,
        proposed_value=5e-3,
        parameter_type="learning_rate"
    )
    
    print(f"Parameter Risk Assessment:")
    print(f"  Risk Score: {risk.risk_score:.3f}")
    print(f"  Severity: {risk.impact_severity}")
    print(f"  Recommendation: {risk.recommendation}")
    
    # Пример системной оценки
    changes = {
        "learning_rate": (1e-3, 5e-3),
        "batch_size": (32, 64),
        "gradient_clip": (1.0, 2.0)
    }
    
    system_risk = risk_module.assess_system_risk(changes)
    
    print(f"\nSystem Risk Assessment:")
    print(f"  Overall Risk: {system_risk.overall_risk_score:.3f}")
    print(f"  Safe to Proceed: {system_risk.is_safe_to_proceed}")
    print(f"  Recommendations: {system_risk.recommendations}") 