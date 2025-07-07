#!/usr/bin/env python3
"""
🔄 FEEDBACK LOOP MANAGER - Система управления обратными связями
==============================================================

Интеллектуальная система управления с использованием Kalman-фильтров
для точного контроля параметров обучения и стабилизации процесса.

Компоненты:
1. KalmanFilter - фильтрация и предсказание состояний
2. PIDController - пропорционально-интегрально-дифференциальное управление
3. SystemIdentifier - идентификация модели системы
4. FeedbackLoopManager - главный менеджер обратных связей

Автор: Enhanced Tacotron2 AI System
Версия: 1.0.0
"""

import numpy as np
import torch
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
from datetime import datetime
import pickle

# Интеграция с существующими системами
try:
    from unified_logging_system import UnifiedLoggingSystem
    UNIFIED_LOGGING_AVAILABLE = True
except ImportError:
    UNIFIED_LOGGING_AVAILABLE = False

try:
    from context_aware_training_manager import TrainingPhase
    CONTEXT_MANAGER_AVAILABLE = True
except ImportError:
    class TrainingPhase(Enum):
        PRE_ALIGNMENT = "pre_alignment"
        ALIGNMENT_LEARNING = "alignment"
        REFINEMENT = "refinement"
        CONVERGENCE = "convergence"
    CONTEXT_MANAGER_AVAILABLE = False


class ControlMode(Enum):
    """Режимы управления"""
    MANUAL = "manual"           # Ручное управление
    PID = "pid"                # PID контроллер
    ADAPTIVE = "adaptive"       # Адаптивное управление
    KALMAN_PID = "kalman_pid"  # Kalman + PID
    MODEL_PREDICTIVE = "mpc"    # Model Predictive Control


@dataclass
class SystemState:
    """Состояние системы обучения"""
    timestamp: float
    
    # Основные метрики
    loss: float
    learning_rate: float
    gradient_norm: float
    attention_quality: float
    
    # Производные метрики
    loss_derivative: float = 0.0
    lr_derivative: float = 0.0
    convergence_rate: float = 0.0
    
    # Системные параметры
    batch_size: int = 32
    epoch: int = 0
    step: int = 0
    
    # Флаги состояния
    is_stable: bool = True
    needs_intervention: bool = False


@dataclass
class ControlAction:
    """Управляющее воздействие"""
    timestamp: float
    action_type: str
    parameter_name: str
    old_value: float
    new_value: float
    confidence: float
    reason: str


class KalmanFilter:
    """🔍 Kalman Filter для фильтрации и предсказания состояний"""
    
    def __init__(self, state_dim: int = 4, obs_dim: int = 4):
        """
        Args:
            state_dim: Размерность вектора состояния [loss, lr, grad_norm, attention]
            obs_dim: Размерность вектора наблюдений
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # Инициализация матриц Kalman фильтра
        self.x = np.zeros(state_dim)  # Вектор состояния
        self.P = np.eye(state_dim) * 10.0  # Ковариационная матрица ошибки
        
        # Матрица перехода состояния (простая модель)
        self.F = np.eye(state_dim)
        # Добавляем слабую связь между состояниями только если размерность позволяет
        if state_dim >= 2:
            self.F[0, 1] = -0.1  # loss зависит от lr
        if state_dim >= 3:
            self.F[1, 2] = -0.05  # lr зависит от gradient norm
        
        # Матрица наблюдений
        self.H = np.eye(obs_dim, state_dim)
        
        # Ковариационные матрицы шума
        self.Q = np.eye(state_dim) * 0.1  # Шум процесса
        self.R = np.eye(obs_dim) * 1.0    # Шум наблюдений
        
        self.is_initialized = False
        self.prediction_history = deque(maxlen=100)
        
    def predict(self) -> np.ndarray:
        """Предсказание следующего состояния"""
        # Предсказание состояния
        self.x = self.F @ self.x
        
        # Предсказание ковариационной матрицы
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.copy()
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Обновление состояния по наблюдению"""
        if not self.is_initialized:
            self.x = measurement[:self.state_dim]
            self.is_initialized = True
            return self.x.copy()
        
        # Инновация (разность между наблюдением и предсказанием)
        y = measurement - self.H @ self.x
        
        # Инновационная ковариационная матрица
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Обновление состояния
        self.x = self.x + K @ y
        
        # Обновление ковариационной матрицы
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P
        
        # Сохранение в историю
        prediction = self.x.copy()
        self.prediction_history.append({
            'timestamp': datetime.now().timestamp(),
            'state': prediction,
            'innovation': y,
            'uncertainty': np.diag(self.P)
        })
        
        return prediction
    
    def get_uncertainty(self) -> np.ndarray:
        """Получение неопределенности оценки"""
        return np.sqrt(np.diag(self.P))
    
    def get_prediction_confidence(self) -> float:
        """Получение уверенности в предсказании"""
        uncertainty = self.get_uncertainty()
        # Нормализация и инверсия неопределенности
        max_uncertainty = np.max(uncertainty)
        if max_uncertainty > 0:
            confidence = 1.0 / (1.0 + max_uncertainty)
        else:
            confidence = 1.0
        return confidence


class PIDController:
    """🎛️ PID Controller для управления параметрами"""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.01,
                 output_limits: Tuple[float, float] = (-10.0, 10.0)):
        self.kp = kp  # Пропорциональный коэффициент
        self.ki = ki  # Интегральный коэффициент  
        self.kd = kd  # Дифференциальный коэффициент
        
        self.output_limits = output_limits
        
        # Внутренние переменные
        self.setpoint = 0.0
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None
        
        self.error_history = deque(maxlen=100)
        self.output_history = deque(maxlen=100)
        
    def set_setpoint(self, setpoint: float):
        """Установка целевого значения"""
        self.setpoint = setpoint
        
    def update(self, current_value: float, dt: Optional[float] = None) -> float:
        """Обновление PID контроллера"""
        if dt is None:
            current_time = datetime.now().timestamp()
            if self.last_time is not None:
                dt = current_time - self.last_time
            else:
                dt = 0.1  # Значение по умолчанию
            self.last_time = current_time
        
        # Вычисление ошибки
        error = self.setpoint - current_value
        
        # Пропорциональная составляющая
        proportional = self.kp * error
        
        # Интегральная составляющая
        self.integral += error * dt
        integral = self.ki * self.integral
        
        # Дифференциальная составляющая
        if dt > 0:
            derivative = self.kd * (error - self.last_error) / dt
        else:
            derivative = 0.0
        
        # Общий выход
        output = proportional + integral + derivative
        
        # Ограничение выхода
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Защита от интегрального насыщения
        if output == self.output_limits[0] or output == self.output_limits[1]:
            self.integral -= error * dt  # Откат интеграла
        
        # Сохранение состояния
        self.last_error = error
        
        # История для анализа
        self.error_history.append(error)
        self.output_history.append(output)
        
        return output
    
    def tune_parameters(self, error_data: List[float]) -> Dict[str, float]:
        """Автоматическая настройка PID параметров методом Ziegler-Nichols"""
        if len(error_data) < 10:
            return {'kp': self.kp, 'ki': self.ki, 'kd': self.kd}
        
        # Простая эвристика для настройки
        error_variance = np.var(error_data)
        error_mean = np.mean(np.abs(error_data))
        
        # Адаптивная настройка
        if error_variance > 1.0:  # Высокая вариабельность
            new_kp = self.kp * 0.8  # Уменьшаем пропорциональный
            new_ki = self.ki  # Оставляем без изменений
            new_kd = self.kd * 1.2  # Увеличиваем дифференциальный
        elif error_mean > 0.5:   # Большая постоянная ошибка
            new_kp = self.kp * 1.1  # Увеличиваем пропорциональный
            new_ki = self.ki * 1.2  # Увеличиваем интегральный
            new_kd = self.kd  # Оставляем без изменений
        else:  # Система стабильна
            new_kp = self.kp
            new_ki = self.ki
            new_kd = self.kd
        
        # Применение ограничений
        new_kp = np.clip(new_kp, 0.1, 10.0)
        new_ki = np.clip(new_ki, 0.01, 1.0)
        new_kd = np.clip(new_kd, 0.001, 0.1)
        
        return {'kp': new_kp, 'ki': new_ki, 'kd': new_kd}
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Получение метрик производительности контроллера"""
        if len(self.error_history) < 5:
            return {}
        
        errors = list(self.error_history)
        outputs = list(self.output_history)
        
        return {
            'steady_state_error': np.mean(errors[-10:]) if len(errors) >= 10 else np.mean(errors),
            'overshoot': np.max(outputs) - self.setpoint if outputs else 0.0,
            'settling_time': len(errors),  # Упрощенная метрика
            'error_variance': np.var(errors),
            'control_effort': np.sum(np.abs(outputs))
        }


class SystemIdentifier:
    """🔬 System Identifier для идентификации модели системы"""
    
    def __init__(self, input_dim: int = 1, output_dim: int = 1, history_length: int = 50):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.history_length = history_length
        
        # История входов и выходов
        self.input_history = deque(maxlen=history_length)
        self.output_history = deque(maxlen=history_length)
        
        # Идентифицированная модель (простая линейная AR модель)
        self.model_order = 3
        self.A_matrix = None  # Коэффициенты авторегресси
        self.B_matrix = None  # Коэффициенты входов
        
        self.model_confidence = 0.0
        self.last_identification_time = None
        
    def add_data_point(self, input_val: np.ndarray, output_val: np.ndarray):
        """Добавление точки данных"""
        self.input_history.append(input_val.copy())
        self.output_history.append(output_val.copy())
        
        # Периодическая идентификация модели
        if len(self.input_history) >= self.model_order * 3:
            if (self.last_identification_time is None or 
                datetime.now().timestamp() - self.last_identification_time > 30):  # Каждые 30 секунд
                self._identify_model()
                self.last_identification_time = datetime.now().timestamp()
    
    def _identify_model(self):
        """Идентификация модели методом наименьших квадратов"""
        if len(self.output_history) < self.model_order + 2:
            return
        
        try:
            # Подготовка данных для идентификации
            X = []  # Регрессоры
            Y = []  # Выходы
            
            for i in range(self.model_order, len(self.output_history)):
                # Авторегрессионные компоненты
                ar_components = []
                for j in range(self.model_order):
                    ar_components.extend(self.output_history[i-j-1])
                
                # Входные компоненты
                input_components = []
                for j in range(self.model_order):
                    if i-j-1 < len(self.input_history):
                        input_components.extend(self.input_history[i-j-1])
                    else:
                        input_components.extend([0.0] * self.input_dim)
                
                X.append(ar_components + input_components)
                Y.append(self.output_history[i])
            
            if len(X) < 3:  # Недостаточно данных
                return
            
            X = np.array(X)
            Y = np.array(Y)
            
            # Решение методом наименьших квадратов
            theta = np.linalg.lstsq(X, Y, rcond=None)[0]
            
            # Разделение коэффициентов
            ar_coef_size = self.model_order * self.output_dim
            self.A_matrix = theta[:ar_coef_size].reshape(self.output_dim, self.model_order)
            self.B_matrix = theta[ar_coef_size:].reshape(self.output_dim, -1)
            
            # Оценка качества модели
            Y_pred = X @ theta
            mse = np.mean((Y - Y_pred) ** 2)
            
            # Конвертация MSE в уверенность
            self.model_confidence = 1.0 / (1.0 + mse)
            
        except np.linalg.LinAlgError:
            # Если система плохо обусловлена
            self.model_confidence = 0.0
    
    def predict(self, steps_ahead: int = 1) -> Optional[np.ndarray]:
        """Предсказание выхода системы"""
        if self.A_matrix is None or len(self.output_history) < self.model_order:
            return None
        
        try:
            # Используем последние значения для предсказания
            prediction = np.zeros(self.output_dim)
            
            for i in range(self.model_order):
                if i < len(self.output_history):
                    prediction += self.A_matrix[:, i] * self.output_history[-(i+1)]
            
            return prediction
            
        except Exception:
            return None
    
    def get_model_quality(self) -> Dict[str, float]:
        """Получение метрик качества модели"""
        return {
            'confidence': self.model_confidence,
            'data_points': len(self.output_history),
            'model_age': (datetime.now().timestamp() - self.last_identification_time 
                         if self.last_identification_time else 0)
        }


class FeedbackController:
    """🎮 Универсальный контроллер обратной связи"""
    
    def __init__(self, parameter_name: str, target_range: Tuple[float, float],
                 control_mode: ControlMode = ControlMode.KALMAN_PID):
        self.parameter_name = parameter_name
        self.target_range = target_range
        self.control_mode = control_mode
        
        # Инициализация компонентов
        self.kalman_filter = KalmanFilter(state_dim=1, obs_dim=1)
        self.pid_controller = PIDController()
        self.system_identifier = SystemIdentifier(input_dim=1, output_dim=1)
        
        # Настройка PID для конкретного параметра
        if parameter_name == "learning_rate":
            self.pid_controller.kp = 0.5
            self.pid_controller.ki = 0.1
            self.pid_controller.kd = 0.02
            self.pid_controller.output_limits = (-0.001, 0.001)
        elif parameter_name == "loss":
            self.pid_controller.kp = 1.0
            self.pid_controller.ki = 0.2
            self.pid_controller.kd = 0.05
            self.pid_controller.output_limits = (-5.0, 5.0)
        
        # Целевое значение (середина диапазона)
        target_value = (target_range[0] + target_range[1]) / 2
        self.pid_controller.set_setpoint(target_value)
        
        self.action_history = deque(maxlen=100)
        self.last_value = None
        
    def process_measurement(self, current_value: float) -> Optional[ControlAction]:
        """Обработка измерения и генерация управляющего воздействия"""
        timestamp = datetime.now().timestamp()
        
        # Фильтрация через Kalman
        measurement = np.array([current_value])
        if self.control_mode in [ControlMode.KALMAN_PID, ControlMode.ADAPTIVE]:
            filtered_value = self.kalman_filter.update(measurement)[0]
            confidence = self.kalman_filter.get_prediction_confidence()
        else:
            filtered_value = current_value
            confidence = 1.0
        
        # Добавление данных в систему идентификации
        if self.last_value is not None:
            input_data = np.array([self.last_value])
            output_data = np.array([filtered_value])
            self.system_identifier.add_data_point(input_data, output_data)
        
        # Проверка необходимости управляющего воздействия
        if self.target_range[0] <= filtered_value <= self.target_range[1]:
            # Значение в допустимом диапазоне
            self.last_value = filtered_value
            return None
        
        # Генерация управляющего воздействия
        if self.control_mode in [ControlMode.PID, ControlMode.KALMAN_PID]:
            control_output = self.pid_controller.update(filtered_value)
            
            # Вычисление нового значения
            if self.parameter_name == "learning_rate":
                # Для learning rate используем мультипликативное изменение
                new_value = current_value * (1.0 + control_output)
                new_value = np.clip(new_value, 1e-6, 0.1)
            else:
                # Для других параметров - аддитивное
                new_value = current_value + control_output
                
            action = ControlAction(
                timestamp=timestamp,
                action_type="pid_control",
                parameter_name=self.parameter_name,
                old_value=current_value,
                new_value=new_value,
                confidence=confidence,
                reason=f"Value {filtered_value:.4f} outside target range {self.target_range}"
            )
            
            self.action_history.append(action)
            self.last_value = filtered_value
            
            return action
        
        self.last_value = filtered_value
        return None
    
    def get_controller_status(self) -> Dict[str, Any]:
        """Получение статуса контроллера"""
        pid_metrics = self.pid_controller.get_performance_metrics()
        model_quality = self.system_identifier.get_model_quality()
        
        return {
            'parameter_name': self.parameter_name,
            'control_mode': self.control_mode.value,
            'target_range': self.target_range,
            'actions_taken': len(self.action_history),
            'kalman_confidence': self.kalman_filter.get_prediction_confidence(),
            'pid_metrics': pid_metrics,
            'model_quality': model_quality,
            'last_uncertainty': self.kalman_filter.get_uncertainty().tolist()
        }


class FeedbackLoopManager:
    """🔄 Главный менеджер системы обратных связей"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.controllers: Dict[str, FeedbackController] = {}
        self.system_state_history = deque(maxlen=1000)
        self.global_control_mode = ControlMode.KALMAN_PID
        
        # Загрузка конфигурации
        self.config = self._load_config(config_file)
        
        # Инициализация контроллеров по умолчанию
        self._initialize_default_controllers()
        
        self.logger = self._setup_logger()
        self.logger.info("🔄 Feedback Loop Manager инициализирован")
        
        # Статистика
        self.total_interventions = 0
        self.successful_interventions = 0
        
    def _setup_logger(self):
        if UNIFIED_LOGGING_AVAILABLE:
            return UnifiedLoggingSystem().register_component("FeedbackLoopManager")
        else:
            logger = logging.getLogger("FeedbackLoopManager")
            logger.setLevel(logging.INFO)
            return logger
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Загрузка конфигурации"""
        default_config = {
            'learning_rate_target': (1e-4, 1e-3),
            'loss_target': (0.5, 5.0),
            'gradient_norm_target': (0.1, 10.0),
            'attention_quality_target': (0.3, 1.0),
            'intervention_threshold': 0.7,
            'max_interventions_per_minute': 5
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"⚠️ Ошибка загрузки конфигурации: {e}")
        
        return default_config
    
    def _initialize_default_controllers(self):
        """Инициализация контроллеров по умолчанию"""
        # Контроллер learning rate
        self.controllers['learning_rate'] = FeedbackController(
            parameter_name='learning_rate',
            target_range=self.config['learning_rate_target'],
            control_mode=self.global_control_mode
        )
        
        # Контроллер loss
        self.controllers['loss'] = FeedbackController(
            parameter_name='loss',
            target_range=self.config['loss_target'],
            control_mode=self.global_control_mode
        )
        
        # Контроллер gradient norm
        self.controllers['gradient_norm'] = FeedbackController(
            parameter_name='gradient_norm',
            target_range=self.config['gradient_norm_target'],
            control_mode=self.global_control_mode
        )
        
        # Контроллер attention quality
        self.controllers['attention_quality'] = FeedbackController(
            parameter_name='attention_quality',
            target_range=self.config['attention_quality_target'],
            control_mode=self.global_control_mode
        )
    
    def update_system_state(self, state: SystemState) -> List[ControlAction]:
        """Обновление состояния системы и генерация управляющих воздействий"""
        # Сохранение состояния
        self.system_state_history.append(state)
        
        # Обработка каждым контроллером
        actions = []
        
        for param_name, controller in self.controllers.items():
            # Извлечение соответствующего значения из состояния
            if hasattr(state, param_name):
                current_value = getattr(state, param_name)
                
                # Обработка контроллером
                action = controller.process_measurement(current_value)
                
                if action is not None:
                    actions.append(action)
                    self.total_interventions += 1
                    
                    self.logger.info(f"🎛️ Управляющее воздействие: {param_name} "
                                   f"{action.old_value:.6f} → {action.new_value:.6f}")
        
        # Проверка стабильности системы
        self._check_system_stability(state)
        
        return actions
    
    def _check_system_stability(self, state: SystemState):
        """Проверка стабильности системы"""
        if len(self.system_state_history) < 10:
            return
        
        # Анализ последних состояний
        recent_states = list(self.system_state_history)[-10:]
        
        # Проверка тренда loss
        losses = [s.loss for s in recent_states]
        loss_trend = np.polyfit(range(len(losses)), losses, 1)[0]
        
        # Проверка вариабельности
        loss_variance = np.var(losses)
        
        # Определение нестабильности
        is_unstable = (
            loss_trend > 0.1 or          # Растущий loss
            loss_variance > 10.0 or      # Высокая вариабельность
            state.gradient_norm > 100.0   # Взрыв градиентов
        )
        
        if is_unstable and state.is_stable:
            state.is_stable = False
            state.needs_intervention = True
            self.logger.warning("⚠️ Обнаружена нестабильность системы")
        elif not is_unstable and not state.is_stable:
            state.is_stable = True
            state.needs_intervention = False
            self.logger.info("✅ Система стабилизирована")
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Получение диагностики системы"""
        diagnostics = {
            'total_interventions': self.total_interventions,
            'success_rate': (self.successful_interventions / max(1, self.total_interventions)),
            'system_state_count': len(self.system_state_history),
            'controllers': {}
        }
        
        # Диагностика каждого контроллера
        for name, controller in self.controllers.items():
            diagnostics['controllers'][name] = controller.get_controller_status()
        
        # Общая стабильность системы
        if self.system_state_history:
            recent_states = list(self.system_state_history)[-5:]
            stability_score = sum(1 for s in recent_states if s.is_stable) / len(recent_states)
            diagnostics['system_stability'] = stability_score
        else:
            diagnostics['system_stability'] = 1.0
        
        return diagnostics
    
    def set_control_mode(self, mode: ControlMode):
        """Установка режима управления для всех контроллеров"""
        self.global_control_mode = mode
        
        for controller in self.controllers.values():
            controller.control_mode = mode
            
        self.logger.info(f"🎛️ Режим управления изменен на: {mode.value}")
    
    def add_custom_controller(self, parameter_name: str, target_range: Tuple[float, float]):
        """Добавление пользовательского контроллера"""
        self.controllers[parameter_name] = FeedbackController(
            parameter_name=parameter_name,
            target_range=target_range,
            control_mode=self.global_control_mode
        )
        
        self.logger.info(f"➕ Добавлен контроллер для параметра: {parameter_name}")
    
    def remove_controller(self, parameter_name: str):
        """Удаление контроллера"""
        if parameter_name in self.controllers:
            del self.controllers[parameter_name]
            self.logger.info(f"➖ Удален контроллер для параметра: {parameter_name}")
    
    def get_recommendations(self) -> List[str]:
        """Получение рекомендаций по улучшению системы"""
        recommendations = []
        
        if len(self.system_state_history) < 50:
            recommendations.append("Накопите больше данных для лучшего анализа (рекомендуется >50 состояний)")
        
        # Анализ производительности контроллеров
        for name, controller in self.controllers.items():
            status = controller.get_controller_status()
            
            if status['kalman_confidence'] < 0.5:
                recommendations.append(f"Низкая уверенность Kalman фильтра для {name}. "
                                     "Рассмотрите настройку параметров шума.")
            
            if 'error_variance' in status.get('pid_metrics', {}):
                if status['pid_metrics']['error_variance'] > 1.0:
                    recommendations.append(f"Высокая вариабельность ошибки для {name}. "
                                         "Рассмотрите перенастройку PID параметров.")
        
        # Общие рекомендации
        diagnostics = self.get_system_diagnostics()
        if diagnostics['system_stability'] < 0.7:
            recommendations.append("Низкая стабильность системы. Рассмотрите более консервативные настройки.")
        
        return recommendations


# Функции создания и интеграции
def create_feedback_loop_manager(config_file: Optional[str] = None) -> FeedbackLoopManager:
    """Создание настроенного Feedback Loop Manager"""
    return FeedbackLoopManager(config_file=config_file)


if __name__ == "__main__":
    # Демонстрация использования
    print("🔄 Feedback Loop Manager демо")
    
    # Создание менеджера
    manager = create_feedback_loop_manager()
    
    # Симуляция состояния системы
    test_state = SystemState(
        timestamp=datetime.now().timestamp(),
        loss=8.5,
        learning_rate=0.0005,
        gradient_norm=2.5,
        attention_quality=0.4
    )
    
    print(f"📊 Обработка состояния системы: loss={test_state.loss}")
    
    # Обработка состояния
    actions = manager.update_system_state(test_state)
    
    if actions:
        print(f"🎛️ Сгенерировано {len(actions)} управляющих воздействий")
        for action in actions:
            print(f"   {action.parameter_name}: {action.old_value:.6f} → {action.new_value:.6f}")
    else:
        print("✅ Система в норме, вмешательство не требуется")
    
    # Диагностика
    diagnostics = manager.get_system_diagnostics()
    print(f"📈 Диагностика системы: стабильность = {diagnostics['system_stability']:.2f}")
    
    # Рекомендации
    recommendations = manager.get_recommendations()
    if recommendations:
        print("💡 Рекомендации:")
        for rec in recommendations:
            print(f"   • {rec}")
    
    print("✅ Feedback Loop Manager готов к использованию!") 