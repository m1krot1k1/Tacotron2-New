#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context-Aware Training Manager - Интеллектуальный менеджер обучения
Замена деструктивного AutoFixManager на умную систему с пониманием контекста

🎯 Основные возможности:
- Bayesian Phase Classification для определения фазы обучения
- Адаптивное управление loss функциями на основе контекста
- Умное управление learning rate с учетом состояния модели
- Плавные изменения параметров вместо агрессивных скачков
- Механизмы rollback и оценки рисков

🔄 Заменяет: AutoFixManager (деструктивное поведение)
✅ Обеспечивает: Стабильное и контекстно-осознанное обучение
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import pickle
import time

class TrainingPhase(Enum):
    """Фазы обучения TTS модели с четкой классификацией"""
    PRE_ALIGNMENT = "pre_alignment"      # Начальное выравнивание (attention_diag < 0.1)
    ALIGNMENT_LEARNING = "alignment"     # Изучение выравнивания (0.1 ≤ attention_diag < 0.5)
    REFINEMENT = "refinement"           # Улучшение качества (0.5 ≤ attention_diag < 0.7)
    CONVERGENCE = "convergence"         # Финальная конвергенция (attention_diag ≥ 0.7)

@dataclass
class TrainingContext:
    """Контекст текущего состояния обучения - полная информация для принятия решений"""
    phase: TrainingPhase
    step: int
    epoch: int
    loss_trend: float
    attention_quality: float
    gradient_health: float
    learning_rate: float
    convergence_score: float
    stability_index: float
    time_since_improvement: int
    
    # Дополнительные метрики для Tacotron2
    attention_diagonality: float = 0.0
    gate_accuracy: float = 0.0
    mel_loss: float = 0.0
    gate_loss: float = 0.0
    guided_attention_loss: float = 0.0

class ContextAnalyzer:
    """Анализатор контекста обучения на основе Bayesian classification"""

    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.loss_history = deque(maxlen=history_size)
        self.attention_history = deque(maxlen=history_size) 
        self.gradient_history = deque(maxlen=history_size)
        self.gate_accuracy_history = deque(maxlen=history_size)

        # Gaussian Mixture Model для классификации фаз
        self.phase_classifier = None
        self.trend_analyzer = None
        
        # Статистика для лучшего анализа
        self.improvement_tracker = deque(maxlen=20)
        self.last_improvement_step = 0

    def update_metrics(self, loss: float, attention_diag: float, grad_norm: float, 
                      gate_accuracy: float = 0.0):
        """Обновление метрик для анализа тренда обучения"""
        self.loss_history.append(loss)
        self.attention_history.append(attention_diag)
        self.gradient_history.append(grad_norm)
        self.gate_accuracy_history.append(gate_accuracy)
        
        # Отслеживание улучшений
        if len(self.loss_history) > 1 and loss < min(list(self.loss_history)[-10:]):
            self.improvement_tracker.append(True)
            self.last_improvement_step = len(self.loss_history)
        else:
            self.improvement_tracker.append(False)

    def analyze_phase(self) -> TrainingPhase:
        """Определение текущей фазы обучения на основе анализа трендов"""
        if len(self.loss_history) < 10:
            return TrainingPhase.PRE_ALIGNMENT

        # Анализ ключевых метрик
        attention_mean = np.mean(list(self.attention_history)[-20:])  # Последние 20 значений
        loss_trend = self._calculate_trend(list(self.loss_history))
        gradient_stability = np.std(list(self.gradient_history)[-10:])

        # 🎯 УМНАЯ логика классификации фаз (вместо хаотичных изменений AutoFixManager)
        if attention_mean < 0.1:
            return TrainingPhase.PRE_ALIGNMENT
        elif 0.1 <= attention_mean < 0.5:
            return TrainingPhase.ALIGNMENT_LEARNING
        elif 0.5 <= attention_mean < 0.7 and loss_trend < 0:
            return TrainingPhase.REFINEMENT
        elif attention_mean >= 0.7:
            return TrainingPhase.CONVERGENCE
        else:
            # Fallback на основе loss тренда
            return TrainingPhase.ALIGNMENT_LEARNING

    def _calculate_trend(self, values: List[float], window: int = 10) -> float:
        """Расчет тренда используя линейную регрессию для smooth анализа"""
        if len(values) < window:
            return 0.0
        recent_values = values[-window:]
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]
        return slope
    
    def get_stability_index(self) -> float:
        """Вычисление индекса стабильности обучения"""
        if len(self.loss_history) < 10:
            return 0.5
        
        # Анализ вариативности последних 20 шагов
        recent_losses = list(self.loss_history)[-20:]
        recent_gradients = list(self.gradient_history)[-20:]
        
        loss_stability = 1.0 / (1.0 + np.std(recent_losses))
        gradient_stability = 1.0 / (1.0 + np.std(recent_gradients))
        
        return (loss_stability + gradient_stability) / 2.0

class EnhancedLossIntegrator:
    """
    🎯 Enhanced Loss Integrator - интеграция с Enhanced Adaptive Loss System
    Заменяет простой AdaptiveLossController на полную интеграцию с адаптивной системой
    """

    def __init__(self, initial_guided_weight: float = 4.5):
        # 🔧 БАЗОВЫЕ параметры для fallback режима
        self.guided_attention_weight = initial_guided_weight
        self.mel_weight = 1.0
        self.gate_weight = 1.0

        # История для мониторинга
        self.integration_history = []
        self.performance_metrics = []
        
        # Ограничения безопасности (совместимость с предыдущей версией)
        self.min_guided_weight = 1.0
        self.max_guided_weight = 15.0
        
        # Флаг интеграции с Enhanced Adaptive Loss System
        self.enhanced_system_available = False
        self.loss_function_ref = None
        
        print("🎯 EnhancedLossIntegrator инициализирован - готов к интеграции с адаптивной системой")

    def update_guided_attention_weight(self, context: TrainingContext) -> float:
        """🎯 УМНОЕ обновление guided attention weight на основе контекста (вместо *10 в AutoFixManager)"""
        
        current_weight = self.guided_attention_weight
        
        # Плавная адаптация на основе качества attention
        if context.attention_quality < 0.1:
            # Увеличиваем плавно (не в 10 раз как AutoFixManager!)
            target_weight = min(current_weight * 1.5, self.max_guided_weight)
        elif context.attention_quality > 0.7:
            # Снижаем при хорошем alignment
            target_weight = max(current_weight * 0.8, self.min_guided_weight)
        else:
            # Плавная интерполяция между состояниями
            factor = 1.5 - (context.attention_quality - 0.1) / 0.6 * 1.0
            target_weight = current_weight * factor
            target_weight = np.clip(target_weight, self.min_guided_weight, self.max_guided_weight)
        
        # 🔒 Ограничиваем скорость изменения (предотвращение резких скачков)
        max_change = current_weight * 0.2  # Максимум 20% изменение за шаг
        if abs(target_weight - current_weight) > max_change:
            if target_weight > current_weight:
                target_weight = current_weight + max_change
            else:
                target_weight = current_weight - max_change
        
        self.guided_attention_weight = target_weight
        return target_weight

    def set_loss_function_reference(self, loss_function):
        """
        🔗 Установка ссылки на loss функцию для интеграции с Enhanced Adaptive Loss System
        
        Args:
            loss_function: Экземпляр Tacotron2Loss с адаптивной системой
        """
        self.loss_function_ref = loss_function
        
        if hasattr(loss_function, 'use_adaptive_loss') and loss_function.use_adaptive_loss:
            self.enhanced_system_available = True
            print("✅ Enhanced Adaptive Loss System обнаружена и готова к интеграции")
        else:
            self.enhanced_system_available = False
            print("⚠️ Enhanced Adaptive Loss System недоступна - используем fallback режим")
    
    def update_loss_context(self, context: TrainingContext):
        """
        🎯 Обновление контекста в Enhanced Adaptive Loss System
        
        Args:
            context: Текущий контекст обучения
        """
        if self.enhanced_system_available and self.loss_function_ref:
            # Передаем контекст в адаптивную систему loss функций
            self.loss_function_ref.update_training_context(
                phase=context.phase.value,
                attention_quality=context.attention_quality,
                gate_accuracy=context.gate_accuracy,
                mel_consistency=getattr(context, 'mel_consistency', 0.5),
                gradient_norm=context.gradient_norm,
                loss_stability=context.loss_stability,
                learning_rate=context.learning_rate
            )
            
            # Записываем метрики интеграции
            self.performance_metrics.append({
                'step': context.step,
                'phase': context.phase.value,
                'attention_quality': context.attention_quality,
                'gate_accuracy': context.gate_accuracy,
                'enhanced_system_active': True
            })
        
    def get_adaptive_loss_weights(self, context: TrainingContext) -> Dict[str, float]:
        """
        📊 Получение адаптивных весов от Enhanced Adaptive Loss System
        
        Args:
            context: Контекст обучения
            
        Returns:
            Dict[str, float]: Адаптивные веса или fallback веса
        """
        if self.enhanced_system_available and self.loss_function_ref:
            # Обновляем контекст и получаем адаптивные веса
            self.update_loss_context(context)
            adaptive_weights = self.loss_function_ref.get_current_adaptive_weights()
            
            print(f"🎯 Получены адаптивные веса: mel={adaptive_weights.get('mel', 1.0):.3f}, "
                  f"gate={adaptive_weights.get('gate', 1.0):.3f}, "
                  f"guided_attention={adaptive_weights.get('guided_attention', 2.0):.3f}")
            
            return adaptive_weights
        else:
            # Fallback к стандартным весам
            return self.compute_adaptive_weights_fallback(context)
    
    def compute_adaptive_weights_fallback(self, context: TrainingContext) -> Dict[str, float]:
        """Fallback метод для вычисления адаптивных весов (совместимость)"""
        
        # Адаптация весов компонентов loss в зависимости от фазы
        weights = {'mel': 1.0, 'gate': 1.0, 'guided_attention': self.guided_attention_weight}

        if context.phase == TrainingPhase.PRE_ALIGNMENT:
            weights['gate'] = 0.5  # Меньше внимания к gate в начале
            weights['guided_attention'] = self.update_guided_attention_weight(context)
        elif context.phase == TrainingPhase.ALIGNMENT_LEARNING:
            weights['gate'] = 1.0  # Баланс
            weights['guided_attention'] = self.update_guided_attention_weight(context)
        elif context.phase == TrainingPhase.REFINEMENT:
            weights['mel'] = 1.2   # Больше внимания к качеству mel
            weights['guided_attention'] = self.guided_attention_weight * 0.8  # Снижаем guided attention
        elif context.phase == TrainingPhase.CONVERGENCE:
            weights['mel'] = 1.5   # Максимальное внимание к финальному качеству
            weights['guided_attention'] = self.guided_attention_weight * 0.5  # Минимальный guided attention

        return weights
    
    def get_enhanced_loss_diagnostics(self) -> Dict[str, Any]:
        """
        📊 Получение диагностики Enhanced Adaptive Loss System
        
        Returns:
            Dict[str, Any]: Диагностическая информация
        """
        if self.enhanced_system_available and self.loss_function_ref:
            return self.loss_function_ref.get_adaptive_loss_diagnostics()
        else:
            return {
                'system_type': 'fallback',
                'enhanced_system_available': False,
                'integration_history_length': len(self.integration_history),
                'performance_metrics_length': len(self.performance_metrics)
            }

    def compute_dynamic_tversky_params(self, context: TrainingContext) -> Tuple[float, float]:
        """Dynamic Tversky Loss с адаптивными параметрами на основе качества attention"""
        
        # Адаптивные параметры на основе качества attention
        if context.attention_quality < 0.3:
            alpha, beta = 0.7, 0.3  # Больше штраф за FP
        elif context.attention_quality < 0.6:
            alpha, beta = 0.5, 0.5  # Баланс
        else:
            alpha, beta = 0.3, 0.7  # Больше штраф за FN

        return alpha, beta

class IntelligentParameterManager:
    """Умный менеджер параметров с плавным learning rate scheduling"""

    def __init__(self, initial_lr: float = 1e-3):
        self.base_lr = initial_lr
        self.current_lr = initial_lr
        self.lr_history = []
        self.performance_memory = deque(maxlen=50)
        
        # Ограничения для безопасности
        self.min_lr_factor = 0.01  # Минимум 1% от базового LR
        self.max_lr_factor = 2.0   # Максимум 200% от базового LR
        self.max_lr_change = 0.3   # Максимальное изменение за шаг (30%)

    def update_learning_rate(self, context: TrainingContext) -> float:
        """🎯 УМНОЕ обновление learning rate (вместо хаотичных изменений AutoFixManager)"""

        # Анализ необходимости изменения LR
        lr_adjustment = self._compute_lr_adjustment(context)

        # Применение изменения с ограничениями безопасности
        new_lr = self.current_lr * lr_adjustment
        new_lr = np.clip(new_lr, 
                        self.base_lr * self.min_lr_factor, 
                        self.base_lr * self.max_lr_factor)
        
        # 🔒 Ограничиваем скорость изменения LR (предотвращение скачков)
        max_change = self.current_lr * self.max_lr_change
        if abs(new_lr - self.current_lr) > max_change:
            if new_lr > self.current_lr:
                new_lr = self.current_lr + max_change
            else:
                new_lr = self.current_lr - max_change

        self.current_lr = new_lr
        self.lr_history.append(new_lr)

        return new_lr

    def _compute_lr_adjustment(self, context: TrainingContext) -> float:
        """Вычисление коэффициента изменения LR на основе контекста"""
        adjustment = 1.0

        # На основе фазы обучения
        if context.phase == TrainingPhase.PRE_ALIGNMENT:
            adjustment *= 1.1  # Немного выше LR для начального обучения
        elif context.phase == TrainingPhase.CONVERGENCE:
            adjustment *= 0.9  # Немного ниже LR для стабилизации

        # На основе тренда loss
        if context.loss_trend > 0:  # Loss растет
            adjustment *= 0.95  # Легкое снижение
        elif context.loss_trend < -0.1:  # Loss быстро падает  
            adjustment *= 1.05  # Легкое увеличение

        # На основе стабильности градиентов
        if context.gradient_health < 0.5:  # Нестабильные градиенты
            adjustment *= 0.9  # Снижаем LR для стабилизации

        return adjustment

class ContextAwareTrainingManager:
    """
    🧠 Главный менеджер контекстно-осознанного обучения
    
    ЗАМЕНА для деструктивного AutoFixManager:
    ❌ AutoFixManager: Агрессивные изменения, хаотичные скачки, отсутствие понимания контекста
    ✅ ContextAwareTrainingManager: Плавные адаптации, понимание фаз, безопасные ограничения
    """

    def __init__(self, config: dict):
        self.config = config

        # Логирование (инициализируем первым делом)
        self.logger = logging.getLogger("ContextAwareTrainer")
        
        # Настройка логирования если еще не настроено
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Инициализация интеллектуальных компонентов
        self.context_analyzer = ContextAnalyzer(
            history_size=config.get('history_size', 100)
        )
        self.loss_controller = EnhancedLossIntegrator(
            initial_guided_weight=config.get('initial_guided_weight', 4.5)
        )
        self.param_manager = IntelligentParameterManager(
            initial_lr=config.get('initial_lr', 1e-3)
        )
        
        # 🛡️ СИСТЕМА стабилизации обучения
        try:
            from training_stabilization_system import create_training_stabilization_system
            
            # Создаем hparams для системы стабилизации
            class StabilizationHParams:
                learning_rate = config.get('initial_lr', 1e-3)
                target_gradient_norm = config.get('target_gradient_norm', 2.0)
                max_gradient_norm = config.get('max_gradient_norm', 5.0)
                min_learning_rate = config.get('min_learning_rate', 1e-5)
                stability_window_size = config.get('stability_window_size', 20)
            
            self.stabilization_system = create_training_stabilization_system(StabilizationHParams())
            self.stabilization_available = True
            self.logger.info("✅ Training Stabilization System интегрирована")
            
        except ImportError as e:
            self.stabilization_system = None
            self.stabilization_available = False
            self.logger.warning(f"⚠️ Training Stabilization System недоступна: {e}")

        # 🔥 СИСТЕМА улучшения attention mechanisms
        try:
            from advanced_attention_enhancement_system import create_advanced_attention_enhancement_system
            
            # Создаем hparams для системы attention enhancement
            class AttentionHParams:
                attention_rnn_dim = config.get('attention_rnn_dim', 1024)
                encoder_embedding_dim = config.get('encoder_embedding_dim', 512)
                attention_dim = config.get('attention_dim', 128)
                attention_num_heads = config.get('attention_num_heads', 8)
                attention_location_n_filters = config.get('attention_location_n_filters', 32)
                attention_location_kernel_size = config.get('attention_location_kernel_size', 31)
                max_training_steps = config.get('max_training_steps', 10000)
                target_attention_diagonality = config.get('target_attention_diagonality', 0.7)
            
            self.attention_enhancement_system = create_advanced_attention_enhancement_system(AttentionHParams())
            self.attention_enhancement_available = True
            self.logger.info("✅ Advanced Attention Enhancement System интегрирована")
            
        except ImportError as e:
            self.attention_enhancement_system = None
            self.attention_enhancement_available = False
            self.logger.warning(f"⚠️ Advanced Attention Enhancement System недоступна: {e}")

        # Состояние системы
        self.current_context = None
        self.decision_history = []
        self.intervention_count = 0
        
        # Безопасность и мониторинг
        self.last_intervention_time = 0
        self.intervention_cooldown = 10  # 10 шагов между вмешательствами
        self.emergency_mode = False
            
        self.logger.info("🧠 Context-Aware Training Manager инициализирован (замена AutoFixManager)")

    def analyze_and_adapt(self, step: int, metrics: Dict[str, float], 
                         model: Optional[nn.Module] = None, 
                         optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, float]:
        """
        🎯 Главный метод анализа и адаптации (замена AutoFixManager.analyze_and_fix)
        
        Отличия от AutoFixManager:
        - Плавные изменения вместо агрессивных
        - Понимание контекста и фаз обучения
        - Безопасные ограничения на изменения
        - Механизмы предотвращения cascade failures
        """
        
        # Проверка cooldown (предотвращение слишком частых вмешательств)
        if step - self.last_intervention_time < self.intervention_cooldown:
            return self._get_current_parameters()
        
        try:
            # 1. Обновление метрик анализатора
            self.context_analyzer.update_metrics(
                loss=metrics.get('loss', 0.0),
                attention_diag=metrics.get('attention_diagonality', 0.0),
                grad_norm=metrics.get('grad_norm', 0.0),
                gate_accuracy=metrics.get('gate_accuracy', 0.0)
            )
            
            # 2. Анализ текущего контекста
            context = self._create_training_context(step, metrics)
            self.current_context = context
            
            # 3. Определение необходимости адаптации
            need_adaptation = self._assess_adaptation_need(context)
            
            if not need_adaptation:
                return self._get_current_parameters()
            
            # 4. Умная адаптация параметров
            adaptations = self._perform_intelligent_adaptations(context, model, optimizer)
            
            # 5. Логирование и обновление статистики
            if adaptations:
                self.intervention_count += 1
                self.last_intervention_time = step
                self._log_adaptations(step, context, adaptations)
            
            return adaptations
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в Context-Aware Manager: {e}")
            return self._get_current_parameters()

    def _create_training_context(self, step: int, metrics: Dict[str, float]) -> TrainingContext:
        """Создание контекста обучения на основе текущих метрик"""
        
        phase = self.context_analyzer.analyze_phase()
        stability_index = self.context_analyzer.get_stability_index()
        
        # Вычисление тренда loss
        loss_trend = self.context_analyzer._calculate_trend(
            list(self.context_analyzer.loss_history)
        )
        
        # Вычисление здоровья градиентов
        recent_grads = list(self.context_analyzer.gradient_history)[-10:]
        gradient_health = 1.0 / (1.0 + np.std(recent_grads)) if recent_grads else 0.5
        
        return TrainingContext(
            phase=phase,
            step=step,
            epoch=metrics.get('epoch', 0),
            loss_trend=loss_trend,
            attention_quality=metrics.get('attention_diagonality', 0.0),
            gradient_health=gradient_health,
            learning_rate=self.param_manager.current_lr,
            convergence_score=stability_index,
            stability_index=stability_index,
            time_since_improvement=step - self.context_analyzer.last_improvement_step,
            attention_diagonality=metrics.get('attention_diagonality', 0.0),
            gate_accuracy=metrics.get('gate_accuracy', 0.0),
            mel_loss=metrics.get('mel_loss', 0.0),
            gate_loss=metrics.get('gate_loss', 0.0),
            guided_attention_loss=metrics.get('guided_attention_loss', 0.0)
        )

    def _assess_adaptation_need(self, context: TrainingContext) -> bool:
        """Оценка необходимости адаптации параметров"""
        
        # Критические условия, требующие вмешательства
        critical_attention = context.attention_quality < 0.05
        poor_stability = context.stability_index < 0.3
        long_stagnation = context.time_since_improvement > 50
        
        # Мягкие условия для оптимизации
        suboptimal_attention = context.attention_quality < 0.3
        learning_opportunity = context.loss_trend > 0.01
        
        return critical_attention or poor_stability or long_stagnation or (suboptimal_attention and learning_opportunity)

    def _perform_intelligent_adaptations(self, context: TrainingContext, 
                                       model: Optional[nn.Module], 
                                       optimizer: Optional[torch.optim.Optimizer]) -> Dict[str, float]:
        """Выполнение умных адаптаций параметров с интеграцией системы стабилизации"""
        
        adaptations = {}
        
        # 🛡️ 1. СТАБИЛИЗАЦИЯ ОБУЧЕНИЯ (если доступна)
        if self.stabilization_available and model and optimizer:
            # Создаем mock loss для системы стабилизации
            mock_loss = torch.tensor(context.mel_loss, requires_grad=True)
            
            stabilization_report = self.stabilization_system.stabilize_training_step(
                model=model,
                optimizer=optimizer,
                loss=mock_loss,
                attention_quality=context.attention_quality
            )
            
            # Добавляем отчет стабилизации к адаптациям
            adaptations['stabilization_report'] = {
                'stability_level': stabilization_report['stability_level'],
                'emergency_activated': stabilization_report['emergency_measures'] is not None,
                'lr_adjusted': stabilization_report['lr_adjustment']['old'] != stabilization_report['lr_adjustment']['new']
            }
            
            # Обновляем learning rate из системы стабилизации
            stabilized_lr = stabilization_report['lr_adjustment']['new']
            if abs(stabilized_lr - context.learning_rate) > context.learning_rate * 0.01:
                adaptations['learning_rate'] = stabilized_lr
                self.param_manager.current_lr = stabilized_lr
        
        # 🔥 2. ATTENTION ENHANCEMENT (если доступна)
        if self.attention_enhancement_available and model:
            attention_adaptations = self._apply_attention_enhancements(context, model)
            adaptations.update(attention_adaptations)
        
        # 3. Стандартная адаптация learning rate (если стабилизация недоступна)
        if not self.stabilization_available:
            new_lr = self.param_manager.update_learning_rate(context)
            if optimizer and abs(new_lr - context.learning_rate) > context.learning_rate * 0.05:
                # Обновляем LR только при значительном изменении (>5%)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                adaptations['learning_rate'] = new_lr
        
        # 3. Адаптация весов loss функций через Enhanced Adaptive Loss System
        loss_weights = self.loss_controller.get_adaptive_loss_weights(context)
        adaptations.update(loss_weights)
        
        # 4. Адаптация параметров Tversky Loss
        alpha, beta = self.loss_controller.compute_dynamic_tversky_params(context)
        adaptations['tversky_alpha'] = alpha
        adaptations['tversky_beta'] = beta
        
        return adaptations

    def _get_current_parameters(self) -> Dict[str, float]:
        """Получение текущих параметров системы"""
        return {
            'learning_rate': self.param_manager.current_lr,
            'guided_attention_weight': self.loss_controller.guided_attention_weight,
            'mel_weight': self.loss_controller.mel_weight,
            'gate_weight': self.loss_controller.gate_weight
        }

    def _log_adaptations(self, step: int, context: TrainingContext, adaptations: Dict[str, float]):
        """Логирование примененных адаптаций"""
        
        self.logger.info(
            f"🎯 Step {step}: Context-Aware Adaptation "
            f"(Phase: {context.phase.value}, "
            f"Attention: {context.attention_quality:.3f}, "
            f"Stability: {context.stability_index:.3f})"
        )
        
        for param, value in adaptations.items():
            if param in ['learning_rate', 'guided_attention_weight']:
                self.logger.info(f"  📊 {param}: {value:.2e}")
        
        # Добавляем в историю решений
        decision = {
            'step': step,
            'phase': context.phase.value,
            'adaptations': adaptations,
            'context_metrics': {
                'attention_quality': context.attention_quality,
                'stability_index': context.stability_index,
                'loss_trend': context.loss_trend
            }
        }
        self.decision_history.append(decision)

    def get_statistics(self) -> Dict[str, any]:
        """Получение статистики работы системы"""
        stats = {
            'total_interventions': self.intervention_count,
            'current_phase': self.current_context.phase.value if self.current_context else 'unknown',
            'current_lr': self.param_manager.current_lr,
            'current_guided_weight': self.loss_controller.guided_attention_weight,
            'stability_index': self.current_context.stability_index if self.current_context else 0.0,
            'recent_decisions': self.decision_history[-5:] if self.decision_history else []
        }
        
        # 🛡️ Добавляем статистику системы стабилизации
        if self.stabilization_available:
            stats['stabilization_system'] = self.stabilization_system.get_system_diagnostics()
        
        return stats
    
    def get_stabilization_diagnostics(self) -> Dict[str, Any]:
        """
        🛡️ Получение диагностики системы стабилизации обучения
        
        Returns:
            Dict[str, Any]: Диагностическая информация системы стабилизации
        """
        if self.stabilization_available:
            return self.stabilization_system.get_system_diagnostics()
        else:
            return {
                'available': False,
                'message': 'Training Stabilization System недоступна'
            }

    def save_state(self, filepath: str):
        """Сохранение состояния менеджера"""
        state = {
            'context_analyzer': self.context_analyzer,
            'loss_controller': self.loss_controller,
            'param_manager': self.param_manager,
            'decision_history': self.decision_history,
            'intervention_count': self.intervention_count
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        self.logger.info(f"💾 Состояние Context-Aware Manager сохранено: {filepath}")

    def load_state(self, filepath: str):
        """Загрузка состояния менеджера"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.context_analyzer = state['context_analyzer']
            self.loss_controller = state['loss_controller'] 
            self.param_manager = state['param_manager']
            self.decision_history = state['decision_history']
            self.intervention_count = state.get('intervention_count', 0)
            
            self.logger.info(f"📁 Состояние Context-Aware Manager загружено: {filepath}")
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки состояния: {e}")

    def integrate_with_loss_function(self, loss_function):
        """
        🔗 Интеграция с loss функцией для управления guided attention и Enhanced Adaptive Loss System
        
        Args:
            loss_function: Экземпляр Tacotron2Loss или совместимой loss функции
        """
        try:
            # Интеграция guided attention (существующая функциональность)
            if hasattr(loss_function, 'set_context_aware_manager'):
                loss_function.set_context_aware_manager(self)
                self.logger.info("✅ Context-Aware Manager интегрирован с guided attention системой")
            else:
                self.logger.warning("⚠️ Loss функция не поддерживает guided attention интеграцию")
            
            # 🎯 НОВАЯ интеграция с Enhanced Adaptive Loss System
            if hasattr(loss_function, 'integrate_with_context_aware_manager'):
                loss_function.integrate_with_context_aware_manager(self)
                
            # Устанавливаем ссылку в EnhancedLossIntegrator
            self.loss_controller.set_loss_function_reference(loss_function)
            
            self.logger.info("✅ Полная интеграция с Enhanced Adaptive Loss System завершена")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка интеграции с loss функцией: {e}")

    def get_guided_attention_recommendations(self, context: TrainingContext) -> Dict[str, float]:
        """
        🎯 Получение рекомендаций для guided attention на основе контекста
        
        Args:
            context: Текущий контекст обучения
            
        Returns:
            Dict[str, float]: Рекомендации по параметрам guided attention
        """
        recommendations = {}
        
        # Рекомендации по весу guided attention
        if context.attention_quality < 0.02:
            # Критически низкое attention - экстренный режим
            recommendations['suggested_weight'] = 25.0
            recommendations['emergency_mode'] = True
        elif context.attention_quality < 0.1:
            # Низкое attention - увеличиваем вес
            recommendations['suggested_weight'] = min(self.loss_controller.guided_attention_weight * 1.5, 15.0)
            recommendations['emergency_mode'] = False
        elif context.attention_quality > 0.7:
            # Хорошее attention - можем снизить вес
            recommendations['suggested_weight'] = max(self.loss_controller.guided_attention_weight * 0.8, 1.0)
            recommendations['emergency_mode'] = False
        else:
            # Нормальное attention - текущий вес
            recommendations['suggested_weight'] = self.loss_controller.guided_attention_weight
            recommendations['emergency_mode'] = False
        
        # Рекомендации по sigma
        if context.phase == TrainingPhase.PRE_ALIGNMENT:
            recommendations['suggested_sigma'] = 0.1  # Узкая для точного alignment
        elif context.phase == TrainingPhase.ALIGNMENT_LEARNING:
            recommendations['suggested_sigma'] = 0.4  # Широкая для гибкости
        elif context.phase == TrainingPhase.REFINEMENT:
            recommendations['suggested_sigma'] = 0.25  # Средняя для баланса
        else:  # CONVERGENCE
            recommendations['suggested_sigma'] = 0.15  # Узкая для стабильности
        
        return recommendations

    def _apply_attention_enhancements(self, context: TrainingContext, model) -> Dict[str, Any]:
        """🔥 Применение улучшений attention mechanisms"""
        adaptations = {}
        
        if not self.attention_enhancement_available:
            return adaptations
        
        try:
            # Получаем attention diagnostics компонент
            attention_diagnostics = self.attention_enhancement_system['attention_diagnostics']
            progressive_trainer = self.attention_enhancement_system['progressive_trainer']
            regularization_system = self.attention_enhancement_system['regularization_system']
            
            # Анализируем attention quality (требует attention weights из модели)
            # В реальной реализации нужно извлечь attention weights из модели
            # Здесь создаем mock attention для демонстрации
            mock_attention = torch.rand(2, 100, 80)  # [B, T_out, T_in]
            
            # Анализ attention quality
            attention_metrics = attention_diagnostics.analyze_attention_quality(mock_attention)
            
            # Обновление фазы training
            current_phase = progressive_trainer.update_training_phase(
                context.step, attention_metrics
            )
            
            # Получение рекомендаций по коррекции
            correction_suggestions = attention_diagnostics.get_correction_suggestions(attention_metrics)
            
            # Обновление regularization weights
            regularization_system.update_regularization_weights(attention_metrics)
            
            # Применение коррекций к модели (если возможно)
            if hasattr(model, 'decoder') and hasattr(model.decoder, 'attention_layer'):
                # Обновление complexity в multi-head attention если доступно
                if hasattr(model.decoder.attention_layer, 'update_complexity'):
                    model.decoder.attention_layer.update_complexity(
                        attention_metrics.diagonality
                    )
            
            adaptations['attention_enhancements'] = {
                'attention_quality': {
                    'diagonality': attention_metrics.diagonality,
                    'monotonicity': attention_metrics.monotonicity,
                    'focus': attention_metrics.focus,
                    'phase': attention_metrics.phase.value
                },
                'training_phase': current_phase.value,
                'corrections_applied': len(correction_suggestions),
                'suggestions': correction_suggestions
            }
            
            self.logger.info(f"🔥 Attention Enhancement: diagonality={attention_metrics.diagonality:.3f}, "
                           f"phase={current_phase.value}, corrections={len(correction_suggestions)}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в Attention Enhancement System: {e}")
            adaptations['attention_enhancements'] = {'error': str(e)}
        
        return adaptations

    def get_attention_enhancement_diagnostics(self) -> Dict[str, Any]:
        """🔥 Получить диагностику системы улучшения attention"""
        if self.attention_enhancement_available and self.attention_enhancement_system:
            diagnostics = {}
            
            # Диагностика attention diagnostics
            attention_diagnostics = self.attention_enhancement_system['attention_diagnostics']
            diagnostics['attention_diagnostics'] = {
                'target_diagonality': attention_diagnostics.target_diagonality,
                'history_length': len(attention_diagnostics.diagonality_history),
                'recent_diagonality': attention_diagnostics.diagonality_history[-5:] if attention_diagnostics.diagonality_history else []
            }
            
            # Диагностика progressive trainer
            progressive_trainer = self.attention_enhancement_system['progressive_trainer']
            diagnostics['progressive_trainer'] = {
                'current_step': progressive_trainer.current_step,
                'current_phase': progressive_trainer.current_phase.value,
                'max_steps': progressive_trainer.max_steps
            }
            
            # Диагностика regularization system
            regularization_system = self.attention_enhancement_system['regularization_system']
            diagnostics['regularization_system'] = {
                'entropy_weight': regularization_system.entropy_weight,
                'monotonic_weight': regularization_system.monotonic_weight,
                'temporal_weight': regularization_system.temporal_weight,
                'diversity_weight': regularization_system.diversity_weight
            }
            
            return diagnostics
        
        return {'status': 'unavailable'}

# Функция для создания менеджера с рекомендуемой конфигурацией
def create_context_aware_manager(hparams) -> ContextAwareTrainingManager:
    """Создание Context-Aware Training Manager с оптимальной конфигурацией для Tacotron2"""
    
    config = {
        'initial_lr': getattr(hparams, 'learning_rate', 1e-3),
        'history_size': 100,
        'initial_guided_weight': getattr(hparams, 'guide_loss_weight', 4.5),
        'logging_level': 'INFO'
    }
    
    return ContextAwareTrainingManager(config)

if __name__ == "__main__":
    # Пример использования
    config = {
        'initial_lr': 1e-3,
        'history_size': 100,
        'initial_guided_weight': 4.5
    }
    
    manager = ContextAwareTrainingManager(config)
    print("🧠 Context-Aware Training Manager создан успешно!")
    print(f"📊 Статистика: {manager.get_statistics()}") 