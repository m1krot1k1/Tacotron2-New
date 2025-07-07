
"""
Context-Aware Training Manager - Умный менеджер обучения с пониманием контекста
Ключевой компонент интеллектуальной системы обучения Tacotron2
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import pickle

class TrainingPhase(Enum):
    """Фазы обучения TTS модели"""
    PRE_ALIGNMENT = "pre_alignment"      # Начальное выравнивание
    ALIGNMENT_LEARNING = "alignment"     # Изучение выравнивания
    REFINEMENT = "refinement"           # Улучшение качества
    CONVERGENCE = "convergence"         # Финальная конвергенция

@dataclass
class TrainingContext:
    """Контекст текущего состояния обучения"""
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

class ContextAnalyzer:
    """Анализатор контекста обучения на основе Bayesian classification"""

    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.loss_history = deque(maxlen=history_size)
        self.attention_history = deque(maxlen=history_size) 
        self.gradient_history = deque(maxlen=history_size)

        # Gaussian Mixture Model для классификации фаз
        self.phase_classifier = None
        self.trend_analyzer = None

    def update_metrics(self, loss: float, attention_diag: float, grad_norm: float):
        """Обновление метрик для анализа"""
        self.loss_history.append(loss)
        self.attention_history.append(attention_diag)
        self.gradient_history.append(grad_norm)

    def analyze_phase(self) -> TrainingPhase:
        """Определение текущей фазы обучения"""
        if len(self.loss_history) < 10:
            return TrainingPhase.PRE_ALIGNMENT

        # Анализ трендов
        loss_trend = self._calculate_trend(list(self.loss_history))
        attention_mean = np.mean(list(self.attention_history))
        gradient_stability = np.std(list(self.gradient_history))

        # Логика классификации фаз
        if attention_mean < 0.1 and loss_trend > 0:
            return TrainingPhase.PRE_ALIGNMENT
        elif 0.1 <= attention_mean < 0.5:
            return TrainingPhase.ALIGNMENT_LEARNING
        elif 0.5 <= attention_mean < 0.7 and loss_trend < 0:
            return TrainingPhase.REFINEMENT
        else:
            return TrainingPhase.CONVERGENCE

    def _calculate_trend(self, values: List[float], window: int = 10) -> float:
        """Расчет тренда используя линейную регрессию"""
        if len(values) < window:
            return 0.0
        recent_values = values[-window:]
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]
        return slope

class AdaptiveLossController:
    """Контроллер адаптивного управления loss функциями"""

    def __init__(self):
        self.guided_attention_weight = 4.5
        self.mel_weight = 1.0
        self.gate_weight = 1.0

        # Параметры Dynamic Tversky Loss
        self.alpha_adaptive = 0.3
        self.beta_adaptive = 0.3

        # История изменений для обучения
        self.weight_history = []
        self.performance_history = []

    def compute_adaptive_loss(self, mel_loss: torch.Tensor, 
                            gate_loss: torch.Tensor,
                            attention_loss: torch.Tensor,
                            context: TrainingContext) -> torch.Tensor:
        """Вычисление адаптивной loss функции на основе контекста"""

        # Адаптация весов на основе фазы обучения
        weights = self._adapt_weights_by_phase(context)

        # Dynamic Tversky для gate loss
        gate_loss_adaptive = self._compute_dynamic_tversky_loss(gate_loss, context)

        # Guided attention с контекстным весом
        attention_weight = self._compute_attention_weight(context)

        total_loss = (weights['mel'] * mel_loss + 
                     weights['gate'] * gate_loss_adaptive +
                     attention_weight * attention_loss)

        return total_loss

    def _adapt_weights_by_phase(self, context: TrainingContext) -> Dict[str, float]:
        """Адаптация весов компонентов loss в зависимости от фазы"""
        weights = {'mel': 1.0, 'gate': 1.0}

        if context.phase == TrainingPhase.PRE_ALIGNMENT:
            weights['gate'] = 0.5  # Меньше внимания к gate в начале
        elif context.phase == TrainingPhase.ALIGNMENT_LEARNING:
            weights['gate'] = 1.0  # Баланс
        elif context.phase == TrainingPhase.REFINEMENT:
            weights['mel'] = 1.2   # Больше внимания к качеству mel
        elif context.phase == TrainingPhase.CONVERGENCE:
            weights['mel'] = 1.5   # Максимальное внимание к финальному качеству

        return weights

    def _compute_dynamic_tversky_loss(self, gate_loss: torch.Tensor, 
                                    context: TrainingContext) -> torch.Tensor:
        """Dynamic Tversky Loss с адаптивными параметрами"""
        # Расчет FP и FN на основе текущих предсказаний
        # Это упрощенная версия - в реальности нужны actual predictions

        # Адаптивные параметры на основе качества attention
        if context.attention_quality < 0.3:
            alpha, beta = 0.7, 0.3  # Больше штраф за FP
        elif context.attention_quality < 0.6:
            alpha, beta = 0.5, 0.5  # Баланс
        else:
            alpha, beta = 0.3, 0.7  # Больше штраф за FN

        # Применение Tversky формулы
        return gate_loss * (alpha + beta)

    def _compute_attention_weight(self, context: TrainingContext) -> float:
        """Вычисление веса guided attention на основе контекста"""
        base_weight = 4.5

        # Адаптация на основе качества attention
        if context.attention_quality < 0.1:
            return base_weight * 2.0  # Увеличиваем при плохом alignment
        elif context.attention_quality > 0.7:
            return base_weight * 0.5  # Уменьшаем при хорошем alignment
        else:
            # Плавная интерполяция
            factor = 2.0 - (context.attention_quality - 0.1) / 0.6 * 1.5
            return base_weight * factor

class IntelligentParameterManager:
    """Умный менеджер параметров с learning rate scheduling"""

    def __init__(self, initial_lr: float = 1e-3):
        self.base_lr = initial_lr
        self.current_lr = initial_lr
        self.lr_history = []
        self.performance_memory = deque(maxlen=50)

    def update_learning_rate(self, context: TrainingContext) -> float:
        """Умное обновление learning rate на основе контекста"""

        # Анализ необходимости изменения LR
        lr_adjustment = self._compute_lr_adjustment(context)

        # Применение изменения с ограничениями
        new_lr = self.current_lr * lr_adjustment
        new_lr = np.clip(new_lr, self.base_lr * 0.01, self.base_lr * 2.0)

        self.current_lr = new_lr
        self.lr_history.append(new_lr)

        return new_lr

    def _compute_lr_adjustment(self, context: TrainingContext) -> float:
        """Вычисление коэффициента изменения LR"""
        adjustment = 1.0

        # На основе фазы обучения
        if context.phase == TrainingPhase.PRE_ALIGNMENT:
            adjustment *= 1.2  # Выше LR для быстрого начального обучения
        elif context.phase == TrainingPhase.CONVERGENCE:
            adjustment *= 0.7  # Ниже LR для стабилизации

        # На основе тренда loss
        if context.loss_trend > 0:  # Loss растет
            adjustment *= 0.8
        elif context.loss_trend < -0.1:  # Loss быстро падает  
            adjustment *= 1.1

        # На основе стабильности градиентов
        if context.gradient_health < 0.5:  # Нестабильные градиенты
            adjustment *= 0.6

        return adjustment

class ContextAwareTrainingManager:
    """Главный менеджер контекстно-осознанного обучения"""

    def __init__(self, config: dict):
        self.config = config

        # Инициализация компонентов
        self.context_analyzer = ContextAnalyzer()
        self.loss_controller = AdaptiveLossController()
        self.param_manager = IntelligentParameterManager()

        # Состояние системы
        self.current_context = None
        self.decision_history = []

        # Логирование
        self.logger = logging.getLogger("ContextAwareTrainer")

    def training_step(self, batch_data: dict, model: nn.Module, 
                     optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Один шаг обучения с контекстным управлением"""

        # 1. Анализ текущего контекста
        context = self._analyze_current_context(batch_data)

        # 2. Адаптация параметров на основе контекста
        new_lr = self.param_manager.update_learning_rate(context)
        self._update_optimizer_lr(optimizer, new_lr)

        # 3. Forward pass
        outputs = model(batch_data)

        # 4. Вычисление адаптивной loss
        loss = self.loss_controller.compute_adaptive_loss(
            outputs['mel_loss'], 
            outputs['gate_loss'],
            outputs['attention_loss'],
            context
        )

        # 5. Backward pass
        optimizer.zero_grad()
        loss.backward()

        # 6. Проверка градиентов и их обрезка при необходимости
        grad_norm = self._handle_gradients(model, context)

        optimizer.step()

        # 7. Обновление истории и контекста
        self._update_training_history(loss.item(), outputs, grad_norm)

        # 8. Логирование решений
        self._log_training_decision(context, new_lr, loss.item())

        return {
            'loss': loss.item(),
            'learning_rate': new_lr,
            'phase': context.phase.value,
            'attention_quality': context.attention_quality
        }

    def _analyze_current_context(self, batch_data: dict) -> TrainingContext:
        """Анализ текущего контекста обучения"""
        # Извлечение метрик из batch_data
        # В реальной реализации здесь будет более сложная логика

        phase = self.context_analyzer.analyze_phase()

        return TrainingContext(
            phase=phase,
            step=len(self.decision_history),
            epoch=0,  # Заполнить из реальных данных
            loss_trend=0.0,  # Вычислить из истории
            attention_quality=0.5,  # Извлечь из outputs
            gradient_health=0.8,  # Вычислить
            learning_rate=self.param_manager.current_lr,
            convergence_score=0.0,
            stability_index=0.7,
            time_since_improvement=0
        )

    def _handle_gradients(self, model: nn.Module, context: TrainingContext) -> float:
        """Умная обработка градиентов"""
        # Вычисление нормы градиентов
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        # Адаптивное обрезание на основе контекста
        if context.phase == TrainingPhase.PRE_ALIGNMENT:
            clip_value = 5.0  # Более строгое обрезание
        elif context.phase == TrainingPhase.CONVERGENCE:
            clip_value = 0.5  # Очень осторожное обрезание
        else:
            clip_value = 1.0  # Стандартное обрезание

        if total_norm > clip_value:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        return total_norm

    def _update_training_history(self, loss: float, outputs: dict, grad_norm: float):
        """Обновление истории обучения"""
        # Извлечение attention диагональности
        attention_diag = 0.5  # Заполнить из реальных данных outputs

        self.context_analyzer.update_metrics(loss, attention_diag, grad_norm)

    def _update_optimizer_lr(self, optimizer: torch.optim.Optimizer, new_lr: float):
        """Обновление learning rate в оптимизаторе"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    def _log_training_decision(self, context: TrainingContext, lr: float, loss: float):
        """Логирование принятых решений"""
        decision = {
            'step': context.step,
            'phase': context.phase.value,
            'learning_rate': lr,
            'loss': loss,
            'attention_quality': context.attention_quality
        }

        self.decision_history.append(decision)

        self.logger.info(
            f"Step {context.step}: Phase={context.phase.value}, "
            f"LR={lr:.2e}, Loss={loss:.4f}, "
            f"Attention={context.attention_quality:.3f}"
        )

    def save_state(self, filepath: str):
        """Сохранение состояния менеджера"""
        state = {
            'context_analyzer': self.context_analyzer,
            'loss_controller': self.loss_controller,
            'param_manager': self.param_manager,
            'decision_history': self.decision_history
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        """Загрузка состояния менеджера"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.context_analyzer = state['context_analyzer']
        self.loss_controller = state['loss_controller'] 
        self.param_manager = state['param_manager']
        self.decision_history = state['decision_history']

# Пример использования:
def example_usage():
    """Пример использования Context-Aware Training Manager"""

    config = {
        'initial_lr': 1e-3,
        'history_size': 100,
        'logging_level': 'INFO'
    }

    # Инициализация менеджера
    trainer = ContextAwareTrainingManager(config)

    # Примерный цикл обучения
    for step in range(1000):
        # Загрузка batch данных (заглушка)
        batch_data = {'input': None, 'target': None}

        # Шаг обучения с контекстным управлением
        metrics = trainer.training_step(batch_data, model=None, optimizer=None)

        # Вывод прогресса каждые 100 шагов
        if step % 100 == 0:
            print(f"Step {step}: {metrics}")

    # Сохранение состояния
    trainer.save_state('context_aware_trainer_state.pkl')

if __name__ == "__main__":
    example_usage()
