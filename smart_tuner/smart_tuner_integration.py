#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Tuner Integration Module
Интеграция Smart Tuner с основным процессом обучения Tacotron2.

Этот модуль обеспечивает:
1. Автоматическое улучшение гиперпараметров в процессе обучения
2. Интеллектуальное управление качеством TTS
3. Адаптивное управление процессом обучения
4. Интеграцию всех компонентов Smart Tuner
"""

import torch
import numpy as np
import yaml
import logging
import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Импорт компонентов Smart Tuner
try:
    from .optimization_engine import OptimizationEngine
    from .early_stop_controller import EarlyStopController
    from .intelligent_epoch_optimizer import IntelligentEpochOptimizer
    from .advanced_quality_controller import AdvancedQualityController
except ImportError:
    # Fallback для случаев, когда модули не найдены
    OptimizationEngine = None
    EarlyStopController = None
    IntelligentEpochOptimizer = None
    AdvancedQualityController = None

class SmartTunerIntegration:
    """
    Главный класс интеграции Smart Tuner с обучением Tacotron2.
    
    Обеспечивает:
    - Умное управление гиперпараметрами
    - Контроль качества в реальном времени  
    - Адаптивное принятие решений
    - Оптимизацию эпох обучения
    """
    
    def __init__(self, config_path: str = "smart_tuner/config.yaml", enable_all_features: bool = True):
        """
        Инициализирует Smart Tuner Integration с полным набором возможностей.
        
        Args:
            config_path: Путь к конфигурационному файлу
            enable_all_features: Включить все функции (по умолчанию True)
        """
        self.config_path = config_path
        self.enable_all_features = enable_all_features
        self.is_initialized = False
        
        # Инициализация базовых компонентов
        self.config = self._load_config()
        self.logger = self._setup_logger()
        
        # Инициализация Telegram монитора
        self.telegram_monitor = None
        
        # История метрик для анализа
        self.training_metrics_history = []
        self.current_epoch = 0
        self.hyperparameter_adjustments = []
        
        # Компоненты Smart Tuner
        self.early_stop_controller = None
        self.quality_controller = None
        self.epoch_optimizer = None
        
        # Инициализация всех компонентов
        if enable_all_features:
            self._initialize_components()
            
        self.logger.info("Smart Tuner Integration инициализирован")
        
        # Инициализация _recent_losses для milestone проверок
        self._recent_losses = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию Smart Tuner."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Конфигурация Smart Tuner не найдена: {self.config_path}")
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Ошибка загрузки конфигурации: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию по умолчанию."""
        return {
            'smart_tuner_enabled': True,
            'optimization_enabled': True,
            'quality_control_enabled': True,
            'early_stopping_enabled': True,
            'adaptive_learning_enabled': True
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Настраивает логирование для Smart Tuner."""
        logger = logging.getLogger('SmartTunerIntegration')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - [SmartTuner] - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_components(self):
        """Инициализирует все компоненты Smart Tuner"""
        try:
            # Импорт компонентов
            from smart_tuner.early_stop_controller import EarlyStopController
            from smart_tuner.advanced_quality_controller import AdvancedQualityController
            from smart_tuner.intelligent_epoch_optimizer import IntelligentEpochOptimizer
            from smart_tuner.telegram_monitor_enhanced import TelegramMonitorEnhanced
            
            # Early Stop Controller
            if EarlyStopController and self.config.get('early_stopping_enabled', True):
                self.early_stop_controller = EarlyStopController(self.config_path)
                self.logger.info("✅ Early Stop Controller инициализирован")
            
            # Telegram Monitor Enhanced
            try:
                self.telegram_monitor = TelegramMonitorEnhanced()
                if self.telegram_monitor.enabled:
                    self.logger.info("✅ Telegram Monitor Enhanced инициализирован")
                else:
                    self.logger.info("📱 Telegram Monitor отключен в конфигурации")
            except Exception as e:
                self.logger.warning(f"Telegram Monitor не удалось инициализировать: {e}")
                self.telegram_monitor = None
            
            # Intelligent Epoch Optimizer
            if IntelligentEpochOptimizer and self.config.get('epoch_optimization_enabled', True):
                self.epoch_optimizer = IntelligentEpochOptimizer(self.config_path)
                self.logger.info("✅ Intelligent Epoch Optimizer инициализирован")
            
            # Контроллер качества
            if AdvancedQualityController and self.config.get('quality_control_enabled', True):
                self.quality_controller = AdvancedQualityController(self.config_path)
                self.logger.info("✅ Advanced Quality Controller инициализирован")
            
            self.is_initialized = True
            self.logger.info("🚀 Smart Tuner Integration успешно инициализирован")
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации Smart Tuner: {e}")
            self.is_initialized = False
    
    def analyze_dataset_for_training(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализирует датасет и рекомендует оптимальные параметры обучения.
        
        Args:
            dataset_info: Информация о датасете (размер, качество, характеристики)
            
        Returns:
            Рекомендации по обучению
        """
        recommendations = {
            'optimal_epochs': 3000,
            'recommended_batch_size': 12,
            'learning_rate_range': (1e-6, 5e-5),
            'quality_expectations': 'good',
            'estimated_training_time_hours': 8.0
        }
        
        if self.epoch_optimizer:
            try:
                analysis = self.epoch_optimizer.analyze_dataset(dataset_info)
                recommendations.update(analysis)
                self.logger.info(f"📊 Анализ датасета завершен: {analysis.get('recommended_epochs_range')}")
            except Exception as e:
                self.logger.error(f"Ошибка анализа датасета: {e}")
        
        return recommendations
    
    def on_training_start(self, initial_hyperparams: Dict[str, Any], dataset_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Вызывается в начале обучения для настройки параметров.
        
        Args:
            initial_hyperparams: Начальные гиперпараметры
            dataset_info: Информация о датасете
            
        Returns:
            Оптимизированные гиперпараметры для старта
        """
        if not self.is_initialized:
            self.logger.warning("Smart Tuner не инициализирован, используются исходные параметры")
            return initial_hyperparams
        
        optimized_params = initial_hyperparams.copy()
        
        # Анализ датасета для оптимизации
        if dataset_info and self.epoch_optimizer:
            dataset_recommendations = self.analyze_dataset_for_training(dataset_info)
            
            # Применяем рекомендации к гиперпараметрам
            if 'optimal_epochs' in dataset_recommendations:
                optimized_params['epochs'] = dataset_recommendations['optimal_epochs']
            
            if 'recommended_batch_size' in dataset_recommendations:
                optimized_params['batch_size'] = dataset_recommendations['recommended_batch_size']
        
        # Логируем изменения
        changes = []
        for key in optimized_params:
            if key in initial_hyperparams and optimized_params[key] != initial_hyperparams[key]:
                changes.append(f"{key}: {initial_hyperparams[key]} → {optimized_params[key]}")
        
        if changes:
            self.logger.info(f"🔧 Оптимизированы параметры: {', '.join(changes)}")
        
        return optimized_params
    
    def on_epoch_start(self, epoch: int, current_hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Вызывается в начале каждой эпохи.
        
        Args:
            epoch: Номер эпохи
            current_hyperparams: Текущие гиперпараметры
            
        Returns:
            Возможно обновленные гиперпараметры
        """
        self.current_epoch = epoch
        
        if not self.is_initialized:
            return current_hyperparams
        
        # Мониторинг прогресса эпох
        if self.epoch_optimizer:
            try:
                # Создаем минимальные метрики для мониторинга
                basic_metrics = {
                    'epoch': epoch,
                    'learning_rate': current_hyperparams.get('learning_rate', 1e-4)
                }
                
                progress_analysis = self.epoch_optimizer.monitor_training_progress(epoch, basic_metrics)
                
                if progress_analysis.get('recommendations'):
                    recommendations = progress_analysis['recommendations']
                    self.logger.info(f"📈 Рекомендации для эпохи {epoch}: {recommendations.get('action', 'continue')}")
                
            except Exception as e:
                self.logger.error(f"Ошибка мониторинга эпохи: {e}")
        
        return current_hyperparams
    
    def on_batch_end(self, epoch: int, batch: int, metrics: Dict[str, float], 
                     model_outputs: Optional[Tuple] = None) -> Dict[str, Any]:
        """
        Вызывается после каждого batch для анализа качества.
        
        Args:
            epoch: Номер эпохи
            batch: Номер batch
            metrics: Метрики обучения
            model_outputs: Выходы модели (mel, gate, attention)
            
        Returns:
            Рекомендации и анализ качества
        """
        analysis_result = {
            'continue_training': True,
            'quality_issues': [],
            'recommended_actions': []
        }
        
        if not self.is_initialized:
            return analysis_result
        
        # Анализ качества через Quality Controller
        if self.quality_controller and model_outputs:
            try:
                mel_outputs, gate_outputs, attention_weights = model_outputs[:3]
                
                quality_analysis = self.quality_controller.analyze_training_quality(
                    epoch=epoch,
                    metrics=metrics,
                    attention_weights=attention_weights,
                    gate_outputs=gate_outputs,
                    mel_outputs=mel_outputs
                )
                
                analysis_result['quality_score'] = quality_analysis.get('overall_quality_score', 0.5)
                analysis_result['quality_issues'] = quality_analysis.get('quality_issues', [])
                analysis_result['recommended_actions'] = quality_analysis.get('recommended_interventions', [])
                
                # Логируем серьезные проблемы качества
                high_severity_issues = [
                    issue for issue in analysis_result['quality_issues'] 
                    if issue.get('severity') == 'high'
                ]
                
                if high_severity_issues:
                    self.logger.warning(f"⚠️ Обнаружены серьезные проблемы качества в эпохе {epoch}")
                    for issue in high_severity_issues:
                        self.logger.warning(f"   - {issue.get('description', 'Unknown issue')}")
                
            except Exception as e:
                self.logger.error(f"Ошибка анализа качества: {e}")
        
        return analysis_result
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], 
                     current_hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Вызывается в конце эпохи для принятия решений.
        
        Args:
            epoch: Номер эпохи
            metrics: Метрики эпохи
            current_hyperparams: Текущие гиперпараметры
            
        Returns:
            Решения и обновленные параметры
        """
        decision_result = {
            'continue_training': True,
            'hyperparameter_updates': {},
            'early_stop': False,
            'reason': 'normal_progress'
        }
        
        if not self.is_initialized:
            return decision_result
        
        # Сохраняем метрики в историю
        self.training_metrics_history.append({
            'epoch': epoch,
            'metrics': metrics.copy(),
            'hyperparams': current_hyperparams.copy()
        })
        
        # Анализ через Early Stop Controller
        if self.early_stop_controller:
            try:
                # Добавляем метрики в контроллер
                self.early_stop_controller.add_metrics(metrics)
                
                # Проверяем раннюю остановку
                should_stop, stop_reason = self.early_stop_controller.should_stop_early(metrics)
                
                if should_stop:
                    decision_result['early_stop'] = True
                    decision_result['reason'] = stop_reason
                    decision_result['continue_training'] = False
                    self.logger.info(f"🛑 Рекомендована ранняя остановка: {stop_reason}")
                    return decision_result
                
                # Получаем рекомендации по улучшению
                recommendations = self.early_stop_controller.decide_next_step(current_hyperparams)
                
                if recommendations.get('action') != 'continue':
                    action = recommendations.get('action')
                    reason = recommendations.get('reason', 'Unknown')
                    
                    self.logger.info(f"🔧 Рекомендация: {action} - {reason}")
                    
                    # Применяем изменения гиперпараметров
                    if 'hyperparameter_updates' in recommendations:
                        decision_result['hyperparameter_updates'] = recommendations['hyperparameter_updates']
                        
                        # Логируем изменения
                        updates = recommendations['hyperparameter_updates']
                        changes = [f"{k}: {current_hyperparams.get(k, 'N/A')} → {v}" for k, v in updates.items()]
                        self.logger.info(f"   Изменения: {', '.join(changes)}")
                
            except Exception as e:
                self.logger.error(f"Ошибка в Early Stop Controller: {e}")
        
        # Уведомление в Telegram о действии Smart Tuner
        if hasattr(self, 'telegram_monitor') and self.telegram_monitor and decision_result.get('hyperparameter_updates'):
            try:
                reasoning = self._get_human_readable_reasoning('hyperparameter_update', metrics, {'epoch': epoch})
                action_details = {
                    'changes': decision_result['hyperparameter_updates'],
                    'trigger_metrics': metrics,
                    'context': {'epoch': epoch}
                }
                
                self.telegram_monitor.send_smart_tuner_action(
                    action_type='hyperparameter_update',
                    action_details=action_details,
                    reasoning=reasoning,
                    step=epoch
                )
            except Exception as e:
                self.logger.warning(f"Ошибка отправки Telegram уведомления: {e}")
        
        return decision_result
    
    def apply_quality_interventions(self, interventions: list, 
                                   current_hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Применяет вмешательства для улучшения качества.
        
        Args:
            interventions: Список вмешательств
            current_hyperparams: Текущие гиперпараметры
            
        Returns:
            Обновленные гиперпараметры
        """
        updated_params = current_hyperparams.copy()
        
        if not self.quality_controller or not interventions:
            return updated_params
        
        for intervention in interventions:
            try:
                # Применяем вмешательство через Quality Controller
                updated_params = self.quality_controller.apply_quality_intervention(
                    intervention, updated_params
                )
                
                self.logger.info(f"✨ Применено вмешательство: {intervention.get('description', 'Unknown')}")
                
            except Exception as e:
                self.logger.error(f"Ошибка применения вмешательства: {e}")
        
        return updated_params
    
    def on_training_complete(self, final_metrics: Dict[str, float], 
                           total_epochs: int, training_time_hours: float) -> Dict[str, Any]:
        """
        Вызывается по завершению обучения для анализа результатов.
        
        Args:
            final_metrics: Финальные метрики
            total_epochs: Общее количество эпох
            training_time_hours: Время обучения в часах
            
        Returns:
            Анализ и рекомендации
        """
        training_summary = {
            'training_success': True,
            'quality_assessment': 'good',
            'efficiency_score': 0.8,
            'recommendations_for_future': [],
            'smart_tuner_interventions': len(self.hyperparameter_adjustments)
        }
        
        if not self.is_initialized:
            return training_summary
        
        try:
            # Сохраняем результаты оптимизации
            if self.epoch_optimizer:
                self.epoch_optimizer.save_optimization_result(
                    final_metrics, total_epochs, training_time_hours * 60
                )
            
            # Оценка качества обучения
            if final_metrics.get('val_loss', float('inf')) < 3.0:
                training_summary['quality_assessment'] = 'excellent'
            elif final_metrics.get('val_loss', float('inf')) < 5.0:
                training_summary['quality_assessment'] = 'good'
            else:
                training_summary['quality_assessment'] = 'needs_improvement'
            
            # Оценка эффективности
            expected_time = total_epochs * 0.1  # Примерно 0.1 часа на эпоху
            efficiency = min(expected_time / training_time_hours, 1.0) if training_time_hours > 0 else 0.5
            training_summary['efficiency_score'] = efficiency
            
            # Получаем сводку качества
            if self.quality_controller:
                quality_summary = self.quality_controller.get_quality_summary()
                training_summary['quality_details'] = quality_summary
            
            self.logger.info(f"🎯 Обучение завершено: качество={training_summary['quality_assessment']}, "
                           f"эффективность={efficiency:.2f}, вмешательств={len(self.hyperparameter_adjustments)}")
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа завершения обучения: {e}")
        
        return training_summary
    
    def get_training_recommendations(self) -> Dict[str, Any]:
        """
        Возвращает рекомендации на основе накопленного опыта.
        
        Returns:
            Рекомендации для будущих обучений
        """
        recommendations = {
            'hyperparameter_suggestions': {},
            'training_strategy': 'standard',
            'expected_quality': 'good',
            'confidence': 0.7
        }
        
        if not self.is_initialized or not self.training_metrics_history:
            return recommendations
        
        try:
            # Анализируем историю метрик
            recent_metrics = self.training_metrics_history[-10:]  # Последние 10 эпох
            
            # Средние значения
            avg_train_loss = np.mean([m['metrics'].get('train_loss', 0) for m in recent_metrics])
            avg_val_loss = np.mean([m['metrics'].get('val_loss', 0) for m in recent_metrics])
            
            # Рекомендации на основе производительности
            if avg_val_loss < 3.0:
                recommendations['training_strategy'] = 'high_quality_focused'
                recommendations['expected_quality'] = 'excellent'
            elif avg_val_loss > 8.0:
                recommendations['training_strategy'] = 'stability_focused'
                recommendations['expected_quality'] = 'needs_improvement'
            
            # Рекомендации по гиперпараметрам
            if len(self.hyperparameter_adjustments) > 0:
                last_adjustment = self.hyperparameter_adjustments[-1]
                recommendations['hyperparameter_suggestions'] = last_adjustment.get('successful_params', {})
            
            self.logger.info(f"📋 Сгенерированы рекомендации: стратегия={recommendations['training_strategy']}")
            
        except Exception as e:
            self.logger.error(f"Ошибка генерации рекомендаций: {e}")
        
        return recommendations
    
    def is_available(self) -> bool:
        """Проверяет доступность Smart Tuner."""
        return self.is_initialized
    
    def get_status(self) -> Dict[str, Any]:
        """Возвращает статус Smart Tuner."""
        return {
            'initialized': self.is_initialized,
            'optimization_engine': self.optimization_engine is not None,
            'early_stop_controller': self.early_stop_controller is not None,
            'epoch_optimizer': self.epoch_optimizer is not None,
            'quality_controller': self.quality_controller is not None,
            'current_epoch': self.current_epoch,
            'interventions_count': len(self.hyperparameter_adjustments)
        }
    
    def _get_human_readable_reasoning(self, decision_type: str, metrics: Dict[str, float], 
                                    context: Dict[str, Any]) -> str:
        """
        🧠 Генерирует понятное человеку объяснение решения Smart Tuner.
        
        Args:
            decision_type: Тип принятого решения
            metrics: Текущие метрики
            context: Контекст обучения
            
        Returns:
            Человекопонятное объяснение на русском языке
        """
        
        attention_diag = metrics.get('attention_diagonality', 0)
        val_loss = metrics.get('val_loss', float('inf'))
        quality_score = metrics.get('quality_score', 0)
        phase = context.get('phase', 'unknown')
        
        # Анализ ситуации и формирование объяснения
        if decision_type == 'learning_rate_reduction':
            if attention_diag < 0.3:
                return (f"🎯 Обнаружена горизонтальная полоса вместо диагонали в attention (диагональность {attention_diag:.3f}). "
                       f"Это критическая проблема! Снижаю learning rate для более аккуратного обучения attention механизма. "
                       f"Цель: получить четкую диагональ для правильного выравнивания текст-аудио.")
            elif attention_diag < 0.6:
                return (f"📊 Attention диагональность {attention_diag:.3f} ниже нормы. "
                       f"Модель плохо выравнивает текст и аудио. Снижаю learning rate для более стабильного обучения. "
                       f"Ожидаю улучшения качества голоса.")
            else:
                return (f"⚡ Обучение идет слишком быстро (loss {val_loss:.4f}). "
                       f"Снижаю learning rate для предотвращения пропуска оптимума и получения лучшего качества.")
        
        elif decision_type == 'dropout_adjustment':
            if quality_score < 0.5:
                return (f"🎵 Качество голоса низкое ({quality_score:.1%}). "
                       f"Возможно, высокий dropout создает артефакты. Снижаю dropout для более четкой генерации. "
                       f"Цель: убрать посторонние шумы и улучшить естественность.")
            else:
                return (f"🛡️ Защищаю модель от переобучения. "
                       f"Качество хорошее ({quality_score:.1%}), но нужно сохранить стабильность. "
                       f"Корректирую dropout для баланса качества и надежности.")
        
        elif decision_type == 'batch_size_optimization':
            if attention_diag < 0.5:
                return (f"🔍 Attention плохо фокусируется (диагональность {attention_diag:.3f}). "
                       f"Большие батчи мешают качественному обучению attention. "
                       f"Уменьшаю batch size для лучшего градиентного обновления. "
                       f"Ожидаю более четкого выравнивания.")
            else:
                return (f"⚡ Оптимизирую скорость обучения. "
                       f"Attention работает хорошо, можно увеличить batch size для ускорения без потери качества.")
        
        elif decision_type == 'guided_attention_boost':
            return (f"🎯 КРИТИЧНО! Диагональность {attention_diag:.3f} означает, что модель не учится правильно выравнивать текст и аудио. "
                   f"Это приведет к неразборчивому голосу с артефактами. "
                   f"Усиливаю guided attention loss для принудительного обучения правильному alignment. "
                   f"Цель: заставить модель создать четкую диагональ.")
        
        elif decision_type == 'phase_transition':
            new_phase = context.get('new_phase', 'unknown')
            if new_phase == 'quality_optimization':
                return (f"🎭 Базовое обучение завершено. Диагональность {attention_diag:.3f} достаточна для перехода. "
                       f"Переключаюсь на фазу качественной оптимизации. "
                       f"Теперь фокус на получении максимально человеческого голоса без артефактов.")
            elif new_phase == 'fine_tuning':
                return (f"🏆 Основные проблемы решены! Качество {quality_score:.1%}. "
                       f"Переходжу к тонкой настройке для достижения студийного качества голоса. "
                       f"Цель: идеальная естественность и отсутствие любых артефактов.")
            else:
                return (f"🔄 Изменение стратегии обучения. Переход в фазу '{new_phase}' для оптимизации процесса.")
        
        elif decision_type == 'early_stop':
            if val_loss == float('inf') or val_loss > 10:
                return (f"🚨 КРИТИЧЕСКАЯ ОШИБКА! Loss взорвался ({val_loss}). "
                       f"Модель полностью разрушена и генерирует только шум. "
                       f"Останавливаю обучение для предотвращения дальнейшего ущерба. "
                       f"Нужно перезапустить с меньшим learning rate.")
            elif attention_diag < 0.1:
                return (f"🛑 Attention полностью сломан (диагональность {attention_diag:.3f}). "
                       f"Модель не способна выравнивать текст и аудио. "
                       f"Дальнейшее обучение бесполезно. Нужна коррекция параметров.")
            else:
                return (f"📉 Обучение застопорилось. Val loss {val_loss:.4f} не улучшается. "
                       f"Останавливаю для экономии ресурсов и анализа проблемы.")
        
        # Общий случай
        return (f"🧠 Smart Tuner обнаружил ситуацию, требующую вмешательства. "
               f"Текущие метрики: диагональность {attention_diag:.3f}, качество {quality_score:.1%}. "
               f"Применяю оптимизацию для улучшения обучения.")
    
    def send_critical_alert_if_needed(self, metrics: Dict[str, float], step: int) -> None:
        """
        🚨 Отправляет критическое предупреждение если обнаружены серьезные проблемы.
        """
        if not hasattr(self, 'telegram_monitor') or not self.telegram_monitor:
            return
            
        try:
            critical_issues = []
            recommendations = []
            
            # Проверка критических проблем
            attention_diag = metrics.get('attention_diagonality', 0)
            if attention_diag < 0.1:
                critical_issues.append("Полное разрушение attention механизма")
                recommendations.extend([
                    "Немедленно снизить learning rate в 10 раз",
                    "Увеличить guided attention weight до 20.0",
                    "Проверить guided attention реализацию"
                ])
            
            val_loss = metrics.get('val_loss', 0)
            if val_loss > 50:
                critical_issues.append("Взрывной рост validation loss")
                recommendations.extend([
                    "Откатиться к предыдущему checkpoint",
                    "Снизить learning rate в 5 раз",
                    "Проверить gradient clipping"
                ])
            
            quality_score = metrics.get('quality_score', 1)
            if quality_score < 0.1:
                critical_issues.append("Критически низкое качество генерации")
                recommendations.extend([
                    "Проверить dropout параметры",
                    "Убедиться в корректности loss функций",
                    "Проанализировать данные обучения"
                ])
            
            # Отправляем предупреждение если есть критические проблемы
            if critical_issues:
                alert_details = {
                    'description': f"Обнаружено {len(critical_issues)} критических проблем на шаге {step}",
                    'metrics': {
                        'attention_diagonality': f"{attention_diag:.4f}",
                        'val_loss': f"{val_loss:.4f}",
                        'quality_score': f"{quality_score:.1%}"
                    },
                    'issues': critical_issues
                }
                
                self.telegram_monitor.send_critical_alert(
                    alert_type="Критические проблемы обучения",
                    details=alert_details,
                    recommendations=recommendations
                )
                
        except Exception as e:
            self.logger.error(f"Ошибка отправки критического предупреждения: {e}")
    
    def send_milestone_achievement(self, metrics: Dict[str, float], step: int) -> None:
        """
        🏆 Отправляет уведомление о достижении важных целей.
        """
        if not hasattr(self, 'telegram_monitor') or not self.telegram_monitor:
            return
            
        try:
            attention_diag = metrics.get('attention_diagonality', 0)
            quality_score = metrics.get('quality_score', 0)
            
            # Проверка достижений
            if attention_diag >= 0.8 and not getattr(self, '_attention_milestone_sent', False):
                achievement = {'diagonality': attention_diag}
                self.telegram_monitor.send_success_milestone(
                    milestone_type='attention_quality',
                    achievement=achievement,
                    step=step
                )
                self._attention_milestone_sent = True
            
            if quality_score >= 0.8 and not getattr(self, '_quality_milestone_sent', False):
                achievement = {'quality_score': quality_score}
                self.telegram_monitor.send_success_milestone(
                    milestone_type='quality_threshold',
                    achievement=achievement,
                    step=step
                )
                self._quality_milestone_sent = True
            
            # Проверка стабильности обучения
            if len(self._recent_losses) >= 10:
                recent_std = np.std(self._recent_losses[-10:])
                if recent_std < 0.01 and not getattr(self, '_stability_milestone_sent', False):
                    achievement = {'stability_metric': recent_std}
                    self.telegram_monitor.send_success_milestone(
                        milestone_type='stable_training', 
                        achievement=achievement,
                        step=step
                    )
                    self._stability_milestone_sent = True
                    
        except Exception as e:
            self.logger.error(f"Ошибка отправки уведомления о достижении: {e}") 