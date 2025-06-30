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
        Инициализация Smart Tuner Integration.
        
        Args:
            config_path: Путь к конфигурации Smart Tuner
            enable_all_features: Включить все возможности (может требовать больше ресурсов)
        """
        self.config_path = config_path
        self.enable_all_features = enable_all_features
        
        # Загрузка конфигурации
        self.config = self._load_config()
        
        # Настройка логирования
        self.logger = self._setup_logger()
        
        # Инициализация компонентов Smart Tuner
        self.optimization_engine = None
        self.early_stop_controller = None
        self.epoch_optimizer = None
        self.quality_controller = None
        
        # Состояние интеграции
        self.is_initialized = False
        self.current_epoch = 0
        self.training_metrics_history = []
        self.hyperparameter_adjustments = []
        
        # Инициализация компонентов
        self._initialize_components()
        
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
        """Инициализирует компоненты Smart Tuner."""
        try:
            # Оптимизационный движок
            if OptimizationEngine and self.config.get('optimization_enabled', True):
                self.optimization_engine = OptimizationEngine(self.config_path)
                self.logger.info("✅ Optimization Engine инициализирован")
            
            # Контроллер раннего останова
            if EarlyStopController and self.config.get('early_stopping_enabled', True):
                self.early_stop_controller = EarlyStopController(self.config_path)
                self.logger.info("✅ Early Stop Controller инициализирован")
            
            # Оптимизатор эпох
            if IntelligentEpochOptimizer and self.config.get('adaptive_learning_enabled', True):
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