#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интеграционный менеджер Smart Tuner для Tacotron2-New
Координирует все улучшения и обеспечивает их правильную работу

Особенности:
- Централизованное управление всеми компонентами
- Автоматическая диагностика и восстановление
- Интеграция с системой мониторинга
- Статистика и рекомендации
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from .ddc_diagnostic import initialize_ddc_diagnostic, get_global_ddc_diagnostic

@dataclass
class AppliedRecommendation:
    """Информация о примененной рекомендации."""
    timestamp: float
    recommendation: str
    action_taken: str
    success: bool
    result_description: str
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComponentStatus:
    """Статус компонента Smart Tuner."""
    name: str
    active: bool
    healthy: bool
    last_check: float
    error_count: int
    recommendations: List[str]
    applied_recommendations: List[AppliedRecommendation] = field(default_factory=list)

class SmartTunerIntegrationManager:
    """
    Центральный менеджер интеграции всех улучшений Smart Tuner.
    
    Компоненты:
    - AdaptiveGradientClipper
    - SafeDDCLoss
    - SmartLRAdapter
    - AdvancedQualityController
    - TelegramMonitor
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.components = {}
        self.start_time = time.time()
        self.total_steps = 0
        self.emergency_mode = False
        
        # История примененных рекомендаций
        self.applied_recommendations: List[AppliedRecommendation] = []
        
        # Инициализация компонентов
        self._initialize_components()
    
    def _initialize_components(self):
        """Инициализирует все компоненты Smart Tuner."""
        try:
            # Gradient Clipper
            from .gradient_clipper import AdaptiveGradientClipper
            self.components['gradient_clipper'] = ComponentStatus(
                name="AdaptiveGradientClipper",
                active=True,
                healthy=True,
                last_check=time.time(),
                error_count=0,
                recommendations=[]
            )
            
            # DDC Loss
            from .safe_ddc_loss import SafeDDCLoss
            self.components['ddc_loss'] = ComponentStatus(
                name="SafeDDCLoss",
                active=True,
                healthy=True,
                last_check=time.time(),
                error_count=0,
                recommendations=[]
            )
            
            # LR Adapter
            from .smart_lr_adapter import SmartLRAdapter
            self.components['lr_adapter'] = ComponentStatus(
                name="SmartLRAdapter",
                active=True,
                healthy=True,
                last_check=time.time(),
                error_count=0,
                recommendations=[]
            )
            
            # DDC Diagnostic
            from .ddc_diagnostic import initialize_ddc_diagnostic
            self.components['ddc_diagnostic'] = ComponentStatus(
                name="DDCLossDiagnostic",
                active=True,
                healthy=True,
                last_check=time.time(),
                error_count=0,
                recommendations=[]
            )
            
            # Инициализируем глобальную диагностику
            initialize_ddc_diagnostic()
            
            self.logger.info("✅ Все компоненты Smart Tuner инициализированы")
            
        except ImportError as e:
            self.logger.error(f"❌ Ошибка инициализации компонентов: {e}")
            self.emergency_mode = True
    
    def step(self, step: int, loss: float, grad_norm: float = None, 
             model=None, optimizer=None) -> Dict[str, Any]:
        """
        Выполняет шаг интеграции всех компонентов.
        
        Args:
            step: Текущий шаг обучения
            loss: Текущий loss
            grad_norm: Норма градиентов
            model: Модель PyTorch
            optimizer: Оптимизатор PyTorch
            
        Returns:
            Словарь с результатами работы всех компонентов
        """
        self.total_steps += 1
        results = {
            'step': step,
            'timestamp': time.time(),
            'components_status': {},
            'recommendations': [],
            'emergency_mode': self.emergency_mode,
            'recommendation_summary': self.get_recommendation_summary(),
            'recent_applied_recommendations': self.applied_recommendations[-3:] if self.applied_recommendations else []
        }
        
        # Проверяем каждый компонент
        for component_name, status in self.components.items():
            try:
                component_result = self._check_component(component_name, step, loss, grad_norm, model, optimizer)
                results['components_status'][component_name] = component_result
                
                # Обновляем статус компонента
                status.last_check = time.time()
                status.healthy = component_result.get('healthy', True)
                status.recommendations = component_result.get('recommendations', [])
                
                # Добавляем рекомендации
                results['recommendations'].extend(component_result.get('recommendations', []))
                
            except Exception as e:
                self.logger.error(f"❌ Ошибка в компоненте {component_name}: {e}")
                status.error_count += 1
                status.healthy = False
                results['components_status'][component_name] = {
                    'healthy': False,
                    'error': str(e),
                    'recommendations': [f"Ошибка в {component_name}: {e}"]
                }
        
        # Проверяем общее состояние системы
        self._check_system_health(results)
        
        return results
    
    def _check_component(self, component_name: str, step: int, loss: float, 
                        grad_norm: float = None, model=None, optimizer=None) -> Dict[str, Any]:
        """Проверяет состояние конкретного компонента."""
        if component_name == 'gradient_clipper':
            return self._check_gradient_clipper(step, grad_norm, model)
        elif component_name == 'ddc_loss':
            return self._check_ddc_loss(step)
        elif component_name == 'lr_adapter':
            return self._check_lr_adapter(step, loss, grad_norm, optimizer)
        elif component_name == 'ddc_diagnostic':
            return self._check_ddc_diagnostic(step, loss)
        else:
            return {'healthy': True, 'recommendations': []}
    
    def _check_gradient_clipper(self, step: int, grad_norm: float, model) -> Dict[str, Any]:
        """Проверяет состояние gradient clipper."""
        try:
            from .gradient_clipper import get_global_clipper
            clipper = get_global_clipper()
            
            if clipper is None:
                return {
                    'healthy': False,
                    'recommendations': ['Gradient clipper не инициализирован']
                }
            
            stats = clipper.get_statistics()
            recommendations = clipper.get_recommendations()
            
            return {
                'healthy': stats['emergency_clips'] == 0,
                'statistics': stats,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'recommendations': [f'Ошибка gradient clipper: {e}']
            }
    
    def _check_ddc_loss(self, step: int) -> Dict[str, Any]:
        """Проверяет состояние DDC loss."""
        try:
            from .safe_ddc_loss import get_global_ddc_loss
            ddc_loss = get_global_ddc_loss()
            
            if ddc_loss is None:
                return {
                    'healthy': False,
                    'recommendations': ['DDC loss не инициализирован']
                }
            
            stats = ddc_loss.get_statistics()
            recommendations = ddc_loss.get_recommendations()
            
            return {
                'healthy': stats['error_rate'] < 0.1,
                'statistics': stats,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'recommendations': [f'Ошибка DDC loss: {e}']
            }
    
    def _check_lr_adapter(self, step: int, loss: float, grad_norm: float, 
                         optimizer) -> Dict[str, Any]:
        """Проверяет состояние LR adapter."""
        try:
            from .smart_lr_adapter import get_global_lr_adapter
            lr_adapter = get_global_lr_adapter()
            
            if lr_adapter is None:
                return {
                    'healthy': False,
                    'recommendations': ['LR adapter не инициализирован']
                }
            
            stats = lr_adapter.get_statistics()
            recommendations = lr_adapter.get_recommendations()
            
            return {
                'healthy': not stats['emergency_mode'],
                'statistics': stats,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'recommendations': [f'Ошибка LR adapter: {e}']
            }
    
    def _check_ddc_diagnostic(self, step: int, loss: float) -> Dict[str, Any]:
        """Проверяет состояние DDC diagnostic."""
        try:
            diagnostic = get_global_ddc_diagnostic()
            
            if diagnostic is None:
                return {
                    'healthy': False,
                    'recommendations': ['DDC diagnostic не инициализирован']
                }
            
            # Добавляем текущее значение loss
            diagnostic.add_loss_value(loss, step)
            
            # Получаем сводку
            summary = diagnostic.get_summary()
            
            # Проверяем необходимость детального анализа
            if summary['status'] == 'analyzed':
                if summary['mismatch_rate'] > 0.5:
                    # Высокий уровень несовпадений - запускаем детальный анализ
                    report = diagnostic.generate_report()
                    fixes = diagnostic.suggest_fixes(report)
                    
                    return {
                        'healthy': False,
                        'statistics': summary,
                        'recommendations': fixes,
                        'detailed_report': report
                    }
            
            return {
                'healthy': True,
                'statistics': summary,
                'recommendations': []
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'recommendations': [f'Ошибка DDC diagnostic: {e}']
            }
    
    def _check_system_health(self, results: Dict[str, Any]):
        """Проверяет общее состояние системы."""
        unhealthy_components = [
            name for name, status in results['components_status'].items()
            if not status.get('healthy', True)
        ]
        
        if len(unhealthy_components) > 0:
            self.emergency_mode = True
            self.logger.warning(f"🚨 Режим экстренной работы активирован. Проблемные компоненты: {unhealthy_components}")
        else:
            self.emergency_mode = False
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Возвращает общую статистику системы."""
        uptime = time.time() - self.start_time
        
        # Собираем статистику всех компонентов
        component_stats = {}
        for name, status in self.components.items():
            component_stats[name] = {
                'active': status.active,
                'healthy': status.healthy,
                'error_count': status.error_count,
                'uptime': uptime - status.last_check
            }
        
        return {
            'uptime': uptime,
            'total_steps': self.total_steps,
            'emergency_mode': self.emergency_mode,
            'components': component_stats,
            'system_health': len([c for c in self.components.values() if c.healthy]) / len(self.components)
        }
    
    def get_recommendations(self) -> List[str]:
        """Возвращает общие рекомендации для системы."""
        recommendations = []
        
        # Проверяем общее состояние
        if self.emergency_mode:
            recommendations.append("🚨 Система в экстренном режиме - проверить все компоненты")
        
        # Проверяем каждый компонент
        for name, status in self.components.items():
            if not status.healthy:
                recommendations.append(f"⚠️ Компонент {name} нездоров - {status.error_count} ошибок")
            recommendations.extend(status.recommendations)
        
        return recommendations
    
    def reset_component(self, component_name: str):
        """Сбрасывает состояние компонента."""
        if component_name in self.components:
            self.components[component_name].error_count = 0
            self.components[component_name].healthy = True
            self.logger.info(f"🔄 Компонент {component_name} сброшен")
    
    def reset_all(self):
        """Сбрасывает состояние всех компонентов."""
        for name in self.components:
            self.reset_component(name)
        self.emergency_mode = False
        self.logger.info("🔄 Все компоненты Smart Tuner сброшены")

    def apply_recommendation(self, component_name: str, recommendation: str, 
                           action: str, success: bool, result: str = "",
                           metrics_before: Dict[str, Any] = None,
                           metrics_after: Dict[str, Any] = None) -> AppliedRecommendation:
        """
        Записывает применение рекомендации.
        
        Args:
            component_name: Название компонента
            recommendation: Текст рекомендации
            action: Действие, которое было выполнено
            success: Успешно ли было применено
            result: Описание результата
            metrics_before: Метрики до применения
            metrics_after: Метрики после применения
            
        Returns:
            Запись о примененной рекомендации
        """
        applied_rec = AppliedRecommendation(
            timestamp=time.time(),
            recommendation=recommendation,
            action_taken=action,
            success=success,
            result_description=result,
            metrics_before=metrics_before or {},
            metrics_after=metrics_after or {}
        )
        
        # Добавляем в общую историю
        self.applied_recommendations.append(applied_rec)
        
        # Добавляем в компонент
        if component_name in self.components:
            self.components[component_name].applied_recommendations.append(applied_rec)
        
        # Логируем применение
        status_emoji = "✅" if success else "❌"
        self.logger.info(f"{status_emoji} РЕКОМЕНДАЦИЯ ПРИМЕНЕНА: {recommendation}")
        self.logger.info(f"   Действие: {action}")
        if result:
            self.logger.info(f"   Результат: {result}")
        
        return applied_rec
    
    def get_recommendation_history(self, component_name: Optional[str] = None) -> List[AppliedRecommendation]:
        """
        Возвращает историю примененных рекомендаций.
        
        Args:
            component_name: Если указан, возвращает только для этого компонента
            
        Returns:
            Список примененных рекомендаций
        """
        if component_name:
            if component_name in self.components:
                return self.components[component_name].applied_recommendations
            return []
        
        return self.applied_recommendations
    
    def get_recommendation_summary(self) -> Dict[str, Any]:
        """
        Возвращает сводку по примененным рекомендациям.
        
        Returns:
            Словарь со статистикой рекомендаций
        """
        total_recommendations = len(self.applied_recommendations)
        successful_recommendations = sum(1 for r in self.applied_recommendations if r.success)
        
        # Группируем по компонентам
        component_stats = {}
        for rec in self.applied_recommendations:
            # Находим компонент для этой рекомендации
            component_name = "unknown"
            for comp_name, comp in self.components.items():
                if rec in comp.applied_recommendations:
                    component_name = comp_name
                    break
            
            if component_name not in component_stats:
                component_stats[component_name] = {"total": 0, "successful": 0}
            
            component_stats[component_name]["total"] += 1
            if rec.success:
                component_stats[component_name]["successful"] += 1
        
        return {
            "total_recommendations": total_recommendations,
            "successful_recommendations": successful_recommendations,
            "success_rate": successful_recommendations / total_recommendations if total_recommendations > 0 else 0,
            "component_stats": component_stats,
            "recent_recommendations": self.applied_recommendations[-5:] if self.applied_recommendations else []
        }


# Глобальный экземпляр менеджера
_global_manager = None

def get_global_manager() -> Optional[SmartTunerIntegrationManager]:
    """Возвращает глобальный экземпляр менеджера."""
    return _global_manager

def set_global_manager(manager: SmartTunerIntegrationManager):
    """Устанавливает глобальный экземпляр менеджера."""
    global _global_manager
    _global_manager = manager

def initialize_smart_tuner():
    """Инициализирует Smart Tuner систему."""
    global _global_manager
    if _global_manager is None:
        _global_manager = SmartTunerIntegrationManager()
    return _global_manager 