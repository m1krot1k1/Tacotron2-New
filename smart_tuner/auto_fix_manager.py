#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoFixManager - Автоматическое исправление критических проблем обучения
Интеллектуальная система самовосстановления для Tacotron2

Особенности:
1. Автоматическое исправление исчезновения градиентов
2. Восстановление attention alignment
3. Адаптация гиперпараметров в реальном времени
4. Интеграция с Smart Tuner и Telegram мониторингом
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time

@dataclass
class FixAction:
    """Действие по исправлению проблемы"""
    name: str
    description: str
    priority: int  # 1-10, где 10 - критично
    applied: bool = False
    success: bool = False
    timestamp: float = 0.0

class AutoFixManager:
    """
    Менеджер автоматического исправления критических проблем
    """
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                 hparams, telegram_monitor=None):
        """
        Инициализация менеджера автоматического исправления
        
        Args:
            model: PyTorch модель
            optimizer: Оптимизатор
            hparams: Гиперпараметры
            telegram_monitor: Telegram мониторинг для уведомлений
        """
        self.model = model
        self.optimizer = optimizer
        self.hparams = hparams
        self.telegram_monitor = telegram_monitor
        self.logger = logging.getLogger('AutoFixManager')
        
        # История исправлений
        self.fix_history = []
        self.last_fix_time = 0
        self.fix_cooldown = 60  # 1 минута между исправлениями
        
        # Пороги для автоматических исправлений
        self.thresholds = {
            'gradient_vanishing': 1e-8,
            'gradient_explosion': 100.0,
            'attention_diagonality_critical': 0.1,
            'attention_diagonality_warning': 0.3,
            'gate_accuracy_critical': 0.3,
            'loss_critical': 50.0,
            'nan_detected': True
        }
        
        # Счетчики проблем
        self.problem_counters = {
            'gradient_vanishing': 0,
            'gradient_explosion': 0,
            'attention_problems': 0,
            'gate_problems': 0,
            'nan_problems': 0
        }
        
        self.logger.info("🤖 AutoFixManager инициализирован")
    
    def analyze_and_fix(self, step: int, metrics: Dict[str, Any], 
                       loss: Optional[torch.Tensor] = None) -> List[FixAction]:
        """
        Анализирует метрики и применяет автоматические исправления
        
        Args:
            step: Текущий шаг обучения
            metrics: Словарь с метриками
            loss: Тензор loss (опционально)
            
        Returns:
            Список примененных исправлений
        """
        current_time = time.time()
        if current_time - self.last_fix_time < self.fix_cooldown:
            return []
        
        applied_fixes = []
        
        try:
            # 1. Проверка исчезновения градиентов
            grad_norm = metrics.get('grad_norm', 0.0)
            if grad_norm < self.thresholds['gradient_vanishing']:
                fixes = self._fix_gradient_vanishing(step, grad_norm, loss)
                applied_fixes.extend(fixes)
                self.problem_counters['gradient_vanishing'] += 1
            
            # 2. Проверка взрыва градиентов
            elif grad_norm > self.thresholds['gradient_explosion']:
                fixes = self._fix_gradient_explosion(step, grad_norm)
                applied_fixes.extend(fixes)
                self.problem_counters['gradient_explosion'] += 1
            
            # 3. Проверка проблем с attention
            attention_diag = metrics.get('attention_diagonality', 1.0)
            if attention_diag < self.thresholds['attention_diagonality_critical']:
                fixes = self._fix_attention_problems(step, attention_diag)
                applied_fixes.extend(fixes)
                self.problem_counters['attention_problems'] += 1
            
            # 4. Проверка проблем с gate
            gate_accuracy = metrics.get('gate_accuracy', 1.0)
            if gate_accuracy < self.thresholds['gate_accuracy_critical']:
                fixes = self._fix_gate_problems(step, gate_accuracy)
                applied_fixes.extend(fixes)
                self.problem_counters['gate_problems'] += 1
            
            # 5. Проверка NaN/Inf
            if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
                fixes = self._fix_nan_problems(step, loss)
                applied_fixes.extend(fixes)
                self.problem_counters['nan_problems'] += 1
            
            # Применяем исправления
            for fix in applied_fixes:
                fix.timestamp = current_time
                fix.applied = True
                self._apply_fix(fix)
            
            # Обновляем время последнего исправления
            if applied_fixes:
                self.last_fix_time = current_time
                self.fix_history.extend(applied_fixes)
                
                # Отправляем уведомление
                self._send_fix_notification(step, applied_fixes)
            
        except Exception as e:
            self.logger.error(f"Ошибка в analyze_and_fix: {e}")
        
        return applied_fixes
    
    def _fix_gradient_vanishing(self, step: int, grad_norm: float, 
                               loss: Optional[torch.Tensor]) -> List[FixAction]:
        """Исправление исчезновения градиентов"""
        fixes = []
        
        # 1. Увеличение масштаба loss
        if loss is not None:
            fix = FixAction(
                name="loss_scaling",
                description=f"Увеличение масштаба loss для восстановления градиентов (grad_norm={grad_norm:.2e})",
                priority=9
            )
            fixes.append(fix)
        
        # 2. Снижение learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = max(current_lr * 0.1, 1e-8)
        
        fix = FixAction(
            name="lr_reduction",
            description=f"Снижение learning rate: {current_lr:.2e} → {new_lr:.2e}",
            priority=8
        )
        fixes.append(fix)
        
        # 3. Увеличение guided attention weight
        if hasattr(self.hparams, 'guide_loss_weight'):
            current_weight = getattr(self.hparams, 'guide_loss_weight', 1.0)
            new_weight = min(current_weight * 5.0, 100.0)
            
            fix = FixAction(
                name="guided_attention_boost",
                description=f"Увеличение guided attention weight: {current_weight} → {new_weight}",
                priority=7
            )
            fixes.append(fix)
        
        return fixes
    
    def _fix_gradient_explosion(self, step: int, grad_norm: float) -> List[FixAction]:
        """Исправление взрыва градиентов"""
        fixes = []
        
        # 1. Усиление gradient clipping
        current_clip = getattr(self.hparams, 'grad_clip_thresh', 1.0)
        new_clip = max(current_clip * 0.1, 0.01)
        
        fix = FixAction(
            name="gradient_clipping",
            description=f"Усиление gradient clipping: {current_clip} → {new_clip}",
            priority=9
        )
        fixes.append(fix)
        
        # 2. Снижение learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = max(current_lr * 0.5, 1e-8)
        
        fix = FixAction(
            name="lr_reduction",
            description=f"Снижение learning rate: {current_lr:.2e} → {new_lr:.2e}",
            priority=8
        )
        fixes.append(fix)
        
        return fixes
    
    def _fix_attention_problems(self, step: int, attention_diag: float) -> List[FixAction]:
        """Исправление проблем с attention alignment"""
        fixes = []
        
        # 1. Критическое увеличение guided attention weight
        if hasattr(self.hparams, 'guide_loss_weight'):
            current_weight = getattr(self.hparams, 'guide_loss_weight', 1.0)
            new_weight = min(current_weight * 10.0, 200.0)
            
            fix = FixAction(
                name="guided_attention_critical_boost",
                description=f"Критическое увеличение guided attention: {current_weight} → {new_weight}",
                priority=10
            )
            fixes.append(fix)
        
        # 2. Снижение dropout для attention
        if hasattr(self.hparams, 'p_attention_dropout'):
            current_dropout = getattr(self.hparams, 'p_attention_dropout', 0.1)
            new_dropout = max(current_dropout * 0.1, 0.001)
            
            fix = FixAction(
                name="attention_dropout_reduction",
                description=f"Снижение attention dropout: {current_dropout} → {new_dropout}",
                priority=7
            )
            fixes.append(fix)
        
        return fixes
    
    def _fix_gate_problems(self, step: int, gate_accuracy: float) -> List[FixAction]:
        """Исправление проблем с gate accuracy"""
        fixes = []
        
        # 1. Снижение gate threshold
        if hasattr(self.hparams, 'gate_threshold'):
            current_threshold = getattr(self.hparams, 'gate_threshold', 0.5)
            new_threshold = max(current_threshold * 0.5, 0.1)
            
            fix = FixAction(
                name="gate_threshold_reduction",
                description=f"Снижение gate threshold: {current_threshold} → {new_threshold}",
                priority=6
            )
            fixes.append(fix)
        
        # 2. Увеличение веса gate loss
        if hasattr(self.hparams, 'gate_loss_weight'):
            current_weight = getattr(self.hparams, 'gate_loss_weight', 1.0)
            new_weight = min(current_weight * 2.0, 10.0)
            
            fix = FixAction(
                name="gate_loss_boost",
                description=f"Увеличение веса gate loss: {current_weight} → {new_weight}",
                priority=5
            )
            fixes.append(fix)
        
        return fixes
    
    def _fix_nan_problems(self, step: int, loss: torch.Tensor) -> List[FixAction]:
        """Исправление NaN/Inf проблем"""
        fixes = []
        
        # 1. Экстремальное снижение learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = max(current_lr * 0.01, 1e-10)
        
        fix = FixAction(
            name="emergency_lr_reduction",
            description=f"Экстремальное снижение LR: {current_lr:.2e} → {new_lr:.2e}",
            priority=10
        )
        fixes.append(fix)
        
        # 2. Принудительная активация guided attention
        if hasattr(self.hparams, 'use_guided_attn'):
            fix = FixAction(
                name="force_guided_attention",
                description="Принудительная активация guided attention",
                priority=9
            )
            fixes.append(fix)
        
        # 3. Отключение fp16
        if hasattr(self.hparams, 'fp16_run') and self.hparams.fp16_run:
            fix = FixAction(
                name="disable_fp16",
                description="Отключение fp16 для стабильности",
                priority=8
            )
            fixes.append(fix)
        
        return fixes
    
    def _apply_fix(self, fix: FixAction):
        """Применяет конкретное исправление"""
        try:
            if fix.name == "lr_reduction":
                current_lr = self.optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * 0.1, 1e-8)
                for group in self.optimizer.param_groups:
                    group['lr'] = new_lr
                fix.success = True
                
            elif fix.name == "gradient_clipping":
                new_clip = max(getattr(self.hparams, 'grad_clip_thresh', 1.0) * 0.1, 0.01)
                setattr(self.hparams, 'grad_clip_thresh', new_clip)
                fix.success = True
                
            elif fix.name == "guided_attention_boost":
                if hasattr(self.hparams, 'guide_loss_weight'):
                    current_weight = getattr(self.hparams, 'guide_loss_weight', 1.0)
                    new_weight = min(current_weight * 5.0, 100.0)
                    setattr(self.hparams, 'guide_loss_weight', new_weight)
                    fix.success = True
                    
            elif fix.name == "guided_attention_critical_boost":
                if hasattr(self.hparams, 'guide_loss_weight'):
                    current_weight = getattr(self.hparams, 'guide_loss_weight', 1.0)
                    new_weight = min(current_weight * 10.0, 200.0)
                    setattr(self.hparams, 'guide_loss_weight', new_weight)
                    fix.success = True
                    
            elif fix.name == "attention_dropout_reduction":
                if hasattr(self.hparams, 'p_attention_dropout'):
                    current_dropout = getattr(self.hparams, 'p_attention_dropout', 0.1)
                    new_dropout = max(current_dropout * 0.1, 0.001)
                    setattr(self.hparams, 'p_attention_dropout', new_dropout)
                    fix.success = True
                    
            elif fix.name == "gate_threshold_reduction":
                if hasattr(self.hparams, 'gate_threshold'):
                    current_threshold = getattr(self.hparams, 'gate_threshold', 0.5)
                    new_threshold = max(current_threshold * 0.5, 0.1)
                    setattr(self.hparams, 'gate_threshold', new_threshold)
                    fix.success = True
                    
            elif fix.name == "emergency_lr_reduction":
                current_lr = self.optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * 0.01, 1e-10)
                for group in self.optimizer.param_groups:
                    group['lr'] = new_lr
                fix.success = True
                
            elif fix.name == "force_guided_attention":
                setattr(self.hparams, 'use_guided_attn', True)
                setattr(self.hparams, 'guide_loss_weight', 100.0)
                fix.success = True
                
            elif fix.name == "disable_fp16":
                setattr(self.hparams, 'fp16_run', False)
                fix.success = True
            
            if fix.success:
                self.logger.info(f"✅ Применено исправление: {fix.description}")
            else:
                self.logger.warning(f"⚠️ Не удалось применить исправление: {fix.name}")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка применения исправления {fix.name}: {e}")
            fix.success = False
    
    def _send_fix_notification(self, step: int, fixes: List[FixAction]):
        """Отправляет уведомление о примененных исправлениях"""
        if not self.telegram_monitor:
            return
        
        try:
            message = f"🔧 **АВТОМАТИЧЕСКИЕ ИСПРАВЛЕНИЯ**\n\n"
            message += f"📍 **Шаг:** {step}\n"
            message += f"🕐 **Время:** {time.strftime('%H:%M:%S')}\n\n"
            
            message += f"📋 **Примененные исправления:**\n"
            for fix in fixes:
                status = "✅" if fix.success else "❌"
                message += f"{status} {fix.description}\n"
            
            message += f"\n🤖 **Система продолжает мониторинг...**"
            
            self.telegram_monitor.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Ошибка отправки уведомления: {e}")
    
    def get_fix_statistics(self) -> Dict[str, Any]:
        """Получает статистику исправлений"""
        return {
            'total_fixes': len(self.fix_history),
            'successful_fixes': len([f for f in self.fix_history if f.success]),
            'problem_counters': self.problem_counters.copy(),
            'recent_fixes': self.fix_history[-10:] if self.fix_history else []
        }
    
    def reset_counters(self):
        """Сбрасывает счетчики проблем"""
        for key in self.problem_counters:
            self.problem_counters[key] = 0 