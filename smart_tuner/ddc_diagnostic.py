#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDCLossDiagnostic - Диагностический модуль для анализа проблем DDC Loss
Основан на анализе несовпадений размеров и паттернов в DDC loss

Особенности:
- Анализ несовпадений размеров тензоров
- Визуализация паттернов и статистики
- Генерация отчетов и рекомендаций
- Интеграция с Smart Tuner
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SizeMismatchInfo:
    """Информация о несовпадении размеров."""
    step: int
    coarse_shape: tuple
    fine_shape: tuple
    coarse_length: int
    fine_length: int
    length_ratio: float
    min_length: int
    max_length: int
    length_diff: int

class DDCLossDiagnostic:
    """
    Диагностический модуль для анализа проблем DDC Loss.
    
    Собирает статистику по:
    - Несовпадениям размеров тензоров
    - Паттернам в attention weights
    - Истории loss значений
    """
    
    def __init__(self, save_dir: str = "smart_tuner/ddc_diagnostics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Данные для анализа
        self.size_mismatches: List[SizeMismatchInfo] = []
        self.attention_stats: Dict[str, Any] = {}
        self.loss_history: List[float] = []
        
        # Логирование
        self.logger = logging.getLogger(__name__)
        
    def analyze_size_mismatch(self, coarse_attention: torch.Tensor, 
                            fine_attention: torch.Tensor, step: int) -> SizeMismatchInfo:
        """
        Анализирует несовпадение размеров и записывает статистику.
        
        Args:
            coarse_attention: Attention от coarse декодера
            fine_attention: Attention от fine декодера
            step: Текущий шаг обучения
            
        Returns:
            Информация о несовпадении размеров
        """
        coarse_shape = coarse_attention.shape
        fine_shape = fine_attention.shape
        
        mismatch_info = SizeMismatchInfo(
            step=step,
            coarse_shape=coarse_shape,
            fine_shape=fine_shape,
            coarse_length=coarse_shape[1] if len(coarse_shape) > 1 else 0,
            fine_length=fine_shape[1] if len(fine_shape) > 1 else 0,
            length_ratio=fine_shape[1] / coarse_shape[1] if coarse_shape[1] > 0 else 0,
            min_length=min(coarse_shape[1], fine_shape[1]) if len(coarse_shape) > 1 else 0,
            max_length=max(coarse_shape[1], fine_shape[1]) if len(coarse_shape) > 1 else 0,
            length_diff=abs(coarse_shape[1] - fine_shape[1]) if len(coarse_shape) > 1 else 0
        )
        
        self.size_mismatches.append(mismatch_info)
        return mismatch_info
    
    def add_loss_value(self, loss_value: float, step: int):
        """Добавляет значение loss в историю."""
        self.loss_history.append(loss_value)
        
        # Ограничиваем размер истории
        if len(self.loss_history) > 10000:
            self.loss_history = self.loss_history[-5000:]
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Генерирует отчет о проблемах DDC Loss.
        
        Returns:
            Словарь с анализом и рекомендациями
        """
        if not self.size_mismatches:
            return {"error": "Нет данных для анализа"}
        
        length_diffs = [info.length_diff for info in self.size_mismatches]
        length_ratios = [info.length_ratio for info in self.size_mismatches if info.length_ratio > 0]
        
        report = {
            'total_samples': len(self.size_mismatches),
            'perfect_matches': sum(1 for diff in length_diffs if diff == 0),
            'average_length_diff': np.mean(length_diffs),
            'max_length_diff': max(length_diffs),
            'min_length_diff': min(length_diffs),
            'std_length_diff': np.std(length_diffs),
            'average_length_ratio': np.mean(length_ratios) if length_ratios else 0,
            'problematic_samples': sum(1 for diff in length_diffs if diff > 100),
            'recommendations': []
        }
        
        # Генерируем рекомендации
        if report['perfect_matches'] / report['total_samples'] < 0.1:
            report['recommendations'].append("Критически мало точных совпадений размеров")
            
        if report['average_length_diff'] > 50:
            report['recommendations'].append("Высокая средняя разность длин - рассмотрите bucket batching")
            
        if report['max_length_diff'] > 200:
            report['recommendations'].append("Обнаружены экстремальные разности длин - нужно ограничить максимальную длину")
            
        if len(length_ratios) > 0 and (np.std(length_ratios) > 0.5):
            report['recommendations'].append("Высокая вариативность соотношений длин - проблемы с reduction factors")
        
        return report
    
    def suggest_fixes(self, report: Dict[str, Any]) -> List[str]:
        """
        Предлагает конкретные исправления на основе анализа.
        
        Args:
            report: Отчет о проблемах
            
        Returns:
            Список предложений по исправлению
        """
        fixes = []
        
        if report['average_length_diff'] > 30:
            fixes.append("Добавьте динамический padding с группировкой по длине")
            fixes.append("Используйте интерполяцию attention векторов")
            
        if report['problematic_samples'] > report['total_samples'] * 0.1:
            fixes.append("Ограничьте максимальную длину последовательностей")
            fixes.append("Добавьте pre-processing для нормализации длин")
            
        if 'length_ratio' in report and report['average_length_ratio'] > 2.0:
            fixes.append("Пересмотрите reduction factors для декодеров")
            fixes.append("Добавьте адаптивное масштабирование attention")
            
        return fixes
    
    def save_diagnostic_data(self, filename: str = "ddc_diagnostic_data.json"):
        """Сохраняет диагностические данные в JSON."""
        import json
        
        data = {
            'size_mismatches': [
                {
                    'step': info.step,
                    'coarse_shape': info.coarse_shape,
                    'fine_shape': info.fine_shape,
                    'coarse_length': info.coarse_length,
                    'fine_length': info.fine_length,
                    'length_ratio': info.length_ratio,
                    'length_diff': info.length_diff
                }
                for info in self.size_mismatches
            ],
            'loss_history': self.loss_history[-1000:],  # Последние 1000 значений
            'attention_stats': self.attention_stats
        }
        
        filepath = self.save_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Диагностические данные сохранены в {filepath}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Возвращает краткую сводку диагностики."""
        if not self.size_mismatches:
            return {"status": "no_data", "message": "Нет данных для анализа"}
        
        report = self.generate_report()
        
        return {
            "status": "analyzed",
            "total_samples": report['total_samples'],
            "mismatch_rate": 1 - (report['perfect_matches'] / report['total_samples']),
            "avg_length_diff": report['average_length_diff'],
            "max_length_diff": report['max_length_diff'],
            "problematic_rate": report['problematic_samples'] / report['total_samples'],
            "recommendations_count": len(report['recommendations'])
        }
    
    def reset(self):
        """Сбрасывает все собранные данные."""
        self.size_mismatches.clear()
        self.attention_stats.clear()
        self.loss_history.clear()
        self.logger.info("🔄 Диагностические данные сброшены")

# Глобальный экземпляр диагностики
_global_ddc_diagnostic = None

def get_global_ddc_diagnostic() -> Optional[DDCLossDiagnostic]:
    """Возвращает глобальный экземпляр диагностики."""
    return _global_ddc_diagnostic

def set_global_ddc_diagnostic(diagnostic: DDCLossDiagnostic):
    """Устанавливает глобальный экземпляр диагностики."""
    global _global_ddc_diagnostic
    _global_ddc_diagnostic = diagnostic

def initialize_ddc_diagnostic(save_dir: str = "smart_tuner/ddc_diagnostics") -> DDCLossDiagnostic:
    """Инициализирует глобальную диагностику DDC."""
    global _global_ddc_diagnostic
    if _global_ddc_diagnostic is None:
        _global_ddc_diagnostic = DDCLossDiagnostic(save_dir)
    return _global_ddc_diagnostic 