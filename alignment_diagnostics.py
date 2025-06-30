#!/usr/bin/env python3
"""
🎯 Диагностическая утилита для анализа Alignment матриц Tacotron2
Автор: AI Assistant для проекта Intelligent TTS Training Pipeline

Этот модуль предоставляет комплексный анализ качества attention alignment
и рекомендации по улучшению обучения модели.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlignmentDiagnostics:
    """Класс для диагностики и анализа alignment матриц."""
    
    def __init__(self):
        self.results = {}
        
    def analyze_alignment_matrix(self, alignment: np.ndarray, 
                               step: int = 0, 
                               text_length: int = None,
                               audio_length: int = None) -> Dict[str, Any]:
        """
        Комплексный анализ alignment матрицы.
        
        Args:
            alignment: numpy array размером (decoder_steps, encoder_steps)
            step: номер шага обучения
            text_length: реальная длина текста (без padding)
            audio_length: реальная длина аудио (без padding)
            
        Returns:
            Словарь с результатами анализа
        """
        logger.info(f"🔍 Анализ alignment матрицы на шаге {step}")
        logger.info(f"📏 Размерность: {alignment.shape}")
        
        results = {
            'step': step,
            'shape': alignment.shape,
            'diagnostics': {},
            'problems': [],
            'recommendations': []
        }
        
        # 1. Базовые статистики
        results['diagnostics']['mean_attention'] = float(np.mean(alignment))
        results['diagnostics']['max_attention'] = float(np.max(alignment))
        results['diagnostics']['min_attention'] = float(np.min(alignment))
        results['diagnostics']['std_attention'] = float(np.std(alignment))
        
        # 2. Анализ диагональности
        diag_score = self._calculate_diagonal_score(alignment)
        results['diagnostics']['diagonal_score'] = diag_score
        
        # 3. Анализ монотонности
        monotonic_score = self._calculate_monotonic_score(alignment)
        results['diagnostics']['monotonic_score'] = monotonic_score
        
        # 4. Анализ фокусировки
        focus_score = self._calculate_focus_score(alignment)
        results['diagnostics']['focus_score'] = focus_score
        
        # 5. Анализ энтропии
        entropy_score = self._calculate_entropy_score(alignment)
        results['diagnostics']['entropy_score'] = entropy_score
        
        # 6. Детекция проблем
        problems = self._detect_problems(alignment, results['diagnostics'])
        results['problems'] = problems
        
        # 7. Генерация рекомендаций
        recommendations = self._generate_recommendations(problems, results['diagnostics'])
        results['recommendations'] = recommendations
        
        # 8. Оценка общего качества
        overall_score = self._calculate_overall_score(results['diagnostics'])
        results['overall_score'] = overall_score
        
        self._log_results(results)
        return results
    
    def _calculate_diagonal_score(self, alignment: np.ndarray) -> float:
        """Вычисляет степень диагональности alignment матрицы."""
        H, W = alignment.shape
        diagonal_sum = 0.0
        total_sum = np.sum(alignment)
        
        if total_sum == 0:
            return 0.0
        
        # Создаем идеальную диагональную маску
        for i in range(H):
            # Пропорциональная позиция на диагонали
            diag_pos = int((i / H) * W)
            # Суммируем веса в окрестности диагонали (±3 позиции)
            for j in range(max(0, diag_pos-3), min(W, diag_pos+4)):
                diagonal_sum += alignment[i, j]
        
        return diagonal_sum / total_sum
    
    def _calculate_monotonic_score(self, alignment: np.ndarray) -> float:
        """Вычисляет степень монотонности alignment."""
        H, W = alignment.shape
        monotonic_violations = 0
        total_transitions = 0
        
        prev_peak = 0
        for i in range(1, H):
            # Находим пик attention для текущей строки
            current_peak = np.argmax(alignment[i])
            
            # Проверяем монотонность
            if current_peak < prev_peak:
                monotonic_violations += 1
            
            prev_peak = current_peak
            total_transitions += 1
        
        if total_transitions == 0:
            return 1.0
            
        return 1.0 - (monotonic_violations / total_transitions)
    
    def _calculate_focus_score(self, alignment: np.ndarray) -> float:
        """Вычисляет степень фокусировки attention."""
        # Вычисляем среднюю концентрацию attention по строкам
        focus_scores = []
        for i in range(alignment.shape[0]):
            row = alignment[i]
            if np.sum(row) > 0:
                # Вычисляем энтропию строки (низкая энтропия = высокая фокусировка)
                row_normalized = row / np.sum(row)
                entropy = -np.sum(row_normalized * np.log(row_normalized + 1e-8))
                # Конвертируем энтропию в score фокусировки (0-1)
                max_entropy = np.log(len(row))
                focus_score = 1.0 - (entropy / max_entropy)
                focus_scores.append(focus_score)
        
        return np.mean(focus_scores) if focus_scores else 0.0
    
    def _calculate_entropy_score(self, alignment: np.ndarray) -> float:
        """Вычисляет энтропию всей alignment матрицы."""
        flat = alignment.flatten()
        if np.sum(flat) == 0:
            return 0.0
        
        flat_normalized = flat / np.sum(flat)
        entropy = -np.sum(flat_normalized * np.log(flat_normalized + 1e-8))
        max_entropy = np.log(len(flat))
        return entropy / max_entropy
    
    def _detect_problems(self, alignment: np.ndarray, diagnostics: Dict) -> List[Dict]:
        """Определяет проблемы в alignment матрице."""
        problems = []
        
        # Проблема 1: Горизонтальная полоса (как на вашем графике)
        if diagnostics['diagonal_score'] < 0.3:
            problems.append({
                'type': 'horizontal_stripe',
                'severity': 'critical',
                'description': 'Attention фокусируется только на первых encoder timesteps',
                'score': diagnostics['diagonal_score']
            })
        
        # Проблема 2: Низкая монотонность
        if diagnostics['monotonic_score'] < 0.5:
            problems.append({
                'type': 'non_monotonic',
                'severity': 'high',
                'description': 'Attention не следует монотонной последовательности',
                'score': diagnostics['monotonic_score']
            })
        
        # Проблема 3: Размытый attention
        if diagnostics['focus_score'] < 0.4:
            problems.append({
                'type': 'unfocused_attention',
                'severity': 'medium',
                'description': 'Attention слишком размыт, нет четкой фокусировки',
                'score': diagnostics['focus_score']
            })
        
        # Проблема 4: Высокая энтропия
        if diagnostics['entropy_score'] > 0.8:
            problems.append({
                'type': 'high_entropy',
                'severity': 'medium',
                'description': 'Слишком равномерное распределение attention весов',
                'score': diagnostics['entropy_score']
            })
        
        # Проблема 5: Очень маленькие веса
        if diagnostics['max_attention'] < 0.1:
            problems.append({
                'type': 'weak_attention',
                'severity': 'high',
                'description': 'Attention веса слишком малы',
                'score': diagnostics['max_attention']
            })
        
        return problems
    
    def _generate_recommendations(self, problems: List[Dict], diagnostics: Dict) -> List[str]:
        """Генерирует рекомендации по улучшению."""
        recommendations = []
        
        for problem in problems:
            if problem['type'] == 'horizontal_stripe':
                recommendations.extend([
                    "🔧 КРИТИЧНО: Увеличить вес guided attention loss в начале обучения",
                    "🔧 Проверить правильность формулы guided attention loss",
                    "🔧 Уменьшить learning rate для стабилизации обучения",
                    "🔧 Увеличить количество шагов до начала decay guided loss",
                    "🔧 Проверить инициализацию attention весов"
                ])
            
            if problem['type'] == 'non_monotonic':
                recommendations.extend([
                    "🔧 Включить forward attention constraint",
                    "🔧 Увеличить sigma параметр в guided attention",
                    "🔧 Добавить monotonic alignment regularization"
                ])
                
            if problem['type'] == 'unfocused_attention':
                recommendations.extend([
                    "🔧 Уменьшить dropout в attention слоях",
                    "🔧 Увеличить attention dimension",
                    "🔧 Проверить location-based attention параметры"
                ])
        
        # Общие рекомендации
        if diagnostics['diagonal_score'] < 0.5:
            recommendations.append("📊 Продолжить обучение - модель еще не научилась alignment")
        
        return list(set(recommendations))  # Убираем дубликаты
    
    def _calculate_overall_score(self, diagnostics: Dict) -> float:
        """Вычисляет общую оценку качества alignment."""
        weights = {
            'diagonal_score': 0.4,
            'monotonic_score': 0.3,
            'focus_score': 0.2,
            'entropy_score': 0.1  # Обратный вес - низкая энтропия лучше
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric == 'entropy_score':
                # Для энтропии: чем меньше, тем лучше
                score += (1.0 - diagnostics[metric]) * weight
            else:
                score += diagnostics[metric] * weight
        
        return score
    
    def _log_results(self, results: Dict):
        """Логирует результаты анализа."""
        logger.info("=" * 60)
        logger.info(f"📊 ДИАГНОСТИКА ALIGNMENT МАТРИЦЫ - ШАГ {results['step']}")
        logger.info("=" * 60)
        
        # Основные метрики
        diag = results['diagnostics']
        logger.info(f"🎯 Диагональность: {diag['diagonal_score']:.3f}")
        logger.info(f"📈 Монотонность: {diag['monotonic_score']:.3f}")
        logger.info(f"🔍 Фокусировка: {diag['focus_score']:.3f}")
        logger.info(f"🌊 Энтропия: {diag['entropy_score']:.3f}")
        logger.info(f"⭐ Общая оценка: {results['overall_score']:.3f}")
        
        # Проблемы
        if results['problems']:
            logger.warning("🚨 ОБНАРУЖЕННЫЕ ПРОБЛЕМЫ:")
            for problem in results['problems']:
                logger.warning(f"  - {problem['type']}: {problem['description']}")
        
        # Рекомендации
        if results['recommendations']:
            logger.info("💡 РЕКОМЕНДАЦИИ:")
            for rec in results['recommendations'][:5]:  # Топ-5 рекомендаций
                logger.info(f"  {rec}")
    
    def visualize_alignment(self, alignment: np.ndarray, 
                          step: int, 
                          save_path: str = None) -> str:
        """Создает визуализацию alignment матрицы с диагностикой."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Основная alignment матрица
        im1 = ax1.imshow(alignment, aspect='auto', origin='lower', cmap='Blues')
        ax1.set_title(f'Alignment Matrix (Step {step})')
        ax1.set_xlabel('Encoder timestep')
        ax1.set_ylabel('Decoder timestep')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Диагональная проекция
        H, W = alignment.shape
        diagonal_profile = []
        for i in range(H):
            diag_pos = int((i / H) * W)
            if diag_pos < W:
                diagonal_profile.append(alignment[i, diag_pos])
            else:
                diagonal_profile.append(0)
        
        ax2.plot(diagonal_profile, 'r-', linewidth=2)
        ax2.set_title('Diagonal Attention Profile')
        ax2.set_xlabel('Decoder timestep')
        ax2.set_ylabel('Attention weight')
        ax2.grid(True, alpha=0.3)
        
        # 3. Attention фокус по временным шагам
        attention_peaks = [np.argmax(alignment[i]) for i in range(H)]
        ideal_peaks = [int((i / H) * W) for i in range(H)]
        
        ax3.plot(attention_peaks, 'b-', label='Actual peaks', linewidth=2)
        ax3.plot(ideal_peaks, 'r--', label='Ideal diagonal', linewidth=2)
        ax3.set_title('Attention Peak Progression')
        ax3.set_xlabel('Decoder timestep')
        ax3.set_ylabel('Encoder timestep')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Attention энтропия по строкам
        entropies = []
        for i in range(H):
            row = alignment[i]
            if np.sum(row) > 0:
                row_norm = row / np.sum(row)
                entropy = -np.sum(row_norm * np.log(row_norm + 1e-8))
                entropies.append(entropy)
            else:
                entropies.append(0)
        
        ax4.plot(entropies, 'g-', linewidth=2)
        ax4.set_title('Attention Entropy per Decoder Step')
        ax4.set_xlabel('Decoder timestep')
        ax4.set_ylabel('Entropy')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"💾 Диагностика сохранена: {save_path}")
            return save_path
        else:
            plt.show()
            return "displayed"

def analyze_current_alignment(alignment_path: str = None, step: int = 500):
    """
    Быстрый анализ текущей alignment матрицы.
    Используйте эту функцию для анализа вашей проблемы.
    """
    logger.info("🚀 Запуск диагностики alignment матрицы")
    
    # Если путь к alignment не указан, создаем симуляцию проблемы пользователя
    if alignment_path is None:
        logger.warning("⚠️ Путь к alignment не указан, создаем симуляцию проблемы")
        # Симулируем проблему: горизонтальная полоса сверху
        alignment = np.zeros((200, 2500))
        # Создаем горизонтальную полосу в верхней части (как на вашем графике)
        alignment[:50, :] = np.random.exponential(0.1, (50, 2500))
        alignment[:10, :] = np.random.exponential(0.3, (10, 2500))  # Еще ярче сверху
    else:
        alignment = np.load(alignment_path)
    
    diagnostics = AlignmentDiagnostics()
    results = diagnostics.analyze_alignment_matrix(alignment, step=step)
    
    # Создаем визуализацию
    vis_path = f"alignment_diagnostics_step_{step}.png"
    diagnostics.visualize_alignment(alignment, step, vis_path)
    
    return results

if __name__ == "__main__":
    # Анализ текущей проблемы пользователя
    results = analyze_current_alignment(step=500)
    
    print("\n" + "="*80)
    print("🎯 ЗАКЛЮЧЕНИЕ ПО ВАШЕЙ ПРОБЛЕМЕ:")
    print("="*80)
    print(f"Общая оценка качества: {results['overall_score']:.1%}")
    print(f"Диагональность: {results['diagnostics']['diagonal_score']:.1%}")
    print(f"Монотонность: {results['diagnostics']['monotonic_score']:.1%}")
    
    if results['problems']:
        print("\n🚨 КРИТИЧЕСКИЕ ПРОБЛЕМЫ:")
        for problem in results['problems']:
            print(f"  • {problem['description']}")
    
    if results['recommendations']:
        print("\n💡 ПЕРВООЧЕРЕДНЫЕ ДЕЙСТВИЯ:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            print(f"{i}. {rec}") 