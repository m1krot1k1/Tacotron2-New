# Создадим код для диагностики и визуализации проблем DDC Loss

diagnostic_code = """
# Диагностический модуль для DDC Loss проблем
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Tuple, List, Dict

class DDCLossDiagnostic:
    def __init__(self):
        self.size_mismatches = []
        self.attention_stats = {}
        self.loss_history = []
        
    def analyze_size_mismatch(self, coarse_attention, fine_attention, step: int):
        \"\"\"
        Анализирует несоответствие размеров и записывает статистику
        \"\"\"
        coarse_shape = coarse_attention.shape
        fine_shape = fine_attention.shape
        
        mismatch_info = {
            'step': step,
            'coarse_shape': coarse_shape,
            'fine_shape': fine_shape,
            'coarse_length': coarse_shape[1] if len(coarse_shape) > 1 else 0,
            'fine_length': fine_shape[1] if len(fine_shape) > 1 else 0,
            'length_ratio': fine_shape[1] / coarse_shape[1] if coarse_shape[1] > 0 else 0,
            'min_length': min(coarse_shape[1], fine_shape[1]) if len(coarse_shape) > 1 else 0,
            'max_length': max(coarse_shape[1], fine_shape[1]) if len(coarse_shape) > 1 else 0,
            'length_diff': abs(coarse_shape[1] - fine_shape[1]) if len(coarse_shape) > 1 else 0
        }
        
        self.size_mismatches.append(mismatch_info)
        return mismatch_info
    
    def visualize_size_patterns(self, save_path='ddc_size_analysis.png'):
        \"\"\"
        Визуализирует паттерны размеров тензоров
        \"\"\"
        if not self.size_mismatches:
            print("Нет данных о размерах для анализа")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # График 1: Длины последовательностей во времени
        steps = [info['step'] for info in self.size_mismatches]
        coarse_lengths = [info['coarse_length'] for info in self.size_mismatches]
        fine_lengths = [info['fine_length'] for info in self.size_mismatches]
        
        axes[0, 0].plot(steps, coarse_lengths, label='Coarse Decoder', alpha=0.7)
        axes[0, 0].plot(steps, fine_lengths, label='Fine Decoder', alpha=0.7)
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Sequence Length')
        axes[0, 0].set_title('Sequence Lengths Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # График 2: Распределение разностей длин
        length_diffs = [info['length_diff'] for info in self.size_mismatches]
        axes[0, 1].hist(length_diffs, bins=30, alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Length Difference')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Length Differences')
        axes[0, 1].grid(True, alpha=0.3)
        
        # График 3: Соотношение длин
        length_ratios = [info['length_ratio'] for info in self.size_mismatches if info['length_ratio'] > 0]
        axes[1, 0].scatter(range(len(length_ratios)), length_ratios, alpha=0.6, color='green')
        axes[1, 0].axhline(y=1.0, color='red', linestyle='--', label='Equal Length')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Fine/Coarse Length Ratio')
        axes[1, 0].set_title('Length Ratio Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # График 4: Тепловая карта размеров
        if len(coarse_lengths) > 10:
            # Создаем 2D массив для тепловой карты
            coarse_bins = np.linspace(min(coarse_lengths), max(coarse_lengths), 20)
            fine_bins = np.linspace(min(fine_lengths), max(fine_lengths), 20)
            
            heatmap_data = np.zeros((len(fine_bins)-1, len(coarse_bins)-1))
            
            for coarse_len, fine_len in zip(coarse_lengths, fine_lengths):
                coarse_idx = np.digitize(coarse_len, coarse_bins) - 1
                fine_idx = np.digitize(fine_len, fine_bins) - 1
                
                if 0 <= coarse_idx < heatmap_data.shape[1] and 0 <= fine_idx < heatmap_data.shape[0]:
                    heatmap_data[fine_idx, coarse_idx] += 1
            
            im = axes[1, 1].imshow(heatmap_data, cmap='Blues', aspect='auto')
            axes[1, 1].set_xlabel('Coarse Length Bins')
            axes[1, 1].set_ylabel('Fine Length Bins')
            axes[1, 1].set_title('Length Correlation Heatmap')
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Диагностические графики сохранены в {save_path}")
    
    def generate_report(self) -> Dict:
        \"\"\"
        Генерирует отчет о проблемах DDC Loss
        \"\"\"
        if not self.size_mismatches:
            return {"error": "Нет данных для анализа"}
        
        length_diffs = [info['length_diff'] for info in self.size_mismatches]
        length_ratios = [info['length_ratio'] for info in self.size_mismatches if info['length_ratio'] > 0]
        
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
    
    def suggest_fixes(self, report: Dict) -> List[str]:
        \"\"\"
        Предлагает конкретные исправления на основе анализа
        \"\"\"
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

# Пример использования диагностического модуля
def example_usage():
    \"\"\"Пример использования DDCLossDiagnostic\"\"\"
    diagnostic = DDCLossDiagnostic()
    
    # Симуляция проблемных данных
    for step in range(100):
        # Создаем тензоры с разными размерами
        coarse_len = np.random.randint(100, 300)
        fine_len = np.random.randint(200, 600)
        
        coarse_attention = torch.randn(4, coarse_len, 512)
        fine_attention = torch.randn(4, fine_len, 512)
        
        diagnostic.analyze_size_mismatch(coarse_attention, fine_attention, step)
    
    # Генерируем отчет и визуализацию
    report = diagnostic.generate_report()
    fixes = diagnostic.suggest_fixes(report)
    
    print("=== ОТЧЕТ О ПРОБЛЕМАХ DDC LOSS ===")
    for key, value in report.items():
        if key != 'recommendations':
            print(f"{key}: {value}")
    
    print("\\n=== РЕКОМЕНДАЦИИ ===")
    for rec in report['recommendations']:
        print(f"- {rec}")
    
    print("\\n=== ПРЕДЛАГАЕМЫЕ ИСПРАВЛЕНИЯ ===")
    for fix in fixes:
        print(f"• {fix}")
    
    # Создаем визуализацию
    diagnostic.visualize_size_patterns()

if __name__ == "__main__":
    example_usage()
"""

# Сохраняем диагностический код
with open("ddc_diagnostic.py", "w", encoding="utf-8") as f:
    f.write(diagnostic_code)

print("Создан файл ddc_diagnostic.py для диагностики проблем DDC Loss")