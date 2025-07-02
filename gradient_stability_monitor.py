"""
Модуль для мониторинга стабильности градиентов во время обучения.
Пока что содержит базовую заглушку для совместимости.
"""

import torch
from typing import Optional, Dict, Any


class GradientStabilityMonitor:
    """
    Мониторинг стабильности градиентов для предотвращения проблем обучения.
    Базовая реализация для совместимости.
    """
    
    def __init__(self):
        """Инициализация монитора градиентов."""
        self.gradient_history = []
        self.max_history_size = 100
        
    def check_gradients(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Проверка градиентов модели на стабильность.
        
        Args:
            model: PyTorch модель для проверки
            
        Returns:
            Словарь с информацией о градиентах
        """
        grad_info = {
            'has_nan': False,
            'has_inf': False,
            'grad_norm': 0.0,
            'max_grad': 0.0,
            'min_grad': 0.0
        }
        
        try:
            total_norm = 0.0
            max_grad = 0.0
            min_grad = float('inf')
            
            for param in model.parameters():
                if param.grad is not None:
                    # Проверка на NaN и Inf
                    if torch.isnan(param.grad).any():
                        grad_info['has_nan'] = True
                    if torch.isinf(param.grad).any():
                        grad_info['has_inf'] = True
                    
                    # Вычисление норм градиентов
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    
                    # Максимальный и минимальный градиенты
                    max_grad = max(max_grad, param.grad.abs().max().item())
                    min_grad = min(min_grad, param.grad.abs().min().item())
            
            grad_info['grad_norm'] = total_norm ** 0.5
            grad_info['max_grad'] = max_grad
            grad_info['min_grad'] = min_grad if min_grad != float('inf') else 0.0
            
        except Exception as e:
            print(f"⚠️ Ошибка при проверке градиентов: {e}")
            
        return grad_info
    
    def log_gradients(self, grad_info: Dict[str, Any], step: int):
        """
        Логирование информации о градиентах.
        
        Args:
            grad_info: Информация о градиентах
            step: Номер шага обучения
        """
        self.gradient_history.append({
            'step': step,
            'grad_info': grad_info
        })
        
        # Ограничиваем размер истории
        if len(self.gradient_history) > self.max_history_size:
            self.gradient_history.pop(0)
    
    def get_stability_status(self) -> str:
        """
        Получение текущего статуса стабильности.
        
        Returns:
            Строка со статусом стабильности
        """
        if not self.gradient_history:
            return "unknown"
        
        last_grad_info = self.gradient_history[-1]['grad_info']
        
        if last_grad_info['has_nan'] or last_grad_info['has_inf']:
            return "unstable"
        elif last_grad_info['grad_norm'] > 10.0:
            return "high_gradients"
        elif last_grad_info['grad_norm'] < 1e-8:
            return "vanishing_gradients"
        else:
            return "stable"
    
    def check_gradient_stability(self, model: torch.nn.Module, loss: torch.Tensor, step: int) -> Dict[str, Any]:
        """
        Проверка стабильности градиентов с обнаружением критических проблем.
        
        Args:
            model: PyTorch модель
            loss: Значение функции потерь
            step: Номер шага обучения
            
        Returns:
            Словарь с результатами проверки стабильности
        """
        result = {
            'explosion_detected': False,
            'nan_detected': False,
            'recommendations': []
        }
        
        try:
            # Проверяем loss на NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                result['nan_detected'] = True
                result['recommendations'].append('Обнаружен NaN/Inf в loss - перезапуск с пониженным learning rate')
                return result
            
            # Проверяем градиенты
            grad_info = self.check_gradients(model)
            
            if grad_info['has_nan'] or grad_info['has_inf']:
                result['nan_detected'] = True
                result['recommendations'].append('Обнаружен NaN/Inf в градиентах')
            
            if grad_info['grad_norm'] > 100.0:  # Порог для взрыва градиентов
                result['explosion_detected'] = True
                result['recommendations'].append(f'Взрыв градиентов (норма: {grad_info["grad_norm"]:.2f})')
            
            # Логируем информацию
            self.log_gradients(grad_info, step)
            
        except Exception as e:
            print(f"⚠️ Ошибка проверки стабильности градиентов: {e}")
            result['recommendations'].append(f'Ошибка проверки: {e}')
        
        return result 