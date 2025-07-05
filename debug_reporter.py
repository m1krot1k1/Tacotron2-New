#!/usr/bin/env python3
"""
🔍 DEBUG REPORTER - Система сбора технической диагностики
Собирает подробную информацию для диагностики проблем обучения

Автор: Smart Assistant
Версия: 1.0
"""

import os
import json
import time
import psutil
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import traceback


class DebugReporter:
    """
    🔍 Система сбора и отправки подробной технической диагностики
    """
    
    def __init__(self, telegram_monitor=None):
        self.telegram_monitor = telegram_monitor
        self.debug_data = []
        self.start_time = time.time()
        self.last_report_step = 0
        self.report_interval = 250  # уменьшено с 1000 для более частого мониторинга
        
        # История для анализа трендов
        self.loss_history = []
        self.attention_history = []
        self.gradient_history = []
        self.restart_history = []
        self.warning_history = []
        
        print("🔍 Debug Reporter инициализирован")
    
    def collect_step_data(self, step: int, metrics: Dict[str, Any], 
                         model=None, y_pred=None, loss_components=None,
                         hparams=None, smart_tuner_decisions=None):
        """
        📊 Собирает данные для одного шага обучения
        """
        try:
            step_data = {
                'timestamp': datetime.now().isoformat(),
                'step': step,
                'training_time_hours': (time.time() - self.start_time) / 3600,
                
                # === 🔥 LOSS АНАЛИЗ ===
                'loss_analysis': self._analyze_loss(loss_components, metrics),
                
                # === 🎯 ATTENTION АНАЛИЗ ===
                'attention_analysis': self._analyze_attention(y_pred),
                
                # === 📈 ГРАДИЕНТЫ ===
                'gradient_analysis': self._analyze_gradients(model),
                
                # === ⚙️ ГИПЕРПАРАМЕТРЫ ===
                'hyperparameters': self._collect_hyperparameters(hparams),
                
                # === 🖥️ СИСТЕМНАЯ ИНФОРМАЦИЯ ===
                'system_info': self._collect_system_info(),
                
                # === 🤖 SMART TUNER РЕШЕНИЯ ===
                'smart_tuner_decisions': smart_tuner_decisions or {},
                
                # === ⚠️ ПРОБЛЕМЫ И ПРЕДУПРЕЖДЕНИЯ ===
                'issues_detected': self._detect_issues(metrics, loss_components, y_pred),
                
                # === 📊 ТРЕНДЫ ===
                'trends': self._analyze_trends(step, metrics)
            }
            
            self.debug_data.append(step_data)
            
            # Ограничиваем размер истории для экономии памяти
            if len(self.debug_data) > 5000:
                self.debug_data = self.debug_data[-3000:]  # Оставляем последние 3000
            
            # Проверяем критические проблемы каждые 10 шагов
            if step % 10 == 0:
                self._check_critical_issues(step, metrics, loss_components)
            
            # Проверяем, нужно ли отправить отчет
            if step - self.last_report_step >= self.report_interval:
                self.send_debug_report(step)
                self.last_report_step = step
                
        except Exception as e:
            print(f"⚠️ Ошибка сбора debug данных: {e}")
    
    def _check_critical_issues(self, step: int, metrics: Dict, loss_components: Dict):
        """
        Проверка критических проблем каждые 10 шагов с автоматическим перезапуском.
        Реализует рекомендации из технического задания.
        """
        try:
            critical_issues = []
            
            # 1. Проверка NaN в loss компонентах
            if loss_components:
                for name, value in loss_components.items():
                    if isinstance(value, (int, float)):
                        if np.isnan(value):
                            critical_issues.append(f"NaN в {name}")
                        elif np.isinf(value):
                            critical_issues.append(f"Inf в {name}")
            
            # 2. Проверка NaN в основных метриках
            nan_metrics = []
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and np.isnan(value):
                    nan_metrics.append(key)
            
            if nan_metrics:
                critical_issues.append(f"NaN в метриках: {', '.join(nan_metrics)}")
            
            # 3. Проверка экстремальных значений
            total_loss = metrics.get('loss', 0)
            if isinstance(total_loss, (int, float)):
                if total_loss > 1000:
                    critical_issues.append(f"Экстремально высокий loss: {total_loss:.2f}")
                elif total_loss < 0:
                    critical_issues.append(f"Отрицательный loss: {total_loss:.2f}")
            
            # 4. Проверка градиентов
            grad_norm = metrics.get('grad_norm', 0)
            if isinstance(grad_norm, (int, float)):
                if grad_norm > 1000:
                    critical_issues.append(f"Взрыв градиентов: {grad_norm:.2f}")
                elif grad_norm < 1e-8:
                    critical_issues.append(f"Исчезновение градиентов: {grad_norm:.2e}")
            
            # 5. Если есть критические проблемы, отправляем уведомление
            if critical_issues:
                self._handle_critical_issues(step, critical_issues)
                
        except Exception as e:
            print(f"⚠️ Ошибка проверки критических проблем: {e}")
    
    def _handle_critical_issues(self, step: int, issues: List[str]):
        """
        Обработка критических проблем с автоматическими действиями.
        """
        try:
            # Логируем проблемы
            issues_text = "; ".join(issues)
            print(f"🚨 КРИТИЧЕСКИЕ ПРОБЛЕМЫ на шаге {step}: {issues_text}")
            
            # Отправляем экстренное уведомление в Telegram
            if self.telegram_monitor:
                try:
                    self.telegram_monitor.send_critical_alert(
                        title="🚨 КРИТИЧЕСКИЕ ПРОБЛЕМЫ ОБУЧЕНИЯ",
                        message=f"Шаг {step}: {issues_text}",
                        severity="critical"
                    )
                except Exception as e:
                    print(f"⚠️ Не удалось отправить критическое уведомление: {e}")
            
            # Сохраняем в историю проблем
            self.warning_history.append({
                'step': step,
                'timestamp': time.time(),
                'issues': issues,
                'severity': 'critical'
            })
            
            # Проверяем, нужен ли автоматический перезапуск
            if self._should_trigger_restart(issues):
                self._trigger_emergency_restart(step, issues)
                
        except Exception as e:
            print(f"⚠️ Ошибка обработки критических проблем: {e}")
    
    def _should_trigger_restart(self, issues: List[str]) -> bool:
        """
        Определяет, нужен ли автоматический перезапуск на основе критических проблем.
        """
        # Автоматический перезапуск при NaN или Inf
        for issue in issues:
            if "NaN" in issue or "Inf" in issue:
                return True
            if "Взрыв градиентов" in issue:
                return True
        return False
    
    def _trigger_emergency_restart(self, step: int, issues: List[str]):
        """
        Запускает процедуру экстренного перезапуска.
        """
        try:
            print(f"🔄 ЭКСТРЕННЫЙ ПЕРЕЗАПУСК на шаге {step}")
            
            # Записываем информацию о перезапуске
            restart_info = {
                'step': step,
                'timestamp': time.time(),
                'reason': 'critical_issues',
                'issues': issues
            }
            
            self.restart_history.append(restart_info)
            
            # Отправляем уведомление о перезапуске
            if self.telegram_monitor:
                try:
                    self.telegram_monitor.send_restart_notification(
                        reason=f"Критические проблемы: {'; '.join(issues)}",
                        step=step
                    )
                except Exception as e:
                    print(f"⚠️ Не удалось отправить уведомление о перезапуске: {e}")
            
            # Сохраняем информацию о перезапуске в файл
            import json
            restart_file = f"emergency_restart_step_{step}.json"
            with open(restart_file, 'w') as f:
                json.dump(restart_info, f, indent=2)
            
            print(f"💾 Информация о перезапуске сохранена в {restart_file}")
            
        except Exception as e:
            print(f"⚠️ Ошибка запуска экстренного перезапуска: {e}")
    
    def _analyze_loss(self, loss_components: Dict, metrics: Dict) -> Dict:
        """🔥 Анализ loss компонентов"""
        analysis = {
            'components': {},
            'total_loss': metrics.get('loss', 0),
            'nan_detected': False,
            'inf_detected': False,
            'problematic_components': [],
            'loss_trend': 'stable',
            'loss_magnitude': 'normal'
        }
        
        if loss_components:
            for name, value in loss_components.items():
                if isinstance(value, (int, float)):
                    analysis['components'][name] = float(value)
                    
                    # Проверка на NaN/Inf
                    if np.isnan(value):
                        analysis['nan_detected'] = True
                        analysis['problematic_components'].append(f"{name}: NaN")
                    elif np.isinf(value):
                        analysis['inf_detected'] = True
                        analysis['problematic_components'].append(f"{name}: Inf")
                    elif abs(value) > 100:
                        analysis['problematic_components'].append(f"{name}: {value:.2f} (очень высокий)")
        
        # Анализ тренда loss
        self.loss_history.append(analysis['total_loss'])
        if len(self.loss_history) > 50:
            self.loss_history = self.loss_history[-50:]
            
        if len(self.loss_history) >= 10:
            recent_trend = np.polyfit(range(len(self.loss_history[-10:])), self.loss_history[-10:], 1)[0]
            if recent_trend > 0.001:
                analysis['loss_trend'] = 'increasing'
            elif recent_trend < -0.001:
                analysis['loss_trend'] = 'decreasing'
        
        # Определение масштаба loss
        if analysis['total_loss'] > 10:
            analysis['loss_magnitude'] = 'very_high'
        elif analysis['total_loss'] > 5:
            analysis['loss_magnitude'] = 'high'
        elif analysis['total_loss'] < 0.1:
            analysis['loss_magnitude'] = 'very_low'
        
        return analysis
    
    def _analyze_attention(self, y_pred) -> Dict:
        """🎯 Анализ attention матрицы"""
        analysis = {
            'diagonality_score': 0.0,
            'monotonicity_score': 0.0,
            'focus_score': 0.0,
            'entropy_score': 0.0,
            'attention_shape': None,
            'attention_problems': [],
            'alignment_quality': 'unknown'
        }
        
        try:
            if y_pred and len(y_pred) >= 4:
                alignments = y_pred[3] if len(y_pred) == 4 else y_pred[4]
                
                if alignments is not None and alignments.numel() > 0:
                    # Берем первый элемент батча
                    attention = alignments[0].detach().cpu().numpy()
                    analysis['attention_shape'] = list(attention.shape)
                    
                    # Диагональность
                    analysis['diagonality_score'] = self._calculate_diagonality(attention)
                    
                    # Монотонность
                    analysis['monotonicity_score'] = self._calculate_monotonicity(attention)
                    
                    # Фокусировка
                    analysis['focus_score'] = self._calculate_focus(attention)
                    
                    # Энтропия
                    analysis['entropy_score'] = self._calculate_entropy(attention)
                    
                    # Проблемы
                    if analysis['diagonality_score'] < 0.2:
                        analysis['attention_problems'].append("Критически низкая диагональность")
                    if analysis['monotonicity_score'] < 0.3:
                        analysis['attention_problems'].append("Плохая монотонность")
                    if analysis['focus_score'] < 0.4:
                        analysis['attention_problems'].append("Размытый фокус")
                    
                    # Общая оценка качества
                    avg_score = (analysis['diagonality_score'] + analysis['monotonicity_score'] + analysis['focus_score']) / 3
                    if avg_score > 0.7:
                        analysis['alignment_quality'] = 'excellent'
                    elif avg_score > 0.5:
                        analysis['alignment_quality'] = 'good'
                    elif avg_score > 0.3:
                        analysis['alignment_quality'] = 'poor'
                    else:
                        analysis['alignment_quality'] = 'critical'
                    
                    # Сохраняем в историю
                    self.attention_history.append({
                        'diagonality': analysis['diagonality_score'],
                        'monotonicity': analysis['monotonicity_score'],
                        'focus': analysis['focus_score']
                    })
                    
                    if len(self.attention_history) > 100:
                        self.attention_history = self.attention_history[-100:]
                        
        except Exception as e:
            analysis['attention_problems'].append(f"Ошибка анализа: {str(e)}")
        
        return analysis
    
    def _analyze_gradients(self, model) -> Dict:
        """📈 Анализ градиентов модели"""
        analysis = {
            'total_grad_norm': 0.0,
            'layer_grad_norms': {},
            'grad_problems': [],
            'grad_status': 'normal',
            'max_grad': 0.0,
            'min_grad': 0.0,
            'nan_gradients': False,
            'zero_gradients': 0
        }
        
        try:
            if model:
                total_norm = 0.0
                grad_values = []
                zero_count = 0
                
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2).item()
                        analysis['layer_grad_norms'][name] = param_norm
                        total_norm += param_norm ** 2
                        
                        grad_flat = param.grad.data.flatten()
                        grad_values.extend(grad_flat.cpu().numpy().tolist())
                        
                        # Проверка на NaN
                        if torch.isnan(param.grad.data).any():
                            analysis['nan_gradients'] = True
                            analysis['grad_problems'].append(f"NaN в {name}")
                        
                        # Подсчет нулевых градиентов
                        zero_count += (param.grad.data == 0).sum().item()
                
                analysis['total_grad_norm'] = total_norm ** 0.5
                analysis['zero_gradients'] = zero_count
                
                if grad_values:
                    analysis['max_grad'] = float(np.max(np.abs(grad_values)))
                    analysis['min_grad'] = float(np.min(np.abs(grad_values)))
                
                # Определение статуса градиентов
                if analysis['total_grad_norm'] > 10.0:
                    analysis['grad_status'] = 'explosion'
                    analysis['grad_problems'].append("Взрыв градиентов")
                elif analysis['total_grad_norm'] < 1e-6:
                    analysis['grad_status'] = 'vanishing'
                    analysis['grad_problems'].append("Затухание градиентов")
                elif analysis['nan_gradients']:
                    analysis['grad_status'] = 'nan'
                
                # Сохраняем в историю
                self.gradient_history.append(analysis['total_grad_norm'])
                if len(self.gradient_history) > 100:
                    self.gradient_history = self.gradient_history[-100:]
                    
        except Exception as e:
            analysis['grad_problems'].append(f"Ошибка анализа градиентов: {str(e)}")
        
        return analysis
    
    def _collect_hyperparameters(self, hparams) -> Dict:
        """⚙️ Сбор текущих гиперпараметров"""
        params = {}
        
        if hparams:
            key_params = [
                'learning_rate', 'batch_size', 'grad_clip_thresh',
                'use_guided_attn', 'guide_loss_weight', 'guide_loss_initial_weight',
                'p_attention_dropout', 'p_decoder_dropout', 'gate_threshold',
                'use_mmi', 'fp16_run', 'epochs'
            ]
            
            for param in key_params:
                if hasattr(hparams, param):
                    params[param] = getattr(hparams, param)
                    
        return params
    
    def _collect_system_info(self) -> Dict:
        """🖥️ Сбор системной информации"""
        info = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'gpu_info': {},
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
        
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    gpu_mem_used = torch.cuda.memory_allocated(i) / (1024**3)
                    info['gpu_info'][f'gpu_{i}'] = {
                        'total_memory_gb': gpu_mem,
                        'used_memory_gb': gpu_mem_used,
                        'memory_percent': (gpu_mem_used / gpu_mem) * 100,
                        'name': torch.cuda.get_device_name(i)
                    }
        except Exception as e:
            info['gpu_info']['error'] = str(e)
        
        return info
    
    def _detect_issues(self, metrics: Dict, loss_components: Dict, y_pred) -> List[str]:
        """⚠️ Обнаружение проблем"""
        issues = []
        
        # Проблемы с loss
        if metrics.get('loss', 0) != metrics.get('loss', 0):  # NaN check
            issues.append("🚨 КРИТИЧНО: Loss стал NaN")
        elif metrics.get('loss', 0) == float('inf'):
            issues.append("🚨 КРИТИЧНО: Loss стал бесконечным")
        elif metrics.get('loss', 0) > 50:
            issues.append("⚠️ ВЫСОКИЙ: Loss очень большой")
        
        # Проблемы с attention
        try:
            if y_pred and len(y_pred) >= 4:
                alignments = y_pred[3] if len(y_pred) == 4 else y_pred[4]
                if alignments is not None:
                    diag = self._calculate_diagonality(alignments[0].detach().cpu().numpy())
                    if diag < 0.1:
                        issues.append("🚨 КРИТИЧНО: Attention диагональность <10%")
                    elif diag < 0.3:
                        issues.append("⚠️ НИЗКАЯ: Attention диагональность <30%")
        except:
            issues.append("❓ НЕИЗВЕСТНО: Не удалось проанализировать attention")
        
        # Проблемы с памятью
        if psutil.virtual_memory().percent > 90:
            issues.append("🖥️ ПАМЯТЬ: Высокое использование ОЗУ")
        
        # Проблемы с GPU
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem_used = torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory
                    if mem_used > 0.95:
                        issues.append(f"🎮 GPU {i}: Критически мало памяти")
        except:
            pass
        
        return issues
    
    def _analyze_trends(self, step: int, metrics: Dict) -> Dict:
        """📊 Анализ трендов"""
        trends = {
            'loss_trend_last_100': 'stable',
            'attention_trend': 'stable',
            'gradient_trend': 'stable',
            'performance_degrading': False
        }
        
        try:
            # Тренд loss
            if len(self.loss_history) >= 20:
                recent_losses = self.loss_history[-20:]
                trend_coef = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
                if trend_coef > 0.01:
                    trends['loss_trend_last_100'] = 'worsening'
                elif trend_coef < -0.01:
                    trends['loss_trend_last_100'] = 'improving'
            
            # Тренд attention
            if len(self.attention_history) >= 10:
                recent_diag = [h['diagonality'] for h in self.attention_history[-10:]]
                trend_coef = np.polyfit(range(len(recent_diag)), recent_diag, 1)[0]
                if trend_coef > 0.01:
                    trends['attention_trend'] = 'improving'
                elif trend_coef < -0.01:
                    trends['attention_trend'] = 'degrading'
            
            # Проверка деградации производительности
            if (trends['loss_trend_last_100'] == 'worsening' and 
                trends['attention_trend'] == 'degrading'):
                trends['performance_degrading'] = True
                
        except Exception as e:
            trends['analysis_error'] = str(e)
        
        return trends
    
    def _calculate_diagonality(self, attention_matrix) -> float:
        """Быстрый расчет диагональности"""
        try:
            if attention_matrix.size == 0:
                return 0.0
            
            mel_len, text_len = attention_matrix.shape
            diagonal_sum = 0.0
            total_sum = attention_matrix.sum()
            
            if total_sum == 0:
                return 0.0
            
            for i in range(mel_len):
                diagonal_pos = int(i * text_len / mel_len)
                if diagonal_pos < text_len:
                    diagonal_sum += attention_matrix[i, diagonal_pos]
            
            return diagonal_sum / total_sum if total_sum > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_monotonicity(self, attention_matrix) -> float:
        """Расчет монотонности"""
        try:
            peaks = np.argmax(attention_matrix, axis=1)
            monotonic = sum(1 for i in range(1, len(peaks)) if peaks[i] >= peaks[i-1])
            return monotonic / max(1, len(peaks) - 1)
        except:
            return 0.0
    
    def _calculate_focus(self, attention_matrix) -> float:
        """Расчет фокусировки"""
        try:
            entropies = []
            for i in range(attention_matrix.shape[0]):
                attention_step = attention_matrix[i] + 1e-8
                # 🔥 ИСПРАВЛЕНИЕ: Проверяем на нули и нормализуем
                attention_step = attention_step / (attention_step.sum() + 1e-8)
                # Маскируем нули для избежания log(0)
                mask = attention_step > 1e-8
                if mask.any():
                    entropy = -np.sum(attention_step[mask] * np.log(attention_step[mask]))
                else:
                    entropy = 0.0
                entropies.append(entropy)
            
            max_entropy = np.log(attention_matrix.shape[1])
            avg_entropy = np.mean(entropies) if entropies else 0.0
            return 1.0 - (avg_entropy / max_entropy) if max_entropy > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_entropy(self, attention_matrix) -> float:
        """Расчет энтропии"""
        try:
            # 🔥 ИСПРАВЛЕНИЕ: Нормализуем матрицу и избегаем log(0)
            attention_matrix = attention_matrix + 1e-8
            attention_matrix = attention_matrix / (attention_matrix.sum() + 1e-8)
            
            # Маскируем очень маленькие значения
            mask = attention_matrix > 1e-8
            if mask.any():
                entropy = -np.sum(attention_matrix[mask] * np.log(attention_matrix[mask]))
            else:
                entropy = 0.0
                
            max_entropy = np.log(attention_matrix.size)
            return entropy / max_entropy if max_entropy > 0 else 0.0
        except:
            return 0.0
    
    def send_debug_report(self, step: int):
        """📱 Отправка debug отчета в Telegram"""
        try:
            if not self.telegram_monitor:
                return
            
            # Создаем debug файл
            debug_filename = f"debug_step_{step}.txt"
            debug_content = self._generate_debug_content(step)
            
            # Сохраняем во временный файл
            with open(debug_filename, 'w', encoding='utf-8') as f:
                f.write(debug_content)
            
            # Отправляем файл
            caption = f"🔍 **Debug Report - Шаг {step}**\n"
            caption += f"📊 **Анализ последних {len(self.debug_data)} шагов**\n"
            caption += f"⏰ **Время обучения:** {(time.time() - self.start_time) / 3600:.1f}ч"
            
            success = self._send_file_to_telegram(debug_filename, caption)
            
            if success:
                print(f"✅ Debug отчет отправлен: {debug_filename}")
            else:
                print(f"❌ Ошибка отправки debug отчета")
            
            # Удаляем временный файл
            try:
                os.remove(debug_filename)
            except:
                pass
                
        except Exception as e:
            print(f"⚠️ Ошибка создания debug отчета: {e}")
    
    def _generate_debug_content(self, step: int) -> str:
        """📝 Генерация содержимого debug файла"""
        content = f"""🔍 TECHNICAL DEBUG REPORT - Шаг {step}
{'='*80}
📅 Время создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
⏰ Время обучения: {(time.time() - self.start_time) / 3600:.2f} часов
📊 Всего данных: {len(self.debug_data)} шагов

{'='*80}
🔥 SUMMARY - КРИТИЧЕСКИЕ ПРОБЛЕМЫ
{'='*80}
"""
        
        # Анализируем последние данные для summary
        if self.debug_data:
            latest = self.debug_data[-1]
            
            content += f"🎯 Текущий шаг: {latest['step']}\n"
            content += f"📉 Loss: {latest['loss_analysis']['total_loss']:.6f}\n"
            content += f"🎯 Attention диагональность: {latest['attention_analysis']['diagonality_score']:.1%}\n"
            content += f"📈 Градиент норма: {latest['gradient_analysis']['total_grad_norm']:.6f}\n"
            content += f"💾 Использование памяти: {latest['system_info']['memory_percent']:.1f}%\n"
            
            # Критические проблемы
            if latest['issues_detected']:
                content += f"\n🚨 КРИТИЧЕСКИЕ ПРОБЛЕМЫ:\n"
                for issue in latest['issues_detected']:
                    content += f"  • {issue}\n"
            
            # Тренды
            trends = latest['trends']
            content += f"\n📊 ТРЕНДЫ:\n"
            content += f"  • Loss: {trends['loss_trend_last_100']}\n"
            content += f"  • Attention: {trends['attention_trend']}\n"
            content += f"  • Деградация: {'ДА' if trends['performance_degrading'] else 'НЕТ'}\n"
        
        content += f"\n{'='*80}\n"
        content += f"📊 ДЕТАЛЬНЫЕ ДАННЫЕ\n"
        content += f"{'='*80}\n"
        
        # Добавляем последние 50 шагов подробно
        recent_data = self.debug_data[-50:] if len(self.debug_data) > 50 else self.debug_data
        
        for i, data in enumerate(recent_data):
            content += f"\n--- ШАГ {data['step']} ---\n"
            content += f"Время: {data['timestamp']}\n"
            
            # Loss анализ
            loss_info = data['loss_analysis']
            content += f"Loss: {loss_info['total_loss']:.6f} ({loss_info['loss_trend']}, {loss_info['loss_magnitude']})\n"
            if loss_info['problematic_components']:
                content += f"Проблемы с loss: {', '.join(loss_info['problematic_components'])}\n"
            
            # Attention анализ
            att_info = data['attention_analysis']
            content += f"Attention: диаг={att_info['diagonality_score']:.3f}, монот={att_info['monotonicity_score']:.3f}, фокус={att_info['focus_score']:.3f}\n"
            if att_info['attention_problems']:
                content += f"Проблемы attention: {', '.join(att_info['attention_problems'])}\n"
            
            # Градиенты
            grad_info = data['gradient_analysis']
            content += f"Градиенты: норма={grad_info['total_grad_norm']:.6f}, статус={grad_info['grad_status']}\n"
            
            # Smart Tuner решения
            if data['smart_tuner_decisions']:
                content += f"Smart Tuner: {json.dumps(data['smart_tuner_decisions'], ensure_ascii=False, indent=2)}\n"
            
            # Системная информация
            sys_info = data['system_info']
            content += f"Система: CPU={sys_info['cpu_percent']:.1f}%, RAM={sys_info['memory_percent']:.1f}%\n"
            
            # Проблемы
            if data['issues_detected']:
                content += f"ПРОБЛЕМЫ: {', '.join(data['issues_detected'])}\n"
            
            content += "\n"
        
        # История перезапусков
        if self.restart_history:
            content += f"\n{'='*80}\n"
            content += f"🔄 ИСТОРИЯ ПЕРЕЗАПУСКОВ\n"
            content += f"{'='*80}\n"
            for restart in self.restart_history:
                content += f"{restart}\n"
        
        content += f"\n{'='*80}\n"
        content += f"📈 СТАТИСТИКА И ТРЕНДЫ\n"
        content += f"{'='*80}\n"
        
        # Статистика loss
        if self.loss_history:
            content += f"Loss история ({len(self.loss_history)} точек):\n"
            content += f"  Мин: {min(self.loss_history):.6f}\n"
            content += f"  Макс: {max(self.loss_history):.6f}\n"
            content += f"  Текущий: {self.loss_history[-1]:.6f}\n"
            content += f"  Среднее (последние 20): {np.mean(self.loss_history[-20:]):.6f}\n"
        
        # Статистика attention
        if self.attention_history:
            recent_diag = [h['diagonality'] for h in self.attention_history[-20:]]
            content += f"Attention диагональность (последние 20):\n"
            content += f"  Мин: {min(recent_diag):.3f}\n"
            content += f"  Макс: {max(recent_diag):.3f}\n"
            content += f"  Среднее: {np.mean(recent_diag):.3f}\n"
        
        content += f"\n{'='*80}\n"
        content += f"🎯 КОНЕЦ ОТЧЕТА\n"
        content += f"{'='*80}\n"
        
        return content
    
    def _send_file_to_telegram(self, filename: str, caption: str) -> bool:
        """📱 Отправка файла в Telegram"""
        try:
            if hasattr(self.telegram_monitor, '_send_document'):
                return self.telegram_monitor._send_document(filename, caption)
            elif hasattr(self.telegram_monitor, 'bot_token') and hasattr(self.telegram_monitor, 'chat_id'):
                # Прямая отправка через API
                import requests
                
                url = f"https://api.telegram.org/bot{self.telegram_monitor.bot_token}/sendDocument"
                
                with open(filename, 'rb') as f:
                    files = {'document': f}
                    data = {
                        'chat_id': self.telegram_monitor.chat_id,
                        'caption': caption,
                        'parse_mode': 'Markdown'
                    }
                    
                    response = requests.post(url, files=files, data=data, timeout=30)
                    response.raise_for_status()
                    return True
            else:
                print("⚠️ Telegram monitor не поддерживает отправку файлов")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка отправки файла в Telegram: {e}")
            return False
    
    def add_restart_info(self, restart_info: str):
        """🔄 Добавление информации о перезапуске"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.restart_history.append(f"[{timestamp}] {restart_info}")
    
    def add_warning(self, warning: str):
        """⚠️ Добавление предупреждения"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.warning_history.append(f"[{timestamp}] {warning}")


# Глобальный экземпляр для использования в train.py
debug_reporter = None

def initialize_debug_reporter(telegram_monitor=None):
    """Инициализация глобального debug reporter"""
    global debug_reporter
    debug_reporter = DebugReporter(telegram_monitor)
    return debug_reporter

def get_debug_reporter():
    """Получение глобального debug reporter"""
    return debug_reporter
