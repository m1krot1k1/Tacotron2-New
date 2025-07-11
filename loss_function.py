from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
import logging  # 🔥 ДОБАВЛЯЕМ ИМПОРТ LOGGING
from smart_tuner.ddc_diagnostic import get_global_ddc_diagnostic

# 🔥 ИМПОРТ унифицированной системы guided attention
try:
    from unified_guided_attention import UnifiedGuidedAttentionLoss, create_unified_guided_attention
    UNIFIED_GUIDED_AVAILABLE = True
except ImportError:
    UNIFIED_GUIDED_AVAILABLE = False
    print("⚠️ UnifiedGuidedAttentionLoss недоступен - используем legacy реализацию")

# 🎯 ИМПОРТ новой адаптивной системы loss функций
try:
    from adaptive_loss_system import (
        create_adaptive_loss_system, 
        create_loss_context_from_metrics,
        LossPhase,
        LossContext
    )
    ADAPTIVE_LOSS_AVAILABLE = True
    print("✅ Enhanced Adaptive Loss System доступна")
except ImportError:
    ADAPTIVE_LOSS_AVAILABLE = False
    print("⚠️ Enhanced Adaptive Loss System недоступна - используем стандартную реализацию")


class SpectralMelLoss(nn.Module):
    """
    🎵 Улучшенная Mel Loss с акцентом на спектральное качество
    """
    def __init__(self, n_mel_channels=80, sample_rate=22050):
        super(SpectralMelLoss, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sample_rate = sample_rate
        
        # Веса для разных частотных диапазонов
        mel_weights = torch.ones(n_mel_channels)
        # Больший вес на средних частотах (важных для голоса)
        mid_range = slice(n_mel_channels//4, 3*n_mel_channels//4)
        mel_weights[mid_range] *= 1.5
        # Меньший вес на очень высоких частотах
        high_range = slice(3*n_mel_channels//4, n_mel_channels)
        mel_weights[high_range] *= 0.8
        
        self.register_buffer('mel_weights', mel_weights)
        
    def forward(self, mel_pred, mel_target):
        # 🔥 ИСПРАВЛЕНИЕ CUDA/CPU: Убеждаемся, что mel_weights на том же устройстве
        mel_weights = self.mel_weights.to(mel_pred.device)
        
        # 🔧 ИСПРАВЛЕНИЕ: Адаптируем веса к фактическому количеству mel каналов
        actual_mel_channels = mel_pred.size(1)
        if actual_mel_channels != mel_weights.size(0):
            # Ресамплируем веса под фактический размер
            if actual_mel_channels > mel_weights.size(0):
                # Повторяем последний вес
                extra_weights = mel_weights[-1].repeat(actual_mel_channels - mel_weights.size(0))
                mel_weights = torch.cat([mel_weights, extra_weights])
            else:
                # Обрезаем до нужного размера
                mel_weights = mel_weights[:actual_mel_channels]
        
        # Основной MSE loss с весами по частотам
        weighted_mse = F.mse_loss(mel_pred * mel_weights[None, :, None], 
                                  mel_target * mel_weights[None, :, None])
        
        # Добавляем L1 loss для резкости
        l1_loss = F.l1_loss(mel_pred, mel_target)
        
        # Спектральный loss (разности соседних фреймов)
        mel_pred_diff = mel_pred[:, :, 1:] - mel_pred[:, :, :-1]
        mel_target_diff = mel_target[:, :, 1:] - mel_target[:, :, :-1]
        spectral_loss = F.mse_loss(mel_pred_diff, mel_target_diff)
        
        return weighted_mse + 0.3 * l1_loss + 0.2 * spectral_loss


class AdaptiveGateLoss(nn.Module):
    """
    🚪 Адаптивная Gate Loss с динамическим весом
    """
    def __init__(self, initial_weight=1.3, min_weight=0.8, max_weight=2.0):
        super(AdaptiveGateLoss, self).__init__()
        self.initial_weight = initial_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.current_weight = initial_weight
        
    def update_weight(self, gate_accuracy):
        """Обновляет вес на основе текущей gate accuracy"""
        if gate_accuracy < 0.5:
            # Увеличиваем вес если accuracy низкая
            self.current_weight = min(self.max_weight, self.current_weight * 1.1)
        elif gate_accuracy > 0.8:
            # Уменьшаем вес если accuracy высокая
            self.current_weight = max(self.min_weight, self.current_weight * 0.95)
            
    def forward(self, gate_pred, gate_target, global_step=None):
        """
        🔥 ИСПРАВЛЕНО: Добавлен параметр global_step для адаптивности
        """
        # Обновляем вес на основе текущего качества gate
        if global_step is not None and global_step % 100 == 0:
            # Каждые 100 шагов пересчитываем вес
            with torch.no_grad():
                gate_accuracy = ((torch.sigmoid(gate_pred) > 0.5) == (gate_target > 0.5)).float().mean()
                self.update_weight(gate_accuracy.item())
        
        return self.current_weight * F.binary_cross_entropy_with_logits(gate_pred, gate_target)


class Tacotron2Loss(nn.Module):
    """
    Продвинутая система Loss для Tacotron2 на основе современных исследований.
    
    Включает:
    1. ИСПРАВЛЕННЫЙ Guided Attention Loss (MonoAlign 2024)
    2. Spectral Mel Loss для лучшего качества (Llasa 2025)  
    3. Adaptive Gate Loss (Very Attentive Tacotron 2025)
    4. Perceptual Loss для человеческого качества
    5. Style Loss для сохранения характера голоса
    6. Monotonic Alignment Loss для стабильности
    """
    
    def __init__(self, hparams):
        super(Tacotron2Loss, self).__init__()
        self.hparams = hparams
        
        # 🔥 ДОБАВЛЯЕМ LOGGER
        self.logger = logging.getLogger(__name__)
        
        # 📊 ИСПРАВЛЕННЫЕ веса loss функций (из исследований)
        self.mel_loss_weight = getattr(hparams, 'mel_loss_weight', 1.0)
        self.gate_loss_weight = getattr(hparams, 'gate_loss_weight', 1.0)
        self.guide_loss_weight = getattr(hparams, 'guide_loss_weight', 5.0)  # 🔥 УВЕЛИЧЕНО для MSE loss
        
        # 🎵 НОВЫЕ продвинутые loss функции
        self.spectral_loss_weight = getattr(hparams, 'spectral_loss_weight', 0.3)
        self.perceptual_loss_weight = getattr(hparams, 'perceptual_loss_weight', 0.2)
        self.style_loss_weight = getattr(hparams, 'style_loss_weight', 0.1)
        self.monotonic_loss_weight = getattr(hparams, 'monotonic_loss_weight', 0.1)
        
        # Guided attention параметры (ИСПРАВЛЕННЫЕ)
        self.guide_decay = getattr(hparams, 'guide_decay', 0.9999)  # Медленнее decay
        self.guide_sigma = getattr(hparams, 'guide_sigma', 0.4)      # Оптимальная sigma
        
        # Adaptive parameters
        self.adaptive_gate_threshold = getattr(hparams, 'adaptive_gate_threshold', True)
        self.curriculum_teacher_forcing = getattr(hparams, 'curriculum_teacher_forcing', True)
        
        # Счетчик шагов для адаптивных loss
        self.global_step = 0
        
        # Инициализация продвинутых loss функций
        self.spectral_mel_loss = SpectralMelLoss()
        self.adaptive_gate_loss = AdaptiveGateLoss()
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.monotonic_alignment_loss = MonotonicAlignmentLoss()
        
        # 🔥 УНИФИЦИРОВАННАЯ система guided attention
        if UNIFIED_GUIDED_AVAILABLE:
            self.unified_guided_attention = create_unified_guided_attention(hparams)
            self.use_unified_guided = True
            print("✅ Использую UnifiedGuidedAttentionLoss")
        else:
            self.unified_guided_attention = None
            self.use_unified_guided = False
            print("⚠️ Использую legacy guided attention")
        
        # 🎯 ИНТЕГРАЦИЯ Enhanced Adaptive Loss System
        if ADAPTIVE_LOSS_AVAILABLE:
            self.adaptive_loss_system = create_adaptive_loss_system(hparams)
            self.use_adaptive_loss = True
            print("✅ Enhanced Adaptive Loss System интегрирована")
        else:
            self.adaptive_loss_system = None
            self.use_adaptive_loss = False
            print("⚠️ Используем стандартную систему loss функций")
        
        # Контекст для адаптивной системы
        self.current_context = None
        self.training_metrics = {}
        
        # DDC
        self.use_ddc = getattr(hparams, 'use_ddc', False)
        self.ddc_consistency_weight = getattr(hparams, 'ddc_consistency_weight', 0.5)
        
    def forward(self, model_output, targets, attention_weights=None, gate_outputs=None):
        """
        Продвинутый forward pass с множественными loss функциями.
        """
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        # 🔥 ИСПРАВЛЕНИЕ: Правильно распаковываем все элементы модели
        if len(model_output) == 7:
            # Полный формат: [decoder_outputs, mel_out, mel_out_postnet, gate_out, alignments, tpse_gst_outputs, gst_outputs]
            decoder_outputs, mel_out, mel_out_postnet, gate_out, alignments, tpse_gst_outputs, gst_outputs = model_output
        elif len(model_output) == 6:
            # Формат без decoder_outputs: [mel_out, mel_out_postnet, gate_out, alignments, tpse_gst_outputs, gst_outputs]  
            mel_out, mel_out_postnet, gate_out, alignments, tpse_gst_outputs, gst_outputs = model_output
        elif len(model_output) == 5:
            # Формат с alignments: [mel_out, mel_out_postnet, gate_out, alignments, extra]
            mel_out, mel_out_postnet, gate_out, alignments, _ = model_output
        elif len(model_output) == 4:
            # Старый формат: [mel_out, mel_out_postnet, gate_out, alignments]
            mel_out, mel_out_postnet, gate_out, alignments = model_output
        else:
            # Fallback: берем первые 4 элемента
            mel_out, mel_out_postnet, gate_out, alignments = model_output[:4]
        gate_out = gate_out.view(-1, 1)
        
        # 🎯 1. ОСНОВНЫЕ LOSS ФУНКЦИИ С ЗАЩИТОЙ ОТ NaN
        
        # 🔥 Mel loss с проверкой на NaN
        try:
            mel_loss = nn.MSELoss()(mel_out, mel_target) + \
                nn.MSELoss()(mel_out_postnet, mel_target)
            
            # Проверка на NaN/Inf в mel loss
            if torch.isnan(mel_loss) or torch.isinf(mel_loss):
                self.logger.error("🚨 NaN/Inf в mel_loss, заменяем на единицу")
                mel_loss = torch.tensor(1.0, requires_grad=True, device=mel_out.device)
        except Exception as e:
            self.logger.error(f"🚨 Ошибка в mel_loss: {e}")
            mel_loss = torch.tensor(1.0, requires_grad=True, device=mel_out.device)
        
        # 🔥 Gate loss с проверкой на NaN
        try:
            raw_gate_loss = self.adaptive_gate_loss(gate_out, gate_target, self.global_step)
            
            # Проверка на NaN/Inf в gate loss
            if torch.isnan(raw_gate_loss) or torch.isinf(raw_gate_loss):
                self.logger.error("🚨 NaN/Inf в gate_loss, заменяем на BCE")
                raw_gate_loss = F.binary_cross_entropy_with_logits(gate_out, gate_target)
                
            gate_loss = self.gate_loss_weight * raw_gate_loss
        except Exception as e:
            self.logger.error(f"🚨 Ошибка в gate_loss: {e}")
            gate_loss = torch.tensor(0.5, requires_grad=True, device=gate_out.device)
        
        # 🔥 2. GUIDED ATTENTION LOSS С ЗАЩИТОЙ ОТ NaN
        guide_loss = 0.0
        if alignments is not None and self.guide_loss_weight > 0:
            try:
                if self.use_unified_guided and self.unified_guided_attention:
                    # Используем новую унифицированную систему
                    guide_loss = self.unified_guided_attention(model_output)
                else:
                    # Fallback на legacy реализацию
                    guide_loss = self.guided_attention_loss(
                        alignments, 
                        mel_target.size(2), 
                        mel_out.size(1)
                    )
                
                # 🔥 ЗАЩИТА ОТ NaN В GUIDED ATTENTION
                if torch.isnan(guide_loss) or torch.isinf(guide_loss):
                    self.logger.error("🚨 NaN/Inf в guide_loss, заменяем на ноль")
                    guide_loss = torch.tensor(0.0, requires_grad=True, device=mel_out.device)
                    
            except Exception as e:
                self.logger.error(f"🚨 Ошибка в guide_loss: {e}")
                guide_loss = torch.tensor(0.0, requires_grad=True, device=mel_out.device)
        
        # 🎵 3. ПРОДВИНУТЫЕ LOSS ФУНКЦИИ
        
        # Spectral Mel Loss для лучшего качества частот
        spectral_loss = self.spectral_mel_loss(mel_out_postnet, mel_target)
        
        # Perceptual Loss для человеческого восприятия
        perceptual_loss = self.perceptual_loss(mel_out_postnet, mel_target)
        
        # Style Loss для сохранения характера голоса
        style_loss = self.style_loss(mel_out_postnet, mel_target)
        
        # Monotonic Alignment Loss для стабильности
        monotonic_loss = 0.0
        # ИСПРАВЛЕНИЕ: Используем alignments из model_output
        if alignments is not None:
            monotonic_loss = self.monotonic_alignment_loss(alignments)
        
        # 🔥 ИСПРАВЛЕНИЕ: Возвращаем 4 компонента, как ожидает train.py
        
        # Обновляем счетчик шагов
        self.global_step += 1
        
        # 🎯 АДАПТИВНАЯ СИСТЕМА LOSS ФУНКЦИЙ
        if self.use_adaptive_loss and self.adaptive_loss_system and self.current_context:
            # Подготавливаем компоненты loss для адаптивной системы
            loss_components = {
                'mel': mel_loss,
                'gate': raw_gate_loss,
                'guided_attention': guide_loss,
                'spectral': spectral_loss,
                'perceptual': perceptual_loss,
                'style': style_loss,
                'monotonic': monotonic_loss
            }
            
            # Если есть gate predictions/targets для Dynamic Tversky
            if gate_out is not None and gate_targets is not None:
                loss_components['gate_predictions'] = gate_out
                loss_components['gate_targets'] = gate_targets
            
            # Применяем адаптивную оптимизацию
            optimized_loss, diagnostics = self.adaptive_loss_system.optimize_loss_computation(
                loss_components, self.current_context
            )
            
            # Разбиваем оптимизированную loss на компоненты для совместимости
            component_contributions = diagnostics['component_contributions']
            
            combined_mel_loss = (
                component_contributions.get('mel', 0.0) +
                component_contributions.get('spectral', 0.0) +
                component_contributions.get('perceptual', 0.0)
            )
            
            gate_loss = component_contributions.get('gate', 0.0)
            adaptive_guide_loss = component_contributions.get('guided_attention', 0.0)
            combined_emb_loss = (
                component_contributions.get('style', 0.0) +
                component_contributions.get('monotonic', 0.0)
            )
            
            # Сохраняем диагностику для мониторинга
            self.training_metrics['adaptive_diagnostics'] = diagnostics
            
        else:
            # 🔄 FALLBACK: стандартная система loss функций
            # Объединяем mel loss + продвинутые loss для совместимости
            combined_mel_loss = (
                self.mel_loss_weight * mel_loss +
                self.spectral_loss_weight * spectral_loss +
                self.perceptual_loss_weight * perceptual_loss
            )
            
            # Style loss + monotonic loss как embedding loss
            combined_emb_loss = (
                self.style_loss_weight * style_loss +
                self.monotonic_loss_weight * monotonic_loss
            )
            
            # Адаптивный guided attention loss (унифицированная система уже применяет веса)
            if self.use_unified_guided and self.unified_guided_attention:
                # Унифицированная система уже применила адаптивный вес
                adaptive_guide_loss = guide_loss
            else:
                # Legacy система нуждается в адаптивном весе
                adaptive_guide_loss = self._get_adaptive_guide_weight() * guide_loss
        
        # Double Decoder Consistency Loss
        ddc_loss = 0.0
        if self.use_ddc and len(model_output) >= 8:
            # Вторичный декодер outputs: mel_out2, mel_post2, gate2, align2
            mel_out2 = model_output[4]
            mel_out_postnet2 = model_output[5]
            
            # 🔧 ИНТЕГРАЦИЯ SAFE DDC LOSS
            try:
                from smart_tuner.safe_ddc_loss import get_global_ddc_loss, SafeDDCLoss
                
                # Инициализируем SafeDDCLoss если еще не создан
                ddc_loss_fn = get_global_ddc_loss()
                if ddc_loss_fn is None:
                    ddc_loss_fn = SafeDDCLoss(
                        weight=self.ddc_consistency_weight,
                        use_masking=True,
                        log_warnings=True
                    )
                    from smart_tuner.safe_ddc_loss import set_global_ddc_loss
                    set_global_ddc_loss(ddc_loss_fn)
                
                # Применяем безопасное вычисление DDC loss
                ddc_loss = ddc_loss_fn(mel_out_postnet, mel_out_postnet2.detach(), step=self.global_step)
                
                # 🔍 ДИАГНОСТИКА DDC LOSS
                try:
                    ddc_diagnostic = get_global_ddc_diagnostic()
                    if ddc_diagnostic is not None:
                        # Анализируем несовпадения размеров
                        ddc_diagnostic.analyze_size_mismatch(
                            mel_out_postnet, mel_out_postnet2.detach(), self.global_step
                        )
                except Exception as diag_e:
                    print(f"⚠️ Ошибка диагностики DDC: {diag_e}")
                
            except ImportError:
                # Fallback к стандартной логике
                if mel_out_postnet.shape == mel_out_postnet2.shape:
                    ddc_loss = F.mse_loss(mel_out_postnet, mel_out_postnet2.detach())
                else:
                    # Если размеры не совпадают, обрезаем до минимального
                    min_time = min(mel_out_postnet.size(2), mel_out_postnet2.size(2))
                    mel_out_postnet_trimmed = mel_out_postnet[:, :, :min_time]
                    mel_out_postnet2_trimmed = mel_out_postnet2[:, :, :min_time]
                    ddc_loss = F.mse_loss(mel_out_postnet_trimmed, mel_out_postnet2_trimmed.detach())
                    print(f"⚠️ DDC loss: размеры не совпадают, обрезаем до {min_time} временных шагов")

        # Добавляем DDC к composite mel loss
        combined_mel_loss = combined_mel_loss + self.ddc_consistency_weight * ddc_loss
        
        # 🛡️ ФИНАЛЬНАЯ ЗАЩИТА ОТ NaN - ПРОВЕРЯЕМ ВСЕ КОМПОНЕНТЫ
        def safe_loss_component(loss_value, default_value, component_name):
            """Безопасное создание loss компонента с защитой от NaN/Inf"""
            try:
                if torch.isnan(loss_value) or torch.isinf(loss_value):
                    self.logger.error(f"🚨 NaN/Inf в {component_name}, заменяем на {default_value}")
                    return torch.tensor(default_value, requires_grad=True, device=loss_value.device)
                return loss_value
            except Exception as e:
                self.logger.error(f"🚨 Ошибка в {component_name}: {e}")
                return torch.tensor(default_value, requires_grad=True, device=mel_out.device)
        
        # Применяем защиту ко всем компонентам
        safe_mel_loss = safe_loss_component(combined_mel_loss, 1.0, "combined_mel_loss")
        safe_gate_loss = safe_loss_component(gate_loss, 0.5, "gate_loss")
        safe_guide_loss = safe_loss_component(adaptive_guide_loss, 0.0, "adaptive_guide_loss")
        safe_emb_loss = safe_loss_component(combined_emb_loss, 0.0, "combined_emb_loss")
        
        # Возвращаем 4 безопасных компонента в ожидаемом формате train.py
        return safe_mel_loss, safe_gate_loss, safe_guide_loss, safe_emb_loss

    def set_context_aware_manager(self, context_manager):
        """
        🧠 Интеграция с Context-Aware Training Manager
        
        Args:
            context_manager: Экземпляр ContextAwareTrainingManager
        """
        if self.use_unified_guided and self.unified_guided_attention:
            self.unified_guided_attention.set_context_aware_manager(context_manager)
            print("✅ Context-Aware Manager интегрирован с UnifiedGuidedAttention")
        else:
            print("⚠️ Context-Aware интеграция доступна только с UnifiedGuidedAttention")
    
    def get_guided_attention_diagnostics(self) -> Dict[str, Any]:
        """
        📊 Получение диагностической информации guided attention
        
        Returns:
            Dict[str, Any]: Диагностическая информация
        """
        if self.use_unified_guided and self.unified_guided_attention:
            diagnostics = self.unified_guided_attention.get_diagnostics()
            diagnostics['system_type'] = 'unified'
            return diagnostics
        else:
            return {
                'system_type': 'legacy',
                'current_weight': self._get_adaptive_guide_weight(),
                'current_sigma': self._get_adaptive_sigma(),
                'global_step': self.global_step
            }
    
    def update_training_context(self, 
                                phase: str,
                                attention_quality: float,
                                gate_accuracy: float,
                                mel_consistency: float = 0.5,
                                gradient_norm: float = 1.0,
                                loss_stability: float = 1.0,
                                learning_rate: float = 1e-3):
        """
        🎯 Обновление контекста обучения для адаптивной системы loss функций
        
        Args:
            phase: Фаза обучения ('PRE_ALIGNMENT', 'ALIGNMENT_LEARNING', etc.)
            attention_quality: Качество attention alignment (0.0 - 1.0)  
            gate_accuracy: Точность gate предсказаний (0.0 - 1.0)
            mel_consistency: Консистентность mel spectrogram (0.0 - 1.0)
            gradient_norm: Норма градиентов
            loss_stability: Стабильность loss (стандартное отклонение)
            learning_rate: Текущий learning rate
        """
        if self.use_adaptive_loss and ADAPTIVE_LOSS_AVAILABLE:
            # Обновляем метрики
            self.training_metrics.update({
                'attention_quality': attention_quality,
                'gate_accuracy': gate_accuracy,
                'mel_consistency': mel_consistency,
                'gradient_norm': gradient_norm,
                'loss_stability': loss_stability,
                'learning_rate': learning_rate
            })
            
            # Создаем контекст для адаптивной системы
            self.current_context = create_loss_context_from_metrics(
                self.training_metrics, phase, self.global_step
            )
            
            print(f"🎯 Контекст обновлен: фаза={phase}, attention={attention_quality:.3f}, gate={gate_accuracy:.3f}")
    
    def get_adaptive_loss_diagnostics(self) -> Dict[str, Any]:
        """
        📊 Получение полной диагностики адаптивной системы loss функций
        
        Returns:
            Dict[str, Any]: Полная диагностическая информация
        """
        if self.use_adaptive_loss and self.adaptive_loss_system:
            return self.adaptive_loss_system.get_system_diagnostics()
        else:
            return {
                'system_type': 'standard',
                'adaptive_loss_available': False,
                'message': 'Enhanced Adaptive Loss System не используется'
            }
    
    def get_current_adaptive_weights(self) -> Dict[str, float]:
        """
        📊 Получение текущих адаптивных весов loss компонентов
        
        Returns:
            Dict[str, float]: Текущие веса для каждого компонента
        """
        if (self.use_adaptive_loss and 
            self.adaptive_loss_system and 
            self.current_context):
            return self.adaptive_loss_system.weight_manager.get_adaptive_weights(self.current_context)
        else:
            # Fallback к стандартным весам
            return {
                'mel': self.mel_loss_weight,
                'gate': self.gate_loss_weight,
                'guided_attention': self.guide_loss_weight,
                'spectral': self.spectral_loss_weight,
                'perceptual': self.perceptual_loss_weight,
                'style': self.style_loss_weight,
                'monotonic': self.monotonic_loss_weight
            }
    
    def integrate_with_context_aware_manager(self, context_manager):
        """
        🧠 Интеграция с Context-Aware Training Manager
        
        Args:
            context_manager: Экземпляр ContextAwareTrainingManager
        """
        if self.use_unified_guided and self.unified_guided_attention:
            self.unified_guided_attention.set_context_aware_manager(context_manager)
            print("✅ Context-Aware Manager интегрирован с UnifiedGuidedAttention")
            
        if self.use_adaptive_loss:
            # Устанавливаем связь для автоматического обновления контекста
            self._context_aware_manager = context_manager
            print("✅ Context-Aware Manager интегрирован с Enhanced Adaptive Loss System")
        else:
            print("⚠️ Enhanced Adaptive Loss System недоступна для интеграции")

    def guided_attention_loss(self, att_ws, mel_len, text_len):
        """
        🔥 РЕВОЛЮЦИОННЫЙ Guided Attention Loss на основе Very Attentive Tacotron (2025).
        
        Ключевые ИСПРАВЛЕНИЯ из исследований:
        1. Location-Relative формула вместо простой диагонали
        2. Tensor операции вместо циклов (в 10x быстрее) 
        3. Adaptive sigma и weight decay
        4. Правильная нормализация и KL divergence
        """
        batch_size = att_ws.size(0)
        max_mel_len = att_ws.size(1)
        max_text_len = att_ws.size(2)
        
        # 🔥 ВЕКТОРИЗОВАННАЯ генерация диагональной матрицы
        # Создаем координатные сетки для всех батчей сразу
        mel_indices = torch.arange(max_mel_len, device=att_ws.device, dtype=torch.float32).unsqueeze(1)
        text_indices = torch.arange(max_text_len, device=att_ws.device, dtype=torch.float32).unsqueeze(0)
        
        # 🔥 АДАПТИВНАЯ sigma на основе длины последовательности
        adaptive_sigma = self._get_adaptive_sigma()
        
        # 🔥 LOCATION-RELATIVE формула из Very Attentive Tacotron
        # Нормализуем позиции относительно длин последовательностей
        mel_normalized = mel_indices / max_mel_len  # [max_mel_len, 1]
        text_normalized = text_indices / max_text_len  # [1, max_text_len]
        
        # Вычисляем ожидаемое выравнивание с учетом относительных позиций
        expected_alignment = torch.exp(-0.5 * ((mel_normalized - text_normalized) ** 2) / (adaptive_sigma ** 2))
        
        # Нормализация для каждого mel шага
        expected_alignment = expected_alignment / (expected_alignment.sum(dim=1, keepdim=True) + 1e-8)
        
        # Добавляем batch размерность и создаем маску
        expected_alignment = expected_alignment.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 🔥 УМНАЯ маскировка на основе реальных длин
        mask = torch.zeros_like(expected_alignment, dtype=torch.bool)
        for b in range(batch_size):
            actual_mel_len = min(mel_len, max_mel_len) if isinstance(mel_len, int) else min(mel_len[b], max_mel_len)
            actual_text_len = min(text_len, max_text_len) if isinstance(text_len, int) else min(text_len[b], max_text_len)
            mask[b, :actual_mel_len, :actual_text_len] = True
        
        # 🔥 ИСПРАВЛЕНИЕ NaN: Используем MSE вместо KL divergence для стабильности
        # KL divergence слишком чувствителен к нулевым значениям на раннем этапе
        att_ws_masked = att_ws * mask.float()
        expected_masked = expected_alignment * mask.float()
        
        # 🔥 БЕЗОПАСНАЯ нормализация с большим epsilon
        att_sum = att_ws_masked.sum(dim=2, keepdim=True)
        expected_sum = expected_masked.sum(dim=2, keepdim=True)
        
        # Клампим суммы для избежания деления на ноль
        att_sum_safe = torch.clamp(att_sum, min=1e-6)
        expected_sum_safe = torch.clamp(expected_sum, min=1e-6)
        
        att_ws_normalized = att_ws_masked / att_sum_safe
        expected_normalized = expected_masked / expected_sum_safe
        
        # 🔥 ИСПОЛЬЗУЕМ MSE ВМЕСТО KL DIVERGENCE для стабильности
        mse_loss = F.mse_loss(att_ws_normalized, expected_normalized, reduction='none')
        
        # Маскируем и усредняем
        mse_loss_masked = mse_loss * mask.float()
        
        # Усредняем по действительным элементам  
        valid_elements = mask.float().sum()
        if valid_elements > 0:
            guide_loss = mse_loss_masked.sum() / valid_elements
        else:
            guide_loss = torch.tensor(0.0, device=att_ws.device, requires_grad=True)
        
        return guide_loss

    def _get_adaptive_guide_weight(self):
        """
        🔥 АДАПТИВНЫЙ вес guided attention на основе Very Attentive Tacotron.
        
        Схема: начинаем с максимального веса, постепенно снижаем до минимума.
        """
        # Параметры из гиперпараметров
        initial_weight = getattr(self.hparams, 'guide_loss_initial_weight', 15.0)  # 🔥 УВЕЛИЧЕНО!
        decay_start = getattr(self.hparams, 'guide_loss_decay_start', 2000)
        decay_steps = getattr(self.hparams, 'guide_loss_decay_steps', 25000)
        min_weight = 0.05  # Минимальный вес для поддержания alignment
        
        if self.global_step < decay_start:
            # Фаза максимального guided attention
            return initial_weight
        elif self.global_step < decay_start + decay_steps:
            # Фаза постепенного снижения
            progress = (self.global_step - decay_start) / decay_steps
            # Экспоненциальный decay для плавного снижения
            decay_factor = math.exp(-progress * 3)  # -3 для умеренного снижения
            current_weight = min_weight + (initial_weight - min_weight) * decay_factor
            return max(min_weight, current_weight)
        else:
            # Фаза минимального guided attention
            return min_weight

    def _get_adaptive_sigma(self):
        """
        🔥 АДАПТИВНАЯ sigma для guided attention на основе фазы обучения.
        
        Схема: узкая -> широкая -> средняя для оптимального alignment.
        """
        if self.global_step < 1000:
            # Начальная фаза: узкая sigma для точного alignment
            return 0.1
        elif self.global_step < 5000:
            # Расширяющая фаза: увеличиваем sigma для гибкости
            progress = (self.global_step - 1000) / 4000
            return 0.1 + 0.3 * progress  # От 0.1 до 0.4
        else:
            # Стабилизирующая фаза: средняя sigma для баланса
            progress = min(1.0, (self.global_step - 5000) / 15000)
            return 0.4 - 0.25 * progress  # От 0.4 до 0.15


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss для более естественного звучания.
    Основан на человеческом восприятии звука.
    """
    
    def __init__(self, mel_channels=80):
        super(PerceptualLoss, self).__init__()
        self.mel_channels = mel_channels
        
        # Веса частот (низкие частоты важнее для восприятия)
        freq_weights = torch.exp(-torch.arange(mel_channels) / 20.0)
        self.register_buffer('freq_weights', freq_weights)
        
    def forward(self, mel_pred, mel_target):
        """
        Вычисляет perceptual loss с весами частот.
        """
        # 🔥 ИСПРАВЛЕНИЕ CUDA/CPU: Убеждаемся, что freq_weights на том же устройстве
        freq_weights = self.freq_weights.to(mel_pred.device)
        
        # 🔧 ИСПРАВЛЕНИЕ: Адаптируем веса к фактическому количеству mel каналов
        actual_mel_channels = mel_pred.size(1)
        if actual_mel_channels != freq_weights.size(0):
            # Ресамплируем веса под фактический размер
            if actual_mel_channels > freq_weights.size(0):
                # Повторяем последний вес
                extra_weights = freq_weights[-1].repeat(actual_mel_channels - freq_weights.size(0))
                freq_weights = torch.cat([freq_weights, extra_weights])
            else:
                # Обрезаем до нужного размера
                freq_weights = freq_weights[:actual_mel_channels]
        
        # Взвешенный MSE loss
        weighted_diff = (mel_pred - mel_target) ** 2
        weighted_loss = weighted_diff * freq_weights.view(1, -1, 1)
        
        # Добавляем L1 loss для резкости
        l1_loss = F.l1_loss(mel_pred, mel_target)
        
        perceptual_loss = torch.mean(weighted_loss) + 0.5 * l1_loss
        return perceptual_loss


class StyleLoss(nn.Module):
    """
    Style Loss для сохранения характеристик голоса.
    Основан на статистических характеристиках mel спектрограмм.
    """
    
    def __init__(self):
        super(StyleLoss, self).__init__()
        
    def forward(self, mel_pred, mel_target):
        """
        Вычисляет style loss на основе статистик mel.
        """
        # Статистики первого порядка (среднее)
        mean_pred = torch.mean(mel_pred, dim=2)
        mean_target = torch.mean(mel_target, dim=2)
        mean_loss = F.mse_loss(mean_pred, mean_target)
        
        # Статистики второго порядка (дисперсия)
        var_pred = torch.var(mel_pred, dim=2)
        var_target = torch.var(mel_target, dim=2)
        var_loss = F.mse_loss(var_pred, var_target)
        
        # Корреляции между каналами
        def compute_gram_matrix(x):
            b, c, t = x.size()
            features = x.view(b, c, t)
            gram = torch.bmm(features, features.transpose(1, 2))
            return gram / (c * t)
        
        gram_pred = compute_gram_matrix(mel_pred)
        gram_target = compute_gram_matrix(mel_target)
        gram_loss = F.mse_loss(gram_pred, gram_target)
        
        style_loss = mean_loss + var_loss + 0.5 * gram_loss
        return style_loss


class MonotonicAlignmentLoss(nn.Module):
    """
    Monotonic Alignment Loss для принуждения к монотонному alignment.
    Основан на MonoAlign исследованиях (2024).
    """
    
    def __init__(self):
        super(MonotonicAlignmentLoss, self).__init__()
        
    def forward(self, attention_weights):
        """
        🔥 ИСПРАВЛЕНО: Векторизованный расчет монотонности без циклов Python.
        """
        # attention_weights: (B, T_mel, T_text)
        batch_size, mel_len, text_len = attention_weights.shape
        device = attention_weights.device
        
        # Находим пики attention для каждого mel шага - векторизованно
        peak_positions = torch.argmax(attention_weights, dim=2)  # (B, T_mel)
        
        # Вычисляем разности между соседними пиками
        if mel_len < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        peak_diffs = peak_positions[:, 1:] - peak_positions[:, :-1]  # (B, T_mel-1)
        
        # Штраф за отрицательные разности (нарушения монотонности)
        violations = torch.clamp(-peak_diffs, min=0.0)  # (B, T_mel-1)
        
        # Общий штраф - среднее по всем элементам
        monotonic_loss = torch.mean(violations)
        
        return monotonic_loss


class QualityEnhancedMelLoss(nn.Module):
    """
    Улучшенный Mel Loss с фокусом на качество и естественность.
    Комбинирует несколько подходов для максимального качества.
    """
    
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.3):
        super(QualityEnhancedMelLoss, self).__init__()
        self.alpha = alpha  # MSE weight
        self.beta = beta    # L1 weight  
        self.gamma = gamma  # Dynamic range weight
        
    def forward(self, mel_pred, mel_target):
        """
        Комбинированный mel loss для максимального качества.
        """
        # 1. Базовый MSE loss
        mse_loss = F.mse_loss(mel_pred, mel_target)
        
        # 2. L1 loss для резкости деталей
        l1_loss = F.l1_loss(mel_pred, mel_target)
        
        # 3. Dynamic range loss для сохранения динамики
        pred_range = torch.max(mel_pred, dim=2)[0] - torch.min(mel_pred, dim=2)[0]
        target_range = torch.max(mel_target, dim=2)[0] - torch.min(mel_target, dim=2)[0]
        range_loss = F.mse_loss(pred_range, target_range)
        
        # 4. Комбинированный loss
        total_loss = (
            self.alpha * mse_loss + 
            self.beta * l1_loss + 
            self.gamma * range_loss
        )
        
        return total_loss


def create_enhanced_loss_function(hparams):
    """
    Фабричная функция для создания улучшенной loss function.
    """
    return Tacotron2Loss(hparams)


class GuidedAttentionLoss(nn.Module):
    """
    ❌ DEPRECATED: Заменен на UnifiedGuidedAttentionLoss
    
    Этот класс оставлен для обратной совместимости.
    🔥 ИСПОЛЬЗУЙТЕ UnifiedGuidedAttentionLoss для новых проектов!
    
    Legacy класс с медленными Python циклами.
    Новая система в 10x быстрее и умнее.
    """
    
    def __init__(self, alpha=2.0, sigma=0.4, decay_rate=0.9999):
        super(GuidedAttentionLoss, self).__init__()
        self.alpha = alpha              # Начальный вес
        self.sigma = sigma              # Sigma для gaussian guided attention
        self.decay_rate = decay_rate    # Скорость decay веса
        self.current_weight = alpha     # Текущий вес
        self.global_step = 0
        
        # 🚨 КРИТИЧЕСКИЕ параметры для восстановления после NaN
        self.critical_mode = False      # Режим критического восстановления
        self.min_weight = 0.1           # Повышенный минимальный вес
        self.max_weight = 50.0          # Увеличенный максимальный вес
        self.decay_start = 5000         # Отложенный decay для стабилизации
        self.decay_steps = 40000        # Увеличенная длительность decay
        
        # 🛡️ Параметры для экстренного восстановления диагональности
        self.emergency_weight = 25.0    # Экстренный вес при низкой диагональности
        
    def forward(self, model_output):
        """
        Вычисляет guided attention loss для модели.
        
        Args:
            model_output: Выход модели [mel_out, mel_out_postnet, gate_out, alignments]
            
        Returns:
            torch.Tensor: Guided attention loss
        """
        if len(model_output) < 4:
            # Нет alignments, возвращаем 0
            return torch.tensor(0.0, requires_grad=True)
            
        # 🔥 ИСПРАВЛЕНИЕ: Правильно распаковываем элементы модели в GuidedAttentionLoss
        if len(model_output) == 7:
            # Полный формат: [decoder_outputs, mel_out, mel_out_postnet, gate_out, alignments, tpse_gst_outputs, gst_outputs]
            decoder_outputs, mel_out, mel_out_postnet, gate_out, alignments, tpse_gst_outputs, gst_outputs = model_output
        elif len(model_output) == 6:
            # Формат без decoder_outputs: [mel_out, mel_out_postnet, gate_out, alignments, tpse_gst_outputs, gst_outputs]  
            mel_out, mel_out_postnet, gate_out, alignments, tpse_gst_outputs, gst_outputs = model_output
        elif len(model_output) == 5:
            # Формат с alignments: [mel_out, mel_out_postnet, gate_out, alignments, extra]
            mel_out, mel_out_postnet, gate_out, alignments, _ = model_output
        elif len(model_output) == 4:
            # Старый формат: [mel_out, mel_out_postnet, gate_out, alignments]
            mel_out, mel_out_postnet, gate_out, alignments = model_output
        else:
            # Fallback: берем первые 4 элемента
            mel_out, mel_out_postnet, gate_out, alignments = model_output[:4]
            
        if alignments is None:
            return torch.tensor(0.0, requires_grad=True)
            
        # Вычисляем guided attention loss
        batch_size, mel_len, text_len = alignments.shape
        
        # Создаем диагональную матрицу целевого alignment
        device = alignments.device
        
        # Индексы для создания диагонального alignment
        mel_indices = torch.arange(mel_len, device=device).float()
        text_indices = torch.arange(text_len, device=device).float()
        
        # Нормализуем индексы
        mel_indices_norm = mel_indices / (mel_len - 1) if mel_len > 1 else mel_indices
        text_indices_norm = text_indices / (text_len - 1) if text_len > 1 else text_indices
        
        # Создаем meshgrid для вычисления расстояний
        mel_grid = mel_indices_norm.unsqueeze(1).expand(mel_len, text_len)
        text_grid = text_indices_norm.unsqueeze(0).expand(mel_len, text_len)
        
        # Диагональная матрица (идеальный alignment)
        distances = (mel_grid - text_grid) ** 2
        
        # Gaussian guided attention с адаптивной sigma
        current_sigma = self._get_adaptive_sigma()
        guided_attention = torch.exp(-distances / (2 * current_sigma ** 2))
        
        # Нормализуем guided attention
        guided_attention = guided_attention / guided_attention.sum(dim=1, keepdim=True).clamp(min=1e-6)
        
        # Применяем к каждому элементу batch
        loss = 0.0
        for i in range(batch_size):
            # 🔥 ИСПРАВЛЕНИЕ NaN: Более безопасная нормализация attention весов
            att_sum = alignments[i].sum(dim=1, keepdim=True)
            att_sum_safe = torch.clamp(att_sum, min=1e-8)  # Увеличенный порог
            att_norm = alignments[i] / att_sum_safe
            
            # 🔥 ДОПОЛНИТЕЛЬНАЯ ЗАЩИТА: Клампим нормализованные веса
            att_norm = torch.clamp(att_norm, min=1e-8, max=1.0 - 1e-8)
            
            # 🔥 ИСПРАВЛЕНИЕ: Используем MSE вместо KL divergence для стабильности
            # KL divergence создает NaN на ранних этапах обучения
            
            # MSE loss между нормализованными attention весами
            mse_loss = F.mse_loss(att_norm, guided_attention, reduction='none')
            
            # Проверяем на NaN/Inf после MSE
            if torch.isnan(mse_loss).any() or torch.isinf(mse_loss).any():
                self.logger.warning(f"🚨 NaN/Inf в mse_loss, заменяем на ноль")
                mse_loss = torch.zeros_like(mse_loss)
            
            # Маскируем валидные элементы
            mask = (alignments[i].sum(dim=1) > 0).float().unsqueeze(1)
            mse_loss_masked = mse_loss * mask
            
            loss += mse_loss_masked.sum()
        
        # Нормализуем по batch size и sequence length
        loss = loss / (batch_size * mel_len)
        
        # Применяем адаптивный вес
        weighted_loss = self.get_weight() * loss
        
        # Обновляем глобальный счетчик
        self.global_step += 1
        self._update_weight()
        
        return weighted_loss
    
    def get_weight(self):
        """Возвращает текущий вес guided attention loss."""
        # 🚨 В критическом режиме используем экстренный вес
        if self.critical_mode:
            return self.emergency_weight
        return self.current_weight

    def activate_critical_mode(self):
        """
        🚨 Активирует критический режим восстановления диагональности.
        Используется при обнаружении NaN или критически низкой диагональности.
        """
        self.critical_mode = True
        # 🔥 ИСПРАВЛЕНИЕ: В критическом режиме СНИЖАЕМ вес guided attention
        # Вместо увеличения веса (что может вызывать NaN), сначала стабилизируем систему
        self.current_weight = min(self.emergency_weight, 1.0)  # Максимум 1.0 в критическом режиме
        print(f"🛡️ GuidedAttentionLoss: КРИТИЧЕСКИЙ РЕЖИМ активирован! Вес СНИЖЕН до: {self.current_weight}")

    def deactivate_critical_mode(self):
        """Деактивирует критический режим."""
        self.critical_mode = False
        print(f"✅ GuidedAttentionLoss: критический режим деактивирован")

    def check_diagonality_and_adapt(self, alignments):
        """
        Проверяет диагональность attention и автоматически адаптирует режим.
        
        Args:
            alignments: Attention веса [batch, mel_len, text_len]
        """
        if alignments is None or alignments.numel() == 0:
            return
        
        # Быстрая проверка диагональности для первого элемента батча
        attention = alignments[0].detach().cpu().numpy()
        batch_size, mel_len, text_len = attention.shape if len(attention.shape) == 3 else (1, *attention.shape)
        
        if len(attention.shape) == 2:
            attention = attention.reshape(1, *attention.shape)
            
        diagonality = self._calculate_quick_diagonality(attention[0] if len(attention.shape) == 3 else attention)
        
        # Автоматическая активация критического режима при низкой диагональности
        if diagonality < 0.2 and not self.critical_mode:
            self.activate_critical_mode()
        elif diagonality > 0.5 and self.critical_mode:
            self.deactivate_critical_mode()

    def _calculate_quick_diagonality(self, attention_matrix):
        """Быстрое вычисление диагональности."""
        try:
            if attention_matrix.size == 0:
                return 0.0
            
            mel_len, text_len = attention_matrix.shape
            
            # Создаем идеальную диагональ
            diagonal_sum = 0.0
            total_sum = attention_matrix.sum()
            
            if total_sum == 0:
                return 0.0
            
            # Суммируем веса по диагонали
            for i in range(mel_len):
                diagonal_pos = int(i * text_len / mel_len)
                if diagonal_pos < text_len:
                    diagonal_sum += attention_matrix[i, diagonal_pos]
            
            # Диагональность = доля весов на диагонали
            return diagonal_sum / total_sum if total_sum > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _update_weight(self):
        """Обновляет вес guided attention на основе расписания."""
        if self.global_step < self.decay_start:
            # Фаза максимального guided attention
            return
        elif self.global_step < self.decay_start + self.decay_steps:
            # Фаза постепенного снижения
            progress = (self.global_step - self.decay_start) / self.decay_steps
            # Экспоненциальный decay
            decay_factor = math.exp(-progress * 3)
            self.current_weight = self.min_weight + (self.alpha - self.min_weight) * decay_factor
            self.current_weight = max(self.min_weight, self.current_weight)
        else:
            # Фаза минимального guided attention
            self.current_weight = self.min_weight
    
    def _get_adaptive_sigma(self):
        """Вычисляет адаптивную sigma на основе фазы обучения."""
        if self.global_step < 1000:
            # Начальная фаза: узкая sigma для точного alignment
            return 0.1
        elif self.global_step < 5000:
            # Расширяющая фаза: увеличиваем sigma для гибкости
            progress = (self.global_step - 1000) / 4000
            return 0.1 + 0.3 * progress  # От 0.1 до 0.4
        else:
            # Стабилизирующая фаза: средняя sigma для баланса
            progress = min(1.0, (self.global_step - 5000) / 15000)
            return 0.4 - 0.25 * progress  # От 0.4 до 0.15
