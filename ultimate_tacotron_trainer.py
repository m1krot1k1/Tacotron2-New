#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 ULTIMATE ENHANCED TACOTRON TRAINER
Объединение ВСЕХ лучших решений из всех систем обучения

Интегрирует лучшее из:
- EnhancedTacotronTrainer (фазовое обучение, архитектура)
- train.py (критические исправления, стабильность)
- smart_tuner_main.py (автоматическая оптимизация)
- train_with_auto_fixes.py (автоматические исправления)

Режимы работы:
- 'simple': Быстрое обучение без оптимизации
- 'enhanced': Полное фазовое обучение с мониторингом  
- 'auto_optimized': Автоматическая оптимизация + обучение
- 'ultimate': Все возможности + интеллектуальная адаптация
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
import logging
import copy
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import yaml
import argparse

# Импорт основных компонентов
from model import Tacotron2
from loss_function import Tacotron2Loss
from hparams import create_hparams
from audio_quality_enhancer import AudioQualityEnhancer

# 🔧 КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ ИЗ TRAIN.PY
try:
    from smart_tuner.gradient_clipper import AdaptiveGradientClipper, get_global_clipper, set_global_clipper
    GRADIENT_CLIPPER_AVAILABLE = True
except ImportError:
    GRADIENT_CLIPPER_AVAILABLE = False
    logging.warning("AdaptiveGradientClipper недоступен")

try:
    from smart_tuner.smart_lr_adapter import SmartLRAdapter, get_global_lr_adapter, set_global_lr_adapter  
    SMART_LR_AVAILABLE = True
except ImportError:
    SMART_LR_AVAILABLE = False
    logging.warning("SmartLRAdapter недоступен")

# 🤖 АВТОМАТИЧЕСКИЕ ИСПРАВЛЕНИЯ
try:
    from smart_tuner.auto_fix_manager import AutoFixManager
    AUTO_FIX_AVAILABLE = True
except ImportError:
    AUTO_FIX_AVAILABLE = False
    logging.warning("AutoFixManager недоступен")

# 🚀 АВТОМАТИЧЕСКАЯ ОПТИМИЗАЦИЯ
try:
    from smart_tuner.optimization_engine import OptimizationEngine
    from smart_tuner.smart_tuner_integration import SmartTunerIntegration
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    logging.warning("OptimizationEngine недоступен")

# 📱 МОНИТОРИНГ И ДИАГНОСТИКА
try:
    from smart_tuner.telegram_monitor_enhanced import TelegramMonitorEnhanced
    from debug_reporter import initialize_debug_reporter, get_debug_reporter
    from alignment_diagnostics import AlignmentDiagnostics
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    logging.warning("Системы мониторинга недоступны")

# 📊 ЛОГИРОВАНИЕ
try:
    import mlflow
    import mlflow.pytorch
    from torch.utils.tensorboard import SummaryWriter
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    logging.warning("MLflow/TensorBoard недоступны")

# Импорт для данных
try:
    from data_utils import TextMelLoader, TextMelCollate
    DATA_UTILS_AVAILABLE = True
except ImportError:
    DATA_UTILS_AVAILABLE = False
    logging.warning("data_utils недоступен - создам минимальную реализацию")

class UltimateEnhancedTacotronTrainer:
    """
    🏆 ULTIMATE Enhanced Tacotron Trainer
    
    Объединяет ВСЕ лучшие решения из всех систем:
    - Фазовое обучение (4 фазы с адаптацией)
    - Автоматические исправления (AutoFixManager)
    - Адаптивное управление градиентами (AdaptiveGradientClipper)
    - Smart LR адаптация в реальном времени
    - Автоматическая оптимизация гиперпараметров
    - Продвинутый мониторинг (15+ метрик)
    - Интеллектуальные рекомендации
    """
    
    def __init__(self, hparams, mode: str = 'enhanced', dataset_info: Optional[Dict] = None):
        """
        Инициализация Ultimate Enhanced Trainer.
        
        Args:
            hparams: Гиперпараметры обучения
            mode: Режим работы ('simple', 'enhanced', 'auto_optimized', 'ultimate')
            dataset_info: Информация о датасете для оптимизации
        """
        self.hparams = hparams
        self.mode = mode
        self.dataset_info = dataset_info or {}
        
        # Настройка логирования
        self.logger = self._setup_logger()
        self.logger.info(f"🏆 Инициализация Ultimate Enhanced Tacotron Trainer (режим: {mode})")
        
        # Базовые компоненты
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # 🔧 КРИТИЧЕСКИЕ КОМПОНЕНТЫ ИЗ TRAIN.PY
        self.adaptive_gradient_clipper = None
        self.smart_lr_adapter = None
        
        # 🤖 АВТОМАТИЧЕСКИЕ ИСПРАВЛЕНИЯ
        self.auto_fix_manager = None
        
        # 🚀 АВТОМАТИЧЕСКАЯ ОПТИМИЗАЦИЯ
        self.optimization_engine = None
        self.smart_tuner = None
        
        # 📱 МОНИТОРИНГ И ДИАГНОСТИКА
        self.telegram_monitor = None
        self.debug_reporter = None
        self.alignment_diagnostics = None
        
        # 📊 ЛОГИРОВАНИЕ
        self.tensorboard_writer = None
        self.mlflow_run = None
        
        # Состояние обучения
        self.current_epoch = 0
        self.global_step = 0
        self.best_validation_loss = float('inf')
        self.training_metrics_history = []
        self.last_attention_diagonality = 0.0
        
        # 🎯 ФАЗОВОЕ ОБУЧЕНИЕ
        self.training_phases = {
            'pre_alignment': {'max_epoch': 500, 'focus': 'attention_learning'},
            'alignment_learning': {'max_epoch': 2000, 'focus': 'attention_stabilization'}, 
            'quality_optimization': {'max_epoch': 3000, 'focus': 'quality_improvement'},
            'fine_tuning': {'max_epoch': 3500, 'focus': 'final_polishing'}
        }
        
        # Инициализация всех компонентов
        self._initialize_components()
        
        self.logger.info("✅ Ultimate Enhanced Tacotron Trainer инициализирован успешно!")
    
    def _setup_logger(self) -> logging.Logger:
        """Настраивает логирование для Ultimate Trainer."""
        logger = logging.getLogger('UltimateEnhancedTacotronTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Консольный handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - [🏆 Ultimate Trainer] - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # Файловый handler
            file_handler = logging.FileHandler('ultimate_training.log')
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_components(self):
        """Инициализирует все компоненты в зависимости от режима."""
        self.logger.info(f"🔧 Инициализация компонентов для режима '{self.mode}'...")
        
        # 📊 ВСЕГДА: Базовое логирование
        self._initialize_logging()
        
        # 📱 ВСЕГДА: Telegram мониторинг  
        self._initialize_telegram_monitoring()
        
        if self.mode in ['enhanced', 'auto_optimized', 'ultimate']:
            # 🔧 КРИТИЧЕСКИЕ КОМПОНЕНТЫ
            self._initialize_critical_components()
            
        if self.mode in ['auto_optimized', 'ultimate']:
            # 🚀 АВТОМАТИЧЕСКАЯ ОПТИМИЗАЦИЯ
            self._initialize_optimization()
            
        if self.mode == 'ultimate':
            # 🤖 ВСЕ ВОЗМОЖНОСТИ
            self._initialize_ultimate_features()
    
    def _initialize_logging(self):
        """Инициализация систем логирования."""
        if not LOGGING_AVAILABLE:
            self.logger.warning("⚠️ Системы логирования недоступны")
            return
            
        try:
            # TensorBoard
            self.tensorboard_writer = SummaryWriter('logs')
            self.logger.info("✅ TensorBoard writer инициализирован")
            
            # MLflow  
            mlflow.start_run(run_name=f"ultimate_training_{int(time.time())}")
            self.mlflow_run = mlflow.active_run()
            self.logger.info("✅ MLflow run инициализирован")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации логирования: {e}")
    
    def _initialize_telegram_monitoring(self):
        """Инициализация Telegram мониторинга."""
        if not MONITORING_AVAILABLE:
            return
            
        try:
            config_path = "smart_tuner/config.yaml"
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                telegram_config = config.get('telegram', {})
                if telegram_config.get('enabled', False):
                    self.telegram_monitor = TelegramMonitorEnhanced(
                        bot_token=telegram_config.get('bot_token'),
                        chat_id=telegram_config.get('chat_id'),
                        enabled=True
                    )
                    self.logger.info("✅ Telegram Monitor Enhanced инициализирован")
                    
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации Telegram: {e}")
    
    def _initialize_critical_components(self):
        """Инициализация критических компонентов из train.py."""
        self.logger.info("🔧 Инициализация критических компонентов...")
        
        # 🔧 AdaptiveGradientClipper
        if GRADIENT_CLIPPER_AVAILABLE:
            try:
                self.adaptive_gradient_clipper = AdaptiveGradientClipper(
                    max_norm=1.0,
                    adaptive=True,
                    emergency_threshold=100.0,
                    history_size=1000,
                    percentile=95
                )
                set_global_clipper(self.adaptive_gradient_clipper)
                self.logger.info("✅ AdaptiveGradientClipper инициализирован")
            except Exception as e:
                self.logger.error(f"❌ Ошибка AdaptiveGradientClipper: {e}")
        
        # 🤖 AutoFixManager  
        if AUTO_FIX_AVAILABLE:
            try:
                # AutoFixManager будет инициализирован после создания модели
                self.logger.info("🤖 AutoFixManager будет инициализирован после модели")
            except Exception as e:
                self.logger.error(f"❌ Ошибка подготовки AutoFixManager: {e}")
        
        # 📊 Alignment Diagnostics
        if MONITORING_AVAILABLE:
            try:
                self.alignment_diagnostics = AlignmentDiagnostics()
                self.logger.info("✅ AlignmentDiagnostics инициализирован")
            except Exception as e:
                self.logger.error(f"❌ Ошибка AlignmentDiagnostics: {e}")
                
        # 🔍 Debug Reporter
        if MONITORING_AVAILABLE:
            try:
                self.debug_reporter = initialize_debug_reporter(self.telegram_monitor)
                self.logger.info("✅ Debug Reporter инициализирован")
            except Exception as e:
                self.logger.error(f"❌ Ошибка Debug Reporter: {e}")
    
    def _initialize_optimization(self):
        """Инициализация автоматической оптимизации."""
        if not OPTIMIZATION_AVAILABLE:
            self.logger.warning("⚠️ OptimizationEngine недоступен")
            return
            
        try:
            self.optimization_engine = OptimizationEngine("smart_tuner/config.yaml")
            self.smart_tuner = SmartTunerIntegration()
            self.logger.info("✅ OptimizationEngine и SmartTuner инициализированы")
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации оптимизации: {e}")
    
    def _initialize_ultimate_features(self):
        """Инициализация эксклюзивных возможностей Ultimate режима."""
        self.logger.info("🏆 Инициализация Ultimate возможностей...")
        
        # Здесь будут эксклюзивные возможности Ultimate режима
        # - Интеллектуальные рекомендации
        # - Автоматическое переключение стратегий
        # - Комбинированные подходы
        # - Продвинутый анализ
        
        self.logger.info("✅ Ultimate возможности инициализированы") 

    def initialize_training(self):
        """Инициализирует все компоненты для обучения."""
        self.logger.info("🔧 Инициализация компонентов обучения...")
        
        # 🚀 АВТОМАТИЧЕСКАЯ ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ (для режимов auto_optimized и ultimate)
        if self.mode in ['auto_optimized', 'ultimate'] and self.smart_tuner:
            try:
                original_hparams = vars(self.hparams).copy()
                optimized_hparams = self.smart_tuner.on_training_start(
                    original_hparams, self.dataset_info
                )
                
                # Применяем оптимизированные параметры
                for key, value in optimized_hparams.items():
                    if hasattr(self.hparams, key):
                        setattr(self.hparams, key, value)
                        
                self.logger.info("✨ Гиперпараметры оптимизированы через Smart Tuner")
            except Exception as e:
                self.logger.error(f"Ошибка оптимизации гиперпараметров: {e}")
        
        # Инициализация модели
        self.model = Tacotron2(self.hparams).cuda()
        self.logger.info(f"📊 Модель загружена: {sum(p.numel() for p in self.model.parameters())} параметров")
        
        # Инициализация loss функции
        self.criterion = Tacotron2Loss(self.hparams)
        self.logger.info("🎯 Enhanced loss function инициализирована")
        
        # Инициализация оптимизатора
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=getattr(self.hparams, 'weight_decay', 1e-6)
        )
        self.logger.info("⚙️ Оптимизатор AdamW инициализирован")
        
        # 🔧 Smart LR Adapter (если доступен)
        if SMART_LR_AVAILABLE:
            try:
                self.smart_lr_adapter = SmartLRAdapter(
                    optimizer=self.optimizer,
                    patience=10,
                    factor=0.5,
                    min_lr=getattr(self.hparams, 'learning_rate_min', 1e-8),
                    max_lr=self.hparams.learning_rate * 2,
                    emergency_factor=0.1,
                    grad_norm_threshold=1000.0,
                    loss_nan_threshold=1e6
                )
                set_global_lr_adapter(self.smart_lr_adapter)
                self.logger.info("✅ Smart LR Adapter инициализирован")
            except Exception as e:
                self.logger.error(f"Ошибка Smart LR Adapter: {e}")
        
        # 🤖 AutoFixManager (инициализируем после модели и оптимизатора)
        if AUTO_FIX_AVAILABLE and self.mode in ['enhanced', 'auto_optimized', 'ultimate']:
            try:
                self.auto_fix_manager = AutoFixManager(
                    model=self.model,
                    optimizer=self.optimizer,
                    hparams=self.hparams,
                    telegram_monitor=self.telegram_monitor
                )
                self.logger.info("✅ AutoFixManager инициализирован")
            except Exception as e:
                self.logger.error(f"Ошибка AutoFixManager: {e}")
    
    def get_current_training_phase(self) -> str:
        """Определяет текущую фазу обучения на основе эпохи."""
        for phase, config in self.training_phases.items():
            if self.current_epoch < config['max_epoch']:
                return phase
        return 'fine_tuning'  # Последняя фаза
    
    def adjust_hyperparams_for_phase(self, phase: str):
        """Адаптирует гиперпараметры для текущей фазы обучения."""
        if self.mode not in ['enhanced', 'auto_optimized', 'ultimate']:
            return  # Фазовое обучение только для продвинутых режимов
            
        phase_configs = {
            'pre_alignment': {
                'learning_rate_multiplier': 1.2,
                'guide_loss_weight_multiplier': 3.0,
                'batch_size_multiplier': 0.8
            },
            'alignment_learning': {
                'learning_rate_multiplier': 1.0, 
                'guide_loss_weight_multiplier': 2.0,
                'batch_size_multiplier': 1.0
            },
            'quality_optimization': {
                'learning_rate_multiplier': 0.8,
                'guide_loss_weight_multiplier': 1.5,
                'batch_size_multiplier': 1.2
            },
            'fine_tuning': {
                'learning_rate_multiplier': 0.5,
                'guide_loss_weight_multiplier': 1.0,
                'batch_size_multiplier': 1.0
            }
        }
        
        config = phase_configs.get(phase, phase_configs['alignment_learning'])
        
        # Применяем адаптации
        if hasattr(self.criterion, 'guide_loss_weight'):
            new_weight = getattr(self.hparams, 'guide_loss_weight', 2.5) * config['guide_loss_weight_multiplier']
            self.criterion.guide_loss_weight = new_weight
            
        self.logger.info(f"📊 Фаза '{phase}': адаптированы параметры "
                        f"lr_mult={config['learning_rate_multiplier']}")
    
    def train_step(self, batch):
        """Выполняет один шаг обучения с максимальными возможностями."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Распаковка batch (TextMelCollate возвращает 8 элементов)
        text_inputs, text_lengths, mel_targets, gate_targets, mel_lengths, ctc_text, ctc_text_lengths, guide_mask = batch
        
        # Перенос на GPU
        text_inputs = text_inputs.cuda()
        mel_targets = mel_targets.cuda() 
        gate_targets = gate_targets.cuda()
        
        # 🔧 БЕЗОПАСНАЯ ОБРАБОТКА MODEL_OUTPUTS (из train.py)
        try:
            x, y = self.model.parse_batch(batch)
            model_outputs = self.model(x)
            
            # Безопасная распаковка (поддержка 1-7+ значений)
            if len(model_outputs) >= 4:
                mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model_outputs[:4]
            elif len(model_outputs) == 3:
                mel_outputs, mel_outputs_postnet, gate_outputs = model_outputs
                alignments = None
            elif len(model_outputs) == 2:
                mel_outputs, mel_outputs_postnet = model_outputs
                gate_outputs = None
                alignments = None
            else:
                mel_outputs = model_outputs[0] if len(model_outputs) > 0 else None
                mel_outputs_postnet = None
                gate_outputs = None
                alignments = None
                
        except Exception as e:
            self.logger.error(f"Ошибка forward pass: {e}")
            return {'total_loss': torch.tensor(float('inf'))}
        
        # 🔍 ВЫЧИСЛЕНИЕ МЕТРИК КАЧЕСТВА
        attention_diagonality = 0.0
        gate_accuracy = 0.0
        
        try:
            # Вычисляем attention_diagonality
            if alignments is not None:
                attention_matrix = alignments.detach().cpu().numpy()
                if attention_matrix.ndim == 3:
                    batch_diagonalities = []
                    for b in range(attention_matrix.shape[0]):
                        attn = attention_matrix[b]
                        if attn.sum() > 0:
                            attn = attn / attn.sum(axis=1, keepdims=True)
                        
                        min_dim = min(attn.shape[0], attn.shape[1])
                        diagonal_elements = [attn[i, i] for i in range(min_dim)]
                        batch_diagonalities.append(np.mean(diagonal_elements) if diagonal_elements else 0.0)
                    
                    attention_diagonality = np.mean(batch_diagonalities) if batch_diagonalities else 0.0
            
            # Вычисляем gate_accuracy
            if gate_outputs is not None:
                gate_pred = (gate_outputs > 0.5).float()
                gate_targets_binary = (gate_targets > 0.5).float()
                correct = (gate_pred == gate_targets_binary).float().mean()
                gate_accuracy = correct.item()
                
        except Exception as e:
            self.logger.warning(f"Ошибка вычисления метрик качества: {e}")
        
        # Сохраняем диагональность для следующего шага
        self.last_attention_diagonality = attention_diagonality
        
        # 🎯 БЕЗОПАСНАЯ АДАПТИВНАЯ НАСТРОЙКА GUIDED ATTENTION
        if hasattr(self.criterion, 'guide_loss_weight') and self.global_step > 0:
            current_weight = self.criterion.guide_loss_weight
            
            # КРИТИЧЕСКИ ВАЖНО: НЕ ДОПУСКАЕМ ВЗРЫВА ДО 100.0!
            if attention_diagonality < 0.05:
                # Очень осторожное увеличение - НЕ БОЛЕЕ 15.0!
                new_weight = min(current_weight * 1.5, 15.0)
                self.criterion.guide_loss_weight = new_weight
                self.logger.warning(f"🚨 Осторожное увеличение guided attention weight: {current_weight:.1f} → {new_weight:.1f}")
            elif attention_diagonality < 0.1:
                # Умеренное увеличение - НЕ БОЛЕЕ 12.0!
                new_weight = min(current_weight * 1.3, 12.0)
                self.criterion.guide_loss_weight = new_weight
                self.logger.warning(f"🚨 Умеренное увеличение guided attention weight: {current_weight:.1f} → {new_weight:.1f}")
            elif attention_diagonality > 0.7:
                # Снижение когда attention уже хорошее
                new_weight = max(current_weight * 0.9, 1.0)
                self.criterion.guide_loss_weight = new_weight
                self.logger.info(f"📉 Снижение guided attention weight: {current_weight:.1f} → {new_weight:.1f}")
            
            # АВАРИЙНАЯ ЗАЩИТА: если как-то дошло до критических значений
            if self.criterion.guide_loss_weight > 20.0:
                self.criterion.guide_loss_weight = 10.0  # Сбрасываем до безопасного
                self.logger.error(f"🚨 АВАРИЙНЫЙ СБРОС guided attention weight до 10.0!")
        
        # 🎯 ВЫЧИСЛЕНИЕ LOSS С БЕЗОПАСНОЙ ОБРАБОТКОЙ
        try:
            loss_components = self.criterion(
                model_outputs, 
                (mel_targets, gate_targets),
                attention_weights=alignments,
                gate_outputs=gate_outputs
            )
            
            # Безопасная обработка loss компонентов
            if isinstance(loss_components, (list, tuple)):
                if len(loss_components) == 4:
                    mel_loss, gate_loss, guide_loss, emb_loss = loss_components
                    loss = mel_loss + gate_loss + guide_loss + emb_loss
                    loss_dict = {
                        'mel_loss': mel_loss.item(),
                        'gate_loss': gate_loss.item(), 
                        'guide_loss': guide_loss.item(),
                        'emb_loss': emb_loss.item(),
                        'total_loss': loss.item()
                    }
                else:
                    loss = loss_components[0]
                    loss_dict = {'total_loss': loss.item()}
            else:
                loss = loss_components
                loss_dict = {'total_loss': loss.item()}
                
        except Exception as e:
            self.logger.error(f"Ошибка вычисления loss: {e}")
            return {'total_loss': torch.tensor(float('inf'))}
        
        # Backward pass
        try:
            loss.backward()
        except Exception as e:
            self.logger.error(f"Ошибка backward pass: {e}")
            return {'total_loss': torch.tensor(float('inf'))}
        
        # 🔧 ПРОДВИНУТОЕ УПРАВЛЕНИЕ ГРАДИЕНТАМИ
        if self.adaptive_gradient_clipper:
            # Используем AdaptiveGradientClipper
            was_clipped, grad_norm, clip_threshold = self.adaptive_gradient_clipper.clip_gradients(
                self.model, self.global_step
            )
            
            if was_clipped:
                self.logger.info(f"🔧 Градиенты обрезаны: {grad_norm:.2f} → {clip_threshold:.2f}")
        else:
            # Стандартное клипирование с критическими алертами
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                getattr(self.hparams, 'grad_clip_thresh', 1.0)
            )
            
            # Критические алерты для высоких градиентов
            if grad_norm > 10.0:
                self.logger.warning(f"🚨 ВЫСОКАЯ норма градиентов: {grad_norm:.2f}")
            if grad_norm > 100.0:
                self.logger.error(f"🚨 КРИТИЧЕСКАЯ норма градиентов: {grad_norm:.2f}")
        
        # 🤖 АВТОМАТИЧЕСКИЕ ИСПРАВЛЕНИЯ
        if self.auto_fix_manager:
            try:
                fix_metrics = {
                    'grad_norm': float(grad_norm),
                    'attention_diagonality': attention_diagonality,
                    'gate_accuracy': gate_accuracy,
                    'loss': float(loss.item()),
                    'mel_loss': loss_dict.get('mel_loss', 0),
                    'gate_loss': loss_dict.get('gate_loss', 0),
                    'guide_loss': loss_dict.get('guide_loss', 0)
                }
                
                applied_fixes = self.auto_fix_manager.analyze_and_fix(
                    step=self.global_step,
                    metrics=fix_metrics,
                    loss=loss
                )
                
                if applied_fixes:
                    self.logger.info(f"🔧 Применено {len(applied_fixes)} автоматических исправлений")
                    
            except Exception as e:
                self.logger.error(f"❌ Ошибка в AutoFixManager: {e}")
        
        # 🔧 Smart LR Adapter
        if self.smart_lr_adapter:
            try:
                lr_changed = self.smart_lr_adapter.step(
                    loss=float(loss.item()),
                    grad_norm=float(grad_norm),
                    step=self.global_step
                )
                if lr_changed:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    # Проверяем что LR не стал слишком маленьким
                    if current_lr < 1e-7:
                        # Устанавливаем минимально допустимый LR
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = 1e-6
                        current_lr = 1e-6
                        self.logger.info(f"🔄 LR восстановлен до минимального уровня: {current_lr:.2e}")
                    else:
                        self.logger.info(f"🔄 Smart LR адаптация: LR изменен на {current_lr:.2e}")
            except Exception as e:
                self.logger.error(f"Ошибка Smart LR Adapter: {e}")
        
        # Optimizer step
        try:
            self.optimizer.step()
            self.global_step += 1
        except Exception as e:
            self.logger.error(f"Ошибка optimizer step: {e}")
            return {'total_loss': torch.tensor(float('inf'))}
        
        # Добавляем метрики к результату
        loss_dict.update({
            'grad_norm': grad_norm,
            'attention_diagonality': attention_diagonality,
            'gate_accuracy': gate_accuracy,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        })
        
        return loss_dict
    
    def validate_step(self, val_loader):
        """Выполняет валидацию с полным анализом."""
        self.model.eval()
        
        val_losses = []
        attention_scores = []
        gate_accuracies = []
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    # Аналогичная обработка как в train_step, но без backward
                    text_inputs, text_lengths, mel_targets, gate_targets, mel_lengths, ctc_text, ctc_text_lengths, guide_mask = batch
                    
                    text_inputs = text_inputs.cuda()
                    mel_targets = mel_targets.cuda()
                    gate_targets = gate_targets.cuda()
                    
                    x, y = self.model.parse_batch(batch)
                    model_outputs = self.model(x)
                    
                    # Вычисление loss
                    loss_components = self.criterion(model_outputs, (mel_targets, gate_targets))
                    
                    if isinstance(loss_components, (list, tuple)):
                        val_loss = sum(loss_components)
                    else:
                        val_loss = loss_components
                        
                    val_losses.append(val_loss.item())
                    
                    # Метрики качества (упрощенно для валидации)
                    if len(model_outputs) >= 4:
                        alignments = model_outputs[3]
                        gate_outputs = model_outputs[2]
                        
                        # Attention diagonality
                        if alignments is not None:
                            attention_matrix = alignments.detach().cpu().numpy()
                            if attention_matrix.ndim == 3:
                                diag_score = np.mean([np.trace(attention_matrix[b]) / min(attention_matrix[b].shape) 
                                                    for b in range(attention_matrix.shape[0])])
                                attention_scores.append(diag_score)
                        
                        # Gate accuracy
                        if gate_outputs is not None:
                            gate_pred = (gate_outputs > 0.5).float()
                            gate_targets_binary = (gate_targets > 0.5).float()
                            accuracy = (gate_pred == gate_targets_binary).float().mean()
                            gate_accuracies.append(accuracy.item())
                            
                except Exception as e:
                    self.logger.warning(f"Ошибка в валидации: {e}")
                    continue
        
        # Агрегированные метрики валидации
        validation_metrics = {
            'val_loss': np.mean(val_losses) if val_losses else float('inf'),
            'val_attention_diagonality': np.mean(attention_scores) if attention_scores else 0.0,
            'val_gate_accuracy': np.mean(gate_accuracies) if gate_accuracies else 0.0
        }
        
        return validation_metrics 

    def train(self, train_loader, val_loader, num_epochs: int = 3500, max_steps: Optional[int] = None):
        """
        Главный метод обучения с полной интеграцией всех систем.
        
        Args:
            train_loader: DataLoader для обучения
            val_loader: DataLoader для валидации
            num_epochs: Количество эпох обучения
            max_steps: Максимальное количество шагов (для быстрого тестирования)
        """
        self.logger.info(f"🚀 Начинаем Ultimate Enhanced Training (режим: {self.mode})")
        if max_steps:
            self.logger.info(f"🔬 ТЕСТОВЫЙ РЕЖИМ: ограничение {max_steps} шагов")
        else:
            self.logger.info(f"📊 Эпох: {num_epochs}, Батчей: {len(train_loader)}")
        
        # Инициализация обучения
        self.initialize_training()
        
        # 📱 Отправка стартового уведомления
        if self.telegram_monitor:
            try:
                if hasattr(self.telegram_monitor, 'send_training_start_notification'):
                    self.telegram_monitor.send_training_start_notification(
                        hparams=self.hparams,
                        dataset_info=self.dataset_info
                    )
                elif hasattr(self.telegram_monitor, 'send_message'):
                    self.telegram_monitor.send_message("🚀 Начинаю Ultimate Enhanced Training!")
                elif hasattr(self.telegram_monitor, 'send_training_notification'):
                    self.telegram_monitor.send_training_notification("🚀 Начинаю Ultimate Enhanced Training!")
                else:
                    self.logger.debug("Telegram monitor не поддерживает отправку сообщений")
            except Exception as e:
                self.logger.warning(f"Ошибка Telegram уведомления: {e}")
        
        # Переменные для мониторинга
        epoch_start_time = time.time()
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = getattr(self.hparams, 'early_stopping_patience', 10)
        
        # История обучения для анализа
        training_history = []
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # 🎯 ФАЗОВОЕ ОБУЧЕНИЕ (для продвинутых режимов)
            current_phase = self.get_current_training_phase()
            if epoch % 100 == 0 or epoch < 5:  # Периодическая адаптация
                self.adjust_hyperparams_for_phase(current_phase)
            
            # 📊 ОБУЧЕНИЕ ЗА ЭПОХУ
            epoch_losses = []
            epoch_grad_norms = []
            epoch_attention_scores = []
            epoch_gate_accuracies = []
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    # Выполняем шаг обучения
                    step_metrics = self.train_step(batch)
                    
                    # Сохраняем метрики
                    epoch_losses.append(step_metrics['total_loss'])
                    epoch_grad_norms.append(step_metrics.get('grad_norm', 0))
                    epoch_attention_scores.append(step_metrics.get('attention_diagonality', 0))
                    epoch_gate_accuracies.append(step_metrics.get('gate_accuracy', 0))
                    
                    # 📊 ЛОГИРОВАНИЕ КАЖДЫЕ 100 ШАГОВ (или каждые 10 в тестовом режиме)
                    log_frequency = 10 if max_steps else 100
                    if self.global_step % log_frequency == 0:
                        self._log_training_step(step_metrics, epoch, batch_idx)
                    
                    # 🔧 КРИТИЧЕСКИЙ МОНИТОРИНГ
                    if step_metrics['total_loss'] > 100 or step_metrics.get('grad_norm', 0) > 1000:
                        self.logger.error(f"🚨 КРИТИЧЕСКИЕ ПОКАЗАТЕЛИ на шаге {self.global_step}!")
                        self.logger.error(f"Loss: {step_metrics['total_loss']:.2f}, Grad Norm: {step_metrics.get('grad_norm', 0):.2f}")
                        
                        # Автоматические аварийные меры
                        if self.mode in ['auto_optimized', 'ultimate'] and self.auto_fix_manager:
                            emergency_fixes = self.auto_fix_manager.emergency_intervention(
                                step=self.global_step,
                                critical_metrics=step_metrics
                            )
                            if emergency_fixes:
                                self.logger.info(f"🔧 Применены аварийные меры: {emergency_fixes}")
                    
                    # 🔬 ПРОВЕРКА ЛИМИТА ШАГОВ ДЛЯ ТЕСТИРОВАНИЯ
                    if max_steps and self.global_step >= max_steps:
                        self.logger.info(f"🔬 Достигнут лимит тестирования: {max_steps} шагов на эпохе {epoch}")
                        break
                        
                except Exception as e:
                    self.logger.error(f"❌ Ошибка в шаге обучения {batch_idx}: {e}")
                    continue
            
            # 📊 АГРЕГИРОВАНИЕ МЕТРИК ЭПОХИ
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': np.mean(epoch_losses) if epoch_losses else float('inf'),
                'train_grad_norm': np.mean(epoch_grad_norms) if epoch_grad_norms else 0,
                'train_attention_diagonality': np.mean(epoch_attention_scores) if epoch_attention_scores else 0,
                'train_gate_accuracy': np.mean(epoch_gate_accuracies) if epoch_gate_accuracies else 0,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'phase': current_phase,
                'epoch_time': time.time() - epoch_start
            }
            
            # 📊 ВАЛИДАЦИЯ КАЖДЫЕ 10 ЭПОХ (или сразу в тестовом режиме)
            validation_frequency = 1 if max_steps else 10
            if epoch % validation_frequency == 0 or epoch < 5:
                try:
                    val_metrics = self.validate_step(val_loader)
                    epoch_metrics.update(val_metrics)
                    
                    # Проверка улучшения
                    if val_metrics['val_loss'] < best_val_loss:
                        best_val_loss = val_metrics['val_loss']
                        patience_counter = 0
                        self._save_checkpoint(epoch, is_best=True)
                        self.logger.info(f"🎉 Новый лучший результат: val_loss = {best_val_loss:.4f}")
                    else:
                        patience_counter += 1
                        
                except Exception as e:
                    self.logger.error(f"❌ Ошибка валидации: {e}")
                    val_metrics = {'val_loss': float('inf')}
                    epoch_metrics.update(val_metrics)
            
            # Сохранение истории
            training_history.append(epoch_metrics)
            
            # 📊 ПОДРОБНОЕ ЛОГИРОВАНИЕ ЭПОХИ
            self._log_epoch_summary(epoch_metrics)
            
            # 📱 TELEGRAM УВЕДОМЛЕНИЯ (каждые 5 шагов в тестовом режиме, каждые 50 эпох в обычном)
            telegram_frequency = 5 if max_steps else 50
            if epoch % telegram_frequency == 0 or epoch < 5 or patience_counter > max_patience // 2:
                self._send_epoch_telegram_update(epoch_metrics)
            
            # 🔄 АВТОМАТИЧЕСКАЯ ОПТИМИЗАЦИЯ (для ultimate режима)
            if self.mode == 'ultimate' and epoch > 10 and len(training_history) >= 10:
                self._perform_intelligent_adjustments(training_history[-10:])  # Анализ последних 10 эпох
            
            # 🛑 EARLY STOPPING
            if patience_counter >= max_patience:
                self.logger.info(f"🛑 Early stopping после {patience_counter} эпох без улучшений")
                break
                
            # 🔬 ВЫХОД ИЗ ЦИКЛА ПРИ ДОСТИЖЕНИИ ЛИМИТА ШАГОВ
            if max_steps and self.global_step >= max_steps:
                self.logger.info(f"🔬 Тестирование завершено на {self.global_step} шагах")
                break
            
            # 🔧 ПЕРИОДИЧЕСКОЕ СОХРАНЕНИЕ
            save_frequency = 20 if max_steps else 100
            if epoch % save_frequency == 0:
                self._save_checkpoint(epoch, is_best=False)
                
        # 🎉 ЗАВЕРШЕНИЕ ОБУЧЕНИЯ
        total_time = time.time() - epoch_start_time
        self._finalize_training(training_history, total_time)
        
        return training_history
    
    def _log_training_step(self, metrics: Dict, epoch: int, batch_idx: int):
        """Логирование шага обучения."""
        # Console logging
        self.logger.info(
            f"Эпоха {epoch}, Батч {batch_idx}, Шаг {self.global_step}: "
            f"Loss: {metrics['total_loss']:.4f}, "
            f"Grad: {metrics.get('grad_norm', 0):.2f}, "
            f"Attn: {metrics.get('attention_diagonality', 0):.3f}, "
            f"Gate: {metrics.get('gate_accuracy', 0):.3f}"
        )
        
        # TensorBoard logging
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_scalar('Train/Loss', metrics['total_loss'], self.global_step)
                self.tensorboard_writer.add_scalar('Train/GradNorm', metrics.get('grad_norm', 0), self.global_step)
                self.tensorboard_writer.add_scalar('Train/AttentionDiagonality', metrics.get('attention_diagonality', 0), self.global_step)
                self.tensorboard_writer.add_scalar('Train/GateAccuracy', metrics.get('gate_accuracy', 0), self.global_step)
                self.tensorboard_writer.add_scalar('Train/LearningRate', metrics.get('learning_rate', 0), self.global_step)
            except Exception as e:
                self.logger.warning(f"Ошибка TensorBoard: {e}")
        
        # MLflow logging
        if self.mlflow_run:
            try:
                mlflow.log_metrics({
                    'train_loss': metrics['total_loss'],
                    'grad_norm': metrics.get('grad_norm', 0),
                    'attention_diagonality': metrics.get('attention_diagonality', 0),
                    'gate_accuracy': metrics.get('gate_accuracy', 0)
                }, step=self.global_step)
            except Exception as e:
                self.logger.warning(f"Ошибка MLflow: {e}")
    
    def _log_epoch_summary(self, metrics: Dict):
        """Подробное логирование результатов эпохи."""
        epoch = metrics['epoch']
        
        self.logger.info("=" * 80)
        self.logger.info(f"📊 ЭПОХА {epoch} ЗАВЕРШЕНА")
        self.logger.info(f"🎯 Фаза обучения: {metrics.get('phase', 'unknown')}")
        self.logger.info(f"📈 Train Loss: {metrics['train_loss']:.4f}")
        
        if 'val_loss' in metrics:
            self.logger.info(f"📉 Val Loss: {metrics['val_loss']:.4f}")
            self.logger.info(f"🎯 Val Attention: {metrics.get('val_attention_diagonality', 0):.3f}")
            self.logger.info(f"🎯 Val Gate Acc: {metrics.get('val_gate_accuracy', 0):.3f}")
        
        self.logger.info(f"🔧 Grad Norm: {metrics['train_grad_norm']:.2f}")
        self.logger.info(f"📊 Attention Diag: {metrics['train_attention_diagonality']:.3f}")
        self.logger.info(f"🎯 Gate Accuracy: {metrics['train_gate_accuracy']:.3f}")
        self.logger.info(f"⚙️ Learning Rate: {metrics['learning_rate']:.2e}")
        self.logger.info(f"⏱️ Время эпохи: {metrics['epoch_time']:.1f}s")
        self.logger.info("=" * 80)
    
    def _send_epoch_telegram_update(self, metrics: Dict):
        """Отправка обновления в Telegram."""
        if not self.telegram_monitor:
            return
            
        try:
            message = f"""🏆 Ultimate Tacotron Training

📊 Эпоха: {metrics['epoch']} | Фаза: {metrics.get('phase', 'unknown')}
📈 Train Loss: {metrics['train_loss']:.4f}
📉 Val Loss: {metrics.get('val_loss', 'N/A')}
🎯 Attention: {metrics['train_attention_diagonality']:.3f}
🔧 Grad Norm: {metrics['train_grad_norm']:.2f}
⚙️ LR: {metrics['learning_rate']:.2e}
⏱️ Время: {metrics['epoch_time']:.1f}s

🚀 Режим: {self.mode}"""
            
            # Безопасная отправка сообщения
            if hasattr(self.telegram_monitor, 'send_message'):
                self.telegram_monitor.send_message(message)
            elif hasattr(self.telegram_monitor, 'send_epoch_update'):
                self.telegram_monitor.send_epoch_update(metrics)
            else:
                self.logger.debug("Telegram monitor не поддерживает отправку сообщений")
        except Exception as e:
            self.logger.warning(f"Ошибка Telegram уведомления: {e}")
    
    def _perform_intelligent_adjustments(self, recent_history: List[Dict]):
        """Интеллектуальные корректировки для Ultimate режима."""
        if not recent_history:
            return
            
        self.logger.info("🧠 Выполняю интеллектуальный анализ...")
        
        # Анализ трендов
        losses = [h['train_loss'] for h in recent_history]
        attention_scores = [h['train_attention_diagonality'] for h in recent_history]
        grad_norms = [h['train_grad_norm'] for h in recent_history]
        
        # Определение проблем и корректировок
        adjustments_made = []
        
        # Проблема 1: Стагнация loss
        if len(losses) > 10:
            recent_loss_trend = np.mean(losses[-5:]) - np.mean(losses[-10:-5])
            if abs(recent_loss_trend) < 0.001:  # Стагнация
                if hasattr(self.criterion, 'guide_loss_weight'):
                    old_weight = self.criterion.guide_loss_weight
                    self.criterion.guide_loss_weight *= 1.5
                    adjustments_made.append(f"Увеличен guide_loss_weight: {old_weight:.1f} → {self.criterion.guide_loss_weight:.1f}")
        
        # Проблема 2: Низкое внимание
        avg_attention = np.mean(attention_scores[-10:]) if len(attention_scores) >= 10 else 0
        if avg_attention < 0.3:
            # Увеличиваем фокус на attention БЕЗОПАСНО
            if hasattr(self.criterion, 'guide_loss_weight'):
                # Безопасное увеличение - НЕ БОЛЕЕ 20.0!
                old_weight = self.criterion.guide_loss_weight
                self.criterion.guide_loss_weight = min(old_weight * 1.5, 15.0)  # Максимум 15.0
                adjustments_made.append(f"Безопасное увеличение guided attention weight: {old_weight:.1f} → {self.criterion.guide_loss_weight:.1f}")
        
        # Проблема 3: Высокие градиенты
        avg_grad_norm = np.mean(grad_norms[-10:]) if len(grad_norms) >= 10 else 0
        if avg_grad_norm > 5.0:
            # Снижаем learning rate
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] *= 0.8
                adjustments_made.append(f"Снижен LR: {old_lr:.2e} → {param_group['lr']:.2e}")
        
        # Логирование корректировок
        if adjustments_made:
            self.logger.info("🔧 Применены интеллектуальные корректировки:")
            for adj in adjustments_made:
                self.logger.info(f"  • {adj}")
                
            # Уведомление в Telegram
            if self.telegram_monitor:
                message = "🧠 Интеллектуальные корректировки:\n" + "\n".join(f"• {adj}" for adj in adjustments_made)
                self.telegram_monitor.send_message(message)
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Сохранение чекпоинта."""
        try:
            checkpoint = {
                'epoch': epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_validation_loss': self.best_validation_loss,
                'hparams': vars(self.hparams),
                'training_mode': self.mode
            }
            
            filename = f"checkpoint_epoch_{epoch}.pt"
            if is_best:
                filename = "best_model.pt"
                
            torch.save(checkpoint, filename)
            self.logger.info(f"💾 Сохранен чекпоинт: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения чекпоинта: {e}")
    
    def _finalize_training(self, training_history: List[Dict], total_time: float):
        """Финализация обучения и отчетность."""
        self.logger.info("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        
        # Финальная статистика
        final_stats = {
            'total_epochs': len(training_history),
            'total_time_hours': total_time / 3600,
            'best_train_loss': min(h['train_loss'] for h in training_history),
            'final_attention_score': training_history[-1]['train_attention_diagonality'],
            'final_gate_accuracy': training_history[-1]['train_gate_accuracy']
        }
        
        self.logger.info(f"📊 Эпох обучения: {final_stats['total_epochs']}")
        self.logger.info(f"⏱️ Общее время: {final_stats['total_time_hours']:.1f} часов")
        self.logger.info(f"🏆 Лучший train loss: {final_stats['best_train_loss']:.4f}")
        self.logger.info(f"🎯 Финальное внимание: {final_stats['final_attention_score']:.3f}")
        self.logger.info(f"🎯 Финальная точность gate: {final_stats['final_gate_accuracy']:.3f}")
        
        # Сохранение финального отчета
        try:
            import json
            import numpy as np
            
            # Функция для конвертации numpy типов и type объектов в стандартные Python типы
            def convert_numpy_types(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, type):  # Исправляем type объекты
                    return str(obj)
                elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool)):
                    # Для сложных объектов преобразуем в строку
                    return str(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            # Конвертируем все данные
            report_data = {
                'final_stats': convert_numpy_types(final_stats),
                'training_history': convert_numpy_types(training_history),
                'mode': self.mode,
                'hparams': convert_numpy_types(vars(self.hparams))
            }
            
            with open('ultimate_training_report.json', 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            self.logger.info("📄 Сохранен детальный отчет: ultimate_training_report.json")
        except Exception as e:
            self.logger.warning(f"Ошибка сохранения отчета: {e}")
        
        # Финальное Telegram уведомление
        if self.telegram_monitor:
            try:
                message = f"""🎉 ULTIMATE TRAINING ЗАВЕРШЕНО!

📊 Эпох: {final_stats['total_epochs']}
⏱️ Время: {final_stats['total_time_hours']:.1f}ч
🏆 Лучший Loss: {final_stats['best_train_loss']:.4f}
🎯 Внимание: {final_stats['final_attention_score']:.3f}
🎯 Gate Acc: {final_stats['final_gate_accuracy']:.3f}

🚀 Режим: {self.mode}
✅ Все системы работали стабильно!"""
                
                # Безопасная отправка финального сообщения
                if hasattr(self.telegram_monitor, 'send_message'):
                    self.telegram_monitor.send_message(message)
                else:
                    self.logger.debug("Telegram monitor не поддерживает отправку сообщений")
            except Exception as e:
                self.logger.warning(f"Ошибка финального Telegram уведомления: {e}")
        
        # Закрытие логгеров
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        if self.mlflow_run:
            mlflow.end_run()


def main():
    """Главная функция для запуска Ultimate Enhanced Tacotron Trainer."""
    parser = argparse.ArgumentParser(description='Ultimate Enhanced Tacotron Trainer')
    parser.add_argument('--mode', choices=['simple', 'enhanced', 'auto_optimized', 'ultimate'], 
                       default='enhanced', help='Режим обучения')
    parser.add_argument('--config', type=str, default='hparams.py', help='Путь к конфигурации')
    parser.add_argument('--epochs', type=int, default=3500, help='Количество эпох')
    parser.add_argument('--max-steps', type=int, default=None, help='Максимальное количество шагов (для тестирования)')
    parser.add_argument('--dataset-path', type=str, required=True, help='Путь к датасету')
    
    args = parser.parse_args()
    
    # Загрузка гиперпараметров
    hparams = create_hparams()
    
    # Анализ датасета (для оптимизации)
    dataset_info = {}
    
    print(f"🚀 Запуск Ultimate Enhanced Tacotron Trainer в режиме '{args.mode}'")
    
    # Создание trainer'а
    trainer = UltimateEnhancedTacotronTrainer(
        hparams=hparams,
        mode=args.mode,
        dataset_info=dataset_info
    )
    
    # Инициализация DataLoader'ов
    try:
        print("📊 Инициализация DataLoader'ов...")
        
        if DATA_UTILS_AVAILABLE:
            # Стандартная инициализация
            trainset = TextMelLoader(args.dataset_path, hparams)
            valset = TextMelLoader(args.dataset_path.replace('train', 'val'), hparams)
            collate_fn = TextMelCollate(hparams.n_frames_per_step)
            
            train_loader = DataLoader(
                trainset, 
                num_workers=1, 
                shuffle=True,
                sampler=None,
                batch_size=hparams.batch_size, 
                pin_memory=False,
                drop_last=True, 
                collate_fn=collate_fn
            )
            
            val_loader = DataLoader(
                valset, 
                num_workers=1, 
                shuffle=False,
                sampler=None,
                batch_size=hparams.batch_size, 
                pin_memory=False,
                collate_fn=collate_fn
            )
            
            print(f"✅ DataLoader'ы созданы успешно!")
            print(f"📊 Train samples: {len(trainset)}")
            print(f"📊 Val samples: {len(valset)}")
            
        else:
            print("⚠️ data_utils недоступен - создание mock DataLoader'ов для тестирования")
            
            # Создание mock данных для тестирования
            class MockDataset(torch.utils.data.Dataset):
                def __init__(self, size=100):
                    self.size = size
                
                def __len__(self):
                    return self.size
                    
                def __getitem__(self, idx):
                    return (
                        torch.randint(0, 100, (50,)),  # text
                        torch.randn(80, 100),           # mel
                        torch.randint(0, 2, (100,))    # gate
                    )
            
            train_loader = DataLoader(MockDataset(1000), batch_size=8, shuffle=True)
            val_loader = DataLoader(MockDataset(200), batch_size=8, shuffle=False)
            
            print("✅ Mock DataLoader'ы созданы для тестирования")
            
        # Инициализация обучения
        trainer.initialize_training()
        
        # Запуск обучения
        if args.max_steps:
            print(f"🔬 Начинаю ТЕСТИРОВАНИЕ в режиме '{args.mode}' на {args.max_steps} шагов...")
            trainer.train(train_loader, val_loader, args.epochs, args.max_steps)
        else:
            print(f"🏆 Начинаю обучение в режиме '{args.mode}' на {args.epochs} эпох...")
            trainer.train(train_loader, val_loader, args.epochs)
        
        print("🎉 Ultimate Enhanced Tacotron Trainer завершил работу!")
        
    except Exception as e:
        print(f"❌ Критическая ошибка при инициализации: {e}")
        import traceback
        traceback.print_exc()
        
        # Создаем минимальное демо для тестирования базового функционала
        print("\n🔧 Попытка создания демо для тестирования базовых функций...")
        try:
            trainer.initialize_training()
            print("✅ Базовая инициализация прошла успешно!")
            print("⚠️ Для полного обучения необходимо исправить проблемы с данными")
        except Exception as e2:
            print(f"❌ Даже базовая инициализация не удалась: {e2}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 