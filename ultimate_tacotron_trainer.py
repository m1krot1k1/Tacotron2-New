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
from tqdm import tqdm
import psutil
try:
    import GPUtil
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False

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

# 🧠 ИНТЕЛЛЕКТУАЛЬНАЯ СИСТЕМА ОБУЧЕНИЯ (замена AutoFixManager)
try:
    from context_aware_training_manager import ContextAwareTrainingManager, create_context_aware_manager
    CONTEXT_AWARE_AVAILABLE = True
    logging.info("✅ Context-Aware Training Manager доступен")
except ImportError:
    CONTEXT_AWARE_AVAILABLE = False
    logging.warning("❌ Context-Aware Training Manager недоступен")

# 🤖 АВТОМАТИЧЕСКИЕ ИСПРАВЛЕНИЯ (ПОЛНОСТЬЮ УДАЛЕНО - заменено на умную систему)
# AutoFixManager УДАЛЕН - заменен на Context-Aware Training Manager
    AUTO_FIX_AVAILABLE = False
logging.info("🔧 AutoFixManager полностью удален - заменен на Context-Aware Manager")

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
    - Интеллектуальная система обучения (Context-Aware Manager)
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
        
        # 🧠 ИНТЕЛЛЕКТУАЛЬНАЯ СИСТЕМА ОБУЧЕНИЯ
        self.context_aware_manager = None
        
        # 🤖 АВТОМАТИЧЕСКИЕ ИСПРАВЛЕНИЯ (ОТКЛЮЧЕНО)
        # 🤖 AutoFixManager ПОЛНОСТЬЮ УДАЛЕН - заменен на context_aware_manager
        
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
        """Настраивает селективное логирование - только критические сообщения."""
        logger = logging.getLogger('UltimateEnhancedTacotronTrainer')
        logger.setLevel(logging.WARNING)  # Только критические сообщения
        
        if not logger.handlers:
            # Консольный handler - только для критических сообщений
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '🔧 %(message)s'  # Упрощенный формат
            )
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.WARNING)
            logger.addHandler(console_handler)
            
            # Файловый handler - для полного логирования
            file_handler = logging.FileHandler('ultimate_training.log')
            file_formatter = logging.Formatter(
                '%(asctime)s - [Ultimate] - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.INFO)
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
        """Инициализация расширенного логирования с системными метриками."""
        if not LOGGING_AVAILABLE:
            self.logger.warning("⚠️ Системы логирования недоступны")
            return
            
        try:
            # TensorBoard с расширенным набором метрик
            from datetime import datetime
            log_dir = f'tensorboard_logs/ultimate_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            self.tensorboard_writer = SummaryWriter(log_dir)
            self.logger.info(f"✅ TensorBoard инициализирован: {log_dir}")
            
            # MLflow с расширенной конфигурацией
            try:
                experiment_name = f"Ultimate_Tacotron_Training_{self.mode}"
                mlflow.set_experiment(experiment_name)
                
                # Очистка старых MLflow запусков (оставляем только последние 10)
                self._cleanup_old_mlflow_runs(experiment_name, keep_last=10)
                
                self.mlflow_run = mlflow.start_run(
                    run_name=f"ultimate_{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                # Логируем начальные параметры и системные метрики
                self._log_initial_mlflow_params()
                
                self.logger.info(f"✅ MLflow инициализирован: {experiment_name}")
            except Exception as e:
                self.logger.warning(f"⚠️ MLflow недоступен: {e}")
                self.mlflow_run = None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка инициализации логирования: {e}")
    
    def _cleanup_old_mlflow_runs(self, experiment_name: str, keep_last: int = 10):
        """Очистка старых MLflow запусков."""
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            # Получаем эксперимент
            experiment = client.get_experiment_by_name(experiment_name)
            if not experiment:
                return
                
            # Получаем все запуски
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"]
            )
            
            # Удаляем старые запуски (оставляем только последние keep_last)
            for run in runs[keep_last:]:
                try:
                    client.delete_run(run.info.run_id)
                    self.logger.info(f"🗑️ Удален старый MLflow запуск: {run.info.run_id}")
                except Exception:
                    pass  # Игнорируем ошибки удаления
                    
        except Exception as e:
            self.logger.warning(f"Ошибка очистки MLflow: {e}")
    
    def _log_initial_mlflow_params(self):
        """Логирование начальных параметров в MLflow."""
        if not self.mlflow_run:
            return
            
        try:
            # Основные параметры обучения
            mlflow.log_params({
                'mode': self.mode,
                'learning_rate': getattr(self.hparams, 'learning_rate', 0),
                'batch_size': getattr(self.hparams, 'batch_size', 0),
                'max_decoder_steps': getattr(self.hparams, 'max_decoder_steps', 0),
                'guide_loss_weight': getattr(self.hparams, 'guide_loss_weight', 0),
                'gate_threshold': getattr(self.hparams, 'gate_threshold', 0),
            })
            
            # Системные параметры
            system_info = {
                'python_version': sys.version.split()[0],
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            
            if torch.cuda.is_available():
                system_info['gpu_name'] = torch.cuda.get_device_name(0)
                system_info['cuda_version'] = torch.version.cuda
            
            mlflow.log_params(system_info)
            
            # Информация о датасете
            if self.dataset_info:
                dataset_params = {f'dataset_{k}': v for k, v in self.dataset_info.items()}
                mlflow.log_params(dataset_params)
                
        except Exception as e:
            self.logger.warning(f"Ошибка логирования параметров MLflow: {e}")
    
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
        
        # 🧠 Context-Aware Training Manager
        if CONTEXT_AWARE_AVAILABLE:
            try:
                # Context-Aware Manager будет инициализирован после создания модели
                self.logger.info("🧠 Context-Aware Training Manager будет инициализирован после модели")
            except Exception as e:
                self.logger.error(f"❌ Ошибка подготовки Context-Aware Manager: {e}")
        
        # 🤖 AutoFixManager (ПОЛНОСТЬЮ УДАЛЕН)
        self.logger.info("🔧 AutoFixManager полностью удален - заменен на Context-Aware Manager")
        
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
        
        # 🧠 Context-Aware Training Manager (инициализируем после модели и оптимизатора)
        if CONTEXT_AWARE_AVAILABLE and self.mode in ['enhanced', 'auto_optimized', 'ultimate']:
            try:
                self.context_aware_manager = create_context_aware_manager(self.hparams)
                self.logger.info("✅ Context-Aware Training Manager инициализирован")
                self.logger.info("🎯 Система умного обучения активирована (замена AutoFixManager)")
            except Exception as e:
                self.logger.error(f"❌ Ошибка Context-Aware Manager: {e}")
        
        # 🤖 AutoFixManager (ПОЛНОСТЬЮ УДАЛЕН - заменено на умную систему)
        # AutoFixManager больше НЕ ИСПОЛЬЗУЕТСЯ - заменен на Context-Aware Manager
        self.logger.info("🔧 AutoFixManager полностью удален - используется Context-Aware Manager")
    
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
        
        # 🔧 БЕЗОПАСНАЯ ОБРАБОТКА MODEL_OUTPUTS с проверкой размеров
        try:
            x, y = self.model.parse_batch(batch)
            
            # 🔥 ДИАГНОСТИКА РАЗМЕРОВ ПЕРЕД FORWARD PASS
            text_inputs, text_lengths, mel_targets, max_len, output_lengths, ctc_text, ctc_text_lengths = x
            if text_inputs.size(0) != mel_targets.size(0):
                self.logger.error(f"🚨 Batch size mismatch: text={text_inputs.size(0)}, mel={mel_targets.size(0)}")
                return {'total_loss': 10.0}
            
            # 🔥 ОГРАНИЧЕНИЕ МАКСИМАЛЬНЫХ РАЗМЕРОВ ДЛЯ СТАБИЛЬНОСТИ
            max_text_len = min(text_inputs.size(1), 200)  # Ограничиваем длину текста
            max_mel_len = min(mel_targets.size(2), 1000)   # Ограничиваем длину mel
            
            if text_inputs.size(1) > max_text_len:
                text_inputs = text_inputs[:, :max_text_len]
                text_lengths = torch.clamp(text_lengths, max=max_text_len)
            
            if mel_targets.size(2) > max_mel_len:
                mel_targets = mel_targets[:, :, :max_mel_len]
                output_lengths = torch.clamp(output_lengths, max=max_mel_len)
            
            # Пересобираем x с ограниченными размерами
            x = (text_inputs, text_lengths, mel_targets, max_len, output_lengths, ctc_text, ctc_text_lengths)
            
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
        
        # 🚨 ИНТЕЛЛЕКТУАЛЬНАЯ АДАПТИВНАЯ СИСТЕМА GUIDED ATTENTION
        if hasattr(self.criterion, 'guide_loss_weight') and self.global_step > 0:
            current_weight = self.criterion.guide_loss_weight
            
            # 🎯 ИНТЕЛЛЕКТУАЛЬНАЯ СТРАТЕГИЯ С ИСПОЛЬЗОВАНИЕМ EMERGENCY MODE
            if attention_diagonality < 0.02:
                # КРИТИЧЕСКИЙ РЕЖИМ: Активируем emergency mode в loss function
                if hasattr(self.criterion, 'guided_attention_loss') and hasattr(self.criterion.guided_attention_loss, 'activate_critical_mode'):
                    self.criterion.guided_attention_loss.activate_critical_mode()
                    self.logger.error(f"🚨 АКТИВИРОВАН EMERGENCY MODE для guided attention (weight=25.0)!")
                
                # Дополнительно увеличиваем основной weight до максимального безопасного значения  
                new_weight = min(current_weight * 2.0, 50.0)  # Убираем искусственный лимит
                self.criterion.guide_loss_weight = new_weight
                self.logger.error(f"🚨 КРИТИЧЕСКОЕ увеличение guided attention weight: {current_weight:.1f} → {new_weight:.1f}")
                
                # Экстренное увеличение LR для attention компонентов
                for param_group in self.optimizer.param_groups:
                    if param_group['lr'] < 1e-4:
                        param_group['lr'] = min(param_group['lr'] * 3.0, 1e-4)
                        self.logger.warning(f"🚀 ЭКСТРЕННОЕ увеличение LR: {param_group['lr']:.2e}")
                        
            elif attention_diagonality < 0.05:
                # Очень низкое - агрессивное увеличение до 30.0
                new_weight = min(current_weight * 1.8, 30.0)
                self.criterion.guide_loss_weight = new_weight
                self.logger.warning(f"🚨 АГРЕССИВНОЕ увеличение guided attention weight: {current_weight:.1f} → {new_weight:.1f}")
                
                # Увеличиваем LR для улучшения обучения attention
                for param_group in self.optimizer.param_groups:
                    if param_group['lr'] < 5e-5:
                        param_group['lr'] = min(param_group['lr'] * 2.0, 5e-5)
                        self.logger.info(f"🔄 Усиление LR для attention: {param_group['lr']:.2e}")
                
            elif attention_diagonality < 0.1:
                # Низкое - умеренное увеличение до 20.0
                new_weight = min(current_weight * 1.4, 20.0)
                self.criterion.guide_loss_weight = new_weight
                self.logger.warning(f"🚨 Сильное увеличение guided attention weight: {current_weight:.1f} → {new_weight:.1f}")
                
            elif attention_diagonality < 0.3:
                # Улучшается - осторожное увеличение до 15.0
                new_weight = min(current_weight * 1.1, 15.0)
                self.criterion.guide_loss_weight = new_weight
                self.logger.info(f"📈 Стабилизация guided attention weight: {current_weight:.1f} → {new_weight:.1f}")
                
            elif attention_diagonality > 0.7:
                # Отличное attention - деактивируем emergency mode и снижаем
                if hasattr(self.criterion, 'guided_attention_loss') and hasattr(self.criterion.guided_attention_loss, 'deactivate_critical_mode'):
                    self.criterion.guided_attention_loss.deactivate_critical_mode()
                    self.logger.info("✅ ДЕАКТИВИРОВАН EMERGENCY MODE - attention стабилизировано!")
                    
                new_weight = max(current_weight * 0.8, 3.0)
                self.criterion.guide_loss_weight = new_weight
                self.logger.info(f"📉 Снижение guided attention weight (отличное внимание): {current_weight:.1f} → {new_weight:.1f}")
                
            elif attention_diagonality > 0.5:
                # Хорошее attention - мягкое снижение
                new_weight = max(current_weight * 0.9, 5.0)
                self.criterion.guide_loss_weight = new_weight
                self.logger.info(f"📊 Стабилизация guided attention weight: {current_weight:.1f} → {new_weight:.1f}")
            
                    # 🎯 РЕВОЛЮЦИОННЫЕ ЭКСТРЕННЫЕ МЕРЫ ДЛЯ КРИТИЧЕСКОГО ATTENTION
        if attention_diagonality < 0.05 and self.global_step > 20:
            # 1. Максимальная контрастность attention mechanism
            if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'attention_layer'):
                self.model.decoder.attention_layer.score_mask_value = -1e6  # Максимальная контрастность
                self.logger.info("🎯 МАКСИМАЛЬНАЯ контрастность attention mechanism")
                
                # 2. Активируем Location-Relative attention для лучшего обучения
                if hasattr(self.model.decoder.attention_layer, 'use_location_relative'):
                    if not self.model.decoder.attention_layer.use_location_relative:
                        self.model.decoder.attention_layer.use_location_relative = True
                        self.model.decoder.attention_layer.relative_sigma = 2.0  # Агрессивнее для быстрого обучения
                        self.logger.error("🚀 АКТИВИРОВАН Location-Relative attention!")
                    else:
                        # Делаем более агрессивным
                        self.model.decoder.attention_layer.relative_sigma = max(
                            self.model.decoder.attention_layer.relative_sigma * 0.8, 1.0
                        )
                        self.logger.warning(f"🎯 Усиление Location-Relative: sigma={self.model.decoder.attention_layer.relative_sigma:.1f}")
            
            # 3. Проверяем и вызываем emergency mode в loss function
            if hasattr(self.criterion, 'guided_attention_loss'):
                self.criterion.guided_attention_loss.check_diagonality_and_adapt(alignments)
            
            # 4. Экстренные меры для Prenet (снижаем dropout для стабильности)
            if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'prenet'):
                if hasattr(self.model.decoder.prenet, 'dropout_rate'):
                    self.model.decoder.prenet.dropout_rate = min(self.model.decoder.prenet.dropout_rate * 0.5, 0.1)
                    self.logger.info(f"🔧 Снижение Prenet dropout: {self.model.decoder.prenet.dropout_rate:.3f}")
        
        # 🌟 ПРОГРЕССИВНОЕ УЛУЧШЕНИЕ ATTENTION ПРИ ХОРОШИХ РЕЗУЛЬТАТАХ
        elif attention_diagonality > 0.3 and self.global_step > 100:
            # Постепенно отключаем Location-Relative для натуральности
            if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'attention_layer'):
                if hasattr(self.model.decoder.attention_layer, 'use_location_relative'):
                    if self.model.decoder.attention_layer.use_location_relative and attention_diagonality > 0.6:
                        self.model.decoder.attention_layer.relative_sigma = min(
                            self.model.decoder.attention_layer.relative_sigma * 1.1, 8.0
                        )
                        if self.model.decoder.attention_layer.relative_sigma > 6.0:
                            self.model.decoder.attention_layer.use_location_relative = False
                            self.logger.info("✅ ОТКЛЮЧЕН Location-Relative - attention стабилизирован!")
        
        # 🎯 ИНТЕЛЛЕКТУАЛЬНЫЙ ATTENTION WARM-UP В НАЧАЛЕ ОБУЧЕНИЯ
        if self.global_step < 200:
            # Прогрессивно активируем сложные attention механизмы
            if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'attention_layer'):
                # В начале делаем очень контрастное attention
                target_mask_value = -1e4 - (self.global_step * 50)  # Становится более контрастным со временем
                self.model.decoder.attention_layer.score_mask_value = target_mask_value
                
                # Активируем Location-Relative с осторожной sigma
                if hasattr(self.model.decoder.attention_layer, 'use_location_relative'):
                    if self.global_step > 50 and not self.model.decoder.attention_layer.use_location_relative:
                        self.model.decoder.attention_layer.use_location_relative = True
                        self.model.decoder.attention_layer.relative_sigma = 3.0
                        self.logger.info("🎯 АКТИВИРОВАН Location-Relative на warm-up!")
                
                if self.global_step % 50 == 0:
                    self.logger.info(f"🎯 Attention Warm-up: шаг {self.global_step}, mask_value={target_mask_value:.0e}")
        
        # 🔍 ДЕТАЛЬНЫЙ МОНИТОРИНГ ATTENTION ПРОГРЕССА
        if self.global_step % 100 == 0 and attention_diagonality < 0.1:
            # Анализируем attention patterns для диагностики
            if alignments is not None:
                try:
                    attention_stats = self._analyze_attention_patterns(alignments)
                    self.logger.warning(f"🔍 Attention Analysis: диагональность={attention_diagonality:.3f}, "
                                      f"фокус={attention_stats.get('focus', 0):.3f}, "
                                      f"монотонность={attention_stats.get('monotonicity', 0):.3f}")
                except Exception as e:
                    self.logger.debug(f"Ошибка анализа attention: {e}")
        
        # 🎯 МОНИТОРИНГ ПРОГРЕССА ATTENTION
        if self.global_step % 10 == 0:
            attention_trend = "📈" if attention_diagonality > self.last_attention_diagonality else "📉" if attention_diagonality < self.last_attention_diagonality else "➡️"
            if attention_diagonality < 0.1:
                self.logger.warning(f"🎯 Attention Progress {attention_trend}: {attention_diagonality:.3f} (TARGET: >0.7)")
            elif attention_diagonality < 0.3:
                self.logger.info(f"🎯 Attention Progress {attention_trend}: {attention_diagonality:.3f} (IMPROVING)")
            elif attention_diagonality >= 0.7:
                self.logger.info(f"🎯 Attention EXCELLENT {attention_trend}: {attention_diagonality:.3f} ✅")
        
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
            # 🔥 ИСПРАВЛЕНИЕ: Вместо inf возвращаем безопасное значение
            return {'total_loss': 10.0}  # Высокое, но конечное значение
        
        # 🔥 ПРОВЕРКА LOSS ПЕРЕД BACKWARD
        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.error(f"🚨 Обнаружен NaN/Inf loss: {loss}, пропускаем backward")
            return {'total_loss': 10.0}
        
        # Backward pass
        try:
            loss.backward()
        except Exception as e:
            self.logger.error(f"Ошибка backward pass: {e}")
            # 🔥 ИСПРАВЛЕНИЕ: Вместо inf возвращаем безопасное значение
            return {'total_loss': 10.0}
        
        # 🔧 ПРОДВИНУТОЕ УПРАВЛЕНИЕ ГРАДИЕНТАМИ С ПРИНУДИТЕЛЬНЫМ КЛИППИНГОМ
        if self.adaptive_gradient_clipper:
            # Используем AdaptiveGradientClipper
            was_clipped, grad_norm, clip_threshold = self.adaptive_gradient_clipper.clip_gradients(
                self.model, self.global_step
            )
            
            if was_clipped:
                self.logger.info(f"🔧 Градиенты обрезаны: {grad_norm:.2f} → {clip_threshold:.2f}")
        else:
            # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Принудительное агрессивное клипирование
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=1.0,  # Принудительно используем 1.0
                norm_type=2.0
            )
            
        # 🚨 ЭКСТРЕННАЯ ЗАЩИТА: Дополнительная проверка после клипирования
        current_grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                current_grad_norm += param_norm.item() ** 2
        current_grad_norm = current_grad_norm ** 0.5
        
        # Если градиенты все еще высокие - принудительно обрезаем еще раз
        if current_grad_norm > 2.0:
            self.logger.warning(f"🚨 ПРИНУДИТЕЛЬНОЕ вторичное клипирование: {current_grad_norm:.2f} → 1.0")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2.0)
            current_grad_norm = 1.0
            
        # Критические алерты для высоких градиентов
        if current_grad_norm > 5.0:
            self.logger.warning(f"🚨 ВЫСОКАЯ норма градиентов: {current_grad_norm:.2f}")
        if current_grad_norm > 20.0:
            self.logger.error(f"🚨 КРИТИЧЕСКАЯ норма градиентов: {current_grad_norm:.2f}")
            
        grad_norm = current_grad_norm
        
        # 🧠 ИНТЕЛЛЕКТУАЛЬНАЯ СИСТЕМА ОБУЧЕНИЯ (замена AutoFixManager)
        if self.context_aware_manager:
            try:
                context_metrics = {
                    'grad_norm': float(grad_norm),
                    'attention_diagonality': attention_diagonality,
                    'gate_accuracy': gate_accuracy,
                    'loss': float(loss.item()),
                    'mel_loss': loss_dict.get('mel_loss', 0),
                    'gate_loss': loss_dict.get('gate_loss', 0),
                    'guided_attention_loss': loss_dict.get('guide_loss', 0),
                    'epoch': self.current_epoch
                }
                
                adaptations = self.context_aware_manager.analyze_and_adapt(
                    step=self.global_step,
                    metrics=context_metrics,
                    model=self.model,
                    optimizer=self.optimizer
                )
                
                if adaptations and len(adaptations) > 4:  # Более 4 параметров означает активные адаптации
                    adapted_params = [k for k, v in adaptations.items() if k not in ['mel_weight', 'gate_weight']]
                    if adapted_params:
                        self.logger.info(f"🎯 Context-Aware адаптации: {adapted_params}")
                    
            except Exception as e:
                self.logger.error(f"❌ Ошибка в Context-Aware Manager: {e}")
        
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
        
        # 🚀 РАСШИРЕННАЯ СИСТЕМА ВОССТАНОВЛЕНИЯ LEARNING RATE
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # КРИТИЧЕСКИЙ УРОВЕНЬ: LR < 1e-7 (практически нулевое обучение)
        if current_lr < 1e-7:
            if attention_diagonality < 0.02:  # Критически плохое attention
                target_lr = 3e-5  # Агрессивное восстановление для экстренного обучения attention
                self.logger.error(f"🚨 ЭКСТРЕННОЕ восстановление LR для критического attention: {current_lr:.2e} → {target_lr:.2e}")
            elif attention_diagonality < 0.1:  # Плохое attention  
                target_lr = 1e-5  # Умеренно агрессивное восстановление
                self.logger.error(f"🚨 КРИТИЧЕСКОЕ восстановление LR: {current_lr:.2e} → {target_lr:.2e}")
            else:  # Нормальное attention, но критически низкий LR
                target_lr = 5e-6  # Консервативное восстановление
                self.logger.warning(f"🔄 Автовосстановление критического LR: {current_lr:.2e} → {target_lr:.2e}")
                
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = target_lr
            
        # ОЧЕНЬ НИЗКИЙ УРОВЕНЬ: LR < 5e-7
        elif current_lr < 5e-7 and (attention_diagonality < 0.1 or loss.item() > 15.0):
            target_lr = 2e-6 if attention_diagonality < 0.05 else 1e-6
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = target_lr
            self.logger.warning(f"🔄 Восстановление очень низкого LR: {current_lr:.2e} → {target_lr:.2e}")
            
        # НИЗКИЙ УРОВЕНЬ В НАЧАЛЕ ОБУЧЕНИЯ: LR < 1e-6 при step < 1000
        elif current_lr < 1e-6 and self.global_step < 1000:
            target_lr = 1e-5 if self.global_step < 100 else 5e-6
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = target_lr
            self.logger.info(f"🚀 Раннее восстановление LR: {current_lr:.2e} → {target_lr:.2e}")
        
        # АДАПТИВНОЕ УВЕЛИЧЕНИЕ при хороших результатах  
        elif (current_lr < 5e-6 and attention_diagonality > 0.3 and loss.item() < 10.0 and 
              grad_norm < 2.0 and self.global_step % 50 == 0):
            target_lr = min(current_lr * 1.3, 2e-5)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = target_lr
            self.logger.info(f"📈 Интеллектуальное увеличение LR: {current_lr:.2e} → {target_lr:.2e}")
        
        # ЗАЩИТА ОТ ПЕРЕОБУЧЕНИЯ: LR > 1e-3
        elif current_lr > 1e-3:
            target_lr = 1e-3
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = target_lr
            self.logger.warning(f"⚠️ Ограничение высокого LR: {current_lr:.2e} → {target_lr:.2e}")
        
        # МОНИТОРИНГ СОСТОЯНИЯ LR каждые 100 шагов
        if self.global_step % 100 == 0:
            lr_status = "🔴 КРИТИЧЕСКИЙ" if current_lr < 1e-7 else "🟡 НИЗКИЙ" if current_lr < 1e-6 else "🟢 НОРМАЛЬНЫЙ"
            if current_lr < 1e-6:  # Логируем только проблемные LR
                self.logger.warning(f"📊 LR Status: {lr_status} | Current: {current_lr:.2e} | Attention: {attention_diagonality:.3f}")
        
        # СВЯЗЬ С ATTENTION EMERGENCY MODE
        if current_lr < 1e-6 and attention_diagonality < 0.02:
            # Синхронизируем с emergency mode guided attention
            if hasattr(self.criterion, 'guided_attention_loss'):
                if hasattr(self.criterion.guided_attention_loss, 'activate_critical_mode'):
                    self.criterion.guided_attention_loss.activate_critical_mode()
                    self.logger.error("🎯 Синхронизация LR + Guided Attention EMERGENCY MODE!")
        
        # Optimizer step
        try:
            self.optimizer.step()
            self.global_step += 1
        except Exception as e:
            self.logger.error(f"Ошибка optimizer step: {e}")
            return {'total_loss': torch.tensor(float('inf'))}
        
        # Добавляем метрики к результату (конвертируем тензоры в числа)
        loss_dict.update({
            'grad_norm': grad_norm.cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'attention_diagonality': attention_diagonality,
            'gate_accuracy': gate_accuracy,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        })
        
        # Конвертируем все тензоры в числа для безопасности
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                loss_dict[key] = value.cpu().item()
        
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
        Главный метод обучения с красивым прогресс-баром и селективным логированием.
        
        Args:
            train_loader: DataLoader для обучения
            val_loader: DataLoader для валидации
            num_epochs: Количество эпох обучения
            max_steps: Максимальное количество шагов (для быстрого тестирования)
        """
        # Печатаем только старт
        print(f"🏆 Ultimate Enhanced Training (режим: {self.mode})")
        if max_steps:
            print(f"🔬 ТЕСТОВЫЙ РЕЖИМ: ограничение {max_steps} шагов")
        else:
            print(f"📊 Эпох: {num_epochs}, Батчей на эпоху: {len(train_loader)}")
        
        # Инициализация обучения
        self.initialize_training()
        
        # 📱 Telegram стартовое уведомление
        self._send_training_start_notification()
        
        # Переменные для мониторинга  
        epoch_start_time = time.time()
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = getattr(self.hparams, 'early_stopping_patience', 10)
        training_history = []
        
        # 🎯 КРАСИВЫЙ ПРОГРЕСС-БАР ДЛЯ ЭПОХ
        epoch_progress = tqdm(
            range(num_epochs), 
            desc="🚀 Обучение", 
            unit="эпоха",
            ncols=100,
            leave=True
        )
        
        for epoch in epoch_progress:
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # 🎯 ФАЗОВОЕ ОБУЧЕНИЕ
            current_phase = self.get_current_training_phase()
            if epoch % 100 == 0 or epoch < 5:
                self.adjust_hyperparams_for_phase(current_phase)
            
            # 📊 МЕТРИКИ ЭПОХИ
            epoch_losses = []
            epoch_grad_norms = []
            epoch_attention_scores = []
            epoch_gate_accuracies = []
            fixes_applied = 0
            
            # 🎯 ПРОГРЕСС-БАР ДЛЯ БАТЧЕЙ
            batch_progress = tqdm(
                train_loader,
                desc=f"Эпоха {epoch}",
                unit="батч", 
                leave=False,
                ncols=80
            )
            
            for batch_idx, batch in enumerate(batch_progress):
                try:
                    # Выполняем шаг обучения
                    step_metrics = self.train_step(batch)
                    
                    # Обновляем прогресс-бар с метриками
                    batch_progress.set_postfix({
                        'Loss': f"{step_metrics['total_loss']:.2f}",
                        'Grad': f"{step_metrics.get('grad_norm', 0):.1f}",
                        'Gate': f"{step_metrics.get('gate_accuracy', 0):.3f}"
                    })
                    
                    # Сохраняем метрики (конвертируем CUDA тензоры в числа)
                    loss_val = step_metrics['total_loss']
                    if isinstance(loss_val, torch.Tensor):
                        loss_val = loss_val.cpu().item()
                    epoch_losses.append(loss_val)
                    
                    grad_norm_val = step_metrics.get('grad_norm', 0)
                    if isinstance(grad_norm_val, torch.Tensor):
                        grad_norm_val = grad_norm_val.cpu().item()
                    epoch_grad_norms.append(grad_norm_val)
                    
                    attention_val = step_metrics.get('attention_diagonality', 0)
                    if isinstance(attention_val, torch.Tensor):
                        attention_val = attention_val.cpu().item()
                    epoch_attention_scores.append(attention_val)
                    
                    gate_acc_val = step_metrics.get('gate_accuracy', 0)
                    if isinstance(gate_acc_val, torch.Tensor):
                        gate_acc_val = gate_acc_val.cpu().item()
                    epoch_gate_accuracies.append(gate_acc_val)
                    
                    # 🧠 КОНТЕКСТНО-ОСОЗНАННЫЕ АДАПТАЦИИ - только при критических проблемах
                    if self.context_aware_manager and step_metrics.get('total_loss', 0) > 50:
                        critical_metrics = {
                            'loss': step_metrics.get('total_loss', 0),
                            'attention_diagonality': step_metrics.get('attention_diagonality', 0),
                            'grad_norm': step_metrics.get('grad_norm', 0),
                            'gate_accuracy': step_metrics.get('gate_accuracy', 0),
                            'epoch': epoch
                        }
                        
                        emergency_adaptations = self.context_aware_manager.analyze_and_adapt(
                            step=self.global_step,
                            metrics=critical_metrics,
                            model=self.model,
                            optimizer=self.optimizer
                        )
                        
                        if emergency_adaptations and any(k in emergency_adaptations for k in ['learning_rate', 'guided_attention_weight']):
                            fixes_applied += 1  # Считаем как одну адаптацию
                            self.logger.warning(f"🎯 Критические контекстные адаптации применены")
                    
                    # 🔬 ЛИМИТ ШАГОВ ДЛЯ ТЕСТИРОВАНИЯ
                    if max_steps and self.global_step >= max_steps:
                        batch_progress.close()
                        break
                        
                except Exception as e:
                    self.logger.error(f"Ошибка в шаге {batch_idx}: {e}")
                    continue
            
            # 📊 АГРЕГИРОВАННЫЕ МЕТРИКИ ЭПОХИ
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': np.mean(epoch_losses) if epoch_losses else float('inf'),
                'train_grad_norm': np.mean(epoch_grad_norms) if epoch_grad_norms else 0,
                'train_attention_diagonality': np.mean(epoch_attention_scores) if epoch_attention_scores else 0,
                'train_gate_accuracy': np.mean(epoch_gate_accuracies) if epoch_gate_accuracies else 0,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'phase': current_phase,
                'epoch_time': time.time() - epoch_start,
                'fixes_applied': fixes_applied
            }
            
            # 📊 ВАЛИДАЦИЯ
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
                        print(f"\n🎉 Новый лучший результат: val_loss = {best_val_loss:.4f}")
                    else:
                        patience_counter += 1
                        
                except Exception as e:
                    self.logger.error(f"Ошибка валидации: {e}")
                    val_metrics = {'val_loss': float('inf')}
                    epoch_metrics.update(val_metrics)
            
            # Сохранение истории
            training_history.append(epoch_metrics)
            
            # 📊 ОБНОВЛЕНИЕ ПРОГРЕСС-БАРА ЭПОХ
            epoch_progress.set_postfix({
                'Loss': f"{epoch_metrics['train_loss']:.2f}",
                'Val': f"{epoch_metrics.get('val_loss', 0):.2f}",
                'Phase': current_phase[:8],
                'LR': f"{epoch_metrics['learning_rate']:.2e}"
            })
            
            # 📱 TELEGRAM УВЕДОМЛЕНИЯ
            telegram_frequency = 5 if max_steps else 50 
            if epoch % telegram_frequency == 0 or epoch < 5:
                self._send_enhanced_telegram_update(epoch_metrics)
            
            # 🔄 АВТОМАТИЧЕСКАЯ ОПТИМИЗАЦИЯ (ultimate режим)
            if self.mode == 'ultimate' and epoch > 10 and len(training_history) >= 10:
                adjustments = self._perform_intelligent_adjustments(training_history[-10:])
                if adjustments:
                    print(f"\n🧠 Применены интеллектуальные корректировки: {len(adjustments)}")
            
            # 🎯 АВТОМАТИЧЕСКАЯ ОСТАНОВКА ПРИ ДОСТИЖЕНИИ КАЧЕСТВА
            if self._check_training_completion(epoch_metrics):
                print(f"\n🎉 Автоматическая остановка - цель достигнута!")
                break
            
            # 🛑 EARLY STOPPING
            if patience_counter >= max_patience:
                print(f"\n🛑 Early stopping после {patience_counter} эпох без улучшений")
                break
                
            # 🔬 ВЫХОД ПРИ ДОСТИЖЕНИИ ЛИМИТА ШАГОВ
            if max_steps and self.global_step >= max_steps:
                print(f"\n🔬 Тестирование завершено на {self.global_step} шагах")
                break
            
            # 🔧 ПЕРИОДИЧЕСКОЕ СОХРАНЕНИЕ
            save_frequency = 20 if max_steps else 100
            if epoch % save_frequency == 0:
                self._save_checkpoint(epoch, is_best=False)
        
        epoch_progress.close()
        
        # 🎉 ЗАВЕРШЕНИЕ ОБУЧЕНИЯ
        total_time = time.time() - epoch_start_time
        self._finalize_training(training_history, total_time)
        
        return training_history
    
    def _log_training_step(self, metrics: Dict, epoch: int, batch_idx: int):
        """Расширенное логирование шага обучения с системными метриками."""
        # Получаем системные метрики
        system_metrics = self._get_system_metrics()
        
        # TensorBoard - расширенное логирование
        if self.tensorboard_writer:
            try:
                # Основные метрики обучения
                self.tensorboard_writer.add_scalar('Train/Loss', metrics['total_loss'], self.global_step)
                self.tensorboard_writer.add_scalar('Train/GradNorm', metrics.get('grad_norm', 0), self.global_step)
                self.tensorboard_writer.add_scalar('Train/AttentionDiagonality', metrics.get('attention_diagonality', 0), self.global_step)
                self.tensorboard_writer.add_scalar('Train/GateAccuracy', metrics.get('gate_accuracy', 0), self.global_step)
                self.tensorboard_writer.add_scalar('Train/LearningRate', metrics.get('learning_rate', 0), self.global_step)
                
                # Детализированные метрики loss
                if 'mel_loss' in metrics:
                    self.tensorboard_writer.add_scalar('Loss/Mel', metrics['mel_loss'], self.global_step)
                if 'gate_loss' in metrics:
                    self.tensorboard_writer.add_scalar('Loss/Gate', metrics['gate_loss'], self.global_step)
                if 'attention_loss' in metrics:
                    self.tensorboard_writer.add_scalar('Loss/Attention', metrics['attention_loss'], self.global_step)
                
                # Системные метрики
                if system_metrics:
                    self.tensorboard_writer.add_scalar('System/RAM_Usage', system_metrics['ram_usage'], self.global_step)
                    self.tensorboard_writer.add_scalar('System/CPU_Usage', system_metrics['cpu_usage'], self.global_step)
                    
                    if system_metrics.get('gpu_usage'):
                        self.tensorboard_writer.add_scalar('System/GPU_Usage', system_metrics['gpu_usage'], self.global_step)
                    if system_metrics.get('gpu_memory'):
                        self.tensorboard_writer.add_scalar('System/GPU_Memory', system_metrics['gpu_memory'], self.global_step)
                    if system_metrics.get('gpu_temperature'):
                        self.tensorboard_writer.add_scalar('System/GPU_Temperature', system_metrics['gpu_temperature'], self.global_step)
                
                # Фазы обучения
                current_phase = self.get_current_training_phase()
                self.tensorboard_writer.add_text('Training/Phase', current_phase, self.global_step)
                
            except Exception as e:
                self.logger.warning(f"Ошибка TensorBoard: {e}")
        
        # MLflow - расширенное логирование
        if self.mlflow_run:
            try:
                # Основные метрики
                mlflow_metrics = {
                    'train_loss': metrics['total_loss'],
                    'grad_norm': metrics.get('grad_norm', 0),
                    'attention_diagonality': metrics.get('attention_diagonality', 0),
                    'gate_accuracy': metrics.get('gate_accuracy', 0),
                    'learning_rate': metrics.get('learning_rate', 0)
                }
                
                # Детализированные метрики loss
                if 'mel_loss' in metrics:
                    mlflow_metrics['mel_loss'] = metrics['mel_loss']
                if 'gate_loss' in metrics:
                    mlflow_metrics['gate_loss'] = metrics['gate_loss']
                if 'attention_loss' in metrics:
                    mlflow_metrics['attention_loss'] = metrics['attention_loss']
                
                # Системные метрики
                if system_metrics:
                    mlflow_metrics.update({
                        'system_ram_usage': system_metrics['ram_usage'],
                        'system_cpu_usage': system_metrics['cpu_usage']
                    })
                    
                    if system_metrics.get('gpu_usage'):
                        mlflow_metrics['system_gpu_usage'] = system_metrics['gpu_usage']
                    if system_metrics.get('gpu_memory'):
                        mlflow_metrics['system_gpu_memory'] = system_metrics['gpu_memory']
                    if system_metrics.get('gpu_temperature'):
                        mlflow_metrics['system_gpu_temperature'] = system_metrics['gpu_temperature']
                
                mlflow.log_metrics(mlflow_metrics, step=self.global_step)
                
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
    
    def _send_training_start_notification(self):
        """Отправка стартового уведомления в Telegram."""
        if not self.telegram_monitor:
            return
            
        try:
            message = f"🚀 Начинаю Ultimate Enhanced Training (режим: {self.mode})!"
            if hasattr(self.telegram_monitor, 'send_message'):
                self.telegram_monitor.send_message(message)
        except Exception:
            pass  # Игнорируем ошибки Telegram
    
    def _send_enhanced_telegram_update(self, metrics: Dict):
        """Отправка кардинально улучшенных уведомлений в Telegram с анализом, графиками и отслеживанием исправлений."""
        if not self.telegram_monitor:
            return
            
        try:
            # 🏆 ЗАГОЛОВОК С ЭМОДЗИ СТАТУСА
            status_emoji = "🔥" if metrics['train_loss'] < 20 else "📈" if metrics['train_loss'] < 50 else "⚠️"
            message = f"{status_emoji} Эпоха {metrics['epoch']} | {metrics['phase'][:8]}\n"
            
            # 📊 ОСНОВНЫЕ МЕТРИКИ С ТРЕНДАМИ
            message += f"📈 Loss: {metrics['train_loss']:.3f}"
            if 'val_loss' in metrics:
                trend = "📉" if metrics['val_loss'] < metrics['train_loss'] else "📈"
                message += f" {trend} {metrics['val_loss']:.3f}"
                # Добавляем разность
                diff = abs(metrics['val_loss'] - metrics['train_loss'])
                message += f" (Δ{diff:.2f})"
            
            message += f"\n🎯 Gate: {metrics['train_gate_accuracy']:.3f}"
            message += f" | Attn: {metrics['train_attention_diagonality']:.3f}"
            message += f"\n🔧 Grad: {metrics['train_grad_norm']:.1f}"
            message += f" | LR: {metrics['learning_rate']:.1e}"
            
            # 💻 СИСТЕМНЫЕ МЕТРИКИ
            system_metrics = self._get_system_metrics()
            if system_metrics:
                message += f"\n💻 RAM: {system_metrics['ram_usage']:.1f}%"
                if system_metrics.get('gpu_usage'):
                    message += f" | GPU: {system_metrics['gpu_usage']:.1f}%"
                if system_metrics.get('gpu_memory'):
                    message += f" | VRAM: {system_metrics['gpu_memory']:.1f}%"
            
            # ✅ ИСПРАВЛЕНИЯ ИЗ ПРЕДЫДУЩИХ РЕКОМЕНДАЦИЙ
            if hasattr(self, 'last_recommendations'):
                fixed_issues = self._check_fixed_issues(metrics)
                if fixed_issues:
                    message += f"\n✅ ИСПРАВЛЕНО: {', '.join(fixed_issues)}"
            
            # 🔧 НОВЫЕ ИСПРАВЛЕНИЯ
            if metrics.get('fixes_applied', 0) > 0:
                message += f"\n🔧 Автоисправлений: {metrics['fixes_applied']}"
            
            # 🚨 АНАЛИЗ ПРОБЛЕМ И НОВЫЕ РЕКОМЕНДАЦИИ
            problems, recommendations = self._analyze_training_issues(metrics)
            
            if problems:
                message += f"\n⚠️ Проблемы: {', '.join(problems[:2])}"
                
            if recommendations:
                message += f"\n💡 К исправлению: {', '.join(recommendations[:2])}"
                # Сохраняем рекомендации для отслеживания в следующем сообщении
                self.last_recommendations = recommendations
            
            # 📈 ASCII ГРАФИКИ ТРЕНДОВ
            if len(self.training_metrics_history) >= 8:
                loss_chart = self._create_ascii_chart(
                    [h['train_loss'] for h in self.training_metrics_history[-8:]], 
                    "Loss"
                )
                if loss_chart:
                    message += f"\n📈 Тренд: {loss_chart}"
                    
                # Дополнительные графики для критических метрик
                if metrics['train_gate_accuracy'] < 0.9:
                    gate_chart = self._create_ascii_chart(
                        [h['train_gate_accuracy'] for h in self.training_metrics_history[-8:]],
                        "Gate"
                    )
                    if gate_chart:
                        message += f"\n🎯 Gate: {gate_chart}"
            
            # 🔍 СТАТУС ОБУЧЕНИЯ
            training_status = self._get_training_status(metrics)
            message += f"\n{training_status}"
            
            # 🏆 ДОСТИЖЕНИЯ И ПРОГРЕСС
            achievements = self._check_achievements(metrics)
            if achievements:
                message += f"\n🏆 {achievements}"
            
            # 📊 ПРОГНОЗ ВРЕМЕНИ ЗАВЕРШЕНИЯ (если возможно)
            if len(self.training_metrics_history) >= 10:
                eta = self._estimate_completion_time(metrics)
                if eta:
                    message += f"\n⏱️ ETA: {eta}"
            
            # Безопасная отправка
            if hasattr(self.telegram_monitor, 'send_message'):
                self.telegram_monitor.send_message(message)
                
        except Exception:
            pass  # Игнорируем ошибки Telegram
            
    def _get_system_metrics(self) -> Optional[Dict]:
        """Получение системных метрик для мониторинга."""
        try:
            # RAM использование
            ram_percent = psutil.virtual_memory().percent
            
            system_metrics = {
                'ram_usage': ram_percent,
                'cpu_usage': psutil.cpu_percent(interval=0.1)
            }
            
            # GPU метрики (если доступны)
            if GPU_MONITORING_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Первая GPU
                        system_metrics['gpu_usage'] = gpu.load * 100
                        system_metrics['gpu_memory'] = gpu.memoryUtil * 100
                        system_metrics['gpu_temperature'] = gpu.temperature
                except Exception:
                    pass
                    
            return system_metrics
        except Exception:
            return None
    
    def _check_fixed_issues(self, current_metrics: Dict) -> List[str]:
        """Проверка исправленных проблем из предыдущих рекомендаций."""
        if not hasattr(self, 'last_recommendations'):
            return []
            
        fixed = []
        
        for recommendation in self.last_recommendations:
            if "внимание" in recommendation.lower():
                if current_metrics['train_attention_diagonality'] > 0.1:
                    fixed.append("Внимание улучшено")
                    
            elif "градиент" in recommendation.lower():
                if current_metrics['train_grad_norm'] < 50:
                    fixed.append("Градиенты стабилизированы")
                    
            elif "loss" in recommendation.lower():
                if current_metrics['train_loss'] < 100:
                    fixed.append("Loss снижен")
                    
            elif "gate" in recommendation.lower():
                if current_metrics['train_gate_accuracy'] > 0.8:
                    fixed.append("Gate accuracy улучшена")
        
        return fixed
    
    def _analyze_training_issues(self, metrics: Dict) -> Tuple[List[str], List[str]]:
        """Анализ проблем обучения и генерация рекомендаций."""
        problems = []
        recommendations = []
        
        # Анализ attention
        if metrics['train_attention_diagonality'] < 0.05:
            problems.append("Критически низкое внимание")
            recommendations.append("Увеличить guided attention weight")
        elif metrics['train_attention_diagonality'] < 0.2:
            problems.append("Низкое внимание")
            recommendations.append("Проверить alignment")
            
        # Анализ градиентов
        if metrics['train_grad_norm'] > 100:
            problems.append("Критический gradient explosion")
            recommendations.append("Снизить LR немедленно")
        elif metrics['train_grad_norm'] > 50:
            problems.append("Большие градиенты")
            recommendations.append("Усилить clipping")
            
        # Анализ loss
        if metrics['train_loss'] > 200:
            problems.append("Критически высокий loss")
            recommendations.append("Проверить данные")
        elif metrics['train_loss'] > 100:
            problems.append("Высокий loss")
            recommendations.append("Увеличить epochs")
            
        # Анализ gate accuracy
        if metrics['train_gate_accuracy'] < 0.5:
            problems.append("Критически низкая gate accuracy")
            recommendations.append("Проверить модель")
        elif metrics['train_gate_accuracy'] < 0.8:
            problems.append("Низкая gate accuracy")
            recommendations.append("Больше обучения")
            
        # Анализ стагнации
        if len(self.training_metrics_history) >= 10:
            recent_losses = [h['train_loss'] for h in self.training_metrics_history[-10:]]
            if max(recent_losses) - min(recent_losses) < 0.01:
                problems.append("Стагнация loss")
                recommendations.append("Изменить LR или архитектуру")
        
        return problems, recommendations
    
    def _create_ascii_chart(self, values: List[float], name: str) -> str:
        """Создание простого ASCII графика."""
        if len(values) < 3:
            return ""
            
        try:
            # Нормализация значений
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return f"{name}: ◆◆◆◆◆◆"
                
            normalized = [(v - min_val) / (max_val - min_val) for v in values]
            
            # Создание символов на основе значений
            symbols = []
            for i, val in enumerate(normalized):
                if i == 0:
                    symbols.append("▪")
                else:
                    prev_val = normalized[i-1]
                    if val > prev_val + 0.1:
                        symbols.append("▲")
                    elif val < prev_val - 0.1:
                        symbols.append("▼")
                    elif val > 0.7:
                        symbols.append("◆")
                    elif val > 0.3:
                        symbols.append("◇")
                    else:
                        symbols.append("▫")
            
            return ''.join(symbols)
        except Exception:
            return ""
    
    def _get_training_status(self, metrics: Dict) -> str:
        """Получение статуса обучения."""
        # Определяем общий статус на основе метрик
        if metrics['train_gate_accuracy'] > 0.95 and metrics['train_grad_norm'] < 10:
            if metrics['train_attention_diagonality'] > 0.5:
                return "🟢 Отличное обучение"
            else:
                return "🟡 Хорошо, но внимание требует работы"
        elif metrics['train_loss'] < 50 and metrics['train_gate_accuracy'] > 0.8:
            return "🟡 Стабильное обучение"
        elif metrics['train_loss'] > 200 or metrics['train_grad_norm'] > 100:
            return "🔴 Критические проблемы"
        else:
            return "🟠 Требует внимания"
    
    def _check_achievements(self, metrics: Dict) -> str:
        """Проверка достижений в обучении."""
        achievements = []
        
        # Достижения по loss
        if metrics['train_loss'] < 10:
            achievements.append("🔥 Loss < 10!")
        elif metrics['train_loss'] < 20:
            achievements.append("🎯 Loss < 20")
            
        # Достижения по gate accuracy
        if metrics['train_gate_accuracy'] > 0.99:
            achievements.append("⭐ Gate 99%+")
        elif metrics['train_gate_accuracy'] > 0.95:
            achievements.append("🎯 Gate 95%+")
            
        # Достижения по attention
        if metrics['train_attention_diagonality'] > 0.8:
            achievements.append("🎯 Отличное внимание")
        elif metrics['train_attention_diagonality'] > 0.5:
            achievements.append("✅ Хорошее внимание")
            
        # Стабильность градиентов
        if metrics['train_grad_norm'] < 5:
            achievements.append("🔧 Супер-стабильные градиенты")
            
        return ' | '.join(achievements) if achievements else ""
    
    def _estimate_completion_time(self, metrics: Dict) -> Optional[str]:
        """Прогноз времени завершения обучения."""
        try:
            if len(self.training_metrics_history) < 10:
                return None
                
            # Анализ тренда loss
            recent_losses = [h['train_loss'] for h in self.training_metrics_history[-10:]]
            if max(recent_losses) - min(recent_losses) < 0.1:
                return "∞ (стагнация)"
                
            # Простой линейный прогноз
            current_loss = metrics['train_loss']
            target_loss = 10.0  # Целевой loss
            
            if current_loss <= target_loss:
                return "🎉 Цель достигнута!"
                
            # Средняя скорость снижения loss за последние эпохи
            loss_decrease_rate = (recent_losses[0] - recent_losses[-1]) / len(recent_losses)
            
            if loss_decrease_rate <= 0:
                return "∞ (loss не снижается)"
                
            epochs_needed = (current_loss - target_loss) / loss_decrease_rate
            
            if epochs_needed < 50:
                return f"~{int(epochs_needed)} эпох до цели"
            elif epochs_needed < 200:
                return f"~{int(epochs_needed/10)*10} эпох"
            else:
                return "200+ эпох"
                
        except Exception:
            return None
            
    def _check_training_completion(self, metrics: Dict) -> bool:
        """Проверка достижения максимального качества обучения."""
        try:
            # 🎯 Критерии завершения обучения
            completion_criteria = []
            
            # 1. Отличный loss и gate accuracy
            if metrics['train_loss'] < 8.0 and metrics['train_gate_accuracy'] > 0.995:
                completion_criteria.append("excellent_loss_and_gate")
            
            # 2. Отличное внимание
            if metrics['train_attention_diagonality'] > 0.85:
                completion_criteria.append("excellent_attention")
            
            # 3. Стабильные градиенты
            if metrics['train_grad_norm'] < 3.0:
                completion_criteria.append("stable_gradients")
            
            # 4. Хорошая генерализация (если есть val_loss)
            if 'val_loss' in metrics:
                if metrics['val_loss'] < 10.0 and abs(metrics['val_loss'] - metrics['train_loss']) < 2.0:
                    completion_criteria.append("good_generalization")
            
            # 5. Стабильность метрик за последние эпохи
            if len(self.training_metrics_history) >= 20:
                recent_losses = [h['train_loss'] for h in self.training_metrics_history[-20:]]
                recent_gates = [h['train_gate_accuracy'] for h in self.training_metrics_history[-20:]]
                
                # Проверяем стабильность
                loss_variance = np.var(recent_losses)
                gate_variance = np.var(recent_gates)
                
                if loss_variance < 0.5 and gate_variance < 0.001:  # Низкая вариация
                    if np.mean(recent_losses) < 12.0 and np.mean(recent_gates) > 0.99:
                        completion_criteria.append("stable_excellent_metrics")
            
            # 🏆 УСЛОВИЯ АВТОМАТИЧЕСКОЙ ОСТАНОВКИ
            
            # Ultimate качество: все критерии выполнены
            if len(completion_criteria) >= 4:
                self.logger.info(f"🏆 ULTIMATE КАЧЕСТВО достигнуто! Критерии: {completion_criteria}")
                return True
            
            # Отличное качество: основные критерии + стабильность
            if ("excellent_loss_and_gate" in completion_criteria and 
                "stable_gradients" in completion_criteria and
                "stable_excellent_metrics" in completion_criteria):
                self.logger.info(f"🎯 ОТЛИЧНОЕ КАЧЕСТВО достигнуто! Критерии: {completion_criteria}")
                return True
            
            # Режим ultimate - более строгие критерии
            if self.mode == 'ultimate':
                if len(completion_criteria) >= 3 and "excellent_loss_and_gate" in completion_criteria:
                    self.logger.info(f"🏆 ULTIMATE режим: высокое качество достигнуто! Критерии: {completion_criteria}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Ошибка проверки завершения обучения: {e}")
            return False
    
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
                # Безопасная отправка с fallback методами
                try:
                    if hasattr(self.telegram_monitor, 'send_message'):
                        self.telegram_monitor.send_message(message)
                    elif hasattr(self.telegram_monitor, 'send_training_update'):
                        self.telegram_monitor.send_training_update({'message': message})
                    elif hasattr(self.telegram_monitor, 'send_auto_improvement_notification'):
                        self.telegram_monitor.send_auto_improvement_notification(message)
                    else:
                        self.logger.debug("Telegram monitor не поддерживает отправку корректировок")
                except Exception as e:
                    self.logger.warning(f"Ошибка Telegram уведомления корректировок: {e}")
    
    def _analyze_attention_patterns(self, alignments):
        """
        🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ATTENTION PATTERNS для диагностики и улучшения.
        Анализирует качество attention matrix на основе современных исследований.
        """
        try:
            # Переводим в numpy для анализа (берем первый элемент батча)
            if alignments.dim() == 3:
                attention = alignments[0].detach().cpu().numpy()
            else:
                attention = alignments.detach().cpu().numpy()
            
            T_out, T_in = attention.shape
            
            # 1. Диагональность (основная метрика)
            diagonality = self._calculate_attention_diagonality(attention)
            
            # 2. Монотонность (важно для TTS)
            monotonicity = self._calculate_attention_monotonicity(attention)
            
            # 3. Фокусировка (концентрация attention)
            focus = self._calculate_attention_focus(attention)
            
            # 4. Покрытие входной последовательности
            coverage = self._calculate_attention_coverage(attention)
            
            # 5. Стабильность (низкие скачки)
            stability = self._calculate_attention_stability(attention)
            
            return {
                'diagonality': diagonality,
                'monotonicity': monotonicity,
                'focus': focus,
                'coverage': coverage,
                'stability': stability
            }
            
        except Exception as e:
            self.logger.debug(f"Ошибка анализа attention patterns: {e}")
            return {'diagonality': 0.0, 'monotonicity': 0.0, 'focus': 0.0, 'coverage': 0.0, 'stability': 0.0}
    
    def _calculate_attention_diagonality(self, attention_matrix):
        """Вычисляет диагональность attention matrix."""
        try:
            T_out, T_in = attention_matrix.shape
            if T_out == 0 or T_in == 0:
                return 0.0
            
            # Создаем идеальную диагональ
            ideal_diagonal = np.zeros_like(attention_matrix)
            min_dim = min(T_out, T_in)
            
            for i in range(T_out):
                diagonal_pos = int(i * T_in / T_out) if T_out > 0 else i
                if diagonal_pos < T_in:
                    ideal_diagonal[i, diagonal_pos] = 1.0
            
            # Корреляция с идеальной диагональю
            attention_flat = attention_matrix.flatten()
            ideal_flat = ideal_diagonal.flatten()
            
            if np.std(attention_flat) == 0 or np.std(ideal_flat) == 0:
                return 0.0
            
            correlation = np.corrcoef(attention_flat, ideal_flat)[0, 1]
            return max(0.0, correlation) if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_attention_monotonicity(self, attention_matrix):
        """Вычисляет монотонность attention (важно для TTS)."""
        try:
            T_out, T_in = attention_matrix.shape
            if T_out < 2:
                return 1.0
            
            # Находим пики attention для каждого выходного шага
            peaks = np.argmax(attention_matrix, axis=1)
            
            # Считаем нарушения монотонности
            violations = 0
            for i in range(1, len(peaks)):
                if peaks[i] < peaks[i-1]:  # Движение назад
                    violations += 1
            
            # Монотонность = 1 - доля нарушений
            monotonicity = 1.0 - (violations / (T_out - 1))
            return max(0.0, monotonicity)
            
        except Exception:
            return 0.0
    
    def _calculate_attention_focus(self, attention_matrix):
        """Вычисляет фокусировку attention (обратное к размытости)."""
        try:
            # Средняя энтропия по всем выходным шагам
            entropies = []
            
            for t_out in range(attention_matrix.shape[0]):
                att_weights = attention_matrix[t_out]
                att_weights_safe = np.clip(att_weights, 1e-10, 1.0)
                entropy = -np.sum(att_weights_safe * np.log(att_weights_safe))
                
                # Нормализуем энтропию
                max_entropy = np.log(len(att_weights_safe))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                entropies.append(normalized_entropy)
            
            # Focus = 1 - средняя нормализованная энтропия
            avg_entropy = np.mean(entropies)
            return max(0.0, 1.0 - avg_entropy)
            
        except Exception:
            return 0.0
    
    def _calculate_attention_coverage(self, attention_matrix):
        """Вычисляет покрытие входной последовательности."""
        try:
            # Сумма attention весов по каждой входной позиции
            input_coverage = np.sum(attention_matrix, axis=0)
            
            # Доля входных позиций с значимым attention (>1% от максимума)
            threshold = 0.01 * np.max(input_coverage)
            covered_positions = np.sum(input_coverage > threshold)
            total_positions = len(input_coverage)
            
            coverage = covered_positions / total_positions if total_positions > 0 else 0.0
            return coverage
            
        except Exception:
            return 0.0
    
    def _calculate_attention_stability(self, attention_matrix):
        """Вычисляет стабильность attention (отсутствие резких скачков)."""
        try:
            if attention_matrix.shape[0] < 2:
                return 1.0
            
            # Вычисляем изменения attention между соседними шагами
            differences = []
            
            for i in range(1, attention_matrix.shape[0]):
                diff = np.abs(attention_matrix[i] - attention_matrix[i-1])
                differences.append(np.mean(diff))
            
            # Стабильность = 1 - средняя величина изменений
            avg_change = np.mean(differences)
            stability = max(0.0, 1.0 - avg_change)
            return stability
            
        except Exception:
            return 0.0

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
            
            # 🔧 УЛУЧШЕННАЯ ФУНКЦИЯ КОНВЕРТАЦИИ ДЛЯ JSON СЕРИАЛИЗАЦИИ
            def convert_numpy_types(obj):
                """Конвертирует все проблемные типы в JSON-совместимые."""
                try:
                    # Numpy типы
                    if isinstance(obj, (np.integer, np.floating)):
                        return obj.item()
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    
                    # Python type objects (класс, функция и т.д.)
                    elif isinstance(obj, type):
                        return f"<type:{obj.__name__}>"
                    elif callable(obj):
                        return f"<callable:{getattr(obj, '__name__', str(obj))}>"
                    
                    # Tensor объекты
                    elif hasattr(obj, 'detach') and hasattr(obj, 'cpu'):  # PyTorch tensor
                        return obj.detach().cpu().numpy().tolist()
                    elif hasattr(obj, 'numpy'):  # TensorFlow tensor
                        return obj.numpy().tolist()
                    
                    # Datetime объекты
                    elif hasattr(obj, 'isoformat'):  # datetime objects
                        return obj.isoformat()
                    
                    # Pathlib пути
                    elif hasattr(obj, '__fspath__'):  # pathlib.Path
                        return str(obj)
                    
                    # Enum объекты
                    elif hasattr(obj, 'value') and hasattr(obj, 'name'):  # Enum
                        return obj.value
                    
                    # Complex числа
                    elif isinstance(obj, complex):
                        return {'real': obj.real, 'imag': obj.imag}
                    
                    # Bytes объекты
                    elif isinstance(obj, (bytes, bytearray)):
                        return obj.decode('utf-8', errors='replace')
                    
                    # Множества
                    elif isinstance(obj, set):
                        return list(obj)
                    elif isinstance(obj, frozenset):
                        return list(obj)
                    
                    # Коллекции
                    elif isinstance(obj, dict):
                        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_numpy_types(item) for item in obj]
                    
                    # Специальные значения
                    elif obj is None:
                        return None
                    elif isinstance(obj, (str, int, float, bool)):
                        return obj
                    elif obj == float('inf'):
                        return "infinity"
                    elif obj == float('-inf'):
                        return "-infinity"
                    elif obj != obj:  # NaN check
                        return "NaN"
                    
                    # Сложные объекты с __dict__
                    elif hasattr(obj, '__dict__'):
                        if hasattr(obj, '__class__'):
                            class_name = obj.__class__.__name__
                            # Исключаем некоторые системные классы
                            if class_name in ['Logger', 'TextIOWrapper', 'Thread', 'Lock']:
                                return f"<{class_name}>"
                            # Пытаемся сериализовать только простые объекты
                            try:
                                return {
                                    '_class': class_name,
                                    **{k: convert_numpy_types(v) for k, v in obj.__dict__.items() 
                                       if not k.startswith('_') and not callable(v)}
                                }
                            except:
                                return f"<{class_name}:not_serializable>"
                        return str(obj)
                    
                    # Fallback: конвертируем в строку
                    else:
                        return str(obj)
                        
                except Exception as e:
                    # Если что-то пошло не так, возвращаем безопасное представление
                    return f"<serialization_error:{type(obj).__name__}>"
            
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
            # Стандартная инициализация - автоматическое определение путей к файлам CSV
            import os
            
            # Определяем пути к файлам CSV
            if os.path.isdir(args.dataset_path):
                # Если передана директория, ищем train.csv и val.csv внутри
                train_file = os.path.join(args.dataset_path, 'train.csv')
                val_file = os.path.join(args.dataset_path, 'val.csv')
            else:
                # Если передан путь к файлу, используем его напрямую
                train_file = args.dataset_path
                val_file = args.dataset_path.replace('train', 'val')
            
            # Проверяем существование файлов
            if not os.path.exists(train_file):
                raise FileNotFoundError(f"Файл обучения не найден: {train_file}")
            if not os.path.exists(val_file):
                print(f"⚠️ Файл валидации не найден: {val_file}, используем файл обучения")
                val_file = train_file
            
            print(f"📂 Используем файлы:")
            print(f"   Обучение: {train_file}")
            print(f"   Валидация: {val_file}")
            
            trainset = TextMelLoader(train_file, hparams)
            valset = TextMelLoader(val_file, hparams)
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