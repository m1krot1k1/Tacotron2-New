#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Tacotron2 Training with Smart Tuner Integration
Революционная система обучения TTS с максимальным качеством

Интегрирует все улучшения из исследований 2024-2025:
- Very Attentive Tacotron (Google, 2025)
- MonoAlign robust alignment (INTERSPEECH 2024)
- XTTS Advanced training practices
- DLPO reinforcement learning
- Style-BERT-VITS2 optimizations
- Smart Tuner intelligent automation
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import yaml

# Импорт основных компонентов
from model import Tacotron2
from loss_function import Tacotron2Loss
from hparams import create_hparams
from audio_quality_enhancer import AudioQualityEnhancer

# Импорт Smart Tuner компонентов
try:
    from smart_tuner.smart_tuner_integration import SmartTunerIntegration
    from smart_tuner.telegram_monitor import TelegramMonitor
    from smart_tuner.integration_manager import SmartTunerIntegrationManager
    SMART_TUNER_AVAILABLE = True
except ImportError:
    SMART_TUNER_AVAILABLE = False
    logging.warning("Smart Tuner не найден, используется стандартное обучение")

# Импорт дополнительных компонентов из train.py
try:
    from debug_reporter import initialize_debug_reporter, get_debug_reporter
    DEBUG_REPORTER_AVAILABLE = True
except ImportError:
    DEBUG_REPORTER_AVAILABLE = False
    logging.warning("Debug Reporter не найден")

# Импорт утилит для метрик качества
try:
    from training_utils.dynamic_padding import DynamicPaddingCollator
    from training_utils.bucket_batching import BucketBatchSampler
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    logging.warning("Утилиты не найдены")

# === MLflow: безопасная инициализация ===
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow не найден, метрики не будут логироваться")

# === TensorBoard: безопасная инициализация ===
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logging.warning("TensorBoard не найден, метрики не будут логироваться")

class EnhancedTacotronTrainer:
    """
    Продвинутый тренер Tacotron2 с интеграцией Smart Tuner и современных техник.
    
    Особенности:
    1. Автоматический контроль качества в реальном времени
    2. Адаптивные гиперпараметры на основе прогресса
    3. Интеллектуальное управление эпохами
    4. Фазовое обучение для максимального качества
    5. Современные loss функции из исследований 2025
    """
    
    def __init__(self, hparams, dataset_info: Optional[Dict] = None):
        """
        Инициализация enhanced тренера.
        
        Args:
            hparams: Гиперпараметры обучения
            dataset_info: Информация о датасете для оптимизации
        """
        self.hparams = hparams
        self.dataset_info = dataset_info or {}
        
        # Настройка логирования
        self.logger = self._setup_logger()
        
        # Инициализация Smart Tuner
        self.smart_tuner = None
        if SMART_TUNER_AVAILABLE:
            try:
                self.smart_tuner = SmartTunerIntegration()
                self.logger.info("🚀 Smart Tuner успешно инициализирован")
            except Exception as e:
                self.logger.error(f"Ошибка инициализации Smart Tuner: {e}")
        
        # 📱 Инициализация Telegram Monitor
        self.telegram_monitor = None
        if SMART_TUNER_AVAILABLE:
            try:
                # Используем TelegramMonitorEnhanced с правильными параметрами
                from smart_tuner.telegram_monitor_enhanced import TelegramMonitorEnhanced
                import yaml
                
                # Загружаем конфиг напрямую
                config_path = "smart_tuner/config.yaml"
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                except Exception as e:
                    self.logger.warning(f"Не удалось загрузить конфиг: {e}")
                    config = {}
                
                telegram_config = config.get('telegram', {})
                bot_token = telegram_config.get('bot_token')
                chat_id = telegram_config.get('chat_id')
                enabled = telegram_config.get('enabled', False)
                
                if bot_token and chat_id and enabled:
                    self.telegram_monitor = TelegramMonitorEnhanced(
                        bot_token=bot_token,
                        chat_id=chat_id,
                        enabled=enabled
                    )
                    self.logger.info("📱 Telegram Monitor Enhanced инициализирован")
                else:
                    self.telegram_monitor = None
                    self.logger.warning("📱 Telegram Monitor отключен (неполные настройки)")
            except Exception as e:
                self.telegram_monitor = None
                self.logger.error(f"Ошибка инициализации Telegram Monitor: {e}")
        
        # Инициализация компонентов
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.audio_enhancer = AudioQualityEnhancer()
        
        # 🔧 ИНТЕГРАЦИЯ ДОПОЛНИТЕЛЬНЫХ КОМПОНЕНТОВ ИЗ TRAIN.PY
        # Integration Manager для координации всех компонентов
        self.integration_manager = None
        if SMART_TUNER_AVAILABLE:
            try:
                self.integration_manager = SmartTunerIntegrationManager()
                self.logger.info("🔧 Integration Manager инициализирован")
            except Exception as e:
                self.logger.error(f"Ошибка инициализации Integration Manager: {e}")
        
        # Debug Reporter для детальной диагностики
        self.debug_reporter = None
        if DEBUG_REPORTER_AVAILABLE:
            try:
                self.debug_reporter = initialize_debug_reporter(self.telegram_monitor)
                self.logger.info("🔍 Debug Reporter инициализирован")
            except Exception as e:
                self.logger.error(f"Ошибка инициализации Debug Reporter: {e}")
        
        # Состояние обучения
        self.current_epoch = 0
        self.global_step = 0
        self.best_validation_loss = float('inf')
        self.training_metrics_history = []
        
        # Фазы обучения
        self.training_phases = {
            'pre_alignment': {'max_epoch': 500, 'focus': 'attention_learning'},
            'alignment_learning': {'max_epoch': 2000, 'focus': 'attention_stabilization'},
            'quality_optimization': {'max_epoch': 3000, 'focus': 'quality_improvement'},
            'fine_tuning': {'max_epoch': 3500, 'focus': 'final_polishing'}
        }
        
        self.logger.info("✅ Enhanced Tacotron Trainer инициализирован")
        
        self.tensorboard_writer = None
        self.tensorboard_logdir = 'logs'  # Можно сделать параметром
        if TENSORBOARD_AVAILABLE:
            try:
                # === Очистка старых логов TensorBoard ===
                if os.path.exists(self.tensorboard_logdir):
                    for file in os.listdir(self.tensorboard_logdir):
                        if file.startswith('events.out.tfevents'):
                            os.remove(os.path.join(self.tensorboard_logdir, file))
                            self.logger.info(f"🗑️ Удален старый лог TensorBoard: {file}")
                self.tensorboard_writer = SummaryWriter(self.tensorboard_logdir)
                self.logger.info(f"✅ TensorBoard writer инициализирован: {self.tensorboard_logdir}")
            except Exception as e:
                self.tensorboard_writer = None
                self.logger.error(f"⚠️ Ошибка инициализации TensorBoard: {e}")
        else:
            self.logger.warning("TensorBoard недоступен, метрики не будут логироваться")
    
    def _setup_logger(self) -> logging.Logger:
        """Настраивает логирование для тренера."""
        logger = logging.getLogger('EnhancedTacotronTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Консольный handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - [Enhanced Tacotron] - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # Файловый handler
            file_handler = logging.FileHandler('enhanced_training.log')
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def initialize_training(self):
        """Инициализирует все компоненты для обучения."""
        self.logger.info("🔧 Инициализация компонентов обучения...")
        
        # Оптимизация гиперпараметров через Smart Tuner
        if self.smart_tuner:
            original_hparams = vars(self.hparams).copy()
            optimized_hparams = self.smart_tuner.on_training_start(
                original_hparams, self.dataset_info
            )
            
            # Применяем оптимизированные параметры
            for key, value in optimized_hparams.items():
                if hasattr(self.hparams, key):
                    setattr(self.hparams, key, value)
                    
            self.logger.info("✨ Гиперпараметры оптимизированы через Smart Tuner")
        
        # Инициализация модели
        self.model = Tacotron2(self.hparams).cuda()
        self.logger.info(f"📊 Модель загружена: {sum(p.numel() for p in self.model.parameters())} параметров")
        
        # Инициализация loss функции с современными улучшениями
        self.criterion = Tacotron2Loss(self.hparams)
        self.logger.info("🎯 Enhanced loss function инициализирована")
        
        # Инициализация оптимизатора
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=getattr(self.hparams, 'weight_decay', 1e-6)
        )
        self.logger.info("⚙️ Оптимизатор AdamW инициализирован")
        
        # 🔧 Инициализация Smart LR Adapter
        try:
            from smart_tuner.smart_lr_adapter import SmartLRAdapter, set_global_lr_adapter
            self.lr_adapter = SmartLRAdapter(
                optimizer=self.optimizer,
                patience=10,
                factor=0.5,
                min_lr=getattr(self.hparams, 'learning_rate_min', 1e-8),
                max_lr=self.hparams.learning_rate * 2,
                emergency_factor=0.1,
                grad_norm_threshold=1000.0,
                loss_nan_threshold=1e6
            )
            set_global_lr_adapter(self.lr_adapter)
            self.logger.info("✅ Smart LR Adapter инициализирован")
        except Exception as e:
            self.lr_adapter = None
            self.logger.warning(f"⚠️ Не удалось инициализировать Smart LR Adapter: {e}")
        
        # Инициализация scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=getattr(self.hparams, 'scheduler_T_0', 1000),
            eta_min=getattr(self.hparams, 'min_learning_rate', 1e-6)
        )
        
        self.logger.info("🚀 Все компоненты успешно инициализированы")
        
        # Инициализация переменных для адаптивной настройки
        self.last_attention_diagonality = 0.0
        
        # === MLflow: инициализация эксперимента ===
        if MLFLOW_AVAILABLE:
            try:
                # Проверяем, есть ли уже активный run
                active_run = mlflow.active_run()
                if active_run is not None:
                    self.logger.info(f"✅ Используем существующий MLflow run: {active_run.info.run_id}")
                else:
                    experiment_name = f"tacotron2_training_{int(time.time())}"
                    mlflow.set_experiment(experiment_name)
                    mlflow.start_run(run_name=f"training_run_{int(time.time())}")
                    self.logger.info(f"✅ MLflow эксперимент инициализирован: {experiment_name}")
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка инициализации MLflow: {e}")
        
        # 🔧 Инициализация AutoFixManager для автоматического исправления проблем
        try:
            from smart_tuner.auto_fix_manager import AutoFixManager
            self.auto_fix_manager = AutoFixManager(
                model=self.model,
                optimizer=self.optimizer,
                hparams=self.hparams,
                telegram_monitor=self.telegram_monitor
            )
            self.logger.info("🤖 AutoFixManager интегрирован")
        except ImportError:
            self.auto_fix_manager = None
            self.logger.warning("⚠️ AutoFixManager не найден - автоматические исправления отключены")
    
    def get_current_training_phase(self) -> str:
        """Определяет текущую фазу обучения."""
        for phase_name, phase_info in self.training_phases.items():
            if self.current_epoch <= phase_info['max_epoch']:
                return phase_name
        return 'fine_tuning'
    
    def adjust_hyperparams_for_phase(self, phase: str):
        """Адаптирует гиперпараметры для текущей фазы обучения."""
        phase_configs = {
            'pre_alignment': {
                'guided_attention_weight': 10.0,
                'learning_rate_multiplier': 1.0,
                'teacher_forcing_ratio': 1.0
            },
            'alignment_learning': {
                'guided_attention_weight': 3.0,
                'learning_rate_multiplier': 0.8,
                'teacher_forcing_ratio': 0.9
            },
            'quality_optimization': {
                'guided_attention_weight': 1.0,
                'learning_rate_multiplier': 0.5,
                'teacher_forcing_ratio': 0.8
            },
            'fine_tuning': {
                'guided_attention_weight': 0.5,
                'learning_rate_multiplier': 0.3,
                'teacher_forcing_ratio': 0.7
            }
        }
        
        if phase in phase_configs:
            config = phase_configs[phase]
            
            # Обновляем веса loss функций
            self.criterion.guide_loss_weight = config['guided_attention_weight']
            
            # Обновляем learning rate
            base_lr = self.hparams.learning_rate
            new_lr = base_lr * config['learning_rate_multiplier']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            self.logger.info(f"🔄 Адаптированы параметры для фазы '{phase}': "
                           f"guided_weight={config['guided_attention_weight']}, "
                           f"lr_mult={config['learning_rate_multiplier']}")
    
    def train_step(self, batch):
        """Выполняет один шаг обучения с enhanced качеством."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Распаковка batch (TextMelCollate возвращает 8 элементов)
        text_inputs, text_lengths, mel_targets, gate_targets, mel_lengths, ctc_text, ctc_text_lengths, guide_mask = batch
        
        # Перенос на GPU
        text_inputs = text_inputs.cuda()
        mel_targets = mel_targets.cuda() 
        gate_targets = gate_targets.cuda()
        
        # Forward pass через parse_batch для правильной обработки всех элементов
        x, y = self.model.parse_batch(batch)
        model_outputs = self.model(x)
        # Модель может возвращать разное количество элементов в зависимости от настроек
        if len(model_outputs) >= 4:
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model_outputs[:4]
        else:
            # Fallback для случая, если модель вернула меньше элементов
            mel_outputs = model_outputs[0] if len(model_outputs) > 0 else None
            mel_outputs_postnet = model_outputs[1] if len(model_outputs) > 1 else None
            gate_outputs = model_outputs[2] if len(model_outputs) > 2 else None
            alignments = model_outputs[3] if len(model_outputs) > 3 else None
        
        # 🔍 ВЫЧИСЛЕНИЕ МЕТРИК КАЧЕСТВА ИЗ ВЫХОДОВ МОДЕЛИ
        attention_diagonality = 0.0
        gate_accuracy = 0.0
        
        # 🔧 АДАПТИВНАЯ НАСТРОЙКА GUIDED ATTENTION
        if hasattr(self.criterion, 'guide_loss_weight'):
            # Если диагональность низкая, увеличиваем guided attention weight
            if self.global_step > 0 and hasattr(self, 'last_attention_diagonality'):
                if self.last_attention_diagonality < 0.05:
                    # Критически низкая диагональность - экстренное увеличение
                    new_weight = min(self.criterion.guide_loss_weight * 3.0, 100.0)
                    self.criterion.guide_loss_weight = new_weight
                    self.logger.warning(f"🚨 КРИТИЧЕСКОЕ увеличение guided attention weight: {new_weight:.1f}")
                elif self.last_attention_diagonality < 0.1:
                    # Очень низкая диагональность - сильное увеличение
                    new_weight = min(self.criterion.guide_loss_weight * 2.5, 75.0)
                    self.criterion.guide_loss_weight = new_weight
                    self.logger.warning(f"🚨 Сильное увеличение guided attention weight: {new_weight:.1f}")
                elif self.last_attention_diagonality < 0.3:
                    # Низкая диагональность - умеренное увеличение
                    new_weight = min(self.criterion.guide_loss_weight * 1.5, 50.0)
                    self.criterion.guide_loss_weight = new_weight
                    self.logger.info(f"📈 Увеличение guided attention weight: {new_weight:.1f}")
                elif self.last_attention_diagonality > 0.7:
                    # Хорошая диагональность - постепенное снижение
                    new_weight = max(self.criterion.guide_loss_weight * 0.9, 1.0)
                    self.criterion.guide_loss_weight = new_weight
                    self.logger.info(f"📉 Снижение guided attention weight: {new_weight:.1f}")
        
        try:
            # Вычисляем attention_diagonality из attention матрицы
            if alignments is not None:
                attention_matrix = alignments.detach().cpu().numpy()
                if attention_matrix.ndim == 3:  # [batch, time, mel_time]
                    # Вычисляем диагональность для каждого элемента batch отдельно
                    batch_diagonalities = []
                    for b in range(attention_matrix.shape[0]):
                        # Нормализуем attention матрицу
                        attn = attention_matrix[b]
                        if attn.sum() > 0:
                            attn = attn / attn.sum(axis=1, keepdims=True)
                        
                        # Вычисляем диагональность как среднее по диагональным элементам
                        min_dim = min(attn.shape[0], attn.shape[1])
                        diagonal_elements = []
                        for i in range(min_dim):
                            diagonal_elements.append(attn[i, i])
                        batch_diagonalities.append(np.mean(diagonal_elements) if diagonal_elements else 0.0)
                    
                    # Берем среднее по batch
                    attention_diagonality = np.mean(batch_diagonalities) if batch_diagonalities else 0.0
                else:
                    attention_diagonality = 0.0
            
            # Вычисляем gate_accuracy из gate outputs
            if gate_outputs is not None:
                # Вычисляем accuracy как процент правильных предсказаний
                gate_pred = (gate_outputs > 0.5).float()
                gate_targets_binary = (gate_targets > 0.5).float()
                correct = (gate_pred == gate_targets_binary).float().mean()
                gate_accuracy = correct.item()
                
        except Exception as e:
            self.logger.warning(f"Ошибка вычисления метрик качества: {e}")
            attention_diagonality = 0.0
            gate_accuracy = 0.0
        
        # Сохраняем диагональность для следующего шага
        self.last_attention_diagonality = attention_diagonality
        
        # Логируем диагональность каждые 100 шагов для отладки
        if self.global_step % 100 == 0:
            self.logger.info(f"📊 Attention диагональность: {attention_diagonality:.4f}")
        
        # Вычисление loss с современными техниками
        loss_components = self.criterion(
            model_outputs, 
            (mel_targets, gate_targets),
            attention_weights=alignments,
            gate_outputs=gate_outputs
        )
        
        # Tacotron2Loss возвращает 4 компонента: mel_loss, gate_loss, guide_loss, emb_loss
        if len(loss_components) == 4:
            mel_loss, gate_loss, guide_loss, emb_loss = loss_components
            # Объединяем все loss в один общий loss
            loss = mel_loss + gate_loss + guide_loss + emb_loss
            # Создаем словарь с детализацией
            loss_dict = {
                'mel_loss': mel_loss.item(),
                'gate_loss': gate_loss.item(),
                'guide_loss': guide_loss.item(),
                'emb_loss': emb_loss.item(),
                'total_loss': loss.item()
            }
        else:
            # Fallback для других loss функций
            loss = loss_components[0] if len(loss_components) > 0 else torch.tensor(0.0)
            loss_dict = {'total_loss': loss.item()}
        
        # Backward pass
        loss.backward()
        
        # Вычисляем grad_norm для мониторинга
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            getattr(self.hparams, 'grad_clip_thresh', 1.0)
        )
        
        # 🔧 АВТОМАТИЧЕСКОЕ ИСПРАВЛЕНИЕ ПРОБЛЕМ через AutoFixManager
        if self.auto_fix_manager:
            try:
                # Собираем метрики для анализа
                fix_metrics = {
                    'grad_norm': float(grad_norm),
                    'attention_diagonality': attention_diagonality,
                    'gate_accuracy': gate_accuracy,
                    'loss': float(loss.item()),
                    'mel_loss': loss_dict.get('mel_loss', 0),
                    'gate_loss': loss_dict.get('gate_loss', 0),
                    'guide_loss': loss_dict.get('guide_loss', 0)
                }
                
                # Анализируем и применяем исправления
                applied_fixes = self.auto_fix_manager.analyze_and_fix(
                    step=self.global_step,
                    metrics=fix_metrics,
                    loss=loss
                )
                
                # Логируем примененные исправления
                if applied_fixes:
                    self.logger.info(f"🔧 Применено {len(applied_fixes)} автоматических исправлений")
                    for fix in applied_fixes:
                        if fix.success:
                            self.logger.info(f"✅ {fix.description}")
                        else:
                            self.logger.warning(f"⚠️ Не удалось: {fix.description}")
                            
            except Exception as e:
                self.logger.error(f"❌ Ошибка в AutoFixManager: {e}")
        
        # 🔧 Проверка на исчезновение градиентов (fallback)
        if grad_norm < 1e-8:
            self.logger.warning(f"⚠️ Исчезновение градиентов: {grad_norm:.2e}")
            # Попытка восстановления
            try:
                # Пересчитываем loss с большим масштабом
                scaled_loss = loss * 10.0
                scaled_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    getattr(self.hparams, 'grad_clip_thresh', 1.0)
                )
                self.logger.info(f"🔄 Градиенты восстановлены: {grad_norm:.2e}")
            except Exception as e:
                self.logger.error(f"❌ Не удалось восстановить градиенты: {e}")
        
        self.optimizer.step()
        
        # 🔧 Применение Smart LR Adapter
        if self.lr_adapter:
            try:
                lr_changed = self.lr_adapter.step(
                    loss=float(loss.item()),
                    grad_norm=float(grad_norm),
                    step=self.global_step
                )
                if lr_changed:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info(f"🔄 Smart LR адаптация: LR изменен на {current_lr:.2e}")
            except Exception as e:
                self.logger.warning(f"⚠️ Ошибка в Smart LR Adapter: {e}")
        
        self.scheduler.step()
        
        # 🔧 ИНТЕГРАЦИЯ С INTEGRATION MANAGER (после backward)
        if self.integration_manager:
            try:
                # Выполняем шаг интеграции всех компонентов
                integration_result = self.integration_manager.step(
                    step=self.global_step,
                    loss=float(loss.item()),
                    grad_norm=float(grad_norm),
                    model=self.model,
                    optimizer=self.optimizer
                )
                
                # Логируем результаты интеграции
                if integration_result.get('emergency_mode'):
                    self.logger.warning(f"🚨 Smart Tuner в экстренном режиме: {integration_result.get('recommendations', [])}")
                    
            except Exception as e:
                self.logger.error(f"Ошибка в Integration Manager: {e}")
        
        # 🔍 DEBUG REPORTER - детальная диагностика
        if self.debug_reporter:
            try:
                # Собираем данные для диагностики
                debug_data = {
                    'step': self.global_step,
                    'epoch': self.current_epoch,
                    'loss': loss.item(),
                    'attention_diagonality': attention_diagonality,
                    'gate_accuracy': gate_accuracy,
                    'mel_outputs': mel_outputs_postnet.detach().cpu().numpy() if mel_outputs_postnet is not None else None,
                    'gate_outputs': gate_outputs.detach().cpu().numpy() if gate_outputs is not None else None,
                    'alignments': alignments.detach().cpu().numpy() if alignments is not None else None,
                }
                
                self.debug_reporter.collect_step_data(
                    step=self.global_step,
                    metrics=debug_data,
                    model=self.model,
                    y_pred=model_outputs,
                    loss_components=loss_dict,
                    hparams=self.hparams,
                    smart_tuner_decisions={}
                )
                
            except Exception as e:
                self.logger.error(f"Ошибка Debug Reporter: {e}")
        
        # Анализ качества через Smart Tuner
        quality_analysis = {}
        if self.smart_tuner:
            try:
                quality_analysis = self.smart_tuner.on_batch_end(
                    self.current_epoch,
                    self.global_step,
                    loss_dict,
                    (mel_outputs_postnet, gate_outputs, alignments)
                )
            except Exception as e:
                self.logger.warning(f"Ошибка анализа качества: {e}")
        
        # 📱 Telegram уведомления каждые 1000 шагов
        if self.telegram_monitor and self.global_step % 1000 == 0:
            try:
                # Обновляем метрики с новыми данными качества
                enhanced_metrics = loss_dict.copy()
                enhanced_metrics.update({
                    'attention_diagonality': attention_diagonality,
                    'gate_accuracy': gate_accuracy,
                    'grad_norm': float(grad_norm),
                })
                
                self.telegram_monitor.send_training_update(
                    step=self.global_step,
                    metrics=enhanced_metrics,
                    attention_weights=alignments,
                    gate_outputs=gate_outputs
                )
            except Exception as e:
                self.logger.warning(f"Ошибка Telegram уведомления: {e}")
        
        # === Логирование в TensorBoard ===
        if self.tensorboard_writer is not None:
            try:
                # Основные метрики
                self.tensorboard_writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.tensorboard_writer.add_scalar("train/attention_diagonality", attention_diagonality, self.global_step)
                self.tensorboard_writer.add_scalar("train/gate_accuracy", gate_accuracy, self.global_step)
                
                # Детализированные loss компоненты
                self.tensorboard_writer.add_scalar("train/mel_loss", loss_dict.get('mel_loss', 0), self.global_step)
                self.tensorboard_writer.add_scalar("train/gate_loss", loss_dict.get('gate_loss', 0), self.global_step)
                self.tensorboard_writer.add_scalar("train/guide_loss", loss_dict.get('guide_loss', 0), self.global_step)
                self.tensorboard_writer.add_scalar("train/emb_loss", loss_dict.get('emb_loss', 0), self.global_step)
                
                # Градиенты и оптимизация
                self.tensorboard_writer.add_scalar("train/grad_norm", float(grad_norm), self.global_step)
                self.tensorboard_writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]['lr'], self.global_step)
                
                # Guided attention weight
                if hasattr(self.criterion, 'guide_loss_weight'):
                    self.tensorboard_writer.add_scalar("train/guided_attention_weight", self.criterion.guide_loss_weight, self.global_step)
                
                # Принудительно сохраняем
                self.tensorboard_writer.flush()
                
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка логирования в TensorBoard: {e}")
        
        # === Логирование в MLflow ===
        if MLFLOW_AVAILABLE:
            try:
                # Основные метрики
                mlflow.log_metric("train.loss", loss.item(), step=self.global_step)
                mlflow.log_metric("train.attention_diagonality", attention_diagonality, step=self.global_step)
                mlflow.log_metric("train.gate_accuracy", gate_accuracy, step=self.global_step)
                
                # Детализированные loss компоненты
                mlflow.log_metric("train.mel_loss", loss_dict.get('mel_loss', 0), step=self.global_step)
                mlflow.log_metric("train.gate_loss", loss_dict.get('gate_loss', 0), step=self.global_step)
                mlflow.log_metric("train.guide_loss", loss_dict.get('guide_loss', 0), step=self.global_step)
                mlflow.log_metric("train.emb_loss", loss_dict.get('emb_loss', 0), step=self.global_step)
                
                # Градиенты и оптимизация
                mlflow.log_metric("train.grad_norm", float(grad_norm), step=self.global_step)
                mlflow.log_metric("train.learning_rate", self.optimizer.param_groups[0]['lr'], step=self.global_step)
                
                # Guided attention weight
                if hasattr(self.criterion, 'guide_loss_weight'):
                    mlflow.log_metric("train.guided_attention_weight", self.criterion.guide_loss_weight, step=self.global_step)
                
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка логирования в MLflow: {e}")
        
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'loss_breakdown': loss_dict,
            'quality_analysis': quality_analysis,
            'attention_diagonality': attention_diagonality,
            'gate_accuracy': gate_accuracy,
            'grad_norm': float(grad_norm)
        }
    
    def validate_step(self, val_loader):
        """Выполняет валидацию с анализом качества."""
        self.model.eval()
        val_losses = []
        quality_metrics = []
        
        with torch.no_grad():
            for batch in val_loader:
                text_inputs, text_lengths, mel_targets, gate_targets, mel_lengths, ctc_text, ctc_text_lengths, guide_mask = batch
                
                # Перенос на GPU
                text_inputs = text_inputs.cuda()
                mel_targets = mel_targets.cuda()
                gate_targets = gate_targets.cuda()
                
                # Forward pass через parse_batch для правильной обработки всех элементов
                x, y = self.model.parse_batch(batch)
                model_outputs = self.model(x)
                # Модель может возвращать разное количество элементов в зависимости от настроек
                if len(model_outputs) >= 4:
                    mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model_outputs[:4]
                else:
                    # Fallback для случая, если модель вернула меньше элементов
                    mel_outputs = model_outputs[0] if len(model_outputs) > 0 else None
                    mel_outputs_postnet = model_outputs[1] if len(model_outputs) > 1 else None
                    gate_outputs = model_outputs[2] if len(model_outputs) > 2 else None
                    alignments = model_outputs[3] if len(model_outputs) > 3 else None
                
                # Вычисление loss
                loss_components = self.criterion(
                    model_outputs,
                    (mel_targets, gate_targets),
                    attention_weights=alignments,
                    gate_outputs=gate_outputs
                )
                
                # Tacotron2Loss возвращает 4 компонента: mel_loss, gate_loss, guide_loss, emb_loss
                if len(loss_components) == 4:
                    mel_loss, gate_loss, guide_loss, emb_loss = loss_components
                    # Объединяем все loss в один общий loss
                    loss = mel_loss + gate_loss + guide_loss + emb_loss
                    # Создаем словарь с детализацией
                    loss_dict = {
                        'mel_loss': mel_loss.item(),
                        'gate_loss': gate_loss.item(),
                        'guide_loss': guide_loss.item(),
                        'emb_loss': emb_loss.item(),
                        'total_loss': loss.item()
                    }
                else:
                    # Fallback для других loss функций
                    loss = loss_components[0] if len(loss_components) > 0 else torch.tensor(0.0)
                    loss_dict = {'total_loss': loss.item()}
                
                val_losses.append(loss.item())
                
                # Анализ качества
                if self.smart_tuner and len(quality_metrics) < 5:  # Анализируем первые 5 батчей
                    try:
                        quality_analysis = self.smart_tuner.on_batch_end(
                            self.current_epoch,
                            self.global_step,
                            loss_dict,
                            (mel_outputs_postnet, gate_outputs, alignments)
                        )
                        quality_metrics.append(quality_analysis.get('quality_score', 0.5))
                    except Exception as e:
                        self.logger.warning(f"Ошибка анализа качества валидации: {e}")
        
        avg_val_loss = np.mean(val_losses)
        avg_quality_score = np.mean(quality_metrics) if quality_metrics else 0.5
        
        # === Логирование в TensorBoard ===
        if self.tensorboard_writer is not None:
            try:
                # Основные метрики валидации
                self.tensorboard_writer.add_scalar("val/loss", avg_val_loss, self.global_step)
                self.tensorboard_writer.add_scalar("val/quality_score", avg_quality_score, self.global_step)
                
                # Дополнительные метрики валидации
                self.tensorboard_writer.add_scalar("val/epoch", self.current_epoch, self.global_step)
                self.tensorboard_writer.add_scalar("val/best_loss", self.best_validation_loss, self.global_step)
                
                # Принудительно сохраняем
                self.tensorboard_writer.flush()
                
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка логирования в TensorBoard (валидация): {e}")
        
        # === Логирование в MLflow ===
        if MLFLOW_AVAILABLE:
            try:
                # Основные метрики валидации
                mlflow.log_metric("val.loss", avg_val_loss, step=self.global_step)
                mlflow.log_metric("val.quality_score", avg_quality_score, step=self.global_step)
                
                # Дополнительные метрики валидации
                mlflow.log_metric("val.epoch", self.current_epoch, step=self.global_step)
                mlflow.log_metric("val.best_loss", self.best_validation_loss, step=self.global_step)
                
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка логирования в MLflow (валидация): {e}")
        
        return {
            'val_loss': avg_val_loss,
            'quality_score': avg_quality_score
        }
    
    def train_epoch(self, train_loader, val_loader):
        """Тренирует одну эпоху с полным мониторингом качества."""
        epoch_start_time = time.time()
        
        # Определяем текущую фазу обучения
        current_phase = self.get_current_training_phase()
        self.adjust_hyperparams_for_phase(current_phase)
        
        # Уведомление Smart Tuner о начале эпохи
        current_hyperparams = {
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'guided_attention_weight': self.criterion.guide_loss_weight,
            'epoch': self.current_epoch,
            'phase': current_phase
        }
        
        if self.smart_tuner:
            try:
                updated_hyperparams = self.smart_tuner.on_epoch_start(
                    self.current_epoch, current_hyperparams
                )
                
                # Применяем обновления гиперпараметров
                if updated_hyperparams != current_hyperparams:
                    self.logger.info("🔧 Smart Tuner обновил гиперпараметры")
            except Exception as e:
                self.logger.error(f"Ошибка Smart Tuner на старте эпохи: {e}")
        
        # Обучение
        train_losses = []
        quality_issues_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            step_result = self.train_step(batch)
            train_losses.append(step_result['loss'])
            
            # Подсчет проблем качества
            quality_analysis = step_result.get('quality_analysis', {})
            quality_issues_count += len(quality_analysis.get('quality_issues', []))
            
            # Логирование прогресса
            if batch_idx % 100 == 0:
                self.logger.info(
                    f"Эпоха {self.current_epoch}, батч {batch_idx}: "
                    f"loss={step_result['loss']:.4f}, "
                    f"quality_score={quality_analysis.get('quality_score', 0):.3f}"
                )
        
        # Валидация
        val_result = self.validate_step(val_loader)
        
        # Метрики эпохи
        epoch_metrics = {
            'train_loss': np.mean(train_losses),
            'val_loss': val_result['val_loss'],
            'quality_score': val_result['quality_score'],
            'epoch_time': time.time() - epoch_start_time,
            'phase': current_phase,
            'quality_issues': quality_issues_count
        }
        
        if self.smart_tuner:
            try:
                decision = self.smart_tuner.on_epoch_end(
                    self.current_epoch, epoch_metrics, current_hyperparams
                )
                
                # 🔥 ИСПРАВЛЕНИЕ: Правильная обработка решений Smart Tuner
                if decision and isinstance(decision, dict):
                    # Обработка решений Smart Tuner
                    if decision.get('early_stop', False):
                        self.logger.info(f"🛑 Smart Tuner рекомендует остановку: {decision.get('reason')}")
                        return False  # Сигнал остановки обучения
                    
                    # Применение обновлений гиперпараметров
                    if decision.get('hyperparameter_updates'):
                        self.apply_hyperparameter_updates(decision['hyperparameter_updates'])
                        self.logger.info("🔧 Smart Tuner применил обновления гиперпараметров")
                        
            except Exception as e:
                self.logger.error(f"Ошибка Smart Tuner на завершении эпохи: {e}")
        
        # Сохраняем метрики эпохи
        self.training_metrics_history.append(epoch_metrics)
        
        # Обновляем лучшую validation loss
        if val_result['val_loss'] < self.best_validation_loss:
            self.best_validation_loss = val_result['val_loss']
            self.logger.info(f"🏆 Новый рекорд validation loss: {self.best_validation_loss:.4f}")
        
        # === Логирование метрик эпохи в TensorBoard ===
        if self.tensorboard_writer is not None:
            try:
                self.tensorboard_writer.add_scalar("epoch/train_loss", epoch_metrics['train_loss'], self.current_epoch)
                self.tensorboard_writer.add_scalar("epoch/val_loss", epoch_metrics['val_loss'], self.current_epoch)
                self.tensorboard_writer.add_scalar("epoch/quality_score", epoch_metrics['quality_score'], self.current_epoch)
                self.tensorboard_writer.add_scalar("epoch/quality_issues", quality_issues_count, self.current_epoch)
                self.tensorboard_writer.add_scalar("epoch/time", epoch_metrics['epoch_time'], self.current_epoch)
                self.tensorboard_writer.add_scalar("epoch/phase", 0 if current_phase == 'pre_alignment' else 1, self.current_epoch)
                self.tensorboard_writer.flush()
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка логирования эпохи в TensorBoard: {e}")
        
        # === Логирование метрик эпохи в MLflow ===
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metric("epoch.train_loss", epoch_metrics['train_loss'], step=self.current_epoch)
                mlflow.log_metric("epoch.val_loss", epoch_metrics['val_loss'], step=self.current_epoch)
                mlflow.log_metric("epoch.quality_score", epoch_metrics['quality_score'], step=self.current_epoch)
                mlflow.log_metric("epoch.quality_issues", quality_issues_count, step=self.current_epoch)
                mlflow.log_metric("epoch.time", epoch_metrics['epoch_time'], step=self.current_epoch)
                mlflow.log_metric("epoch.phase", current_phase, step=self.current_epoch)
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка логирования эпохи в MLflow: {e}")
        
        # 🎵 Генерация тестового аудио каждые 5000 шагов
        if self.telegram_monitor and self.global_step % 5000 == 0:
            try:
                self.telegram_monitor.generate_and_send_test_audio(
                    step=self.global_step,
                    model=self.model,
                    hparams=self.hparams,
                    device=self.device
                )
            except Exception as e:
                self.logger.warning(f"Ошибка генерации тестового аудио: {e}")
        
        # 📱 Обычное Telegram уведомление каждые 1000 шагов
        if self.telegram_monitor and self.global_step % 1000 == 0:
            try:
                # Собираем метрики для отправки
                telegram_metrics = {
                    'step': self.global_step,
                    'train_loss': train_loss,
                    'val_loss': getattr(self, 'last_val_loss', 0),
                    'attention_diagonality': attention_diagonality,
                    'gate_accuracy': gate_accuracy,
                    'quality_score': overall_quality,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'phase': current_phase,
                    'epoch': getattr(self, 'current_epoch', 0)
                }
                
                # Отправляем обновление с изображениями attention
                self.telegram_monitor.send_training_update(
                    step=self.global_step,
                    metrics=telegram_metrics,
                    alignments=alignments
                )
                
            except Exception as e:
                self.logger.warning(f"Ошибка Telegram уведомления: {e}")
        
        # Логирование результатов эпохи
        self.logger.info(
            f"📊 Эпоха {self.current_epoch} завершена [{current_phase}]: "
            f"train_loss={epoch_metrics['train_loss']:.4f}, "
            f"val_loss={epoch_metrics['val_loss']:.4f}, "
            f"quality={epoch_metrics['quality_score']:.3f}, "
            f"проблем={quality_issues_count}, "
            f"время={epoch_metrics['epoch_time']:.1f}с"
        )
        
        return True  # Продолжить обучение
    
    def apply_hyperparameter_updates(self, updates: Dict[str, Any]):
        """Применяет обновления гиперпараметров от Smart Tuner."""
        for param_name, new_value in updates.items():
            if param_name == 'learning_rate':
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_value
                self.logger.info(f"🔧 Обновлен learning_rate: {new_value}")
                
            elif param_name == 'guide_loss_weight':
                self.criterion.guide_loss_weight = new_value
                self.logger.info(f"🔧 Обновлен guide_loss_weight: {new_value}")
                
            elif hasattr(self.hparams, param_name):
                setattr(self.hparams, param_name, new_value)
                self.logger.info(f"🔧 Обновлен {param_name}: {new_value}")
    
    def save_checkpoint(self, filename: str, metrics: Dict[str, Any]):
        """Сохраняет checkpoint модели."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_validation_loss': self.best_validation_loss,
            'hparams': vars(self.hparams),
            'metrics': metrics,
            'training_history': self.training_metrics_history
        }
        
        torch.save(checkpoint, filename)
        self.logger.info(f"💾 Checkpoint сохранен: {filename}")
    
    def train(self, train_loader, val_loader, max_epochs: Optional[int] = None):
        """
        Главный цикл обучения с полной интеграцией Smart Tuner.
        
        Args:
            train_loader: DataLoader для обучающих данных
            val_loader: DataLoader для валидационных данных  
            max_epochs: Максимальное количество эпох (None = определяется автоматически)
        """
        self.logger.info("🚀 Начинаем enhanced обучение Tacotron2")
        
        # Инициализация
        self.initialize_training()
        
        # Определение количества эпох
        if max_epochs is None:
            if self.smart_tuner and hasattr(self.smart_tuner, 'epoch_optimizer'):
                try:
                    analysis = self.smart_tuner.analyze_dataset_for_training(self.dataset_info)
                    max_epochs = analysis.get('optimal_epochs', 3000)
                    self.logger.info(f"📊 Smart Tuner рекомендует {max_epochs} эпох")
                except Exception as e:
                    self.logger.warning(f"Ошибка анализа датасета: {e}")
                    max_epochs = 3000
            else:
                max_epochs = 3000
        
        training_start_time = time.time()
        
        try:
            # Основной цикл обучения
            for epoch in range(max_epochs):
                self.current_epoch = epoch
                
                # Тренировка эпохи
                should_continue = self.train_epoch(train_loader, val_loader)
                
                if not should_continue:
                    self.logger.info("🏁 Обучение остановлено по рекомендации Smart Tuner")
                    break
                
                # Периодическое сохранение
                if epoch % 100 == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', 
                                       self.training_metrics_history[-1])
            
            # Завершение обучения
            total_training_time = (time.time() - training_start_time) / 3600  # в часах
            
            final_metrics = self.training_metrics_history[-1] if self.training_metrics_history else {}
            
            # Финальное сохранение
            self.save_checkpoint('final_model.pth', final_metrics)
            
            # Уведомление Smart Tuner о завершении
            if self.smart_tuner:
                try:
                    training_summary = self.smart_tuner.on_training_complete(
                        final_metrics, self.current_epoch, total_training_time
                    )
                    self.logger.info(f"📈 Smart Tuner анализ: {training_summary}")
                except Exception as e:
                    self.logger.error(f"Ошибка финального анализа: {e}")
            
            self.logger.info(
                f"🎉 Обучение завершено! Эпох: {self.current_epoch}, "
                f"Время: {total_training_time:.1f}ч, "
                f"Лучший val_loss: {self.best_validation_loss:.4f}"
            )
            
        except KeyboardInterrupt:
            self.logger.info("⏹️ Обучение прервано пользователем")
            self.save_checkpoint('interrupted_model.pth', 
                               self.training_metrics_history[-1] if self.training_metrics_history else {})
        
        except Exception as e:
            self.logger.error(f"❌ Ошибка обучения: {e}")
            # Сохраняем checkpoint при ошибке
            if self.training_metrics_history:
                self.save_checkpoint('error_model.pth', self.training_metrics_history[-1])
            raise
        
        finally:
            # Финальная статистика
            if self.training_metrics_history:
                self._print_training_summary()
        
        # === Завершение TensorBoard (только при нормальном завершении) ===
        if self.tensorboard_writer is not None:
            try:
                self.tensorboard_writer.close()
                self.logger.info("✅ TensorBoard writer закрыт")
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка закрытия TensorBoard: {e}")
        
        # === Завершение MLflow (только при нормальном завершении) ===
        if MLFLOW_AVAILABLE:
            try:
                mlflow.end_run()
                self.logger.info("✅ MLflow run завершен")
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка завершения MLflow run: {e}")
    
    def _print_training_summary(self):
        """Выводит сводку по обучению."""
        metrics = self.training_metrics_history
        
        if not metrics:
            return
        
        first_epoch = metrics[0]
        last_epoch = metrics[-1]
        
        train_loss_improvement = first_epoch['train_loss'] - last_epoch['train_loss']
        val_loss_improvement = first_epoch['val_loss'] - last_epoch['val_loss']
        quality_improvement = last_epoch.get('quality_score', 0) - first_epoch.get('quality_score', 0)
        
        self.logger.info("📋 СВОДКА ОБУЧЕНИЯ:")
        self.logger.info(f"   Эпох обучено: {len(metrics)}")
        self.logger.info(f"   Улучшение train_loss: {train_loss_improvement:.4f}")
        self.logger.info(f"   Улучшение val_loss: {val_loss_improvement:.4f}")
        self.logger.info(f"   Улучшение качества: {quality_improvement:.3f}")
        self.logger.info(f"   Лучший val_loss: {self.best_validation_loss:.4f}")
        
        # Фазы обучения
        phases_used = set(m.get('phase', 'unknown') for m in metrics)
        self.logger.info(f"   Использованные фазы: {', '.join(phases_used)}")
        
        # Среднее время эпохи
        avg_epoch_time = np.mean([m.get('epoch_time', 0) for m in metrics])
        self.logger.info(f"   Среднее время эпохи: {avg_epoch_time:.1f}с")

    # === 🔥 МИГРАЦИЯ ФУНКЦИЙ ИЗ TRAIN.PY ===
    
    def reduce_tensor(self, tensor, n_gpus):
        """Reduce tensor across GPUs (для distributed training)."""
        if n_gpus > 1:
            rt = tensor.clone()
            dist.all_reduce(rt, op=dist.ReduceOp.SUM)
            rt /= n_gpus
            return rt
        return tensor

    def init_distributed(self, hparams, n_gpus, rank, group_name):
        """Инициализация distributed training."""
        assert torch.cuda.is_available(), "Distributed mode requires CUDA."
        self.logger.info("Initializing Distributed")

        # Set cuda device so everything is done on the right GPU.
        torch.cuda.set_device(rank % torch.cuda.device_count())

        # Initialize distributed communication
        dist.init_process_group(
            backend=hparams.dist_backend,
            init_method=hparams.dist_url,
            world_size=n_gpus,
            rank=rank,
            group_name=group_name,
        )

        self.logger.info("Done initializing distributed")

    def load_model(self, hparams):
        """Загрузка модели с поддержкой distributed training."""
        model = Tacotron2(hparams).cuda()
        
        if hparams.distributed_run:
            from distributed import apply_gradient_allreduce
            model = apply_gradient_allreduce(model)

        return model

    def warm_start_model(self, checkpoint_path, model, ignore_layers, exclude=None):
        """Warm start модели из checkpoint."""
        assert os.path.isfile(checkpoint_path)
        self.logger.info(f"Warm starting model from checkpoint '{checkpoint_path}'")
        
        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_dict = checkpoint_dict["state_dict"]
        
        self.logger.info(f"ignoring layers: {ignore_layers}")
        if len(ignore_layers) > 0 or exclude:
            model_dict = {
                k: v
                for k, v in model_dict.items()
                if k not in ignore_layers and (not exclude or exclude not in k)
            }
        
        model.load_state_dict(model_dict, strict=False)
        return model

    def load_checkpoint(self, checkpoint_path, model, optimizer):
        """Загрузка checkpoint."""
        assert os.path.isfile(checkpoint_path)
        self.logger.info(f"Loading checkpoint '{checkpoint_path}'")
        
        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint_dict['state_dict'])
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        learning_rate = checkpoint_dict['learning_rate']
        iteration = checkpoint_dict['iteration']
        
        self.logger.info(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")
        return model, optimizer, learning_rate, iteration

    def save_checkpoint_legacy(self, model, optimizer, learning_rate, iteration, filepath):
        """Сохранение checkpoint в legacy формате (для совместимости)."""
        self.logger.info(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
        torch.save({'iteration': iteration,
                   'state_dict': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'learning_rate': learning_rate}, filepath)

    def setup_mixed_precision(self, hparams):
        """Настройка mixed precision (FP16/AMP)."""
        self.apex_available = False
        self.use_native_amp = False
        self.scaler = None

        if hparams.fp16_run:
            try:
                from apex import amp
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level="O2"
                )
                self.apex_available = True
                self.logger.info("✅ NVIDIA Apex успешно загружен для FP16 обучения")
            except ImportError:
                try:
                    from torch.amp import GradScaler, autocast
                    self.model = self.model.float()
                    self.scaler = GradScaler("cuda")
                    self.use_native_amp = True
                    self.logger.info("✅ Переключаемся на torch.amp (PyTorch Native AMP)")
                except ImportError as e:
                    hparams.fp16_run = False
                    self.logger.warning(f"❌ Mixed precision недоступна: {e}. FP16 отключён.")

    def setup_loss_functions(self, hparams):
        """Настройка всех loss функций."""
        # Основной loss
        self.criterion = Tacotron2Loss(hparams)
        
        # MMI Loss
        self.mmi_loss = None
        if hparams.use_mmi:
            try:
                from mmi_loss import MMI_loss
                self.mmi_loss = MMI_loss(hparams.mmi_map, hparams.mmi_weight)
                self.logger.info("✅ MMI loss загружен")
            except ImportError as e:
                self.logger.warning(f"⚠️ MMI loss недоступен: {e}")

        # Guided Attention Loss
        self.guide_loss = None
        if hparams.use_guided_attn:
            try:
                from loss_function import GuidedAttentionLoss
                self.guide_loss = GuidedAttentionLoss(alpha=hparams.guided_attn_weight)
                self.logger.info("✅ Guided Attention Loss загружен")
            except ImportError as e:
                self.logger.warning(f"⚠️ Guided Attention Loss недоступен: {e}")

    def setup_smart_tuner_components(self):
        """Настройка всех Smart Tuner компонентов."""
        # AdvancedQualityController
        try:
            from smart_tuner.advanced_quality_controller import AdvancedQualityController
            self.quality_ctrl = AdvancedQualityController()
            self.logger.info("🤖 AdvancedQualityController активирован")
        except Exception as e:
            self.quality_ctrl = None
            self.logger.warning(f"⚠️ Не удалось инициализировать AdvancedQualityController: {e}")

        # ParamScheduler
        try:
            from smart_tuner.param_scheduler import ParamScheduler
            self.sched_ctrl = ParamScheduler()
            self.logger.info("📅 ParamScheduler активирован")
        except Exception as e:
            self.sched_ctrl = None
            self.logger.warning(f"⚠️ Не удалось инициализировать ParamScheduler: {e}")

        # EarlyStopController
        try:
            from smart_tuner.early_stop_controller import EarlyStopController
            self.stop_ctrl = EarlyStopController()
            self.logger.info("🛑 EarlyStopController активирован")
        except Exception as e:
            self.stop_ctrl = None
            self.logger.warning(f"⚠️ Не удалось инициализировать EarlyStopController: {e}")

        # Debug Reporter
        try:
            from debug_reporter import initialize_debug_reporter
            self.debug_reporter = initialize_debug_reporter(self.telegram_monitor)
            self.logger.info("🔍 Debug Reporter активирован")
        except Exception as e:
            self.debug_reporter = None
            self.logger.warning(f"⚠️ Не удалось инициализировать Debug Reporter: {e}")

    def calculate_global_mean(self, data_loader, global_mean_npy):
        """Вычисление глобального среднего для нормализации."""
        if global_mean_npy and os.path.exists(global_mean_npy):
            self.logger.info(f"Loading global mean from {global_mean_npy}")
            return np.load(global_mean_npy)
        
        self.logger.info("Computing global mean...")
        global_mean = 0.0
        count = 0
        
        for batch in data_loader:
            mel = batch[1]  # mel spectrogram
            global_mean += mel.sum().item()
            count += mel.numel()
        
        global_mean /= count
        self.logger.info(f"Global mean computed: {global_mean}")
        
        if global_mean_npy:
            np.save(global_mean_npy, global_mean)
        
        return global_mean


def prepare_dataloaders(hparams):
    """
    Подготавливает train и val DataLoader с поддержкой dynamic padding, bucket batching и distributed.
    Возвращает train_loader, val_loader.
    """
    from data_utils import TextMelLoader, TextMelCollate
    try:
        from training_utils.dynamic_padding import DynamicPaddingCollator
        from training_utils.bucket_batching import BucketBatchSampler
    except ImportError:
        DynamicPaddingCollator = None
        BucketBatchSampler = None

    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)

    use_bucket_batching = getattr(hparams, 'use_bucket_batching', True)
    use_dynamic_padding = getattr(hparams, 'use_dynamic_padding', True)

    if use_dynamic_padding and DynamicPaddingCollator is not None:
        collate_fn = DynamicPaddingCollator(pad_value=0.0, n_frames_per_step=hparams.n_frames_per_step)
    else:
        collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if use_bucket_batching and BucketBatchSampler is not None:
        train_sampler = BucketBatchSampler(trainset, hparams.batch_size)
        shuffle = False
    else:
        if getattr(hparams, 'distributed_run', False):
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(trainset)
            shuffle = False
        else:
            train_sampler = None
            shuffle = True

    from torch.utils.data import DataLoader
    if use_bucket_batching and BucketBatchSampler is not None:
        train_loader = DataLoader(
            trainset,
            num_workers=1,
            pin_memory=False,
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
        )
    else:
        train_loader = DataLoader(
            trainset,
            num_workers=1,
            shuffle=shuffle,
            sampler=train_sampler,
            batch_size=hparams.batch_size,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_fn,
        )
    val_loader = DataLoader(
        valset,
        num_workers=1,
        shuffle=False,
        batch_size=hparams.batch_size,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def main():
    """Главная функция для запуска enhanced обучения."""
    # Создание гиперпараметров
    hparams = create_hparams()

    # Информация о датасете (может быть получена из анализа данных)
    dataset_info = {
        'total_duration_minutes': 120,  # Пример: 2 часа аудио
        'num_speakers': 1,
        'voice_complexity': 'moderate',  # simple, moderate, complex, very_complex
        'audio_quality': 'good',         # poor, fair, good, excellent
        'language': 'en'
    }

    # Подготовка DataLoader'ов
    train_loader, val_loader = prepare_dataloaders(hparams)

    # Создание тренера
    trainer = EnhancedTacotronTrainer(hparams, dataset_info)

    # Запуск обучения
    trainer.train(train_loader, val_loader)

    print("🚀 Enhanced Tacotron2 Training System завершил обучение!")


if __name__ == "__main__":
    main() 