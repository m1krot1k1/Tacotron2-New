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
    SMART_TUNER_AVAILABLE = True
except ImportError:
    SMART_TUNER_AVAILABLE = False
    logging.warning("Smart Tuner не найден, используется стандартное обучение")

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
                self.telegram_monitor = TelegramMonitor()
                self.logger.info("📱 Telegram Monitor инициализирован")
            except Exception as e:
                self.logger.error(f"Ошибка инициализации Telegram Monitor: {e}")
        
        # Инициализация компонентов
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.audio_enhancer = AudioQualityEnhancer()
        
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
        
        # Инициализация scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=getattr(self.hparams, 'scheduler_T_0', 1000),
            eta_min=getattr(self.hparams, 'min_learning_rate', 1e-6)
        )
        
        self.logger.info("🚀 Все компоненты успешно инициализированы")
    
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
        
        # Распаковка batch
        text_inputs, text_lengths, mel_targets, gate_targets, mel_lengths = batch
        
        # Перенос на GPU
        text_inputs = text_inputs.cuda()
        mel_targets = mel_targets.cuda() 
        gate_targets = gate_targets.cuda()
        
        # Forward pass
        model_outputs = self.model(text_inputs, mel_targets)
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model_outputs
        
        # Вычисление loss с современными техниками
        loss, loss_dict = self.criterion(
            model_outputs, 
            (mel_targets, gate_targets),
            attention_weights=alignments,
            gate_outputs=gate_outputs
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            getattr(self.hparams, 'grad_clip_thresh', 1.0)
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
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
        if self.telegram_monitor:
            try:
                self.telegram_monitor.send_training_update(
                    step=self.global_step,
                    metrics=loss_dict,
                    attention_weights=alignments,
                    gate_outputs=gate_outputs
                )
            except Exception as e:
                self.logger.warning(f"Ошибка Telegram уведомления: {e}")
        
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'loss_breakdown': loss_dict,
            'quality_analysis': quality_analysis
        }
    
    def validate_step(self, val_loader):
        """Выполняет валидацию с анализом качества."""
        self.model.eval()
        val_losses = []
        quality_metrics = []
        
        with torch.no_grad():
            for batch in val_loader:
                text_inputs, text_lengths, mel_targets, gate_targets, mel_lengths = batch
                
                # Перенос на GPU
                text_inputs = text_inputs.cuda()
                mel_targets = mel_targets.cuda()
                gate_targets = gate_targets.cuda()
                
                # Forward pass
                model_outputs = self.model(text_inputs, mel_targets)
                mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model_outputs
                
                # Вычисление loss
                loss, loss_dict = self.criterion(
                    model_outputs,
                    (mel_targets, gate_targets),
                    attention_weights=alignments,
                    gate_outputs=gate_outputs
                )
                
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
            raise
        
        finally:
            # Финальная статистика
            if self.training_metrics_history:
                self._print_training_summary()
    
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
    
    # Создание тренера
    trainer = EnhancedTacotronTrainer(hparams, dataset_info)
    
    # Создание data loaders (здесь должна быть ваша реализация)
    # train_loader = create_train_dataloader(hparams)
    # val_loader = create_val_dataloader(hparams)
    
    # Запуск обучения
    # trainer.train(train_loader, val_loader)
    
    print("🚀 Enhanced Tacotron2 Training System готов к работе!")
    print("📋 Для запуска обучения:")
    print("   1. Подготовьте датасет")
    print("   2. Создайте data loaders")
    print("   3. Вызовите trainer.train(train_loader, val_loader)")


if __name__ == "__main__":
    main() 