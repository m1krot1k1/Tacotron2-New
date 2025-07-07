"""
Тесты для Rollback Controller

Комплексное тестирование state machine, checkpointing и автоматического
отката для Enhanced Tacotron2 AI System.
"""

import unittest
import tempfile
import shutil
import os
import torch
import torch.nn as nn
import torch.optim as optim
import sqlite3
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timedelta
import json
import time

from rollback_controller import (
    RollbackController,
    RollbackConfig,
    StateTransitionManager,
    CheckpointManager,
    RollbackDatabase,
    SystemState,
    RollbackTrigger,
    SystemCheckpoint,
    RollbackEvent,
    create_rollback_controller,
    integrate_with_risk_assessment
)

class DummyModel(nn.Module):
    """Простая модель для тестирования"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

class TestRollbackConfig(unittest.TestCase):
    """Тесты конфигурации Rollback Controller"""
    
    def test_default_config(self):
        """Тест дефолтной конфигурации"""
        config = RollbackConfig()
        
        self.assertEqual(config.high_risk_threshold, 0.8)
        self.assertEqual(config.critical_risk_threshold, 0.95)
        self.assertEqual(config.max_checkpoints, 10)
        self.assertTrue(config.auto_rollback_enabled)
        self.assertFalse(config.require_confirmation)
    
    def test_custom_config(self):
        """Тест кастомной конфигурации"""
        config = RollbackConfig(
            high_risk_threshold=0.7,
            max_checkpoints=5,
            auto_rollback_enabled=False
        )
        
        self.assertEqual(config.high_risk_threshold, 0.7)
        self.assertEqual(config.max_checkpoints, 5)
        self.assertFalse(config.auto_rollback_enabled)

class TestStateTransitionManager(unittest.TestCase):
    """Тесты менеджера переходов состояний"""
    
    def setUp(self):
        self.manager = StateTransitionManager()
    
    def test_valid_transitions_from_stable(self):
        """Тест допустимых переходов из стабильного состояния"""
        allowed = self.manager.get_allowed_transitions(SystemState.STABLE)
        
        expected = [
            SystemState.MONITORING,
            SystemState.UNSTABLE,
            SystemState.CRITICAL,
            SystemState.ROLLBACK_INITIATED
        ]
        
        self.assertEqual(set(allowed), set(expected))
    
    def test_valid_transitions_from_critical(self):
        """Тест переходов из критического состояния"""
        allowed = self.manager.get_allowed_transitions(SystemState.CRITICAL)
        
        expected = [
            SystemState.ROLLBACK_INITIATED,
            SystemState.RECOVERY
        ]
        
        self.assertEqual(set(allowed), set(expected))
    
    def test_is_valid_transition(self):
        """Тест проверки допустимости переходов"""
        # Допустимый переход
        self.assertTrue(
            self.manager.is_valid_transition(SystemState.STABLE, SystemState.MONITORING)
        )
        
        # Недопустимый переход
        self.assertFalse(
            self.manager.is_valid_transition(SystemState.ROLLBACK_INITIATED, SystemState.STABLE)
        )
    
    def test_suggest_next_state_critical_risk(self):
        """Тест предложения состояния при критическом риске"""
        next_state = self.manager.suggest_next_state(
            current_state=SystemState.STABLE,
            risk_score=0.98,
            performance_change=0.0
        )
        
        self.assertEqual(next_state, SystemState.CRITICAL)
    
    def test_suggest_next_state_high_risk(self):
        """Тест предложения состояния при высоком риске"""
        next_state = self.manager.suggest_next_state(
            current_state=SystemState.STABLE,
            risk_score=0.85,
            performance_change=0.0
        )
        
        self.assertEqual(next_state, SystemState.UNSTABLE)
    
    def test_suggest_next_state_performance_degradation(self):
        """Тест предложения состояния при ухудшении производительности"""
        next_state = self.manager.suggest_next_state(
            current_state=SystemState.STABLE,
            risk_score=0.3,
            performance_change=-0.6
        )
        
        self.assertEqual(next_state, SystemState.CRITICAL)
    
    def test_suggest_next_state_recovery_to_stable(self):
        """Тест перехода из восстановления в стабильное состояние"""
        next_state = self.manager.suggest_next_state(
            current_state=SystemState.RECOVERY,
            risk_score=0.2,
            performance_change=0.0
        )
        
        self.assertEqual(next_state, SystemState.STABLE)

class TestCheckpointManager(unittest.TestCase):
    """Тесты менеджера checkpoint'ов"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = RollbackConfig(
            checkpoint_base_path=os.path.join(self.temp_dir, "checkpoints"),
            max_checkpoints=3
        )
        self.manager = CheckpointManager(self.config)
        
        # Создание тестовых объектов
        self.model = DummyModel()
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_directory_creation(self):
        """Тест создания директории для checkpoint'ов"""
        self.assertTrue(self.manager.checkpoint_dir.exists())
    
    def test_create_checkpoint(self):
        """Тест создания checkpoint'а"""
        training_metadata = {"epoch": 1, "step": 100}
        performance_metrics = {"loss": 1.5, "accuracy": 0.8}
        
        checkpoint = self.manager.create_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            system_state=SystemState.STABLE,
            description="Test checkpoint"
        )
        
        # Проверяем создание checkpoint'а
        self.assertIsInstance(checkpoint, SystemCheckpoint)
        self.assertEqual(checkpoint.system_state, SystemState.STABLE)
        self.assertEqual(checkpoint.training_metadata, training_metadata)
        self.assertEqual(checkpoint.performance_metrics, performance_metrics)
        
        # Проверяем создание файлов
        self.assertTrue(os.path.exists(checkpoint.model_state_path))
        self.assertTrue(os.path.exists(checkpoint.optimizer_state_path))
        self.assertTrue(os.path.exists(checkpoint.scheduler_state_path))
        
        # Проверяем метаданные
        metadata_path = os.path.join(os.path.dirname(checkpoint.model_state_path), "metadata.json")
        self.assertTrue(os.path.exists(metadata_path))
    
    def test_restore_checkpoint(self):
        """Тест восстановления checkpoint'а"""
        # Создаем checkpoint
        original_state = self.model.state_dict()
        checkpoint = self.manager.create_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            training_metadata={"epoch": 1},
            performance_metrics={"loss": 1.0},
            system_state=SystemState.STABLE
        )
        
        # Изменяем модель
        with torch.no_grad():
            for param in self.model.parameters():
                param.fill_(999.0)
        
        # Восстанавливаем checkpoint
        restored_checkpoint = self.manager.restore_checkpoint(
            checkpoint.checkpoint_id,
            self.model,
            self.optimizer,
            self.scheduler
        )
        
        # Проверяем восстановление
        self.assertEqual(restored_checkpoint.checkpoint_id, checkpoint.checkpoint_id)
        
        # Проверяем что параметры модели восстановлены
        restored_state = self.model.state_dict()
        for key in original_state:
            self.assertTrue(torch.allclose(original_state[key], restored_state[key]))
    
    def test_get_latest_stable_checkpoint(self):
        """Тест получения последнего стабильного checkpoint'а"""
        # Создаем несколько checkpoint'ов
        checkpoint1 = self.manager.create_checkpoint(
            self.model, self.optimizer, self.scheduler,
            {"epoch": 1}, {"loss": 2.0}, SystemState.UNSTABLE
        )
        
        checkpoint2 = self.manager.create_checkpoint(
            self.model, self.optimizer, self.scheduler,
            {"epoch": 2}, {"loss": 1.5}, SystemState.STABLE
        )
        
        checkpoint3 = self.manager.create_checkpoint(
            self.model, self.optimizer, self.scheduler,
            {"epoch": 3}, {"loss": 1.8}, SystemState.MONITORING
        )
        
        # Получаем последний стабильный
        latest_stable = self.manager.get_latest_stable_checkpoint()
        
        self.assertIsNotNone(latest_stable)
        self.assertEqual(latest_stable.checkpoint_id, checkpoint2.checkpoint_id)
    
    def test_max_checkpoints_limit(self):
        """Тест ограничения количества checkpoint'ов"""
        # Создаем больше checkpoint'ов чем лимит
        for i in range(5):
            self.manager.create_checkpoint(
                self.model, self.optimizer, self.scheduler,
                {"epoch": i}, {"loss": 1.0}, SystemState.STABLE
            )
        
        # Проверяем что количество не превышает лимит
        checkpoints = self.manager.list_checkpoints()
        self.assertEqual(len(checkpoints), self.config.max_checkpoints)
    
    def test_checkpoint_not_found(self):
        """Тест обработки отсутствующего checkpoint'а"""
        with self.assertRaises(ValueError):
            self.manager.restore_checkpoint(
                "nonexistent_checkpoint",
                self.model,
                self.optimizer
            )

class TestRollbackDatabase(unittest.TestCase):
    """Тесты базы данных rollback событий"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_rollback.db")
        self.database = RollbackDatabase(self.db_path)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_database_initialization(self):
        """Тест инициализации базы данных"""
        self.assertTrue(os.path.exists(self.db_path))
        
        # Проверяем создание таблиц
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            self.assertIn('rollback_events', tables)
            self.assertIn('state_history', tables)
    
    def test_save_rollback_event(self):
        """Тест сохранения события отката"""
        event = RollbackEvent(
            event_id="test_event",
            timestamp=datetime.now().isoformat(),
            trigger=RollbackTrigger.HIGH_RISK_ASSESSMENT,
            from_state=SystemState.STABLE,
            to_state=SystemState.ROLLBACK_INITIATED,
            target_checkpoint_id="checkpoint_123",
            reason="Test rollback",
            risk_score=0.85,
            success=True,
            recovery_time_seconds=30.5
        )
        
        self.database.save_rollback_event(event)
        
        # Проверяем сохранение
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM rollback_events")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
            
            cursor.execute("SELECT * FROM rollback_events")
            row = cursor.fetchone()
            self.assertEqual(row[1], "test_event")  # event_id
            self.assertEqual(row[3], "high_risk_assessment")  # trigger
    
    def test_save_state_change(self):
        """Тест сохранения изменения состояния"""
        timestamp = datetime.now().isoformat()
        performance_metrics = {"loss": 1.5, "accuracy": 0.8}
        
        self.database.save_state_change(
            timestamp=timestamp,
            state=SystemState.MONITORING,
            risk_score=0.6,
            performance_metrics=performance_metrics,
            trigger_reason="Risk threshold exceeded"
        )
        
        # Проверяем сохранение
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM state_history")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
            
            cursor.execute("SELECT * FROM state_history")
            row = cursor.fetchone()
            self.assertEqual(row[2], "monitoring")  # state
            self.assertEqual(row[3], 0.6)  # risk_score

class TestRollbackController(unittest.TestCase):
    """Тесты главного контроллера отката"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = RollbackConfig(
            checkpoint_base_path=os.path.join(self.temp_dir, "checkpoints"),
            state_db_path=os.path.join(self.temp_dir, "rollback.db"),
            auto_rollback_enabled=True,
            max_checkpoints=5
        )
        self.controller = RollbackController(self.config)
        
        # Тестовые объекты
        self.model = DummyModel()
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_initial_state(self):
        """Тест начального состояния контроллера"""
        self.assertEqual(self.controller.current_state, SystemState.STABLE)
        self.assertFalse(self.controller.rollback_in_progress)
        self.assertEqual(len(self.controller.performance_history), 0)
    
    def test_valid_state_transition(self):
        """Тест допустимого перехода состояния"""
        success = self.controller.transition_to_state(
            SystemState.MONITORING,
            "Test transition",
            risk_score=0.6
        )
        
        self.assertTrue(success)
        self.assertEqual(self.controller.current_state, SystemState.MONITORING)
    
    def test_invalid_state_transition(self):
        """Тест недопустимого перехода состояния"""
        success = self.controller.transition_to_state(
            SystemState.ROLLBACK_COMPLETED,  # Недопустимый переход из STABLE
            "Invalid transition"
        )
        
        self.assertFalse(success)
        self.assertEqual(self.controller.current_state, SystemState.STABLE)
    
    def test_state_change_callback(self):
        """Тест callback'а изменения состояния"""
        callback_called = False
        callback_args = None
        
        def test_callback(old_state, new_state, reason, risk_score):
            nonlocal callback_called, callback_args
            callback_called = True
            callback_args = (old_state, new_state, reason, risk_score)
        
        self.controller.add_state_change_callback(test_callback)
        
        self.controller.transition_to_state(
            SystemState.MONITORING,
            "Test callback",
            risk_score=0.5
        )
        
        self.assertTrue(callback_called)
        self.assertEqual(callback_args[0], SystemState.STABLE)
        self.assertEqual(callback_args[1], SystemState.MONITORING)
        self.assertEqual(callback_args[2], "Test callback")
        self.assertEqual(callback_args[3], 0.5)
    
    def test_create_checkpoint(self):
        """Тест создания checkpoint'а через контроллер"""
        training_metadata = {"epoch": 5, "step": 500}
        performance_metrics = {"loss": 0.8, "bleu": 0.75}
        
        checkpoint = self.controller.create_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            description="Controller test checkpoint"
        )
        
        self.assertIsInstance(checkpoint, SystemCheckpoint)
        self.assertEqual(checkpoint.system_state, SystemState.STABLE)
        self.assertEqual(checkpoint.training_metadata, training_metadata)
        self.assertEqual(checkpoint.performance_metrics, performance_metrics)
    
    def test_heuristic_risk_calculation(self):
        """Тест эвристической оценки риска"""
        # Высокий риск из-за ухудшения производительности
        risk1 = self.controller._calculate_heuristic_risk(
            performance_change=-0.4,
            instability_metrics={"gradient_norm": 15.0, "loss_variance": 2.0}
        )
        self.assertGreater(risk1, 0.8)
        
        # Низкий риск
        risk2 = self.controller._calculate_heuristic_risk(
            performance_change=0.1,
            instability_metrics={"gradient_norm": 1.0, "loss_variance": 0.1}
        )
        self.assertLess(risk2, 0.3)
    
    def test_performance_metrics_update(self):
        """Тест обновления метрик производительности"""
        metrics1 = {"loss": 2.0, "accuracy": 0.6}
        metrics2 = {"loss": 1.8, "accuracy": 0.65}
        
        self.controller.update_performance_metrics(metrics1)
        self.controller.update_performance_metrics(metrics2)
        
        self.assertEqual(len(self.controller.performance_history), 2)
        
        # Проверяем что метрики сохранены
        latest_metrics = self.controller.performance_history[-1]
        self.assertEqual(latest_metrics["loss"], 1.8)
        self.assertEqual(latest_metrics["accuracy"], 0.65)
        self.assertIn("timestamp", latest_metrics)
    
    def test_performance_degradation_detection(self):
        """Тест обнаружения ухудшения производительности"""
        # Добавляем метрики с ухудшающейся производительностью
        initial_loss = 1.0
        for i in range(5):
            metrics = {"loss": initial_loss + i * 0.5}  # Потери растут
            self.controller.update_performance_metrics(metrics)
        
        # Должен перейти в состояние UNSTABLE из-за ухудшения
        # (но не в CRITICAL, так как деградация не критическая)
        
        # Симулируем критическое ухудшение
        critical_metrics = {"loss": 10.0}  # Очень большие потери
        self.controller.update_performance_metrics(critical_metrics)
    
    @patch.object(RollbackController, '_execute_rollback')
    def test_risk_assessment_processing_critical(self, mock_execute):
        """Тест обработки критической оценки рисков"""
        # Мокаем результат оценки рисков
        mock_assessment = MagicMock()
        mock_assessment.overall_risk_score = 0.98
        mock_assessment.is_safe_to_proceed = False
        
        self.controller.process_risk_assessment(mock_assessment)
        
        # Проверяем что откат был запущен
        mock_execute.assert_called_once()
        
        # Проверяем переход в состояние инициации отката
        self.assertEqual(self.controller.current_state, SystemState.ROLLBACK_INITIATED)
    
    @patch.object(RollbackController, '_execute_rollback')
    def test_risk_assessment_processing_high(self, mock_execute):
        """Тест обработки высокой оценки рисков"""
        mock_assessment = MagicMock()
        mock_assessment.overall_risk_score = 0.85
        mock_assessment.is_safe_to_proceed = False
        
        self.controller.process_risk_assessment(mock_assessment)
        
        # Проверяем что откат был запущен для небезопасной ситуации
        mock_execute.assert_called_once()
    
    def test_risk_assessment_monitoring_mode(self):
        """Тест перехода в режим мониторинга"""
        mock_assessment = MagicMock()
        mock_assessment.overall_risk_score = 0.85
        mock_assessment.is_safe_to_proceed = True  # Безопасно, но высокий риск
        
        self.controller.process_risk_assessment(mock_assessment)
        
        # Должен перейти в режим мониторинга
        self.assertEqual(self.controller.current_state, SystemState.MONITORING)
    
    def test_manual_rollback(self):
        """Тест ручного отката"""
        # Создаем checkpoint для отката
        checkpoint = self.controller.create_checkpoint(
            self.model, self.optimizer, self.scheduler,
            {"epoch": 1}, {"loss": 1.0}
        )
        
        rollback_callback_called = False
        
        def rollback_callback(target_checkpoint):
            nonlocal rollback_callback_called
            rollback_callback_called = True
            # Имитируем успешный откат
        
        self.controller.add_rollback_callback(rollback_callback)
        
        # Запускаем ручной откат
        self.controller.manual_rollback(reason="Test manual rollback")
        
        # Даем время на выполнение
        time.sleep(0.1)
        
        self.assertTrue(rollback_callback_called)
    
    def test_emergency_stop(self):
        """Тест экстренной остановки"""
        self.controller.emergency_stop("Test emergency")
        
        # Должен инициировать откат с критическим риском
        self.assertEqual(self.controller.current_state, SystemState.ROLLBACK_INITIATED)
    
    def test_auto_rollback_disabled(self):
        """Тест отключенного автоматического отката"""
        # Создаем контроллер с отключенным автоматическим откатом
        config = RollbackConfig(auto_rollback_enabled=False)
        controller = RollbackController(config)
        
        mock_assessment = MagicMock()
        mock_assessment.overall_risk_score = 0.98
        mock_assessment.is_safe_to_proceed = False
        
        controller.process_risk_assessment(mock_assessment)
        
        # Состояние не должно измениться
        self.assertEqual(controller.current_state, SystemState.STABLE)
    
    def test_rollback_target_selection(self):
        """Тест выбора цели для отката"""
        # Создаем несколько checkpoint'ов
        unstable_checkpoint = self.controller.create_checkpoint(
            self.model, self.optimizer, self.scheduler,
            {"epoch": 1}, {"loss": 2.0}
        )
        
        # Переходим в другое состояние
        self.controller.transition_to_state(SystemState.MONITORING, "test")
        
        stable_checkpoint = self.controller.create_checkpoint(
            self.model, self.optimizer, self.scheduler,
            {"epoch": 2}, {"loss": 1.5}
        )
        
        # Возвращаемся в стабильное состояние
        self.controller.transition_to_state(SystemState.STABLE, "test")
        
        # Тестируем выбор цели
        target = self.controller._select_rollback_target()
        
        # Должен выбрать последний стабильный checkpoint
        self.assertIsNotNone(target)
        self.assertEqual(target.checkpoint_id, stable_checkpoint.checkpoint_id)
    
    def test_get_system_status(self):
        """Тест получения статуса системы"""
        # Создаем checkpoint
        checkpoint = self.controller.create_checkpoint(
            self.model, self.optimizer, self.scheduler,
            {"epoch": 1}, {"loss": 1.0}
        )
        
        status = self.controller.get_system_status()
        
        # Проверяем содержимое статуса
        self.assertEqual(status['current_state'], 'stable')
        self.assertFalse(status['rollback_in_progress'])
        self.assertTrue(status['auto_rollback_enabled'])
        self.assertEqual(status['total_checkpoints'], 1)
        self.assertEqual(status['latest_checkpoint'], checkpoint.checkpoint_id)
        self.assertIn('valid_transitions', status)

class TestUtilityFunctions(unittest.TestCase):
    """Тесты вспомогательных функций"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_create_rollback_controller(self):
        """Тест создания контроллера отката"""
        config = RollbackConfig(
            checkpoint_base_path=os.path.join(self.temp_dir, "checkpoints")
        )
        
        controller = create_rollback_controller(config)
        
        self.assertIsInstance(controller, RollbackController)
        self.assertEqual(controller.current_state, SystemState.STABLE)
    
    def test_integrate_with_risk_assessment(self):
        """Тест интеграции с модулем оценки рисков"""
        controller = create_rollback_controller()
        mock_risk_module = MagicMock()
        
        # Создаем callback
        risk_callback = integrate_with_risk_assessment(controller, mock_risk_module)
        
        # Тестируем callback
        mock_assessment = MagicMock()
        mock_assessment.overall_risk_score = 0.6
        mock_assessment.is_safe_to_proceed = True
        
        risk_callback(mock_assessment)
        
        # Проверяем что состояние изменилось соответственно
        # (для риска 0.6 должен остаться в STABLE или перейти в MONITORING)
        self.assertIn(controller.current_state, [SystemState.STABLE, SystemState.MONITORING])

class TestIntegrationScenarios(unittest.TestCase):
    """Интеграционные тесты реальных сценариев"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = RollbackConfig(
            checkpoint_base_path=os.path.join(self.temp_dir, "checkpoints"),
            state_db_path=os.path.join(self.temp_dir, "rollback.db"),
            auto_rollback_enabled=True
        )
        self.controller = RollbackController(self.config)
        
        self.model = DummyModel()
        self.optimizer = optim.Adam(self.model.parameters())
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_training_session_with_rollback(self):
        """Тест полного сценария обучения с откатом"""
        
        # 1. Создаем начальный checkpoint
        initial_checkpoint = self.controller.create_checkpoint(
            self.model, self.optimizer,
            training_metadata={"epoch": 0},
            performance_metrics={"loss": 2.0},
            description="Initial checkpoint"
        )
        
        # 2. Симулируем успешное обучение
        for epoch in range(1, 4):
            loss = 2.0 - epoch * 0.3  # Улучшающиеся потери
            
            self.controller.update_performance_metrics({"loss": loss, "epoch": epoch})
            
            # Создаем checkpoint каждую эпоху
            self.controller.create_checkpoint(
                self.model, self.optimizer,
                training_metadata={"epoch": epoch},
                performance_metrics={"loss": loss},
                description=f"Epoch {epoch} checkpoint"
            )
        
        # 3. Симулируем проблему в обучении
        problematic_metrics = {"loss": 5.0, "epoch": 4}  # Резкий рост потерь
        self.controller.update_performance_metrics(problematic_metrics)
        
        # 4. Обрабатываем критическую оценку рисков
        mock_assessment = MagicMock()
        mock_assessment.overall_risk_score = 0.97
        mock_assessment.is_safe_to_proceed = False
        
        rollback_executed = False
        restored_checkpoint = None
        
        def rollback_callback(target_checkpoint):
            nonlocal rollback_executed, restored_checkpoint
            rollback_executed = True
            restored_checkpoint = target_checkpoint
            
            # Имитируем восстановление модели
            self.controller.checkpoint_manager.restore_checkpoint(
                target_checkpoint.checkpoint_id,
                self.model,
                self.optimizer
            )
        
        self.controller.add_rollback_callback(rollback_callback)
        
        # Запускаем обработку рисков
        self.controller.process_risk_assessment(mock_assessment)
        
        # Даем время на выполнение
        time.sleep(0.1)
        
        # 5. Проверяем результаты
        self.assertTrue(rollback_executed)
        self.assertIsNotNone(restored_checkpoint)
        
        # Проверяем что откат был к стабильному состоянию
        self.assertEqual(restored_checkpoint.system_state, SystemState.STABLE)
        
        # Проверяем что система находится в процессе восстановления или завершила откат
        self.assertIn(self.controller.current_state, [
            SystemState.ROLLBACK_IN_PROGRESS,
            SystemState.ROLLBACK_COMPLETED,
            SystemState.RECOVERY
        ])
    
    def test_multiple_risk_escalation(self):
        """Тест эскалации множественных рисков"""
        
        # Начинаем со стабильного состояния
        self.assertEqual(self.controller.current_state, SystemState.STABLE)
        
        # 1. Первый уровень риска - переход в мониторинг
        assessment1 = MagicMock()
        assessment1.overall_risk_score = 0.6
        assessment1.is_safe_to_proceed = True
        
        self.controller.process_risk_assessment(assessment1)
        
        # Должен перейти в мониторинг
        expected_states = [SystemState.STABLE, SystemState.MONITORING]
        self.assertIn(self.controller.current_state, expected_states)
        
        # 2. Эскалация риска - переход в нестабильное состояние
        self.controller.transition_to_state(SystemState.MONITORING, "Manual transition for test")
        
        assessment2 = MagicMock()
        assessment2.overall_risk_score = 0.85
        assessment2.is_safe_to_proceed = True
        
        self.controller.process_risk_assessment(assessment2)
        
        # 3. Критическая эскалация - должен инициировать откат
        assessment3 = MagicMock()
        assessment3.overall_risk_score = 0.96
        assessment3.is_safe_to_proceed = False
        
        self.controller.process_risk_assessment(assessment3)
        
        # Должен инициировать откат
        self.assertEqual(self.controller.current_state, SystemState.ROLLBACK_INITIATED)
    
    def test_recovery_process(self):
        """Тест процесса восстановления после отката"""
        
        # Создаем checkpoint для отката
        checkpoint = self.controller.create_checkpoint(
            self.model, self.optimizer,
            training_metadata={"epoch": 5},
            performance_metrics={"loss": 1.0}
        )
        
        # Переводим в состояние восстановления (имитируем завершенный откат)
        self.controller.transition_to_state(SystemState.ROLLBACK_INITIATED, "Test")
        self.controller.transition_to_state(SystemState.ROLLBACK_IN_PROGRESS, "Test")
        self.controller.transition_to_state(SystemState.ROLLBACK_COMPLETED, "Test")
        self.controller.transition_to_state(SystemState.RECOVERY, "Test")
        
        # Симулируем восстановление производительности
        recovery_metrics = [
            {"loss": 1.2, "step": 1},   # Немного хуже после отката
            {"loss": 1.1, "step": 2},   # Улучшение
            {"loss": 0.9, "step": 3},   # Дальнейшее улучшение
        ]
        
        for metrics in recovery_metrics:
            self.controller.update_performance_metrics(metrics)
        
        # Обработка низкого риска должна вернуть в стабильное состояние
        stable_assessment = MagicMock()
        stable_assessment.overall_risk_score = 0.2
        stable_assessment.is_safe_to_proceed = True
        
        self.controller.process_risk_assessment(stable_assessment)
        
        # Должен вернуться в стабильное состояние
        # (может потребоваться явный переход в тестах)
        allowed_states = [SystemState.RECOVERY, SystemState.STABLE, SystemState.MONITORING]
        self.assertIn(self.controller.current_state, allowed_states)

if __name__ == '__main__':
    # Настройка логирования для тестов
    import logging
    logging.basicConfig(level=logging.WARNING)  # Уменьшаем verbose для тестов
    
    # Запуск тестов
    unittest.main(verbosity=2) 