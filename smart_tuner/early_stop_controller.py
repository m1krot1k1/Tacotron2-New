"""
Early Stop Controller для Smart Tuner V2
Интеллектуальный контроль раннего останова и проактивное вмешательство в обучение.
Теперь с адаптивным советником на основе базы знаний и TTS-специфичными возможностями.
"""

import yaml
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import sqlite3
import json
import os

# Настраиваем логирование только для критически важных сообщений
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - [%(levelname)s] - (EarlyStopController) - %(message)s')

class EarlyStopController:
    """
    Усовершенствованный контроллер для TTS, который совмещает:
    1. Проактивные меры: пытается "вылечить" обучение, если оно идет не так.
    2. Ранний останов: останавливает безнадежное обучение для экономии ресурсов.
    3. Адаптивное обучение: использует базу знаний для принятия более умных решений со временем.
    4. TTS-специфичная диагностика: понимает особенности обучения Tacotron2.
    5. Фазовое обучение: адаптируется к различным фазам обучения TTS модели.
    """
    
    def __init__(self, config_path: str = "smart_tuner/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.advisor_config = self.config.get('adaptive_advisor', {})
        self.db_path = self.advisor_config.get('db_path', 'smart_tuner/advisor_kb.db')
        
        self.metrics_history = []
        self.last_action_step = 0
        self.last_action_info = {}
        
        # TTS-специфичные настройки
        self.tts_phase_config = self.config.get('tts_phase_training', {})
        self.tts_metrics_config = self.config.get('tts_metrics', {})
        self.current_phase = "pre_alignment"  # Начальная фаза
        self.phase_start_step = 0

        # Создаем "пустой" логгер, который ничего не делает
        class DummyLogger:
            def info(self, *args, **kwargs): pass
            def debug(self, *args, **kwargs): pass
            def warning(self, *args, **kwargs): pass
            def error(self, *args, **kwargs): pass
            def critical(self, *args, **kwargs): pass
        
        self.logger = DummyLogger()
        self._init_kb()

    def _init_kb(self):
        """Инициализирует базу знаний (SQLite) с TTS-специфичными полями."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Создаем расширенную таблицу для TTS
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                problem_context TEXT NOT NULL,
                action_taken TEXT NOT NULL,
                outcome_metrics TEXT,
                reward REAL,
                tts_phase TEXT,
                attention_score REAL,
                gate_accuracy REAL,
                mel_quality REAL
            )
            ''')
            conn.commit()
            conn.close()
            self.logger.info(f"TTS-специфичная база знаний инициализирована: {self.db_path}")
        except Exception as e:
            self.logger.error(f"Не удалось инициализировать базу знаний: {e}")

    def add_metrics(self, metrics: Dict[str, float]):
        """Добавляет новый набор метрик в историю с TTS-специфичной обработкой."""
        # Стандартные метрики для базовой функциональности
        required_base_metrics = ['train_loss', 'val_loss', 'grad_norm']
        
        # TTS-специфичные метрики (опциональные)
        tts_metrics = [
            'attention_alignment_score', 'gate_accuracy', 'mel_quality_score',
            'attention_entropy', 'gate_precision', 'gate_recall'
        ]
        
        # Проверяем наличие базовых метрик
        if all(k in metrics for k in required_base_metrics):
            # Обогащаем метрики TTS-специфичной информацией
            enriched_metrics = metrics.copy()
            enriched_metrics['step'] = len(self.metrics_history)
            enriched_metrics['tts_phase'] = self._determine_training_phase(enriched_metrics)
            
            self.metrics_history.append(enriched_metrics)
            self._update_current_phase(enriched_metrics)
            
            self.logger.debug(f"Добавлены метрики для шага {enriched_metrics['step']}, фаза: {enriched_metrics['tts_phase']}")
    
    def _determine_training_phase(self, metrics: Dict[str, float]) -> str:
        """Определяет текущую фазу обучения TTS на основе метрик."""
        if not self.tts_phase_config.get('enabled', False):
            return "standard"
            
        current_step = metrics.get('step', len(self.metrics_history))
        phases = self.tts_phase_config.get('phases', {})
        
        # Проверяем переход между фазами на основе метрик
        attention_score = metrics.get('attention_alignment_score', 0.0)
        gate_accuracy = metrics.get('gate_accuracy', 0.0)
        
        if current_step < 50 or attention_score < 0.5:
            return "pre_alignment"
        elif attention_score >= 0.5 and attention_score < 0.8:
            return "alignment_learning"  
        elif attention_score >= 0.8 and gate_accuracy >= 0.7:
            return "fine_tuning"
        else:
            return "alignment_learning"  # Фаза по умолчанию
    
    def _update_current_phase(self, metrics: Dict[str, float], telegram_monitor=None):
        """Обновляет текущую фазу обучения и логирует переходы."""
        new_phase = metrics.get('tts_phase', self.current_phase)
        if new_phase != self.current_phase:
            old_phase = self.current_phase
            self.logger.info(f"🔄 Переход фазы обучения: {old_phase} → {new_phase}")
            self.current_phase = new_phase
            self.phase_start_step = metrics.get('step', len(self.metrics_history))
            
            # 📱 Отправляем Telegram уведомление о смене фазы
            if telegram_monitor:
                step = metrics.get('step', len(self.metrics_history))
                
                # Формируем достижения предыдущей фазы
                achievements = []
                if old_phase == 'pre_alignment' and new_phase == 'alignment_learning':
                    achievements.append("Базовая инициализация attention завершена")
                elif old_phase == 'alignment_learning' and new_phase == 'quality_optimization':
                    achievements.append(f"Диагональность attention достигла {metrics.get('attention_diagonality', 0):.1%}")
                    achievements.append("Выравнивание текст-аудио стабилизировано")
                elif old_phase == 'quality_optimization' and new_phase == 'fine_tuning':
                    achievements.append(f"Точность gate достигла {metrics.get('gate_accuracy', 0):.1%}")
                    achievements.append("Качество mel-спектрограмм оптимизировано")
                
                try:
                    telegram_monitor.send_training_phase_notification(
                        old_phase=old_phase,
                        new_phase=new_phase,
                        step=step,
                        achievements=achievements
                    )
                except Exception as e:
                    self.logger.warning(f"⚠️ Не удалось отправить уведомление о смене фазы: {e}")

    def decide_next_step(self, current_hparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Главный метод для TTS-оптимизированного принятия решений.
        """
        current_step = len(self.metrics_history)
        min_history = self.advisor_config.get('min_history_for_decision', 200)
        
        if current_step < min_history:
            return {'action': 'continue', 'reason': f'Накопление данных для TTS анализа ({current_step}/{min_history})'}
        
        # 1. Оценка предыдущего действия
        evaluation_window = self.advisor_config.get('evaluation_window', 100)
        if self.last_action_step > 0 and (current_step - self.last_action_step) >= evaluation_window:
            self._evaluate_last_action_tts()

        # 2. TTS-специфичный анализ проблем
        if self.last_action_step == 0:
            problem_context = self._diagnose_tts_problems()
            if problem_context:
                # 3. Получение лучшего действия с учетом TTS фазы
                recommended_action = self._get_best_tts_action(problem_context)
                
                if recommended_action and recommended_action.get('name') != 'continue':
                    # 4. Применяем TTS-специфичное действие
                    self.last_action_step = current_step
                    self.last_action_info = {
                        "context": problem_context,
                        "action": recommended_action,
                        "start_metrics": self.metrics_history[-1],
                        "tts_phase": self.current_phase
                    }
                    
                    db_id = self._log_tts_event_to_kb(problem_context, recommended_action)
                    self.last_action_info['db_id'] = db_id

                    return self._create_tts_response_from_action(recommended_action, current_hparams)

        return {'action': 'continue', 'reason': f'TTS обучение стабильно в фазе {self.current_phase}'}
    
    def _diagnose_tts_problems(self) -> Optional[Dict[str, Any]]:
        """
        TTS-специфичная диагностика проблем с учетом фазы обучения.
        """
        conf = self.advisor_config.get('diagnostics', {})
        history_len = len(self.metrics_history)
        last_metrics = self.metrics_history[-1]
        current_phase = last_metrics.get('tts_phase', self.current_phase)

        # Приоритет 1: Критические проблемы TTS
        
        # Взрыв градиентов (особенно критично для TTS)
        instability_conf = conf.get('instability', {})
        grad_threshold = instability_conf.get('grad_norm_threshold', 200.0)
        if last_metrics['grad_norm'] > grad_threshold:
            self.logger.warning(f"🚨 TTS нестабильность: grad_norm={last_metrics['grad_norm']:.2f} > {grad_threshold}")
            return {
                "problem_type": "instability", 
                "grad_norm": last_metrics['grad_norm'],
                "tts_phase": current_phase,
                "severity": "critical"
            }

        # Attention failure (специфично для TTS)
        attention_failure_conf = conf.get('attention_failure', {})
        if 'attention_alignment_score' in last_metrics:
            min_alignment = attention_failure_conf.get('min_alignment_score', 0.7)
            if last_metrics['attention_alignment_score'] < min_alignment and current_phase != "pre_alignment":
                self.logger.warning(f"🎯 TTS attention failure: score={last_metrics['attention_alignment_score']:.3f} < {min_alignment}")
                return {
                    "problem_type": "attention_failure",
                    "attention_score": last_metrics['attention_alignment_score'],
                    "tts_phase": current_phase,
                    "severity": "high"
                }

        # Gate collapse (специфично для TTS)
        gate_collapse_conf = conf.get('gate_collapse', {})
        if 'gate_accuracy' in last_metrics:
            min_gate_acc = gate_collapse_conf.get('min_gate_accuracy', 0.8)
            if last_metrics['gate_accuracy'] < min_gate_acc and current_phase == "fine_tuning":
                self.logger.warning(f"🚪 TTS gate collapse: accuracy={last_metrics['gate_accuracy']:.3f} < {min_gate_acc}")
                return {
                    "problem_type": "gate_collapse",
                    "gate_accuracy": last_metrics['gate_accuracy'],
                    "tts_phase": current_phase,
                    "severity": "high"
                }

        # Приоритет 2: Стандартные проблемы с TTS-адаптацией

        # Переобучение (с учетом TTS специфики)
        overfitting_conf = conf.get('overfitting', {})
        window = overfitting_conf.get('window_size', 50)
        if history_len >= window:
            overfitting_gap = last_metrics['val_loss'] - last_metrics['train_loss']
            threshold = overfitting_conf.get('threshold', 5.0)
            if overfitting_gap > threshold:
                past_gaps = [m['val_loss'] - m['train_loss'] for m in self.metrics_history[-window:]]
                if len(past_gaps) > 10 and all(g1 <= g2 for g1, g2 in zip(past_gaps[-10:-5], past_gaps[-5:])):
                    self.logger.warning(f"📈 TTS переобучение: gap={overfitting_gap:.3f} > {threshold}")
                    return {
                        "problem_type": "overfitting", 
                        "gap": overfitting_gap,
                        "tts_phase": current_phase,
                        "severity": "medium"
                    }
        
        # Стагнация (с TTS-адаптированными порогами)
        stagnation_conf = conf.get('stagnation', {})
        window = stagnation_conf.get('window_size', 150)
        if history_len >= window:
            recent_val_losses = [m['val_loss'] for m in self.metrics_history[-window:]]
            improvement = recent_val_losses[0] - recent_val_losses[-1]
            min_delta = stagnation_conf.get('min_delta', 0.0005)
            
            if improvement < min_delta:
                self.logger.warning(f"📊 TTS стагнация: improvement={improvement:.5f} < {min_delta}")
                return {
                    "problem_type": "stagnation", 
                    "improvement": improvement,
                    "tts_phase": current_phase,
                    "severity": "low"
                }
        
        return None

    def _get_best_tts_action(self, context: Dict) -> Dict[str, Any]:
        """
        Получает лучшее действие для TTS с учетом фазы обучения и исторического опыта.
        """
        problem_type = context.get("problem_type")
        tts_phase = context.get("tts_phase", self.current_phase)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ищем релевантные действия с учетом TTS фазы
            cursor.execute("""
                SELECT action_taken, reward, attention_score, gate_accuracy FROM knowledge_base
                WHERE json_extract(problem_context, '$.problem_type') = ? 
                AND (tts_phase = ? OR tts_phase IS NULL)
                AND reward IS NOT NULL
                ORDER BY reward DESC
            """, (problem_type, tts_phase))
            
            records = cursor.fetchall()
            conn.close()

            if not records:
                self.logger.warning(f"Для TTS проблемы '{problem_type}' в фазе '{tts_phase}' нет опыта. Использую действие по умолчанию.")
                return self._get_default_tts_action(problem_type, tts_phase)

            # Анализируем лучшие действия для текущей TTS фазы
            action_scores = {}
            for action_str, reward, att_score, gate_acc in records:
                action = json.loads(action_str)
                action_name = action['name']
                
                # Бонус за TTS-специфичные улучшения
                tts_bonus = 0.0
                if att_score and att_score > 0.8:
                    tts_bonus += 0.1
                if gate_acc and gate_acc > 0.85:
                    tts_bonus += 0.1
                    
                adjusted_reward = reward + tts_bonus
                
                if action_name not in action_scores:
                    action_scores[action_name] = []
                action_scores[action_name].append(adjusted_reward)
            
            # Находим лучшее действие
            best_action_name = None
            max_avg_reward = -float('inf')
            
            for action_name, rewards in action_scores.items():
                avg_reward = sum(rewards) / len(rewards)
                self.logger.info(f"TTS анализ для '{problem_type}' в фазе '{tts_phase}': '{action_name}' награда {avg_reward:.4f}")
                if avg_reward > max_avg_reward:
                    max_avg_reward = avg_reward
                    best_action_name = action_name
            
            min_reward_threshold = self.advisor_config.get('min_reward_threshold', -0.1)
            if max_avg_reward < min_reward_threshold:
                self.logger.warning(f"Лучшее TTS действие '{best_action_name}' имеет низкую награду {max_avg_reward:.4f}. Пропускаю.")
                return {'name': 'continue'}

            # Загружаем полное TTS действие
            return self._get_default_tts_action(problem_type, tts_phase, preferred_action=best_action_name)
            
        except Exception as e:
            self.logger.error(f"Ошибка при поиске TTS действия: {e}")
            return self._get_default_tts_action(problem_type, tts_phase)

    def _get_default_tts_action(self, problem_type: str, tts_phase: str = None, preferred_action: str = None) -> Dict[str, Any]:
        """
        Возвращает TTS-специфичное действие по умолчанию с учетом фазы обучения.
        """
        default_actions = self.advisor_config.get('default_actions', {})
        
        if problem_type in default_actions:
            action = default_actions[problem_type].copy()
            
            # Адаптируем действие к TTS фазе
            if tts_phase and 'params' in action:
                phase_configs = self.tts_phase_config.get('phases', {})
                if tts_phase in phase_configs:
                    phase_config = phase_configs[tts_phase]
                    
                    # Корректируем параметры на основе фазы
                    if 'learning_rate_multiplier' in action['params'] and 'learning_rate_multiplier' in phase_config:
                        action['params']['learning_rate_multiplier'] *= phase_config['learning_rate_multiplier']
                    
                    if 'guided_attention_weight' in phase_config:
                        action['params']['guided_attention_weight'] = phase_config['guided_attention_weight']
            
            self.logger.info(f"Использую TTS действие по умолчанию для '{problem_type}' в фазе '{tts_phase}': {action['name']}")
            return action
        
        self.logger.warning(f"Нет TTS действия по умолчанию для '{problem_type}'")
        return {'name': 'continue'}

    def _evaluate_last_action_tts(self):
        """
        Оценивает результат последнего действия с учетом TTS-специфичных метрик.
        """
        if not self.last_action_info:
            return

        try:
            current_step = len(self.metrics_history)
            start_step = self.last_action_step
            evaluation_window = min(50, current_step - start_step)
            
            if evaluation_window < 10:
                return  # Недостаточно данных для оценки

            # Получаем метрики до и после действия
            before_metrics = self.last_action_info["start_metrics"]
            after_metrics = self.metrics_history[-evaluation_window:]
            
            # Вычисляем TTS-специфичную награду
            reward = self._calculate_tts_reward(before_metrics, after_metrics)
            
            # Обновляем базу знаний
            db_id = self.last_action_info.get('db_id')
            if db_id:
                self._update_kb_with_tts_reward(db_id, reward, after_metrics[-1])
            
            action_name = self.last_action_info["action"]["name"]
            tts_phase = self.last_action_info.get("tts_phase", "unknown")
            self.logger.info(f"🔍 TTS оценка действия '{action_name}' в фазе '{tts_phase}': награда {reward:.4f}")
            
            # Сброс для следующего цикла
            self.last_action_step = 0
            self.last_action_info = {}
            
        except Exception as e:
            self.logger.error(f"Ошибка при TTS оценке действия: {e}")

    def _calculate_tts_reward(self, before_metrics: Dict, after_metrics: List[Dict]) -> float:
        """
        Вычисляет награду с учетом TTS-специфичных улучшений.
        """
        if not after_metrics:
            return -1.0
            
        # Базовые веса из конфигурации
        reward_weights = self.advisor_config.get('reward_function', {}).get('weights', {})
        
        total_reward = 0.0
        total_weight = 0.0
        
        # Стандартные метрики
        if 'val_loss' in reward_weights:
            before_val_loss = before_metrics.get('val_loss', float('inf'))
            after_val_loss = np.mean([m.get('val_loss', float('inf')) for m in after_metrics[-10:]])
            val_loss_improvement = (before_val_loss - after_val_loss) / before_val_loss if before_val_loss > 0 else 0
            
            weight = reward_weights['val_loss']
            total_reward += val_loss_improvement * weight
            total_weight += weight
        
        # TTS-специфичные метрики
        tts_metrics = ['attention_alignment_score', 'gate_accuracy', 'mel_quality_score']
        for metric in tts_metrics:
            if metric in reward_weights:
                before_val = before_metrics.get(metric, 0.0)
                after_vals = [m.get(metric, 0.0) for m in after_metrics[-10:] if metric in m]
                if after_vals:
                    after_val = np.mean(after_vals)
                    improvement = (after_val - before_val) / max(before_val, 0.1)
                    
                    weight = reward_weights[metric]
                    total_reward += improvement * weight
                    total_weight += weight
        
        # Нормализация
        if total_weight > 0:
            normalized_reward = total_reward / total_weight
        else:
            normalized_reward = -0.5  # Штраф за отсутствие данных
            
        return max(-2.0, min(2.0, normalized_reward))  # Ограничиваем диапазон

    def _log_tts_event_to_kb(self, context: Dict, action: Dict) -> int:
        """
        Логирует TTS событие в базу знаний с расширенными полями.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Извлекаем TTS-специфичные данные
            last_metrics = self.metrics_history[-1] if self.metrics_history else {}
            tts_phase = context.get('tts_phase', self.current_phase)
            attention_score = last_metrics.get('attention_alignment_score')
            gate_accuracy = last_metrics.get('gate_accuracy')  
            mel_quality = last_metrics.get('mel_quality_score')
            
            cursor.execute("""
                INSERT INTO knowledge_base 
                (problem_context, action_taken, tts_phase, attention_score, gate_accuracy, mel_quality)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                json.dumps(context),
                json.dumps(action),
                tts_phase,
                attention_score,
                gate_accuracy,
                mel_quality
            ))
            
            event_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            self.logger.debug(f"TTS событие {event_id} записано в базу знаний")
            return event_id
            
        except Exception as e:
            self.logger.error(f"Ошибка записи TTS события в БЗ: {e}")
            return -1

    def _update_kb_with_tts_reward(self, db_id: int, reward: float, final_metrics: Dict):
        """
        Обновляет запись в базе знаний с TTS наградой и финальными метриками.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE knowledge_base 
                SET reward = ?, outcome_metrics = ?, attention_score = ?, gate_accuracy = ?, mel_quality = ?
                WHERE id = ?
            """, (
                reward,
                json.dumps(final_metrics),
                final_metrics.get('attention_alignment_score'),
                final_metrics.get('gate_accuracy'),
                final_metrics.get('mel_quality_score'),
                db_id
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"TTS награда {reward:.4f} обновлена для события {db_id}")
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления TTS награды в БЗ: {e}")

    def _create_tts_response_from_action(self, action: Dict, hparams: Dict, 
                                       step: int = 0, telegram_monitor=None) -> Dict:
        """
        Создает TTS-специфичный ответ на основе действия.
        """
        response = {
            'action': action['name'],
            'reason': f"TTS проактивная мера в фазе {self.current_phase}",
            'tts_phase': self.current_phase,
            'hparams_changes': {},
            'training_changes': {}
        }
        
        params = action.get('params', {})
        
        # Обработчики TTS-специфичных действий
        if action['name'] == 'guided_attention_boost':
            if 'guide_loss_weight_multiplier' in params:
                current_weight = hparams.get('guided_attention_weight', 1.0)
                new_weight = current_weight * params['guide_loss_weight_multiplier']
                response['hparams_changes']['guided_attention_weight'] = new_weight
                
            if 'learning_rate_multiplier' in params:
                current_lr = hparams.get('learning_rate', 0.001)
                new_lr = current_lr * params['learning_rate_multiplier']
                response['hparams_changes']['learning_rate'] = new_lr
                
        elif action['name'] == 'attention_regularization':
            if 'attention_dropout_increase' in params:
                current_dropout = hparams.get('attention_dropout', 0.1)
                new_dropout = min(0.5, current_dropout + params['attention_dropout_increase'])
                response['hparams_changes']['attention_dropout'] = new_dropout
                
            if 'gate_threshold_adjust' in params:
                current_threshold = hparams.get('gate_threshold', 0.5)
                new_threshold = current_threshold + params['gate_threshold_adjust']
                response['hparams_changes']['gate_threshold'] = max(0.1, min(0.9, new_threshold))
                
        elif action['name'] == 'attention_recovery':
            response['hparams_changes']['use_guided_attention'] = params.get('use_guided_attention', True)
            response['hparams_changes']['guided_attention_weight'] = params.get('guide_loss_weight', 2.0)
            
        elif action['name'] == 'gate_regularization':
            response['hparams_changes']['gate_loss_weight'] = params.get('gate_loss_weight', 1.5)
            response['hparams_changes']['gate_threshold'] = params.get('gate_threshold', 0.5)
            
        elif action['name'] == 'adaptive_learning_boost':
            if 'learning_rate_multiplier' in params:
                current_lr = hparams.get('learning_rate', 0.001)
                new_lr = current_lr * params['learning_rate_multiplier']
                response['hparams_changes']['learning_rate'] = new_lr
                
        self.logger.info(f"🎯 TTS действие '{action['name']}' подготовлено для фазы '{self.current_phase}'")
        
        # 📱 Отправляем Telegram уведомление об адаптивном действии
        if telegram_monitor and response['hparams_changes']:
            old_params = {}  
            new_params = response['hparams_changes']
            
            # Получаем старые значения для сравнения
            for param_name, new_value in new_params.items():
                old_params[param_name] = hparams.get(param_name, 'не задано')
            
            reason = response['reason']
            try:
                telegram_monitor.send_auto_improvement_notification(
                    improvement_type=action['name'],
                    old_params=old_params,
                    new_params=new_params,
                    reason=reason,
                    step=step
                )
            except Exception as e:
                self.logger.warning(f"⚠️ Не удалось отправить Telegram уведомление о действии: {e}")
        
        return response

    def should_stop_early(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        TTS-специфичная проверка раннего останова с мультикритериальной оценкой.
        """
        early_stop_config = self.config.get('early_stopping', {})
        if not early_stop_config.get('enabled', True):
            return False, "Early stopping отключен"
            
        multi_criteria = early_stop_config.get('multi_criteria', {})
        if not multi_criteria.get('enabled', False):
            # Стандартная проверка
            return self._standard_early_stop_check(metrics)
            
        # Мультикритериальная TTS проверка
        criteria = multi_criteria.get('criteria', {})
        stop_reasons = []
        
        for criterion_name, criterion_config in criteria.items():
            metric_name = criterion_name.replace('_', '.')
            if metric_name in metrics:
                should_stop, reason = self._check_single_criterion(
                    metrics[metric_name], criterion_name, criterion_config
                )
                if should_stop:
                    stop_reasons.append(reason)
        
        # Останавливаем, если выполнены критерии для основных TTS метрик
        critical_stops = [r for r in stop_reasons if any(word in r.lower() for word in ['attention', 'gate', 'validation'])]
        if len(critical_stops) >= 2:  # Минимум 2 критических критерия
            combined_reason = " и ".join(critical_stops[:2])
            return True, f"TTS мультикритериальный останов: {combined_reason}"
            
        return False, "TTS обучение продолжается"

    def _standard_early_stop_check(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """Стандартная проверка раннего останова."""
        early_stop_config = self.config.get('early_stopping', {})
        
        monitor_metric = early_stop_config.get('monitor', 'validation.loss')
        if monitor_metric not in metrics:
            return False, f"Метрика {monitor_metric} не найдена"
            
        patience = early_stop_config.get('patience', 150)
        min_delta = early_stop_config.get('min_delta', 0.0005)
        
        if len(self.metrics_history) < patience:
            return False, "Недостаточно истории для проверки"
            
        recent_values = [m.get(monitor_metric, float('inf')) for m in self.metrics_history[-patience:]]
        if len(recent_values) < patience:
            return False, "Недостаточно данных по метрике"
            
        best_value = min(recent_values[:-patience//2])
        current_value = recent_values[-1]
        
        if current_value - best_value > min_delta:
            return True, f"Ранний останов: {monitor_metric} не улучшается {patience} эпох"
            
        return False, "Метрика улучшается"

    def _check_single_criterion(self, current_value: float, criterion_name: str, config: Dict) -> Tuple[bool, str]:
        """Проверка одного критерия раннего останова."""
        patience = config.get('patience', 100)
        min_delta = config.get('min_delta', 0.001)
        mode = config.get('mode', 'min')
        
        if len(self.metrics_history) < patience:
            return False, f"Недостаточно истории для {criterion_name}"
            
        # Получаем историю для этого критерия
        metric_key = criterion_name.replace('_', '.')
        recent_values = []
        for m in self.metrics_history[-patience:]:
            if metric_key in m:
                recent_values.append(m[metric_key])
                
        if len(recent_values) < patience // 2:
            return False, f"Недостаточно данных для {criterion_name}"
            
        if mode == 'min':
            best_value = min(recent_values[:-patience//3])
            improvement = best_value - current_value
        else:  # mode == 'max'
            best_value = max(recent_values[:-patience//3])
            improvement = current_value - best_value
            
        if improvement < min_delta:
            return True, f"{criterion_name} стагнация"
            
        return False, f"{criterion_name} улучшается"

    def reset(self):
        """Сброс состояния контроллера для нового эксперимента."""
        self.metrics_history.clear()
        self.last_action_step = 0
        self.last_action_info.clear()
        self.current_phase = "pre_alignment"
        self.phase_start_step = 0
        self.logger.info("TTS EarlyStopController сброшен для нового эксперимента")

    def get_tts_training_summary(self) -> Dict[str, Any]:
        """Возвращает сводку TTS обучения с фазовой статистикой."""
        if not self.metrics_history:
            return {"status": "no_data"}
            
        last_metrics = self.metrics_history[-1]
        total_steps = len(self.metrics_history)
        
        # Статистика по фазам
        phase_stats = {}
        current_phase_count = 0
        for m in self.metrics_history:
            phase = m.get('tts_phase', 'unknown')
            if phase not in phase_stats:
                phase_stats[phase] = 0
            phase_stats[phase] += 1
            if phase == self.current_phase:
                current_phase_count += 1
        
        return {
            "status": "training",
            "current_phase": self.current_phase,
            "current_phase_duration": current_phase_count,
            "total_steps": total_steps,
            "phase_distribution": phase_stats,
            "latest_metrics": {
                "val_loss": last_metrics.get('val_loss', 0.0),
                "attention_score": last_metrics.get('attention_alignment_score', 0.0),
                "gate_accuracy": last_metrics.get('gate_accuracy', 0.0),
                "grad_norm": last_metrics.get('grad_norm', 0.0)
            },
            "tts_health": self._evaluate_tts_health(last_metrics)
        }
    
    def _evaluate_tts_health(self, metrics: Dict[str, float]) -> str:
        """Оценивает общее здоровье TTS обучения."""
        issues = []
        
        if metrics.get('grad_norm', 0) > 300:
            issues.append("high_gradients")
        if metrics.get('attention_alignment_score', 1.0) < 0.5:
            issues.append("poor_attention")
        if metrics.get('gate_accuracy', 1.0) < 0.7:
            issues.append("gate_problems")
        if metrics.get('val_loss', 0) > 10.0:
            issues.append("high_loss")
            
        if not issues:
            return "healthy"
        elif len(issues) == 1:
            return "minor_issues"
        elif len(issues) == 2:
            return "moderate_issues"
        else:
            return "critical_issues"