"""
Early Stop Controller для Smart Tuner V2
Интеллектуальный контроль раннего останова и проактивное вмешательство в обучение.
Теперь с адаптивным советником на основе базы знаний.
"""

import yaml
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
import sqlite3
import json
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - (EarlyStopController) - %(message)s')

class EarlyStopController:
    """
    Контроллер, который совмещает:
    1. Проактивные меры: пытается "вылечить" обучение, если оно идет не так.
    2. Ранний останов: останавливает безнадежное обучение для экономии ресурсов.
    3. Адаптивное обучение: использует базу знаний для принятия более умных решений со временем.
    """
    
    def __init__(self, config_path: str = "smart_tuner/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.advisor_config = self.config.get('adaptive_advisor', {})
        self.db_path = self.advisor_config.get('db_path', 'smart_tuner/advisor_kb.db')
        
        self.metrics_history = []
        self.last_action_step = 0 # Шаг, на котором было предпринято последнее действие
        self.last_action_info = {} # Информация о последнем действии для последующей оценки

        self.logger = logging.getLogger(self.__class__.__name__)
        self._init_kb()
        
        self.logger.info("Adaptive EarlyStopController инициализирован с базой знаний.")

    def _init_kb(self):
        """Инициализирует базу знаний (SQLite)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Создаем таблицу, если она не существует
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                problem_context TEXT NOT NULL,
                action_taken TEXT NOT NULL,
                outcome_metrics TEXT,
                reward REAL
            )
            ''')
            conn.commit()
            conn.close()
            self.logger.info(f"База знаний успешно инициализирована по пути: {self.db_path}")
        except Exception as e:
            self.logger.error(f"Не удалось инициализировать базу знаний: {e}")

    def add_metrics(self, metrics: Dict[str, float]):
        """Добавляет новый набор метрик в историю."""
        required_metrics = ['train_loss', 'val_loss', 'grad_norm'] # Будет расширяться
        if all(k in metrics for k in required_metrics):
            self.metrics_history.append(metrics)
    
    def decide_next_step(self, current_hparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Главный метод, принимающий решение о следующем шаге.
        """
        current_step = len(self.metrics_history)
        if current_step < self.advisor_config.get('min_history_for_decision', 10):
            return {'action': 'continue', 'reason': 'Накопление данных для принятия решения'}
        
        # 1. Если пришло время, оцениваем результат предыдущего действия
        evaluation_window = self.advisor_config.get('evaluation_window', 10)
        if self.last_action_step > 0 and (current_step - self.last_action_step) >= evaluation_window:
            self._evaluate_last_action()

        # 2. Анализируем текущую ситуацию, только если не ждем оценки прошлого действия
        if self.last_action_step == 0:
            problem_context = self._diagnose_problem()
            if problem_context:
                # 3. Обращаемся к "советнику" за лучшим действием
                recommended_action = self._get_best_action(problem_context)
                
                if recommended_action and recommended_action.get('name') != 'continue':
                    # 4. Применяем действие
                    self.last_action_step = current_step
                    # Запоминаем всю необходимую информацию для будущей оценки
                    self.last_action_info = {
                        "context": problem_context,
                        "action": recommended_action,
                        "start_metrics": self.metrics_history[-1]
                    }
                    
                    # Логируем само намерение, но без результата и оценки
                    db_id = self._log_event_to_kb(problem_context, recommended_action)
                    self.last_action_info['db_id'] = db_id

                    return self._create_response_from_action(recommended_action, current_hparams)

        return {'action': 'continue', 'reason': 'Ситуация стабильна или ожидается оценка действия'}
    
    def _diagnose_problem(self) -> Dict[str, Any] or None:
        """
        Анализирует "приборную панель" и ставит "диагноз".
        Возвращает словарь с описанием проблемы или None, если все в порядке.
        """
        conf = self.advisor_config.get('diagnostics', {})
        history_len = len(self.metrics_history)
        last_metrics = self.metrics_history[-1]

        # --- Диагноз №1: Нестабильность (самый высокий приоритет) ---
        instability_conf = conf.get('instability', {})
        if last_metrics['grad_norm'] > instability_conf.get('grad_norm_threshold', 50.0):
            self.logger.warning("Диагноз: Нестабильность (взрыв градиентов).")
            return {"problem_type": "instability", "grad_norm": last_metrics['grad_norm']}

        # --- Диагноз №2: Переобучение ---
        overfitting_conf = conf.get('overfitting', {})
        window = overfitting_conf.get('window_size', 10)
        if history_len >= window:
            overfitting_gap = last_metrics['val_loss'] - last_metrics['train_loss']
            if overfitting_gap > overfitting_conf.get('threshold', 0.1):
                # Проверяем, что разрыв устойчиво растет
                past_gaps = [m['val_loss'] - m['train_loss'] for m in self.metrics_history[-window:]]
                if all(g1 <= g2 for g1, g2 in zip(past_gaps, past_gaps[1:])):
                     self.logger.warning("Диагноз: Переобучение (разрыв между train/val loss растет).")
                     return {"problem_type": "overfitting", "gap": overfitting_gap}
        
        # --- Диагноз №3: Стагнация ---
        stagnation_conf = conf.get('stagnation', {})
        window = stagnation_conf.get('window_size', 20)
        if history_len >= window:
            recent_val_losses = [m['val_loss'] for m in self.metrics_history[-window:]]
            improvement = recent_val_losses[0] - recent_val_losses[-1]
            if improvement < stagnation_conf.get('min_delta', 0.005):
                self.logger.warning("Диагноз: Стагнация (val_loss на плато).")
                return {"problem_type": "stagnation", "improvement": improvement}
        
        return None
    
    def _get_best_action(self, context: Dict) -> Dict[str, Any]:
        """
        Запрашивает у базы знаний лучшее действие для данного контекста.
        Если релевантного опыта нет, возвращает действие по умолчанию.
        """
        problem_type = context.get("problem_type")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ищем все релевантные действия и их награды для данного типа проблемы
            cursor.execute("""
                SELECT action_taken, reward FROM knowledge_base
                WHERE json_extract(problem_context, '$.problem_type') = ? AND reward IS NOT NULL
            """, (problem_type,))
            
            records = cursor.fetchall()
            conn.close()

            if not records:
                self.logger.warning(f"Для проблемы '{problem_type}' в базе знаний нет опыта. Использую действие по умолчанию.")
                return self._get_default_action(problem_type)

            # Агрегируем опыт: считаем среднюю награду для каждого уникального действия
            action_rewards = {}
            for action_str, reward in records:
                action = json.loads(action_str)
                action_name = action['name'] # Уникальный ключ для действия
                if action_name not in action_rewards:
                    action_rewards[action_name] = []
                action_rewards[action_name].append(reward)
            
            # Находим действие с лучшей средней наградой
            best_action_name = None
            max_avg_reward = -float('inf')
            
            for action_name, rewards in action_rewards.items():
                avg_reward = sum(rewards) / len(rewards)
                self.logger.info(f"Анализ опыта для '{problem_type}': Действие '{action_name}' имеет среднюю награду {avg_reward:.4f} на основе {len(rewards)} случаев.")
                if avg_reward > max_avg_reward:
                    max_avg_reward = avg_reward
                    best_action_name = action_name
            
            # Если лучшее действие имеет отрицательную награду, возможно, лучше ничего не делать
            if max_avg_reward < self.advisor_config.get('min_reward_threshold', 0):
                 self.logger.warning(f"Лучшее действие '{best_action_name}' имеет отрицательную среднюю награду. Пропускаю ход.")
                 return {'name': 'continue'}

            # Загружаем полное описание лучшего действия из конфига (или можно хранить в БЗ)
            # Для простоты пока ищем в default_actions
            # TODO: Сделать действия более независимыми от конфига
            for action_type, action_details in self.advisor_config.get('default_actions', {}).items():
                if action_details['name'] == best_action_name:
                    self.logger.info(f"На основе исторического опыта выбрано действие: '{best_action_name}'")
                    return action_details

        except Exception as e:
            self.logger.error(f"Ошибка при запросе к базе знаний: {e}. Использую действие по умолчанию.")
            return self._get_default_action(problem_type)

        return self._get_default_action(problem_type) # На случай если что-то пошло не так

    def _get_default_action(self, problem_type: str) -> Dict[str, Any]:
        """Возвращает действие по умолчанию из конфигурации."""
        default_actions = self.advisor_config.get('default_actions', {})
        if problem_type in default_actions:
            self.logger.info(f"Применяю действие по умолчанию для '{problem_type}' из конфига.")
            return default_actions[problem_type]
        
        self.logger.warning(f"Не найдено действий по умолчанию для проблемы: {problem_type}. Продолжаю без изменений.")
        return {'name': 'continue'}

    def _evaluate_last_action(self):
        """Оценивает результат последнего действия и записывает "награду" в базу знаний."""
        start_metrics = self.last_action_info['start_metrics']
        end_metrics = self.metrics_history[-1]
        action_name = self.last_action_info['action']['name']
        
        conf = self.advisor_config.get('reward_function', {})
        weights = conf.get('weights', {'val_loss': 1.0, 'overfitting_gap': 0.5})

        # Компонент 1: Улучшение val_loss
        val_loss_improvement = start_metrics['val_loss'] - end_metrics['val_loss']
        
        # Компонент 2: Улучшение разрыва переобучения
        start_gap = start_metrics['val_loss'] - start_metrics['train_loss']
        end_gap = end_metrics['val_loss'] - end_metrics['train_loss']
        gap_improvement = start_gap - end_gap
        
        # Компонент 3 (опционально): Стабилизация градиентов
        # ... можно добавить в будущем ...

        # Итоговая награда
        reward = (weights.get('val_loss', 1.0) * val_loss_improvement +
                  weights.get('overfitting_gap', 0.5) * gap_improvement)

        # Штраф за бесполезное действие
        if abs(val_loss_improvement) < conf.get('action_inaction_threshold', 0.0001):
            reward -= conf.get('inaction_penalty', 0.01)
            self.logger.info(f"Действие '{action_name}' было практически бесполезным, применен штраф.")

        self.logger.info(f"Оценка действия '{action_name}': val_loss_imp={val_loss_improvement:.4f}, gap_imp={gap_improvement:.4f}. Итоговая награда: {reward:.4f}")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
            UPDATE knowledge_base 
            SET outcome_metrics = ?, reward = ?
            WHERE id = ?
            ''', (json.dumps(end_metrics), reward, self.last_action_info['db_id']))
            conn.commit()
            conn.close()
            self.logger.info(f"База знаний обновлена с результатом для действия ID: {self.last_action_info['db_id']}.")
        except Exception as e:
            self.logger.error(f"Не удалось обновить запись в базе знаний: {e}")

        # Сбрасываем, чтобы система могла анализировать ситуацию заново
        self.last_action_step = 0
        self.last_action_info = {}
        
    def _log_event_to_kb(self, context: Dict, action: Dict) -> int:
        """Записывает событие (контекст и предпринятое действие) в базу знаний."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO knowledge_base (problem_context, action_taken) VALUES (?, ?)",
                           (json.dumps(context), json.dumps(action)))
            conn.commit()
            last_id = cursor.lastrowid
            conn.close()
            return last_id
        except Exception as e:
            self.logger.error(f"Не удалось записать событие в базу знаний: {e}")
            return -1

    def _create_response_from_action(self, action: Dict, hparams: Dict) -> Dict:
        """Создает стандартизированный ответ на основе выбранного действия."""
        action_name = action.get('name')
        params = action.get('params', {})

        if action_name == 'stop_run':
            problem_type = self.last_action_info.get("context", {}).get("problem_type", "неизвестная проблема")
            self.logger.critical(f"Действие: ОСТАНОВКА ОБУЧЕНИЯ. Причина: диагностирована проблема '{problem_type}'.")
            return {'action': 'stop', 'reason': f'Диагностирована критическая проблема: {problem_type}'}
        
        if action_name == 'change_lr':
            multiplier = params.get('multiplier', 0.5)
            # Убедимся, что learning_rate есть в hparams
            if 'learning_rate' not in hparams:
                 self.logger.error("Невозможно изменить learning_rate, так как он отсутствует в переданных гиперпараметрах.")
                 return {'action': 'continue', 'reason': 'Ошибка при изменении LR: ключ отсутствует в hparams.'}

            current_lr = hparams['learning_rate']
            new_lr = current_lr * multiplier
            
            problem_type = self.last_action_info.get("context", {}).get("problem_type", "неизвестная")
            self.logger.info(f"Действие: изменяю learning_rate с {current_lr:.6f} на {new_lr:.6f} из-за проблемы '{problem_type}'")
            return {
                'action': 'update_hparams',
                'params': {'learning_rate': new_lr},
                'reason': f'Снижение LR для борьбы с проблемой: {problem_type}'
            }
            
        # Сюда можно будет добавить другие действия, например, 'increase_regularization'
        
        self.logger.warning(f"Получено неизвестное действие '{action_name}'. Продолжаю без изменений.")
        return {'action': 'continue', 'reason': f'Неизвестное действие: {action_name}'}

    def reset(self):
        """Сбрасывает состояние контроллера для нового run."""
        self.metrics_history = []
        self.last_action_step = 0
        self.last_action_info = {}
        self.logger.info("Adaptive EarlyStopController был полностью сброшен.")