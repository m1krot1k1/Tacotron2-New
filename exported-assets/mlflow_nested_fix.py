
# ФАЙЛ: smart_tuner_main.py
# ПРОБЛЕМА: MLflow parameter overwrite error

import mlflow
import uuid

class MLflowManager:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.parent_run = None

    def start_parent_run(self, run_name=None):
        '''
        Запуск родительского run для оптимизации
        '''
        mlflow.set_experiment(self.experiment_name)

        self.parent_run = mlflow.start_run(
            run_name=run_name or f"smart_tuner_{int(time.time())}"
        )

        # Логируем общие параметры только один раз
        mlflow.log_param("optimization_engine", "smart_tuner_v2")
        mlflow.log_param("framework", "tacotron2")

        print(f"✅ Родительский run запущен: {self.parent_run.info.run_id}")
        return self.parent_run

    def start_trial_run(self, trial_number, trial_params):
        '''
        Запуск дочернего run для каждого trial
        '''
        if not self.parent_run:
            raise Exception("Родительский run не запущен!")

        # Создаем уникальное имя для trial
        trial_name = f"trial_{trial_number}_{uuid.uuid4().hex[:8]}"

        trial_run = mlflow.start_run(
            run_name=trial_name,
            nested=True
        )

        # Логируем параметры trial без конфликтов
        trial_params_prefixed = {
            f"trial_{trial_number}_{k}": v for k, v in trial_params.items()
        }

        mlflow.log_params(trial_params_prefixed)
        mlflow.log_param("trial_number", trial_number)

        print(f"✅ Trial run запущен: {trial_number}")
        return trial_run

    def log_trial_metrics(self, metrics, step=None):
        '''
        Безопасное логирование метрик trial
        '''
        try:
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and not math.isnan(value):
                    mlflow.log_metric(metric_name, value, step=step)
        except Exception as e:
            print(f"⚠️ Ошибка логирования метрик: {e}")

    def end_trial_run(self):
        '''
        Завершение trial run
        '''
        try:
            mlflow.end_run()
        except Exception as e:
            print(f"⚠️ Ошибка завершения trial run: {e}")

    def end_parent_run(self):
        '''
        Завершение родительского run
        '''
        try:
            if self.parent_run:
                mlflow.end_run()
                self.parent_run = None
        except Exception as e:
            print(f"⚠️ Ошибка завершения parent run: {e}")

# ИСПОЛЬЗОВАНИЕ:
# mlflow_manager = MLflowManager("tacotron2_optimization")
# mlflow_manager.start_parent_run("smart_tuner_experiment")
# 
# for trial in trials:
#     trial_run = mlflow_manager.start_trial_run(trial.number, trial.params)
#     # ... обучение ...
#     mlflow_manager.log_trial_metrics({"loss": loss_value})
#     mlflow_manager.end_trial_run()
#
# mlflow_manager.end_parent_run()
