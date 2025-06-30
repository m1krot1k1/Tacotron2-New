class AutoParamController:
    """Простой контроллер автоматической настройки гиперпараметров во время обучения.

    Алгоритм (можно усложнять позже):
      • если диагональность attention (alignment_score) < threshold -> удвоить вес guided attention (max 10.0).
      • если минимальный validation loss за последние N проверок не улучшается 3 проверки подряд — снижать LR вдвое (не ниже min_lr).
    """

    def __init__(self, optimizer, guide_loss, hparams, writer=None,
                 align_threshold: float = 0.15,
                 max_guided_weight: float = 10.0,
                 min_lr: float = 1e-6,
                 lr_decay_factor: float = 0.5,
                 patience: int = 3):
        self.optimizer = optimizer
        self.guide_loss = guide_loss
        self.hparams = hparams
        self.writer = writer
        self.align_threshold = align_threshold
        self.max_guided_weight = max_guided_weight
        self.min_lr = min_lr
        self.lr_decay_factor = lr_decay_factor
        self.patience = patience

        self._best_val_loss = None
        self._no_improve_steps = 0

    # -----------------------------------------------------
    def _log(self, message: str):
        if self.writer is not None:
            # Записываем текстовое сообщение в TensorBoard
            try:
                self.writer.add_text("AutoTune", message)
            except Exception:
                pass
        print(message)

    # -----------------------------------------------------
    def after_validation(self, iteration: int, val_loss: float, alignment_score: float):
        """Вызывается сразу после валидации."""
        # 1. Guided Attention Weight
        try:
            if alignment_score is not None and alignment_score < self.align_threshold:
                new_weight = min(self.max_guided_weight, self.guide_loss.get_weight() * 2)
                if new_weight > self.guide_loss.get_weight():
                    # обновляем GuideLoss и hparams
                    self.guide_loss.alpha = new_weight
                    self.guide_loss.current_weight = new_weight
                    self.hparams.guided_attn_weight = new_weight
                    self._log(f"🔧 AutoTune: Увеличили guided_attn_weight до {new_weight} (iter {iteration})")
                    if self.writer:
                        self.writer.add_scalar("autotune.guided_attn_weight", new_weight, iteration)
        except Exception as e:
            self._log(f"⚠️ AutoTune (guided_attn) ошибка: {e}")

        # 2. Learning rate scheduler на основе отсутсвия улучшений val_loss
        try:
            if self._best_val_loss is None or val_loss < self._best_val_loss - 1e-4:
                self._best_val_loss = val_loss
                self._no_improve_steps = 0
            else:
                self._no_improve_steps += 1

            if self._no_improve_steps >= self.patience:
                # снижаем LR
                for group in self.optimizer.param_groups:
                    new_lr = max(self.min_lr, group['lr'] * self.lr_decay_factor)
                    if new_lr < group['lr']:
                        group['lr'] = new_lr
                self._no_improve_steps = 0
                # Обновляем hparams.learning_rate для консистентности
                self.hparams.learning_rate = self.optimizer.param_groups[0]['lr']
                self._log(f"🔧 AutoTune: Снизили learning rate до {self.hparams.learning_rate} (iter {iteration})")
                if self.writer:
                    self.writer.add_scalar("autotune.learning_rate", self.hparams.learning_rate, iteration)
        except Exception as e:
            self._log(f"⚠️ AutoTune (lr) ошибка: {e}") 