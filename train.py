import os
import time
import argparse
import math
import random
import numpy as np
import copy

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import gradient_adaptive_factor
from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from hparams import create_hparams
from plotting_utils import (
    plot_alignment_to_numpy,
    plot_spectrogram_to_numpy,
    plot_gate_outputs_to_numpy,
)
from text import symbol_to_id
from utils import to_gpu

# from logger import Tacotron2Logger  # Не используется
from smart_tuner.advanced_quality_controller import AdvancedQualityController
from smart_tuner.intelligent_epoch_optimizer import IntelligentEpochOptimizer
from smart_tuner.param_scheduler import ParamScheduler
from smart_tuner.early_stop_controller import EarlyStopController
from gradient_stability_monitor import GradientStabilityMonitor
from debug_reporter import initialize_debug_reporter, get_debug_reporter

# MLflow for experiment tracking
try:
    import mlflow
    import mlflow.pytorch  # позволяет логировать модели PyTorch

    MLFLOW_AVAILABLE = True

    # Импортируем улучшенное логирование
    try:
        from mlflow_metrics_enhancer import (
            log_enhanced_training_metrics,
            log_system_metrics,
        )

        ENHANCED_LOGGING = True
        print("✅ Улучшенное MLflow логирование активировано")
    except ImportError:
        ENHANCED_LOGGING = False
        # MLflow логирование (скрыто для чистоты логов)

except ImportError:
    MLFLOW_AVAILABLE = False
    ENHANCED_LOGGING = False

# Подавляем лишние warning'и
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*torch.cuda.*DtypeTensor.*"
)
warnings.filterwarnings("ignore", message=".*deprecated.*")


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

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

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

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
    return train_loader, valset, collate_fn


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    # FP16 будет обрабатываться через AMP, модель остается в FP32

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers, exclude=None):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    # Добавляем weights_only=False для совместимости с PyTorch 2.6+
    checkpoint_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model_dict = checkpoint_dict["state_dict"]
    # model_dict['embedding.embedding.weight'] = model_dict['embedding.weight']
    # del model_dict['embedding.weight']
    print("ignoring layers:", ignore_layers)
    if len(ignore_layers) > 0 or exclude:
        model_dict = {
            k: v
            for k, v in model_dict.items()
            if k not in ignore_layers and (not exclude or exclude not in k)
        }

        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict, strict=False)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    # Добавляем weights_only=False для совместимости с PyTorch 2.6+
    checkpoint_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint_dict["state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    learning_rate = checkpoint_dict["learning_rate"]
    iteration = checkpoint_dict["iteration"]
    print("Loaded checkpoint '{}' from iteration {}".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, filepath
        )
    )
    torch.save(
        {
            "iteration": iteration,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        filepath,
    )


def validate(
    model,
    criterion,
    valset,
    iteration,
    batch_size,
    n_gpus,
    collate_fn,
    writer,
    distributed_run=False,
    rank=1,
    minimize=False,
):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(
            valset,
            sampler=val_sampler,
            num_workers=1,
            shuffle=False,
            batch_size=batch_size,
            pin_memory=False,
            collate_fn=collate_fn,
        )
        model.decoder.p_teacher_forcing = 0.0
        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x, minimize)
            _loss = criterion(y_pred, y)
            loss = sum(_loss)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    # 🔥 ФИНАЛЬНОЕ возвращение в train режим
    model.train()
    model.decoder.p_teacher_forcing = 1.0
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))

        # Логирование метрик и изображений напрямую через writer
        writer.add_scalar("validation.loss", val_loss, iteration)

        # Логирование изображений (взято из Tacotron2Logger)
        try:
            # 🔥 ИСПРАВЛЕНИЕ: Правильный inference с корректными размерами
            with torch.no_grad():
                # Используем eval режим для предотвращения BatchNorm ошибок
                model.eval()

                # 🔥 ВАЖНО: Используем данные из validation для правильных размеров
                # Вместо inference используем training forward pass для корректных размеров
                try:
                    # Используем реальные данные из validation batch
                    validation_outputs = model(x, minimize=False)

                    # validation_outputs: [decoder_outputs, mel_outputs, mel_outputs_postnet, gate_outputs, alignments, ...]
                    if len(validation_outputs) >= 5:
                        (
                            decoder_outputs_val,
                            mel_outputs_val,
                            mel_outputs_postnet_val,
                            gate_outputs_val,
                            alignments_val,
                        ) = validation_outputs[:5]

                        # Для изображений используем validation outputs (они корректного размера)
                        inference_outputs = [
                            None,
                            mel_outputs_val,
                            mel_outputs_postnet_val,
                            gate_outputs_val,
                            alignments_val,
                        ]
                        print(
                            f"✅ Validation forward pass: mel={mel_outputs_postnet_val.shape if mel_outputs_postnet_val is not None else 'None'}, "
                            f"gate={gate_outputs_val.shape if gate_outputs_val is not None else 'None'}, "
                            f"align={alignments_val.shape if alignments_val is not None else 'None'}"
                        )
                    else:
                        print(
                            f"⚠️ Validation outputs неполные: {len(validation_outputs)} элементов"
                        )
                        inference_outputs = None

                except Exception as val_e:
                    print(f"⚠️ Ошибка validation forward pass: {val_e}")

                    # Fallback к inference с более безопасными параметрами
                    try:
                        # Берем первый элемент батча
                        input_text = x[0][:1] if x[0].size(0) > 0 else x[0]

                        if input_text.size(0) == 0:
                            print("⚠️ Пустой батч для создания изображений")
                            inference_outputs = None
                        else:
                            inference_outputs = model.inference(input_text)
                            print(f"📝 Fallback inference завершен")
                    except Exception as inf_e:
                        print(f"⚠️ Ошибка fallback inference: {inf_e}")
                        inference_outputs = None

            # inference возвращает [None, mel_outputs, mel_outputs_postnet, gate_outputs, alignments, emb_gst]
            if inference_outputs is not None and len(inference_outputs) >= 5:
                (
                    _,
                    mel_outputs_inf,
                    mel_outputs_postnet_inf,
                    gate_outputs_inf,
                    alignments_inf,
                ) = inference_outputs[:5]
                mel_targets, gate_targets = y[0], y[1]

                print(f"🖼️ Создаем изображения для TensorBoard (iteration {iteration})")

                # plot distribution of parameters (только каждые 500 итераций для экономии времени)
                if iteration % 500 == 0:
                    for tag, value in model.named_parameters():
                        tag = tag.replace(".", "/")
                        writer.add_histogram(tag, value.data.cpu().numpy(), iteration)

                idx = 0  # Используем первый элемент из батча

                # 🔥 ИСПРАВЛЕННОЕ создание изображений с проверкой размеров

                # Alignment изображение
                if alignments_inf is not None and alignments_inf.size(0) > idx:
                    try:
                        alignment_data = alignments_inf[idx].data.cpu().numpy()
                        if alignment_data.shape[0] > 1 and alignment_data.shape[1] > 1:
                            alignment_img = plot_alignment_to_numpy(alignment_data.T)
                            writer.add_image(
                                "alignment", alignment_img, iteration, dataformats="HWC"
                            )
                            print(
                                f"✅ Alignment изображение создано: {alignment_img.shape}"
                            )
                        else:
                            print(
                                f"⚠️ Alignment матрица слишком маленькая: {alignment_data.shape}"
                            )
                    except Exception as e:
                        print(f"❌ Ошибка создания alignment изображения: {e}")

                # Mel target изображение
                if mel_targets.size(0) > idx:
                    try:
                        mel_target_data = mel_targets[idx].data.cpu().numpy()
                        if (
                            mel_target_data.shape[0] > 1
                            and mel_target_data.shape[1] > 1
                        ):
                            mel_target_img = plot_spectrogram_to_numpy(mel_target_data)
                            writer.add_image(
                                "mel_target",
                                mel_target_img,
                                iteration,
                                dataformats="HWC",
                            )
                            print(
                                f"✅ Mel target изображение создано: {mel_target_img.shape}"
                            )
                        else:
                            print(
                                f"⚠️ Mel target слишком маленький: {mel_target_data.shape}"
                            )
                    except Exception as e:
                        print(f"❌ Ошибка создания mel target изображения: {e}")

                # Mel predicted изображение
                if mel_outputs_inf is not None and mel_outputs_inf.size(0) > idx:
                    try:
                        mel_pred_data = mel_outputs_inf[idx].data.cpu().numpy()
                        if mel_pred_data.shape[0] > 1 and mel_pred_data.shape[1] > 1:
                            mel_pred_img = plot_spectrogram_to_numpy(mel_pred_data)
                            writer.add_image(
                                "mel_predicted",
                                mel_pred_img,
                                iteration,
                                dataformats="HWC",
                            )
                            print(
                                f"✅ Mel predicted изображение создано: {mel_pred_img.shape}"
                            )
                        else:
                            print(
                                f"⚠️ Mel predicted слишком маленький: {mel_pred_data.shape}"
                            )
                    except Exception as e:
                        print(f"❌ Ошибка создания mel predicted изображения: {e}")

                # Gate outputs изображение
                if (
                    gate_outputs_inf is not None
                    and gate_outputs_inf.size(0) > idx
                    and gate_targets.size(0) > idx
                ):
                    try:
                        gate_target_data = gate_targets[idx].data.cpu().numpy()
                        gate_pred_data = (
                            torch.sigmoid(gate_outputs_inf[idx]).data.cpu().numpy()
                        )

                        if len(gate_target_data) > 1 and len(gate_pred_data) > 1:
                            gate_img = plot_gate_outputs_to_numpy(
                                gate_target_data, gate_pred_data
                            )
                            writer.add_image(
                                "gate", gate_img, iteration, dataformats="HWC"
                            )
                            print(f"✅ Gate изображение создано: {gate_img.shape}")
                        else:
                            print(
                                f"⚠️ Gate данные слишком маленькие: target={len(gate_target_data)}, pred={len(gate_pred_data)}"
                            )
                    except Exception as e:
                        print(f"❌ Ошибка создания gate изображения: {e}")

                # Принудительно сохраняем в TensorBoard
                writer.flush()
                print(f"🔄 TensorBoard данные сохранены для итерации {iteration}")

                # Сохраняем данные для внешних контроллеров
                try:
                    model.last_validation_alignments = alignments_inf
                    model.last_validation_gate_outputs = gate_outputs_inf
                    model.last_validation_mel_outputs = mel_outputs_postnet_inf
                except Exception:
                    pass

            else:
                print(f"⚠️ Inference не вернул корректные данные для изображений")

            # 🔥 ВАЖНО: Возвращаем модель в train режим
            model.train()

        except Exception as e:
            print(f"❌ Общая ошибка при создании изображений: {e}")
            # Fallback - создаем простые изображения
            try:
                mel_targets, gate_targets = y[0], y[1]
                if mel_targets.size(0) > 0:
                    mel_target_img = plot_spectrogram_to_numpy(
                        mel_targets[0].data.cpu().numpy()
                    )
                    writer.add_image(
                        "mel_target_fallback",
                        mel_target_img,
                        iteration,
                        dataformats="HWC",
                    )
                    print(f"✅ Fallback mel target изображение создано")
            except Exception as fallback_e:
                print(f"❌ Даже fallback изображение не удалось создать: {fallback_e}")

        # 🔥 ОБЯЗАТЕЛЬНО: Убеждаемся что модель в train режиме
        model.train()

        if MLFLOW_AVAILABLE:
            validation_metrics = {
                "validation.loss": val_loss,
                "validation.step": iteration,
            }

            # Дополнительные метрики из модели
            if hasattr(model, "decoder") and hasattr(
                model.decoder, "attention_weights"
            ):
                try:
                    # Логируем метрики attention
                    attention_weights = model.decoder.attention_weights
                    if attention_weights is not None:
                        validation_metrics["validation.attention_entropy"] = float(
                            torch.mean(
                                torch.sum(
                                    -attention_weights
                                    * torch.log(attention_weights + 1e-8),
                                    dim=-1,
                                )
                            )
                        )
                except Exception as e:
                    print(f"⚠️ Ошибка при вычислении attention entropy: {e}")

            # Метрики из alignments
            if alignments_inf is not None:
                try:
                    # Диагональность alignment матрицы
                    alignment_diag = torch.diagonal(alignments_inf[0], dim1=-2, dim2=-1)
                    align_score = float(torch.mean(alignment_diag))
                    validation_metrics["validation.alignment_score"] = align_score
                    # сохраняем для AutoParamController
                    try:
                        model.last_validation_alignment_score = align_score
                    except Exception:
                        pass
                    # Фокусировка attention (концентрация на диагонали)
                    attention_focus = torch.max(alignments_inf[0], dim=-1)[0]
                    validation_metrics["validation.attention_focus"] = float(
                        torch.mean(attention_focus)
                    )
                except Exception as e:
                    print(f"⚠️ Ошибка при вычислении attention метрик: {e}")

            # Убеждаемся, что атрибут модели существует даже при отсутствии align_score
            if not hasattr(model, "last_validation_alignment_score"):
                model.last_validation_alignment_score = None

            # Метрики из gate outputs
            if gate_outputs_inf is not None:
                try:
                    gate_probs = torch.sigmoid(gate_outputs_inf[0])
                    validation_metrics["validation.gate_mean"] = float(
                        torch.mean(gate_probs)
                    )
                    validation_metrics["validation.gate_std"] = float(
                        torch.std(gate_probs)
                    )
                except Exception as e:
                    print(f"⚠️ Ошибка при вычислении gate метрик: {e}")

            if ENHANCED_LOGGING:
                # Расширенное логирование validation метрик
                log_enhanced_training_metrics(validation_metrics, iteration)
            else:
                # Базовое MLflow логирование
                for metric_name, metric_value in validation_metrics.items():
                    mlflow.log_metric(metric_name, metric_value, step=iteration)

    return val_loss


def calculate_global_mean(data_loader, global_mean_npy):
    if global_mean_npy and os.path.exists(global_mean_npy):
        global_mean = np.load(global_mean_npy)
        return to_gpu(torch.tensor(global_mean))
    sums = []
    frames = []
    print("\nCalculating global mean...\n")
    for i, batch in enumerate(data_loader):
        (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
            ctc_text,
            ctc_text_lengths,
            _guide_mask,
        ) = batch
        # padded values are 0.
        sums.append(mel_padded.double().sum(dim=(0, 2)))
        frames.append(output_lengths.double().sum())
    global_mean = sum(sums) / sum(frames)
    global_mean = global_mean.float()
    if global_mean_npy:
        np.save(global_mean_npy, global_mean)
    return to_gpu(global_mean)


def train(
    output_directory,
    log_directory,
    checkpoint_path,
    warm_start,
    ignore_mmi_layers,
    ignore_gst_layers,
    ignore_tsgst_layers,
    n_gpus,
    rank,
    group_name,
    hparams,
    # Параметры для интеграции со Smart Tuner
    smart_tuner_trial=None,
    smart_tuner_logger=None,
    tensorboard_writer=None,
    telegram_monitor=None,
):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    warm_start(bool): load model from checkpoint
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=hparams.weight_decay
    )

    # ---------- FP16 / Mixed Precision ---------
    apex_available = False  # Флаг наличия NVIDIA Apex
    use_native_amp = False  # Флаг использования torch.cuda.amp
    scaler = None  # GradScaler для native AMP

    if hparams.fp16_run:
        try:
            from apex import amp  # type: ignore

            model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
            apex_available = True
            print("✅ NVIDIA Apex успешно загружен для FP16 обучения")
        except ImportError:
            # Apex недоступен – используем встроенный AMP PyTorch
            try:
                from torch.amp import GradScaler, autocast

                # Отключаем FP16 для модели, используем только AMP
                model = model.float()  # Убеждаемся, что модель в FP32
                scaler = GradScaler("cuda")
                use_native_amp = True
                print(
                    "✅ NVIDIA Apex не найден. Переключаемся на torch.amp (PyTorch Native AMP)"
                )
            except ImportError as e:
                # Даже native AMP недоступен – отключаем FP16
                hparams.fp16_run = False
                print(f"❌ Mixed precision недоступна: {e}. FP16 отключён.")
    # -------------------------------------------

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss(hparams)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer
            )
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_main_node = rank == 0

    # 💡 Используем переданный writer, а не создаем новый
    writer = tensorboard_writer if is_main_node else None

    if is_main_node and writer is None:
        # Для обратной совместимости, если train.py запускается напрямую
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_directory)

    # Инициализация loss функций
    mmi_loss = None
    guide_loss = None
    
    if hparams.use_mmi:
        from mmi_loss import MMI_loss

        mmi_loss = MMI_loss(hparams.mmi_map, hparams.mmi_weight)
        print("✅ MMI loss загружен")

    if hparams.use_guided_attn:
        from loss_function import GuidedAttentionLoss

        guide_loss = GuidedAttentionLoss(alpha=hparams.guided_attn_weight)
        print("✅ Guided Attention Loss загружен")

    # --- Auto Hyper-parameter Controller ---
    quality_ctrl = None
    if is_main_node:
        try:
            quality_ctrl = AdvancedQualityController()
            print("🤖 AdvancedQualityController активирован")
        except Exception as e:
            print(f"⚠️ Не удалось инициализировать AdvancedQualityController: {e}")

    # --- ParamScheduler и EarlyStopController ---
    sched_ctrl = None
    stop_ctrl = None
    if is_main_node:
        try:
            sched_ctrl = ParamScheduler()
            print("📅 ParamScheduler активирован")
        except Exception as e:
            print(f"⚠️ Не удалось инициализировать ParamScheduler: {e}")

        try:
            stop_ctrl = EarlyStopController()
            print("🛑 EarlyStopController активирован")
        except Exception as e:
            print(f"⚠️ Не удалось инициализировать EarlyStopController: {e}")

    # --- 🔍 Debug Reporter ---
    debug_reporter = None
    if is_main_node:
        try:
            debug_reporter = initialize_debug_reporter(telegram_monitor)
            print("🔍 Debug Reporter активирован - отчеты каждые 1000 шагов")
        except Exception as e:
            print(f"⚠️ Не удалось инициализировать Debug Reporter: {e}")

    global_mean = calculate_global_mean(train_loader, hparams.global_mean_npy)

    # Переменные для отслеживания validation loss для отправки аудио
    last_validation_loss = None
    last_audio_step = 0

    # === EMA и авто-LR переменные ===
    grad_norm_ema = 0.0  # экспоненциальное скользящее среднее нормы градиента
    ema_beta = 0.95      # коэффициент EMA
    lr_adjust_interval = 10  # интервал корректировки LR в шагах

    # ================ MAIN TRAINNIG LOOP ===================
    print(
        f"🚀 Начинаем обучение: epochs={hparams.epochs}, batch_size={hparams.batch_size}, dataset_size={len(train_loader)}"
    )

    # Логирование параметров модели в MLflow
    model_params = {
        "model.total_params": sum(p.numel() for p in model.parameters()),
        "model.trainable_params": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "hparams.epochs": hparams.epochs,
        "hparams.grad_clip_thresh": hparams.grad_clip_thresh,
        "hparams.fp16_run": hparams.fp16_run,
        "hparams.use_mmi": hparams.use_mmi,
        "hparams.use_guided_attn": hparams.use_guided_attn,
        "dataset.train_size": len(train_loader),
        "dataset.val_size": len(valset),
    }

    # Создаем nested run для каждого trial
    if smart_tuner_trial is not None:
        # Для Smart Tuner создаем nested run
        trial_run_name = f"trial_{smart_tuner_trial.number}"
        with mlflow.start_run(nested=True, run_name=trial_run_name):
            try:
                mlflow.log_params(model_params)
                mlflow.log_param("hparams.batch_size_init", hparams.batch_size)
                mlflow.log_param("hparams.learning_rate_init", hparams.learning_rate)
                print(f"📊 Параметры trial {smart_tuner_trial.number} залогированы в MLflow")
            except Exception as e:
                print(f"📊 Ошибка логирования параметров trial: {e}")
    else:
        # Для обычного запуска
        try:
            mlflow.log_params(model_params)
            mlflow.log_param("hparams.batch_size_init", hparams.batch_size)
            mlflow.log_param("hparams.learning_rate_init", hparams.learning_rate)
            print(f"📊 Параметры модели залогированы в MLflow: {model_params['model.total_params']} параметров")
        except Exception as e:
            print(f"📊 Ошибка логирования параметров: {e}")

    # --- Intelligent Epoch Optimizer ---
    optimizer_epochs = None
    if is_main_node:
        try:
            optimizer_epochs = IntelligentEpochOptimizer()
            # Создаем dataset_meta из доступной информации
            dataset_meta = {
                "total_duration_hours": len(train_loader)
                * hparams.batch_size
                * 0.1,  # примерная оценка
                "quality_metrics": {
                    "background_noise_level": 0.3,
                    "voice_consistency": 0.8,
                    "speech_clarity": 0.85,
                },
                "voice_features": {
                    "has_accent": False,
                    "emotional_range": "neutral",
                    "speaking_style": "normal",
                    "pitch_range_semitones": 12,
                },
            }
            analysis = optimizer_epochs.analyze_dataset(dataset_meta)
            if "recommended_epochs" in analysis:
                hparams.epochs = analysis["recommended_epochs"]
                print(f"🔧 Epochs set to {hparams.epochs} (было {hparams.epochs})")
        except Exception as e:
            print(f"⚠️ IntelligentEpochOptimizer ошибка: {e}")

    gradient_monitor = GradientStabilityMonitor()
    restart_attempts = 0
    max_restart_attempts = 3
    safe_hparams_history = []
    
    def get_safe_hparams(hparams, attempt):
        """
        🛡️ УЛЬТРА-безопасные параметры для критического восстановления после NaN/Inf
        Агрессивно снижает все параметры для максимальной стабильности
        """
        new_hparams = copy.deepcopy(hparams)
        
        # 🔥 ЭКСТРЕМАЛЬНОЕ снижение learning rate (каждая попытка в 10 раз меньше)
        new_hparams.learning_rate = max(new_hparams.learning_rate * (0.1 ** (attempt + 1)), 1e-8)
        
        # 📦 Агрессивное уменьшение batch size для стабильности
        new_hparams.batch_size = max(1, int(new_hparams.batch_size * (0.5 ** (attempt + 1))))
        
        # 🎯 КРИТИЧЕСКОЕ увеличение guided attention для восстановления alignment
        if hasattr(new_hparams, 'guide_loss_initial_weight'):
            new_hparams.guide_loss_initial_weight = min(1000.0, max(10.0, new_hparams.guide_loss_initial_weight * (3.0 ** (attempt + 1))))
        else:
            new_hparams.guide_loss_initial_weight = 10.0 * (3.0 ** (attempt + 1))
        
        # ✂️ ЭКСТРЕМАЛЬНО строгое клипирование градиентов
        new_hparams.grad_clip_thresh = max(0.001, new_hparams.grad_clip_thresh * (0.1 ** (attempt + 1)))
        
        # 🚫 Отключаем все "продвинутые" функции для максимальной стабильности
        if hasattr(new_hparams, 'use_mmi'):
            new_hparams.use_mmi = False
        if hasattr(new_hparams, 'use_audio_quality_enhancement'):
            new_hparams.use_audio_quality_enhancement = False
        
        # 🛡️ ПРИНУДИТЕЛЬНАЯ активация guided attention
        new_hparams.use_guided_attn = True
        
        # 🎛️ Минимизируем dropout для большей предсказуемости
        if hasattr(new_hparams, 'dropout_rate'):
            new_hparams.dropout_rate = min(0.005, new_hparams.dropout_rate * 0.1)
        if hasattr(new_hparams, 'encoder_dropout_rate'):
            new_hparams.encoder_dropout_rate = min(0.001, new_hparams.encoder_dropout_rate * 0.1)
        if hasattr(new_hparams, 'postnet_dropout_rate'):
            new_hparams.postnet_dropout_rate = min(0.001, new_hparams.postnet_dropout_rate * 0.1)
        if hasattr(new_hparams, 'p_attention_dropout'):
            new_hparams.p_attention_dropout = min(0.01, new_hparams.p_attention_dropout * 0.1)
        if hasattr(new_hparams, 'p_decoder_dropout'):
            new_hparams.p_decoder_dropout = min(0.01, new_hparams.p_decoder_dropout * 0.1)
        
        # 🚪 Консервативный gate threshold
        if hasattr(new_hparams, 'gate_threshold'):
            new_hparams.gate_threshold = 0.5  # Сбалансированный порог
        
        # 🔄 Принудительная реинициализация модели при критических попытках
        if attempt >= 2:
            new_hparams.force_model_reinit = True
            new_hparams.xavier_init = True  # Принудительная Xavier инициализация
        
        # 📊 Детальное логирование для диагностики
        print(f"\n🛡️ [КРИТИЧЕСКОЕ ВОССТАНОВЛЕНИЕ] Попытка {attempt+1}:")
        print(f"  🔥 learning_rate: {hparams.learning_rate:.8f} → {new_hparams.learning_rate:.8f} (снижено в {hparams.learning_rate/new_hparams.learning_rate:.1f}x)")
        print(f"  📦 batch_size: {hparams.batch_size} → {new_hparams.batch_size}")
        print(f"  🎯 guide_loss_weight: {getattr(hparams, 'guide_loss_initial_weight', 1.0):.2f} → {new_hparams.guide_loss_initial_weight:.2f}")
        print(f"  ✂️ grad_clip_thresh: {hparams.grad_clip_thresh:.4f} → {new_hparams.grad_clip_thresh:.4f}")
        print(f"  🛡️ use_guided_attn: ПРИНУДИТЕЛЬНО включен")
        print(f"  🚫 Отключены: MMI, audio_enhancement")
        print(f"  🎛️ Минимизированы все dropout значения")
        
        return new_hparams

    while restart_attempts <= max_restart_attempts:
        for epoch in range(epoch_offset, hparams.epochs):
            print("Epoch: {} / {}".format(epoch, hparams.epochs))
            for i, batch in enumerate(train_loader):
                # 🔥 ДОПОЛНИТЕЛЬНАЯ ЗАЩИТА: Убеждаемся что модель в train режиме
                model.train()

                start = time.perf_counter()
                model.zero_grad()

                x, y = model.parse_batch(batch)

                # Forward pass с учётом выбранной схемы mixed precision
                if hparams.fp16_run and use_native_amp:
                    with autocast("cuda"):
                        try:
                            y_pred = model(x)
                        except Exception as e:
                            print(f"⚠️ Ошибка forward pass модели: {e}")
                            y_pred = None

                        # total loss
                        if y_pred is not None:
                            try:
                                loss_taco, loss_gate, loss_atten, loss_emb = criterion(y_pred, y)
                            except Exception as e:
                                print(f"⚠️ Ошибка criterion: {e}")
                                # Безопасное получение device из x (tuple)
                                device = x[0].device if isinstance(x, tuple) and len(x) > 0 else 'cuda'
                                loss_taco = torch.tensor(0.0, device=device, requires_grad=True)
                                loss_gate = torch.tensor(0.0, device=device, requires_grad=True)
                                loss_atten = torch.tensor(0.0, device=device, requires_grad=True)
                                loss_emb = torch.tensor(0.0, device=device, requires_grad=True)
                        else:
                            # Если y_pred None, создаем нулевые loss
                            # Безопасное получение device из x (tuple)
                            device = x[0].device if isinstance(x, tuple) and len(x) > 0 else 'cuda'
                            loss_taco = torch.tensor(0.0, device=device, requires_grad=True)
                            loss_gate = torch.tensor(0.0, device=device, requires_grad=True)
                            loss_atten = torch.tensor(0.0, device=device, requires_grad=True)
                            loss_emb = torch.tensor(0.0, device=device, requires_grad=True)
                        
                        # Безопасное получение device
                        if y_pred is not None and len(y_pred) > 1 and y_pred[1] is not None:
                            device = y_pred[1].device  # mel_outputs всегда тензор
                        else:
                            device = x[0].device if isinstance(x, tuple) and len(x) > 0 else 'cuda'
                        try:
                            loss_guide = (
                                guide_loss(y_pred)
                                if hparams.use_guided_attn and guide_loss is not None and y_pred is not None
                                else torch.tensor(0.0, device=device, requires_grad=True)
                            )
                        except Exception as e:
                            print(f"⚠️ Ошибка guide_loss: {e}")
                            loss_guide = torch.tensor(0.0, device=device, requires_grad=True)
                        
                        try:
                            loss_mmi = (
                                mmi_loss(y_pred[1], y[0])
                                if hparams.use_mmi and mmi_loss is not None and y_pred is not None and y_pred[1] is not None
                                else torch.tensor(0.0, device=device, requires_grad=True)
                            )
                        except Exception as e:
                            print(f"⚠️ Ошибка mmi_loss: {e}")
                            loss_mmi = torch.tensor(0.0, device=device, requires_grad=True)
                        try:
                            loss = (
                                0.4 * loss_taco +
                                0.3 * loss_atten +
                                0.3 * loss_gate +
                                loss_guide +
                                loss_mmi +
                                loss_emb
                            )
                        except Exception as e:
                            print(f"⚠️ Ошибка вычисления loss: {e}")
                            loss = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    try:
                        y_pred = model(x)
                    except Exception as e:
                        print(f"⚠️ Ошибка forward pass модели: {e}")
                        y_pred = None
                    # total loss
                    if y_pred is not None:
                        try:
                            loss_taco, loss_gate, loss_atten, loss_emb = criterion(y_pred, y)
                        except Exception as e:
                            print(f"⚠️ Ошибка criterion: {e}")
                            # Безопасное получение device из x (tuple)
                            device = x[0].device if isinstance(x, tuple) and len(x) > 0 else 'cuda'
                            loss_taco = torch.tensor(0.0, device=device, requires_grad=True)
                            loss_gate = torch.tensor(0.0, device=device, requires_grad=True)
                            loss_atten = torch.tensor(0.0, device=device, requires_grad=True)
                            loss_emb = torch.tensor(0.0, device=device, requires_grad=True)
                    else:
                        # Если y_pred None, создаем нулевые loss
                        # Безопасное получение device из x (tuple)
                        device = x[0].device if isinstance(x, tuple) and len(x) > 0 else 'cuda'
                        loss_taco = torch.tensor(0.0, device=device, requires_grad=True)
                        loss_gate = torch.tensor(0.0, device=device, requires_grad=True)
                        loss_atten = torch.tensor(0.0, device=device, requires_grad=True)
                        loss_emb = torch.tensor(0.0, device=device, requires_grad=True)
                    # Безопасное получение device
                    if y_pred is not None and len(y_pred) > 1 and y_pred[1] is not None:
                        device = y_pred[1].device  # mel_outputs всегда тензор
                    else:
                        device = x[0].device if isinstance(x, tuple) and len(x) > 0 else 'cuda'
                    try:
                        loss_guide = (
                            guide_loss(y_pred)
                            if hparams.use_guided_attn and guide_loss is not None and y_pred is not None
                            else torch.tensor(0.0, device=device, requires_grad=True)
                        )
                    except Exception as e:
                        print(f"⚠️ Ошибка guide_loss: {e}")
                        loss_guide = torch.tensor(0.0, device=device, requires_grad=True)
                    
                    try:
                        loss_mmi = (
                            mmi_loss(y_pred[1], y[0])
                            if hparams.use_mmi and mmi_loss is not None and y_pred is not None and y_pred[1] is not None
                            else torch.tensor(0.0, device=device, requires_grad=True)
                        )
                    except Exception as e:
                        print(f"⚠️ Ошибка mmi_loss: {e}")
                        loss_mmi = torch.tensor(0.0, device=device, requires_grad=True)
                    try:
                        loss = (
                            0.4 * loss_taco +
                            0.3 * loss_atten +
                            0.3 * loss_gate +
                            loss_guide +
                            loss_mmi +
                            loss_emb
                        )
                    except Exception as e:
                        print(f"⚠️ Ошибка вычисления loss: {e}")
                        loss = torch.tensor(0.0, device=device, requires_grad=True)

                if hparams.distributed_run:
                    reduced_loss = reduce_tensor(loss.data, n_gpus).item() if loss is not None else 0.0
                else:
                    reduced_loss = loss.item() if loss is not None else 0.0
                    reduced_taco_loss = loss_taco.item() if loss_taco is not None else 0.0
                    reduced_atten_loss = loss_atten.item() if loss_atten is not None else 0.0
                    reduced_mi_loss = loss_mmi.item() if loss_mmi is not None else 0.0
                    reduced_guide_loss = (
                        loss_guide.item() if hparams.use_guided_attn and loss_guide is not None else 0.0
                    )
                    reduced_gate_loss = loss_gate.item() if loss_gate is not None else 0.0
                    reduced_emb_loss = loss_emb.item() if loss_emb is not None else 0.0

                # Backward pass
                if hparams.fp16_run and apex_available:
                    if loss is not None:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        print("⚠️ Предупреждение: loss is None, пропускаем backward pass")
                elif hparams.fp16_run and use_native_amp:
                    if loss is not None:
                        scaler.scale(loss).backward()
                    else:
                        print("⚠️ Предупреждение: loss is None, пропускаем backward pass")
                else:
                    if loss is not None:
                        if not hparams.fp16_run:
                            # --- Dynamic Loss Scaling (FP32 режим) ---
                            scaled_loss = loss * dyn_loss_scale
                            scaled_loss.backward()
                            # Unscale градиенты
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data.div_(dyn_loss_scale)
                        else:
                            loss.backward()
                    else:
                        print("⚠️ Предупреждение: loss is None, пропускаем backward pass")

                if loss is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), hparams.grad_clip_thresh
                    )
                    # --- EMA нормы градиента и авто-коррекция LR ---
                    grad_norm_ema = ema_beta * grad_norm_ema + (1 - ema_beta) * float(grad_norm)
                    if (iteration % lr_adjust_interval) == 0 and iteration > 0:
                        current_lr = optimizer.param_groups[0]["lr"]
                        new_lr = current_lr
                        if grad_norm_ema > 10.0:
                            new_lr = max(hparams.learning_rate_min, current_lr * 0.5)
                        elif grad_norm_ema < 0.1:
                            new_lr = min(hparams.learning_rate * 2, current_lr * 1.1)
                        if abs(new_lr - current_lr) > 1e-12:
                            for g in optimizer.param_groups:
                                g["lr"] = new_lr
                            if debug_reporter:
                                debug_reporter.add_warning(
                                    f"LR auto-adjust: grad_norm_ema={grad_norm_ema:.3f}, lr {current_lr:.2e} → {new_lr:.2e}"
                                )
                            print(f"🔄 LR auto-adjust: {current_lr:.6e} → {new_lr:.6e} (grad_norm_ema={grad_norm_ema:.3f})")
                    # --- Быстрая проверка NaN/Inf каждые 10 шагов ---
                    if (iteration % 10) == 0 and (torch.isnan(loss) or torch.isinf(loss)):
                        print("🚨 [Auto-Recover] NaN/Inf обнаружен в loss – уменьшаем LR и пропускаем шаг")
                        for g in optimizer.param_groups:
                            g["lr"] = max(hparams.learning_rate_min, g["lr"] * 0.5)
                        # Уменьшаем масштаб динамического loss scaling
                        dyn_loss_scale = max(1.0, dyn_loss_scale / loss_scale_factor)
                        optimizer.zero_grad(set_to_none=True)
                        if debug_reporter:
                            debug_reporter.add_warning("NaN/Inf в loss: auto LR halved и шаг пропущен")
                        continue  # переход к следующей итерации
                else:
                    grad_norm = 0.0

                # Optimizer step с учётом схемы mixed precision
                if loss is not None:
                    if hparams.fp16_run and use_native_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                else:
                    print("⚠️ Предупреждение: loss is None, пропускаем optimizer step")

                # Мониторинг градиентов
                if loss is not None:
                    monitor_result = gradient_monitor.check_gradient_stability(model, loss, i + epoch * len(train_loader))
                else:
                    monitor_result = {'explosion_detected': False, 'nan_detected': False, 'recommendations': ['Loss is None']}
                if monitor_result['explosion_detected'] or monitor_result['nan_detected']:
                    print("\n🚨 [Smart Restart] Обнаружена критическая нестабильность на шаге {}!".format(i + epoch * len(train_loader)))
                    print("Причина: {}".format('Взрыв градиентов' if monitor_result['explosion_detected'] else 'NaN/Inf в loss'))
                    print("Рекомендации:")
                    for rec in monitor_result['recommendations']:
                        print("  - ", rec)
                    # Сохраняем параметры и шаг
                    safe_hparams_history.append(copy.deepcopy(hparams))
                    # Перезапуск с более безопасными параметрами
                    restart_attempts += 1
                    if restart_attempts > max_restart_attempts:
                        print("❌ Достигнут лимит попыток перезапуска. Остановка обучения.")
                        return
                    hparams = get_safe_hparams(hparams, restart_attempts)
                    
                    # 🔄 Критическая реинициализация модели при необходимости
                    if hasattr(hparams, 'force_model_reinit') and hparams.force_model_reinit:
                        print("🔄 КРИТИЧЕСКАЯ РЕИНИЦИАЛИЗАЦИЯ МОДЕЛИ...")
                        try:
                            model = load_model(hparams)
                            # Принудительная Xavier инициализация
                            if hasattr(hparams, 'xavier_init') and hparams.xavier_init:
                                for name, param in model.named_parameters():
                                    if len(param.shape) > 1:
                                        torch.nn.init.xavier_uniform_(param)
                                    else:
                                        torch.nn.init.zeros_(param)
                                print("✅ Xavier инициализация применена")
                            
                            # Обновляем optimizer с новой моделью
                            optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)
                            print("✅ Модель и оптимизатор переинициализированы")
                        except Exception as e:
                            print(f"⚠️ Ошибка реинициализации модели: {e}")
                            # Продолжаем с существующей моделью
                    
                    # Обновляем learning rate в оптимизаторе
                    for g in optimizer.param_groups:
                        g["lr"] = hparams.learning_rate
                    # Сохраняем чекпоинт для отладки
                    save_checkpoint(model, optimizer, hparams.learning_rate, i + epoch * len(train_loader), os.path.join(output_directory, f"restart_checkpoint_{restart_attempts}.pt"))
                    print(f"[Smart Restart] Перезапуск обучения с новыми параметрами (попытка {restart_attempts})...\n")
                    # break оба цикла
                    break

                # 🛡️ КРИТИЧЕСКАЯ ЗАЩИТА ОТ NaN LOSS
                if loss is not None:
                    # Проверяем все компоненты loss на NaN/Inf
                    loss_components = {
                        'total_loss': reduced_loss,
                        'taco_loss': reduced_taco_loss,
                        'atten_loss': reduced_atten_loss,
                        'mi_loss': reduced_mi_loss,
                        'guide_loss': reduced_guide_loss,
                        'gate_loss': reduced_gate_loss,
                        'emb_loss': reduced_emb_loss
                    }
                    
                    nan_detected = False
                    problematic_components = []
                    
                    for component_name, component_value in loss_components.items():
                        if torch.isnan(torch.tensor(component_value)) or torch.isinf(torch.tensor(component_value)):
                            nan_detected = True
                            problematic_components.append(f"{component_name}: {component_value}")
                    
                    if nan_detected:
                        print("\n🚨 [КРИТИЧЕСКАЯ ОШИБКА] NaN/Inf обнаружен в loss!")
                        print(f"Проблемные компоненты: {problematic_components}")
                        print(f"Шаг: {i + epoch * len(train_loader)}")
                        
                        # 🔍 Записываем критическую ошибку в debug reporter
                        if debug_reporter:
                            try:
                                critical_info = f"NaN/Inf на шаге {iteration}. Компоненты: {', '.join(problematic_components)}"
                                debug_reporter.add_warning(critical_info)
                                print("🔍 Критическая ошибка записана в Debug Reporter")
                            except Exception as e:
                                print(f"⚠️ Ошибка записи в Debug Reporter: {e}")
                        
                        # 📱 СРОЧНОЕ Telegram уведомление о критической ошибке
                        if telegram_monitor:
                            try:
                                critical_message = f"🚨 **КРИТИЧЕСКАЯ ОШИБКА!**\n\n"
                                critical_message += f"❌ **Loss стал NaN на шаге {iteration}**\n"
                                critical_message += f"🔥 **Компоненты с ошибками:**\n"
                                for comp in problematic_components:
                                    critical_message += f"  • {comp}\n"
                                critical_message += f"\n🔄 **АВТОМАТИЧЕСКИЙ ПЕРЕЗАПУСК в процессе...**\n"
                                critical_message += f"⚙️ Снижаем learning_rate и укрепляем стабильность"
                                
                                telegram_monitor._send_text_message(critical_message)
                                print("📱 Критическое уведомление отправлено в Telegram")
                            except Exception as e:
                                print(f"⚠️ Ошибка отправки критического уведомления: {e}")
                        
                        # Принудительное завершение этой итерации
                        print("🔄 Инициирую умный перезапуск с защищенными параметрами...")
                        restart_attempts += 1
                        if restart_attempts > max_restart_attempts:
                            print("❌ Достигнут лимит попыток перезапуска. Остановка обучения.")
                            return
                        
                        # 🛡️ Активируем критический режим guided attention
                        if guide_loss is not None and hasattr(guide_loss, 'activate_critical_mode'):
                            guide_loss.activate_critical_mode()
                            print("🎯 Guided Attention переведен в КРИТИЧЕСКИЙ режим восстановления")
                        
                        # Сохраняем старые параметры для уведомления
                        old_learning_rate = hparams.learning_rate
                        old_batch_size = hparams.batch_size
                        
                        # Создаем защищенные параметры
                        hparams = get_safe_hparams(hparams, restart_attempts)
                        print(f"[Smart Restart] Перезапуск с УЛЬТРА-безопасными параметрами (попытка {restart_attempts})...\n")
                        
                        # 🔍 Записываем информацию о перезапуске в debug reporter
                        if debug_reporter:
                            try:
                                restart_info = f"Перезапуск #{restart_attempts}: learning_rate {old_learning_rate:.8f}→{hparams.learning_rate:.8f}, batch_size {old_batch_size}→{hparams.batch_size}, причина: NaN/Inf"
                                debug_reporter.add_restart_info(restart_info)
                                print("🔍 Информация о перезапуске записана в Debug Reporter")
                            except Exception as e:
                                print(f"⚠️ Ошибка записи перезапуска в Debug Reporter: {e}")
                        
                        # 📱 TELEGRAM уведомление о критическом перезапуске
                        if telegram_monitor:
                            try:
                                restart_message = f"🔄 **АВТОМАТИЧЕСКИЙ ПЕРЕЗАПУСК #{restart_attempts}**\n\n"
                                restart_message += f"🚨 **Причина:** NaN/Inf в loss components\n"
                                restart_message += f"🛡️ **Действия:**\n"
                                restart_message += f"  • 🔥 Learning rate: {old_learning_rate:.8f} → {hparams.learning_rate:.8f}\n"
                                restart_message += f"  • 📦 Batch size: {old_batch_size} → {hparams.batch_size}\n"
                                restart_message += f"  • 🎯 Guided attention усилен до {getattr(hparams, 'guide_loss_initial_weight', 'активирован')}\n"
                                restart_message += f"  • ✂️ Grad clipping: {hparams.grad_clip_thresh:.4f} (строже)\n"
                                restart_message += f"  • 🛡️ Критический режим guided attention активирован\n"
                                restart_message += f"\n⏰ **Перезапуск через 3 секунды...**"
                                
                                telegram_monitor._send_text_message(restart_message)
                                print("📱 Уведомление о перезапуске отправлено в Telegram")
                            except Exception as e:
                                print(f"⚠️ Ошибка отправки уведомления о перезапуске: {e}")
                        
                        # Даем время на обработку уведомления
                        time.sleep(3)
                        break

                if is_main_node:
                    try:
                        duration = time.perf_counter() - start
                    except Exception as e:
                        print(f"⚠️ Ошибка вычисления duration: {e}")
                        duration = 0.0
                    try:
                        print(
                            "Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                                iteration, reduced_loss, grad_norm, duration
                            )
                        )
                    except Exception as e:
                        print(f"⚠️ Ошибка вывода метрик: {e}")
                        print(f"Train loss {iteration} N/A Grad Norm N/A N/A s/it")

                    # Обновляем learning_rate переменную из optimizer (мог измениться авто-контроллером)
                    try:
                        learning_rate = optimizer.param_groups[0]["lr"]
                    except (IndexError, KeyError) as e:
                        print(f"⚠️ Ошибка получения learning_rate: {e}")
                        learning_rate = hparams.learning_rate

                    # --- EarlyStopController: добавляем метрики ---
                    if stop_ctrl:
                        try:
                            stop_ctrl.add_metrics(
                                {
                                    "train_loss": reduced_loss,
                                    "grad_norm": grad_norm,
                                    "learning_rate": learning_rate,
                                    "guide_loss": (
                                        reduced_guide_loss
                                        if hparams.use_guided_attn and reduced_guide_loss is not None
                                        else 0.0
                                    ),
                                    "gate_loss": reduced_gate_loss,
                                }
                            )
                        except Exception as e:
                            print(f"⚠️ EarlyStopController add_metrics ошибка: {e}")

                    # Логирование в TensorBoard
                    try:
                        writer.add_scalar("training.loss", reduced_loss, iteration)
                        writer.add_scalar("training.taco_loss", reduced_taco_loss, iteration)
                        writer.add_scalar("training.atten_loss", reduced_atten_loss, iteration)
                        writer.add_scalar("training.mi_loss", reduced_mi_loss, iteration)
                        writer.add_scalar("training.guide_loss", reduced_guide_loss if reduced_guide_loss is not None else 0.0, iteration)
                        writer.add_scalar("training.gate_loss", reduced_gate_loss, iteration)
                        writer.add_scalar("training.emb_loss", reduced_emb_loss, iteration)
                        writer.add_scalar("grad.norm", grad_norm, iteration)
                        writer.add_scalar("learning.rate", learning_rate, iteration)
                        writer.add_scalar("duration", duration, iteration)
                    except Exception as e:
                        print(f"⚠️ Ошибка логирования в TensorBoard: {e}")
                    if hparams.use_guided_attn and guide_loss is not None:
                        try:
                            writer.add_scalar(
                                "training.guide_loss_weight", guide_loss.get_weight(), iteration
                            )
                        except Exception as e:
                            print(f"⚠️ Ошибка получения guide_loss weight: {e}")

                    # Логирование в MLflow
                    if MLFLOW_AVAILABLE:
                        training_metrics = {
                            "training.loss": reduced_loss,
                            "training.taco_loss": reduced_taco_loss,
                            "training.atten_loss": reduced_atten_loss,
                            "training.mi_loss": reduced_mi_loss,
                            "training.guide_loss": reduced_guide_loss if reduced_guide_loss is not None else 0.0,
                            "training.gate_loss": reduced_gate_loss,
                            "training.emb_loss": reduced_emb_loss,
                            "grad.norm": grad_norm,
                            "learning.rate": learning_rate,
                            "duration": duration,
                            "batch_size": hparams.batch_size,
                            "learning_rate": learning_rate,
                        }
                        for metric_name, metric_value in training_metrics.items():
                            try:
                                mlflow.log_metric(metric_name, metric_value, step=iteration)
                            except Exception as e:
                                print(f"⚠️ Ошибка логирования в MLflow для {metric_name}: {e}")

                    # 🎯 Автоматическая проверка и адаптация guided attention
                    if guide_loss is not None and hasattr(guide_loss, 'check_diagonality_and_adapt') and y_pred is not None:
                        try:
                            # Извлекаем alignments из y_pred
                            alignments = None
                            if len(y_pred) >= 4:
                                alignments = y_pred[3] if len(y_pred) == 4 else y_pred[4]  # Alignments обычно 4-й элемент
                            
                            if alignments is not None:
                                guide_loss.check_diagonality_and_adapt(alignments)
                        except Exception as e:
                            print(f"⚠️ Ошибка проверки диагональности: {e}")

                    # 📱 Telegram уведомления каждые 100 шагов
                    if is_main_node and telegram_monitor:
                        try:
                            if iteration % 100 == 0:
                                print(f"🚀 Отправляем Telegram уведомление для шага {iteration}")

                                # Метрики для Telegram
                                telegram_metrics = {
                                    "loss": reduced_loss,
                                    "train_loss": reduced_loss,  # Дублируем для совместимости
                                    "mel_loss": reduced_taco_loss,
                                    "gate_loss": reduced_gate_loss,
                                    "guide_loss": reduced_guide_loss if reduced_guide_loss is not None else 0.0,
                                    "grad_norm": grad_norm,
                                    "learning_rate": learning_rate,
                                    "epoch": epoch,
                                    "batch_size": hparams.batch_size,
                                    "guide_loss_weight": hparams.guide_loss_weight if hasattr(hparams, 'guide_loss_weight') and hparams.guide_loss_weight is not None else 1.0,
                                    "gate_threshold": hparams.gate_threshold if hasattr(hparams, 'gate_threshold') and hparams.gate_threshold is not None else 0.5,
                                }
                                # Добавляем validation loss если он доступен
                                if last_validation_loss is not None:
                                    telegram_metrics["validation_loss"] = last_validation_loss
                                    telegram_metrics["val_loss"] = last_validation_loss

                                # 🤖 Собираем информацию о решениях Smart Tuner
                                smart_tuner_decisions = {}

                                # Информация от AdvancedQualityController
                                if quality_ctrl:
                                    try:
                                        quality_summary = quality_ctrl.get_quality_summary()
                                        if quality_summary:
                                            smart_tuner_decisions["quality_controller"] = {
                                                "active": True,
                                                "status": "Анализ качества",
                                                "summary": quality_summary,
                                            }
                                    except Exception as e:
                                        print(f"⚠️ Ошибка получения quality summary: {e}")

                                # Информация от ParamScheduler
                                if sched_ctrl:
                                    try:
                                        sched_status = sched_ctrl.get_status()
                                        if sched_status:
                                            smart_tuner_decisions["param_scheduler"] = {
                                                "active": True,
                                                "status": sched_status.get(
                                                    "phase", "Активен"
                                                ),
                                                "current_params": sched_status.get(
                                                    "current_params", {}
                                                ),
                                            }
                                    except Exception as e:
                                        print(f"⚠️ Ошибка получения scheduler status: {e}")

                                # Информация от EarlyStopController
                                if stop_ctrl:
                                    try:
                                        stop_status = stop_ctrl.get_status()
                                        if stop_status:
                                            smart_tuner_decisions[
                                                "early_stop_controller"
                                            ] = {
                                                "active": True,
                                                "status": stop_status.get(
                                                    "status", "Мониторинг"
                                                ),
                                                "patience_remaining": stop_status.get(
                                                    "patience_remaining", "N/A"
                                                ),
                                            }
                                    except Exception as e:
                                        print(
                                            f"⚠️ Ошибка получения stop controller status: {e}"
                                        )

                                # Информация от IntelligentEpochOptimizer
                                if optimizer_epochs:
                                    try:
                                        epoch_status = optimizer_epochs.get_status()
                                        if epoch_status:
                                            smart_tuner_decisions["epoch_optimizer"] = {
                                                "active": True,
                                                "status": epoch_status.get(
                                                    "status", "Оптимизация"
                                                ),
                                                "recommended_epochs": epoch_status.get(
                                                    "recommended_epochs", "N/A"
                                                ),
                                            }
                                    except Exception as e:
                                        print(
                                            f"⚠️ Ошибка получения epoch optimizer status: {e}"
                                        )

                                # Собираем изменения параметров
                                param_changes = {}
                                if hasattr(model, "last_param_changes"):
                                    param_changes = model.last_param_changes

                                if param_changes:
                                    smart_tuner_decisions["parameter_changes"] = (
                                        param_changes
                                    )

                                # Рекомендации от всех контроллеров
                                all_recommendations = []
                                if quality_ctrl and hasattr(
                                    quality_ctrl, "get_recommendations"
                                ):
                                    try:
                                        quality_recs = quality_ctrl.get_recommendations()
                                        all_recommendations.extend(quality_recs)
                                    except Exception:
                                        pass

                                if sched_ctrl and hasattr(
                                    sched_ctrl, "get_recommendations"
                                ):
                                    try:
                                        sched_recs = sched_ctrl.get_recommendations()
                                        all_recommendations.extend(sched_recs)
                                    except Exception:
                                        pass

                                if all_recommendations:
                                    smart_tuner_decisions["recommendations"] = (
                                        all_recommendations[:3]
                                    )  # До 3 рекомендаций

                                # Предупреждения
                                warnings = []
                                if reduced_loss > 5.0:
                                    warnings.append(
                                        "Высокий loss - возможны проблемы с обучением"
                                    )
                                if grad_norm > 10.0:
                                    warnings.append(
                                        "Высокий grad_norm - возможен взрыв градиентов"
                                    )
                                if learning_rate > 0.01:
                                    warnings.append(
                                        "Высокий learning rate - возможна нестабильность"
                                    )

                                if warnings:
                                    smart_tuner_decisions["warnings"] = warnings

                                print(f"   - smart_tuner_decisions: {smart_tuner_decisions}")

                                # 🔍 Собираем данные для DEBUG REPORTER
                                if debug_reporter:
                                    try:
                                        # Собираем loss компоненты для детальной диагностики
                                        loss_components = {
                                            'total_loss': reduced_loss,
                                            'mel_loss': reduced_taco_loss if 'reduced_taco_loss' in locals() else 0.0,
                                            'gate_loss': reduced_gate_loss if 'reduced_gate_loss' in locals() else 0.0,
                                        }
                                        
                                        # Добавляем guided attention loss если есть
                                        if guide_loss is not None and y_pred is not None:
                                            try:
                                                if len(y_pred) >= 4:
                                                    alignments = y_pred[3] if len(y_pred) == 4 else y_pred[4]
                                                    if alignments is not None:
                                                        # 🔥 ИСПРАВЛЕНИЕ: Используем правильный формат для guided loss
                                                        guided_loss_result = guide_loss(y_pred)  # Передаем весь y_pred
                                                        # Обрабатываем разные типы возвращаемых значений
                                                        if isinstance(guided_loss_result, tuple):
                                                            guided_loss_val = guided_loss_result[0]
                                                        else:
                                                            guided_loss_val = guided_loss_result
                                                        loss_components['guided_loss'] = guided_loss_val.item()
                                            except Exception as e:
                                                print(f"⚠️ Ошибка вычисления guided loss для debug: {e}")
                                        
                                        # Добавляем MMI loss если есть
                                        if mmi_loss is not None and y_pred is not None:
                                            try:
                                                # Безопасная обработка y_pred для MMI
                                                if isinstance(y_pred, (list, tuple)) and len(y_pred) > 1:
                                                    mel_outputs = y_pred[1]
                                                    if hasattr(mel_outputs, 'shape'):  # Проверяем что это тензор
                                                        # 🔥 ИСПРАВЛЕНИЕ: Приводим типы данных к одному формату
                                                        mel_outputs = mel_outputs.float()  # Приводим к float32
                                                        mel_target = y[0].float()  # Приводим к float32
                                                        mmi_loss_val = mmi_loss(mel_outputs, mel_target)
                                                        loss_components['mmi_loss'] = mmi_loss_val.item()
                                            except Exception as e:
                                                print(f"⚠️ Ошибка вычисления MMI loss для debug: {e}")
                                        
                                        # Собираем основные метрики
                                        debug_metrics = {
                                            'loss': reduced_loss,
                                            'grad_norm': grad_norm,
                                            'learning_rate': learning_rate,
                                            'batch_size': hparams.batch_size,
                                            'iteration': iteration,
                                            'epoch': epoch,
                                            'diagonality': 0.0,  # Будет вычислено в debug_reporter
                                            'quality': 0.0,      # Будет вычислено в debug_reporter
                                        }
                                        
                                        # Отправляем данные в debug reporter
                                        debug_reporter.collect_step_data(
                                            step=iteration,
                                            metrics=debug_metrics,
                                            model=model,
                                            y_pred=y_pred,
                                            loss_components=loss_components,
                                            hparams=hparams,
                                            smart_tuner_decisions=smart_tuner_decisions
                                        )
                                        
                                    except Exception as e:
                                        print(f"⚠️ Ошибка сбора debug данных: {e}")

                                # ГАРАНТИРОВАННО отправляем графики (send_plots=True)
                                try:
                                    result = telegram_monitor.send_training_update(
                                        step=iteration,
                                        metrics=telegram_metrics,
                                        smart_tuner_decisions=smart_tuner_decisions
                                    )
                                except Exception as e:
                                    print(f"⚠️ Ошибка отправки Telegram уведомления: {e}")
                                    result = False
                                print(f"📱 Telegram уведомление {'УСПЕШНО' if result else 'НЕ'} отправлено для шага {iteration}")

                        except Exception as e:
                            print(f"⚠️ Ошибка Telegram уведомления: {e}")
                            import traceback
                            print(f"   Traceback: {traceback.format_exc()}")

                    # --- Отправка аудио каждые 500 шагов ---
                    if is_main_node and telegram_monitor:
                        try:
                            # Проверяем, нужно ли отправить аудио (только каждые 500 шагов)
                            if iteration % 500 == 0 and iteration != 0:
                                print(f"🎵 Отправляем тестовый аудио для шага {iteration}")

                                # Тестовый текст с ударениями для Tacotron2
                                test_text = "Привет! Как дела? Сегодня прекрасная погода для прогулки в парке."

                                # Генерируем и отправляем аудио
                                try:
                                    audio_result = (
                                        telegram_monitor.generate_and_send_test_audio(
                                            model=model, step=iteration, test_text=test_text
                                        )
                                    )
                                except Exception as e:
                                    print(f"⚠️ Ошибка генерации аудио: {e}")
                                    audio_result = False

                                if audio_result:
                                    print(
                                        f"✅ Аудиофайлы успешно отправлены для шага {iteration}"
                                    )
                                    last_audio_step = iteration
                                else:
                                    print(
                                        f"⚠️ Не удалось отправить аудиофайлы для шага {iteration}"
                                    )

                        except Exception as e:
                            print(f"⚠️ Ошибка отправки аудио: {e}")
                            import traceback
                            print(f"   Traceback: {traceback.format_exc()}")

                if iteration % hparams.validation_freq == 0:
                    print(f"🔍 Выполняем валидацию на итерации {iteration}")
                    try:
                        val_loss = validate(
                            model,
                            criterion,
                            valset,
                            iteration,
                            hparams.batch_size,
                            n_gpus,
                            collate_fn,
                            writer,
                            hparams.distributed_run,
                            rank,
                        )
                    except Exception as e:
                        print(f"⚠️ Ошибка валидации: {e}")
                        val_loss = float('inf')
                    print(f"📊 Validation loss: {val_loss}")

                    # Auto hyper-parameter tuning (on main node)
                    if is_main_node and quality_ctrl:
                        # Формируем метрики для анализа
                        align_score = getattr(
                            model, "last_validation_alignment_score", None
                        )
                        metrics_dict = {
                            "val_loss": val_loss,
                            "attention_alignment_score": (
                                align_score if align_score is not None else 0.0
                            ),
                        }
                        attention_w = getattr(model, "last_validation_alignments", None)
                        gate_out = getattr(model, "last_validation_gate_outputs", None)
                        mel_out = getattr(model, "last_validation_mel_outputs", None)
                        try:
                            analysis = quality_ctrl.analyze_training_quality(
                                epoch=iteration,
                                metrics=metrics_dict,
                                attention_weights=attention_w,
                                gate_outputs=gate_out,
                                mel_outputs=mel_out,
                            )
                            for intrv in analysis.get("recommended_interventions", []):
                                new_hp = quality_ctrl.apply_quality_intervention(
                                    intrv, vars(hparams), step=iteration
                                )
                                # Применяем изменения к объекту hparams и модели
                                for k, v in new_hp.items():
                                    if hasattr(hparams, k):
                                        old_value = getattr(hparams, k)
                                        setattr(hparams, k, v)
                                        # Отслеживаем изменение параметра
                                        quality_ctrl.track_parameter_change(
                                            k,
                                            old_value,
                                            v,
                                            f"Quality intervention: {intrv.get('type', 'unknown')}",
                                            iteration,
                                        )
                                    if (
                                        k in ["guide_loss_weight", "guided_attn_weight"]
                                        and hparams.use_guided_attn
                                    ):
                                        guide_loss.alpha = v
                                        guide_loss.current_weight = v
                                    # Синхронизируем оба возможных названия в hparams
                                    setattr(hparams, "guided_attn_weight", v)
                                    setattr(hparams, "guide_loss_weight", v)
                                    if k == "learning_rate":
                                        for g in optimizer.param_groups:
                                            g["lr"] = v
                                    if k == "gate_threshold" and hasattr(
                                        model.decoder, "gate_threshold"
                                    ):
                                        model.decoder.gate_threshold = v

                                # Сохраняем изменения параметров в модели для Telegram
                                if not hasattr(model, "last_param_changes"):
                                    model.last_param_changes = {}

                                for k, v in new_hp.items():
                                    if hasattr(hparams, k):
                                        old_value = getattr(hparams, k)
                                        model.last_param_changes[k] = {
                                            "old_value": old_value,
                                            "new_value": v,
                                            "reason": f"Quality intervention: {intrv.get('type', 'unknown')}",
                                        }
                        except Exception as e:
                            print(f"⚠️ AdvancedQualityController ошибка: {e}")

                    # --- EarlyStopController: анализ и решения ---
                    if is_main_node and stop_ctrl:
                        try:
                            # Добавляем validation метрики
                            align_score = getattr(
                                model, "last_validation_alignment_score", None
                            )
                            stop_ctrl.add_metrics(
                                {
                                    "val_loss": val_loss,
                                    "attention_alignment_score": (
                                        align_score if align_score is not None else 0.0
                                    ),
                                }
                            )

                            # Получаем решение контроллера
                            decision = stop_ctrl.decide_next_step(vars(hparams))
                            if decision["action"] == "apply_patch":
                                hparams_dict = decision["new_hparams"]
                                for k, v in hparams_dict.items():
                                    if hasattr(hparams, k):
                                        setattr(hparams, k, v)
                                    if k == "learning_rate":
                                        for g in optimizer.param_groups:
                                            g["lr"] = v
                                    if (
                                        k in ["guide_loss_weight", "guided_attn_weight"]
                                        and hparams.use_guided_attn
                                    ):
                                        guide_loss.alpha = v
                                        guide_loss.current_weight = v
                                print(
                                    f"🛠  EarlyStop/Rescue применил патч: {decision['reason']}"
                                )

                            # Проверяем ранний останов
                            should_stop, reason = stop_ctrl.should_stop_early(
                                {"val_loss": val_loss}
                            )
                            if should_stop:
                                print(f"🟥 Ранний останов: {reason}")
                                return {
                                    "validation_loss": val_loss,
                                    "iteration": iteration,
                                    "checkpoint_path": None,
                                    "early_stop_reason": reason,
                                }
                        except Exception as e:
                            print(f"⚠️ EarlyStopController ошибка: {e}")

                if is_main_node and (iteration % hparams.iters_per_checkpoint == 0):
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration)
                    )
                    try:
                        save_checkpoint(
                            model, optimizer, learning_rate, iteration, checkpoint_path
                        )
                    except Exception as e:
                        print(f"⚠️ Ошибка сохранения чекпоинта: {e}")

                iteration += 1
            else:
                continue
            break
        else:
            # Если обучение завершилось без критических ошибок
            break

    # Сохраняем финальные метрики для Smart Tuner
    if is_main_node:
        print(f"🏁 Обучение завершено после {iteration} итераций")
        val_loss = validate(
            model,
            criterion,
            valset,
            iteration,
            hparams.batch_size,
            n_gpus,
            collate_fn,
            writer,
            hparams.distributed_run,
            rank,
        )

        # Сохраняем финальный checkpoint
        final_checkpoint_path = os.path.join(
            output_directory, f"checkpoint_final_{iteration}"
        )
        save_checkpoint(
            model, optimizer, learning_rate, iteration, final_checkpoint_path
        )

        final_metrics = {
            "validation_loss": val_loss,
            "iteration": iteration,
            "checkpoint_path": final_checkpoint_path,
        }
        print(f"📊 Финальные метрики: {final_metrics}")

        # --- Финальные сводки от контроллеров ---
        if is_main_node:
            try:
                if quality_ctrl:
                    quality_summary = quality_ctrl.get_quality_summary()
                    print(f"🎯 Quality Summary: {quality_summary}")

                if stop_ctrl:
                    stop_summary = stop_ctrl.get_tts_training_summary()
                    print(f"🛑 EarlyStop Summary: {stop_summary}")

                if optimizer_epochs:
                    epoch_summary = optimizer_epochs.get_optimization_summary()
                    print(f"📈 Epoch Optimization Summary: {epoch_summary}")
            except Exception as e:
                print(f"⚠️ Ошибка при получении финальных сводок: {e}")

        if writer:
            writer.close()
        return final_metrics

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_directory", type=str, help="directory to save checkpoints"
    )
    parser.add_argument(
        "-l", "--log_directory", type=str, help="directory to save tensorboard logs"
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        default=None,
        required=False,
        help="checkpoint path",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="load model weights only, ignore specified layers",
    )
    parser.add_argument(
        "--ignore-mmi-layers",
        action="store_true",
        help="load model weights only, ignore specified layers",
    )
    parser.add_argument(
        "--ignore-gst-layers",
        action="store_true",
        help="load model weights only, ignore specified layers",
    )
    parser.add_argument(
        "--ignore-tsgst-layers",
        action="store_true",
        help="load model weights only, ignore specified layers",
    )
    parser.add_argument("--no-dga", action="store_true", help="do not use DGA")
    parser.add_argument(
        "--n_gpus", type=int, default=1, required=False, help="number of gpus"
    )
    parser.add_argument(
        "--rank", type=int, default=0, required=False, help="rank of current gpu"
    )
    parser.add_argument(
        "--group_name",
        type=str,
        default="group_name",
        required=False,
        help="Distributed group name",
    )
    parser.add_argument(
        "--hparams", type=str, required=False, help="comma separated name=value pairs"
    )

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)
    hparams.no_dga = True

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Use MMI:", hparams.use_mmi)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(
        args.output_directory,
        args.log_directory,
        args.checkpoint_path,
        args.warm_start,
        args.ignore_mmi_layers,
        args.ignore_gst_layers,
        args.ignore_tsgst_layers,
        args.n_gpus,
        args.rank,
        args.group_name,
        hparams,
    )
