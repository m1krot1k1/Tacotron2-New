import os
import time
import argparse
import math
import random
import numpy as np

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
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy, plot_gate_outputs_to_numpy
from text import symbol_to_id
from utils import to_gpu
# from logger import Tacotron2Logger  # Не используется
from auto_param_controller import AutoParamController

# MLflow for experiment tracking
try:
    import mlflow
    import mlflow.pytorch  # позволяет логировать модели PyTorch
    MLFLOW_AVAILABLE = True
    
    # Импортируем улучшенное логирование
    try:
        from mlflow_metrics_enhancer import log_enhanced_training_metrics, log_system_metrics
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
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.cuda.*DtypeTensor.*")
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
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

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

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
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
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_dict = checkpoint_dict['state_dict']
    # model_dict['embedding.embedding.weight'] = model_dict['embedding.weight']
    # del model_dict['embedding.weight']
    print("ignoring layers:",ignore_layers)
    if len(ignore_layers) > 0 or exclude:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers and (not exclude or exclude not in k)}
        
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict, strict=False)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    # Добавляем weights_only=False для совместимости с PyTorch 2.6+
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, writer, distributed_run=False, rank=1, minimize=False):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)
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
                        decoder_outputs_val, mel_outputs_val, mel_outputs_postnet_val, gate_outputs_val, alignments_val = validation_outputs[:5]
                        
                        # Для изображений используем validation outputs (они корректного размера)
                        inference_outputs = [None, mel_outputs_val, mel_outputs_postnet_val, gate_outputs_val, alignments_val]
                        print(f"✅ Validation forward pass: mel={mel_outputs_postnet_val.shape if mel_outputs_postnet_val is not None else 'None'}, "
                              f"gate={gate_outputs_val.shape if gate_outputs_val is not None else 'None'}, "
                              f"align={alignments_val.shape if alignments_val is not None else 'None'}")
                    else:
                        print(f"⚠️ Validation outputs неполные: {len(validation_outputs)} элементов")
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
                _, mel_outputs_inf, mel_outputs_postnet_inf, gate_outputs_inf, alignments_inf = inference_outputs[:5]
                mel_targets, gate_targets = y[0], y[1]
                
                print(f"🖼️ Создаем изображения для TensorBoard (iteration {iteration})")
                
                # plot distribution of parameters (только каждые 500 итераций для экономии времени)
                if iteration % 500 == 0:
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram(tag, value.data.cpu().numpy(), iteration)
                
                idx = 0  # Используем первый элемент из батча
                
                # 🔥 ИСПРАВЛЕННОЕ создание изображений с проверкой размеров
                
                # Alignment изображение
                if alignments_inf is not None and alignments_inf.size(0) > idx:
                    try:
                        alignment_data = alignments_inf[idx].data.cpu().numpy()
                        if alignment_data.shape[0] > 1 and alignment_data.shape[1] > 1:
                            alignment_img = plot_alignment_to_numpy(alignment_data.T)
                            writer.add_image("alignment", alignment_img, iteration, dataformats='HWC')
                            print(f"✅ Alignment изображение создано: {alignment_img.shape}")
                        else:
                            print(f"⚠️ Alignment матрица слишком маленькая: {alignment_data.shape}")
                    except Exception as e:
                        print(f"❌ Ошибка создания alignment изображения: {e}")
                
                # Mel target изображение
                if mel_targets.size(0) > idx:
                    try:
                        mel_target_data = mel_targets[idx].data.cpu().numpy()
                        if mel_target_data.shape[0] > 1 and mel_target_data.shape[1] > 1:
                            mel_target_img = plot_spectrogram_to_numpy(mel_target_data)
                            writer.add_image("mel_target", mel_target_img, iteration, dataformats='HWC')
                            print(f"✅ Mel target изображение создано: {mel_target_img.shape}")
                        else:
                            print(f"⚠️ Mel target слишком маленький: {mel_target_data.shape}")
                    except Exception as e:
                        print(f"❌ Ошибка создания mel target изображения: {e}")
                
                # Mel predicted изображение
                if mel_outputs_inf is not None and mel_outputs_inf.size(0) > idx:
                    try:
                        mel_pred_data = mel_outputs_inf[idx].data.cpu().numpy()
                        if mel_pred_data.shape[0] > 1 and mel_pred_data.shape[1] > 1:
                            mel_pred_img = plot_spectrogram_to_numpy(mel_pred_data)
                            writer.add_image("mel_predicted", mel_pred_img, iteration, dataformats='HWC')
                            print(f"✅ Mel predicted изображение создано: {mel_pred_img.shape}")
                        else:
                            print(f"⚠️ Mel predicted слишком маленький: {mel_pred_data.shape}")
                    except Exception as e:
                        print(f"❌ Ошибка создания mel predicted изображения: {e}")
                
                # Gate outputs изображение
                if gate_outputs_inf is not None and gate_outputs_inf.size(0) > idx and gate_targets.size(0) > idx:
                    try:
                        gate_target_data = gate_targets[idx].data.cpu().numpy()
                        gate_pred_data = torch.sigmoid(gate_outputs_inf[idx]).data.cpu().numpy()
                        
                        if len(gate_target_data) > 1 and len(gate_pred_data) > 1:
                            gate_img = plot_gate_outputs_to_numpy(gate_target_data, gate_pred_data)
                            writer.add_image("gate", gate_img, iteration, dataformats='HWC')
                            print(f"✅ Gate изображение создано: {gate_img.shape}")
                        else:
                            print(f"⚠️ Gate данные слишком маленькие: target={len(gate_target_data)}, pred={len(gate_pred_data)}")
                    except Exception as e:
                        print(f"❌ Ошибка создания gate изображения: {e}")
                        
                # Принудительно сохраняем в TensorBoard
                writer.flush()
                print(f"🔄 TensorBoard данные сохранены для итерации {iteration}")
                
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
                    mel_target_img = plot_spectrogram_to_numpy(mel_targets[0].data.cpu().numpy())
                    writer.add_image("mel_target_fallback", mel_target_img, iteration, dataformats='HWC')
                    print(f"✅ Fallback mel target изображение создано")
            except Exception as fallback_e:
                print(f"❌ Даже fallback изображение не удалось создать: {fallback_e}")
                
        # 🔥 ОБЯЗАТЕЛЬНО: Убеждаемся что модель в train режиме
        model.train()

        if MLFLOW_AVAILABLE:
            validation_metrics = {
                "validation.loss": val_loss,
                "validation.step": iteration
            }
            
            # Дополнительные метрики из модели
            if hasattr(model, 'decoder') and hasattr(model.decoder, 'attention_weights'):
                try:
                    # Логируем метрики attention
                    attention_weights = model.decoder.attention_weights
                    if attention_weights is not None:
                        validation_metrics["validation.attention_entropy"] = float(
                            torch.mean(torch.sum(-attention_weights * torch.log(attention_weights + 1e-8), dim=-1))
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
                    validation_metrics["validation.attention_focus"] = float(torch.mean(attention_focus))
                except Exception as e:
                    print(f"⚠️ Ошибка при вычислении attention метрик: {e}")
            
            # Убеждаемся, что атрибут модели существует даже при отсутствии align_score
            if not hasattr(model, 'last_validation_alignment_score'):
                model.last_validation_alignment_score = None
            
            # Метрики из gate outputs
            if gate_outputs_inf is not None:
                try:
                    gate_probs = torch.sigmoid(gate_outputs_inf[0])
                    validation_metrics["validation.gate_mean"] = float(torch.mean(gate_probs))
                    validation_metrics["validation.gate_std"] = float(torch.std(gate_probs))
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
    print('\nCalculating global mean...\n')
    for i, batch in enumerate(data_loader):
        (text_padded, input_lengths, mel_padded, gate_padded,
         output_lengths, ctc_text, ctc_text_lengths, _guide_mask) = batch
        # padded values are 0.
        sums.append(mel_padded.double().sum(dim=(0, 2)))
        frames.append(output_lengths.double().sum())
    global_mean = sum(sums) / sum(frames)
    global_mean = global_mean.float()
    if global_mean_npy:
        np.save(global_mean_npy, global_mean)
    return to_gpu(global_mean)

def train(output_directory, log_directory, checkpoint_path, warm_start, ignore_mmi_layers, ignore_gst_layers, ignore_tsgst_layers, n_gpus,
          rank, group_name, hparams,
          # Параметры для интеграции со Smart Tuner
          smart_tuner_trial=None,
          smart_tuner_logger=None,
          tensorboard_writer=None,
          telegram_monitor=None):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    # ---------- FP16 / Mixed Precision ---------
    apex_available = False  # Флаг наличия NVIDIA Apex
    use_native_amp = False  # Флаг использования torch.cuda.amp
    scaler = None           # GradScaler для native AMP

    if hparams.fp16_run:
        try:
            from apex import amp
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2')
            apex_available = True
            print("✅ NVIDIA Apex успешно загружен для FP16 обучения")
        except ImportError:
            # Apex недоступен – используем встроенный AMP PyTorch
            try:
                from torch.amp import GradScaler, autocast
                # Отключаем FP16 для модели, используем только AMP
                model = model.float()  # Убеждаемся, что модель в FP32
                scaler = GradScaler('cuda')
                use_native_amp = True
                print("✅ NVIDIA Apex не найден. Переключаемся на torch.amp (PyTorch Native AMP)")
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
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_main_node = (rank == 0)

    # 💡 Используем переданный writer, а не создаем новый
    writer = tensorboard_writer if is_main_node else None

    if is_main_node and writer is None:
        # Для обратной совместимости, если train.py запускается напрямую
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_directory)

    if hparams.use_mmi:
        from mmi_loss import MMI_loss
        mmi_loss = MMI_loss(hparams.mmi_map, hparams.mmi_weight)
        print("✅ MMI loss загружен")

    if hparams.use_guided_attn:
        from loss_function import GuidedAttentionLoss
        guide_loss = GuidedAttentionLoss(alpha=hparams.guided_attn_weight)
        print("✅ Guided Attention Loss загружен")

    # --- Auto Hyper-parameter Controller ---
    auto_ctrl = None
    if is_main_node and hparams.use_guided_attn:
        try:
            auto_ctrl = AutoParamController(optimizer=optimizer,
                                            guide_loss=guide_loss,
                                            hparams=hparams,
                                            writer=writer)
            print("🤖 AutoParamController активирован")
        except Exception as e:
            print(f"⚠️ Не удалось инициализировать AutoParamController: {e}")

    global_mean = calculate_global_mean(train_loader, hparams.global_mean_npy)

    # ================ MAIN TRAINNIG LOOP ===================
    print(f"🚀 Начинаем обучение: epochs={hparams.epochs}, batch_size={hparams.batch_size}, dataset_size={len(train_loader)}")
    
    # Логирование параметров модели в MLflow
    if is_main_node and MLFLOW_AVAILABLE:
        model_params = {
            "model.total_params": sum(p.numel() for p in model.parameters()),
            "model.trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "hparams.batch_size": hparams.batch_size,
            "hparams.learning_rate": hparams.learning_rate,
            "hparams.epochs": hparams.epochs,
            "hparams.grad_clip_thresh": hparams.grad_clip_thresh,
            "hparams.fp16_run": hparams.fp16_run,
            "hparams.use_mmi": hparams.use_mmi,
            "hparams.use_guided_attn": hparams.use_guided_attn,
            "dataset.train_size": len(train_loader),
            "dataset.val_size": len(valset)
        }
        
        # Логируем параметры
        mlflow.log_params(model_params)
        print(f"📊 Параметры модели залогированы в MLflow: {model_params['model.total_params']} параметров")
    
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {} / {}".format(epoch, hparams.epochs))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            model.zero_grad()
            
            x, y = model.parse_batch(batch)

            # Forward pass с учётом выбранной схемы mixed precision
            if hparams.fp16_run and use_native_amp:
                with autocast('cuda'):
                    y_pred = model(x)
                    
                    # total loss
                    loss_taco, loss_gate, loss_atten, loss_emb = criterion(y_pred, y)
                    loss_guide = guide_loss(y_pred) if hparams.use_guided_attn else torch.tensor(0.0, device=y_pred[0].device)
                    loss_mmi = mmi_loss(y_pred[1], y[0]) if hparams.use_mmi else torch.tensor(0.0, device=y_pred[0].device)
                    loss = loss_taco + loss_gate + loss_atten + loss_guide + loss_mmi + loss_emb
            else:
                y_pred = model(x)
                # total loss
                loss_taco, loss_gate, loss_atten, loss_emb = criterion(y_pred, y)
                loss_guide = guide_loss(y_pred) if hparams.use_guided_attn else torch.tensor(0.0, device=y_pred[0].device)
                loss_mmi = mmi_loss(y_pred[1], y[0]) if hparams.use_mmi else torch.tensor(0.0, device=y_pred[0].device)
                loss = loss_taco + loss_gate + loss_atten + loss_guide + loss_mmi + loss_emb

            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
                reduced_taco_loss = loss_taco.item()
                reduced_atten_loss = loss_atten.item()
                reduced_mi_loss = loss_mmi.item()
                reduced_guide_loss = loss_guide.item() if hparams.use_guided_attn else 0.0
                reduced_gate_loss = loss_gate.item()
                reduced_emb_loss = loss_emb.item()

            # Backward pass
            if hparams.fp16_run and apex_available:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif hparams.fp16_run and use_native_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), hparams.grad_clip_thresh)
            
            # Optimizer step с учётом схемы mixed precision
            if hparams.fp16_run and use_native_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if is_main_node:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, grad_norm, duration))
                
                # Обновляем learning_rate переменную из optimizer (мог измениться авто-контроллером)
                learning_rate = optimizer.param_groups[0]['lr']

                # Логирование в TensorBoard
                writer.add_scalar("training.loss", reduced_loss, iteration)
                writer.add_scalar("training.taco_loss", reduced_taco_loss, iteration)
                writer.add_scalar("training.atten_loss", reduced_atten_loss, iteration)
                writer.add_scalar("training.mi_loss", reduced_mi_loss, iteration)
                writer.add_scalar("training.guide_loss", reduced_guide_loss, iteration)
                writer.add_scalar("training.gate_loss", reduced_gate_loss, iteration)
                writer.add_scalar("training.emb_loss", reduced_emb_loss, iteration)
                writer.add_scalar("grad.norm", grad_norm, iteration)
                writer.add_scalar("learning.rate", learning_rate, iteration)
                writer.add_scalar("duration", duration, iteration)
                if hparams.use_guided_attn:
                     writer.add_scalar("training.guide_loss_weight", guide_loss.get_weight(), iteration)

                # Логирование в MLflow
                if MLFLOW_AVAILABLE:
                    training_metrics = {
                        "training.loss": reduced_loss,
                        "training.taco_loss": reduced_taco_loss,
                        "training.atten_loss": reduced_atten_loss,
                        "training.mi_loss": reduced_mi_loss,
                        "training.guide_loss": reduced_guide_loss,
                        "training.gate_loss": reduced_gate_loss,
                        "training.emb_loss": reduced_emb_loss,
                        "grad.norm": grad_norm,
                        "learning.rate": learning_rate,
                        "duration": duration,
                        "epoch": epoch,
                        "iteration": iteration
                    }
                    
                    if hparams.use_guided_attn:
                        training_metrics["training.guide_loss_weight"] = guide_loss.get_weight()
                    
                    if ENHANCED_LOGGING:
                        # Используем расширенное логирование
                        log_enhanced_training_metrics(training_metrics, iteration)
                        log_system_metrics(iteration)
                    else:
                        # Базовое MLflow логирование
                        for metric_name, metric_value in training_metrics.items():
                            mlflow.log_metric(metric_name, metric_value, step=iteration)
                
                # 📱 Telegram уведомления каждые 100 шагов (чаще для отладки)
                if telegram_monitor:
                    try:

                        
                        if iteration % 100 == 0:
                            print(f"🚀 Отправляем Telegram уведомление для шага {iteration}")
                            
                            # Получаем attention weights из y_pred
                            attention_weights = None
                            gate_outputs = None
                            
                            if len(y_pred) >= 5:
                                attention_weights = y_pred[4] if y_pred[4] is not None else None
                            if len(y_pred) >= 4:
                                gate_outputs = y_pred[3] if y_pred[3] is not None else None
                            
                            print(f"   - attention_weights: {attention_weights.shape if attention_weights is not None else 'None'}")
                            print(f"   - gate_outputs: {gate_outputs.shape if gate_outputs is not None else 'None'}")
                            
                            # Метрики для Telegram
                            telegram_metrics = {
                                "loss": reduced_loss,
                                "mel_loss": reduced_taco_loss,
                                "gate_loss": reduced_gate_loss,
                                "guide_loss": reduced_guide_loss,
                                "grad_norm": grad_norm,
                                "learning_rate": learning_rate,
                                "epoch": epoch
                            }
                            
                            print(f"   - telegram_metrics: {telegram_metrics}")
                            
                            result = telegram_monitor.send_training_update(
                                step=iteration,
                                metrics=telegram_metrics,
                                attention_weights=attention_weights,
                                gate_outputs=gate_outputs
                            )
                            
                            print(f"📱 Telegram уведомление {'УСПЕШНО' if result else 'НЕ'} отправлено для шага {iteration}")
                        else:
                            print(f"   - Пропускаем шаг {iteration} (не кратен 100)")
                        
                    except Exception as e:
                        print(f"⚠️ Ошибка Telegram уведомления: {e}")
                        import traceback
                        print(f"   Traceback: {traceback.format_exc()}")

            if (iteration % hparams.validation_freq == 0):
                print(f"🔍 Выполняем валидацию на итерации {iteration}")
                val_loss = validate(model, criterion, valset, iteration, hparams.batch_size, n_gpus, collate_fn, writer, hparams.distributed_run, rank)
                print(f"📊 Validation loss: {val_loss}")
                # Auto hyper-parameter tuning (on main node)
                if is_main_node and auto_ctrl:
                    align_score = getattr(model, 'last_validation_alignment_score', None)
                    auto_ctrl.after_validation(iteration, val_loss, align_score)

            if is_main_node and (iteration % hparams.iters_per_checkpoint == 0):
                checkpoint_path = os.path.join(
                    output_directory, "checkpoint_{}".format(iteration))
                save_checkpoint(model, optimizer, learning_rate, iteration,
                                checkpoint_path)

            iteration += 1

    # Сохраняем финальные метрики для Smart Tuner
    if is_main_node:
        print(f"🏁 Обучение завершено после {iteration} итераций")
        val_loss = validate(model, criterion, valset, iteration, hparams.batch_size, n_gpus, collate_fn, writer, hparams.distributed_run, rank)
        
        # Сохраняем финальный checkpoint
        final_checkpoint_path = os.path.join(output_directory, f"checkpoint_final_{iteration}")
        save_checkpoint(model, optimizer, learning_rate, iteration, final_checkpoint_path)
        
        final_metrics = {
            "validation_loss": val_loss,
            "iteration": iteration,
            "checkpoint_path": final_checkpoint_path
        }
        print(f"📊 Финальные метрики: {final_metrics}")
        
        if writer:
            writer.close()
        return final_metrics
    
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm-start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--ignore-mmi-layers', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--ignore-gst-layers', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--ignore-tsgst-layers', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--no-dga', action='store_true',
                        help='do not use DGA')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

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
    
    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.ignore_mmi_layers, args.ignore_gst_layers, args.ignore_tsgst_layers, args.n_gpus, args.rank, args.group_name, hparams)
