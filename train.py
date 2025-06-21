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
from logger import Tacotron2Logger
from hparams import create_hparams
from utils import to_gpu
from text import symbol_to_id

# MLflow for experiment tracking
try:
    import mlflow
    import mlflow.pytorch  # позволяет логировать модели PyTorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
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


def prepare_directories_and_logger(output_directory, log_directory, rank, experiment_name=None):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        # Логи TensorBoard должны быть в log_directory, а не в output_directory
        if not os.path.isdir(log_directory):
            os.makedirs(log_directory)
            os.chmod(log_directory, 0o775)
        # Передаем имя эксперимента в logger для красивого отображения в TensorBoard
        logger = Tacotron2Logger(log_directory, run_name=experiment_name)
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = np.finfo('float16').min

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
             collate_fn, logger=None, distributed_run=False, rank=1, minimize=False):
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

    model.train()
    model.decoder.p_teacher_forcing = 1.0
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        if logger:
            logger.log_validation(val_loss, model, y, y_pred, iteration)
        if MLFLOW_AVAILABLE:
            mlflow.log_metric("validation.loss", val_loss, step=iteration)
    
    return val_loss  # Возвращаем validation loss для scheduler'а

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
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    # torch.autograd.set_detect_anomaly(True)

    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)
    if hparams.drop_frame_rate > 0.:
        global_mean = calculate_global_mean(train_loader, hparams.global_mean_npy)
        hparams.global_mean = global_mean
    
    hparams.end_symbols_ids = [symbol_to_id[s] for s in '?!.']


    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        try:
            from apex import amp
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2')
            hparams.use_builtin_amp = False
        except ImportError:
            # Используем встроенный в PyTorch AMP если apex недоступен
            print("Apex недоступен, используем встроенный PyTorch AMP")
            hparams.use_builtin_amp = True
    else:
        hparams.use_builtin_amp = False
    
    # Создаем scaler для встроенного AMP
    if hparams.fp16_run and hparams.use_builtin_amp:
        # Используем новую сигнатуру для PyTorch 2.6+
        try:
            scaler = torch.amp.GradScaler('cuda')
        except (AttributeError, TypeError):
            # Fallback для старых версий PyTorch
            scaler = torch.cuda.amp.GradScaler()

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    # Извлекаем имя эксперимента из пути для красивого отображения в TensorBoard
    experiment_name = os.path.basename(output_directory)
    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank, experiment_name)

    # ---------------- MLflow ----------------
    if MLFLOW_AVAILABLE and rank == 0:
        try:
            mlflow.set_tracking_uri("http://localhost:5000")
        except Exception:
            pass  # fallback to default URI
        experiment_name = os.path.basename(output_directory)
        # Проверяем, есть ли уже активный run от Smart Tuner
        existing_run_id = os.getenv('MLFLOW_RUN_ID')
        if existing_run_id:
            # Используем существующий run, созданный Smart Tuner
            mlflow.start_run(run_id=existing_run_id)
            print(f"🔗 Подключились к существующему MLflow run: {existing_run_id}")
        else:
            # Создаем новый run (обычный режим)
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()
        
        # Добавим тег, чтобы было понятно, что это вложенный/дочерний процесс
        # Это также поможет нам понять, что train.py был запущен тюнером
        if os.getenv('MLFLOW_RUN_ID'):
            mlflow.set_tag("run_type", "smart_tuner_child")
            
        # Логируем базовые гиперпараметры
        mlflow.log_params({
            "batch_size": hparams.batch_size,
            "learning_rate": learning_rate,
            "fp16": hparams.fp16_run,
            "gaf": hparams.use_gaf,
            "dataset": os.path.basename(hparams.training_files)
        })

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start or ignore_mmi_layers or ignore_gst_layers or ignore_tsgst_layers:
            layers = []
            exclude = None
            if warm_start:
                layers += hparams.ignore_layers
            if ignore_mmi_layers:
                layers += hparams.mmi_ignore_layers
            if ignore_gst_layers:
                exclude = 'gst.'
            if ignore_tsgst_layers:
                exclude = 'tpse_gst.'
            model = warm_start_model(
                checkpoint_path, model, layers, exclude)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    criterion = Tacotron2Loss(hparams, iteration=iteration)

    model.train()
    is_overflow = False
    
    # Learning Rate Scheduler - отслеживание validation loss
    best_val_loss = float('inf')
    patience_counter = 0
    # Early stopping variables
    early_best_val_loss = float('inf')
    early_patience_counter = 0
    stop_training = False
    
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()

            # --- Learning Rate Warm-up ---
            if iteration < hparams.warmup_steps:
                learning_rate = hparams.learning_rate * (iteration + 1) / hparams.warmup_steps
            else:
                # После прогрева можно использовать основную стратегию (например, decay)
                # Текущая логика decay уже есть ниже, в секции валидации
                pass # Просто используем текущий learning_rate

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            
            # Используем autocast для forward pass если включен FP16
            if hparams.fp16_run and hparams.use_builtin_amp:
                with torch.cuda.amp.autocast():
                    y_pred = model(x)
                    _loss = criterion(y_pred, y)
                    loss = sum(_loss)
            else:
                y_pred = model(x)
                _loss = criterion(y_pred, y)
                loss = sum(_loss)
            guide_loss_val = _loss[2]
            gate_loss = _loss[1]
            emb_loss = _loss[3]
            
            # --- Расчет динамического веса для Guide Loss ---
            guide_loss_weight = hparams.guide_loss_initial_weight
            if hparams.use_dynamic_guide_loss and iteration > hparams.guide_loss_decay_start:
                decay_progress = min(1.0, (iteration - hparams.guide_loss_decay_start) / hparams.guide_loss_decay_steps)
                guide_loss_weight = hparams.guide_loss_initial_weight * (1.0 - decay_progress)

            # Пересчитываем общую потерю с учетом веса
            # _loss[2] - это guide_loss
            loss = _loss[0] + _loss[1] + (guide_loss_val * guide_loss_weight) + _loss[3]

            if model.mi is not None:
                # transpose to [b, T, dim]
                decoder_outputs = y_pred[0].transpose(2, 1)
                ctc_text, ctc_text_lengths, aco_lengths = x[-2], x[-1], x[4]
                mi_loss = model.mi(decoder_outputs, ctc_text, aco_lengths, ctc_text_lengths)
                taco_loss = loss
                if hparams.use_gaf:
                    if i % hparams.update_gaf_every_n_step == 0:
                        safe_loss = 0. * sum([x.sum() for x in model.parameters()])
                        gaf = gradient_adaptive_factor.calc_grad_adapt_factor(
                            taco_loss + safe_loss, mi_loss + safe_loss, model.parameters(), optimizer)
                        gaf = min(gaf, hparams.max_gaf)
                else:
                    gaf = 1.0
                loss = loss + gaf * mi_loss
            else:
                taco_loss = sum([_loss[0],_loss[1]])
                mi_loss = torch.tensor([-1.0])
                gaf = -1.0

            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
                taco_loss = reduce_tensor(taco_loss.data, n_gpus).item()
                mi_loss = reduce_tensor(mi_loss.data, n_gpus).item()
                guide_loss = reduce_tensor(guide_loss_val.data, n_gpus).item()
                gate_loss = reduce_tensor(gate_loss.data, n_gpus).item()
                emb_loss = reduce_tensor(emb_loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
                taco_loss = taco_loss.item()
                mi_loss = mi_loss.item()
                guide_loss = guide_loss_val.item()
                gate_loss = gate_loss.item()
                emb_loss = emb_loss.item()


            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
            if hparams.fp16_run:
                if hparams.use_builtin_amp:
                    # Используем встроенный PyTorch AMP
                    scaler.scale(loss).backward()
                else:
                    # Используем apex AMP
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                if hparams.use_builtin_amp:
                    # Используем встроенный PyTorch AMP
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), hparams.grad_clip_thresh)
                    is_overflow = math.isnan(grad_norm) or math.isinf(grad_norm)
                else:
                    # Используем apex AMP
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), hparams.grad_clip_thresh)
                    is_overflow = math.isnan(grad_norm) or math.isinf(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm) or math.isinf(grad_norm)

            if not is_overflow:
                if hparams.fp16_run and hparams.use_builtin_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            elif rank == 0:
                print(f"Gradient overflow. Skipping step {iteration}...")

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.4f} mi_loss {:.4f} guide_loss {:.4f} gate_loss {:.4f} emb_loss {:.4f} Grad Norm {:.4f} {:.2f}s/it".format(
                    iteration, taco_loss, mi_loss, guide_loss, gate_loss, emb_loss, grad_norm, duration))
                logger.log_training(
                    reduced_loss, taco_loss, mi_loss, guide_loss, gate_loss, emb_loss, grad_norm,
                    learning_rate, duration, iteration, guide_loss_weight)

                # MLflow metrics
                if MLFLOW_AVAILABLE:
                    mlflow.log_metrics({
                        "training.loss": reduced_loss,
                        "training.taco_loss": taco_loss,
                        "training.mi_loss": mi_loss,
                        "training.guide_loss": guide_loss,
                        "training.gate_loss": gate_loss,
                        "training.emb_loss": emb_loss,
                        "grad_norm": grad_norm,
                        "learning_rate": learning_rate,
                        "duration": duration,
                        "guide_loss_weight": guide_loss_weight
                    }, step=iteration)

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                val_loss = validate(model, criterion, valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank)
                
                # Learning Rate Scheduler based on validation loss
                if rank == 0:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        print(f"📈 Новый лучший validation loss: {val_loss:.4f}")
                    else:
                        patience_counter += 1
                        print(f"⏳ Validation loss не улучшился: {patience_counter}/{hparams.learning_rate_decay_patience}")
                        
                        if patience_counter >= hparams.learning_rate_decay_patience:
                            old_lr = learning_rate
                            learning_rate = max(learning_rate * hparams.learning_rate_decay, 
                                              hparams.min_learning_rate)
                            if learning_rate != old_lr:
                                print(f"🔻 Learning Rate снижен: {old_lr:.6f} → {learning_rate:.6f}")
                                patience_counter = 0
                            else:
                                print(f"⚠️ Learning Rate достиг минимума: {learning_rate:.6f}")
                
                # ---------- Early Stopping Logic ----------
                if rank == 0 and hparams.early_stopping:
                    improvement = early_best_val_loss - val_loss
                    if improvement > hparams.early_stopping_min_delta:
                        early_best_val_loss = val_loss
                        early_patience_counter = 0
                        print(f"✅ EarlyStopping: улучшение {improvement:.6f}, счетчик сброшен")
                    else:
                        early_patience_counter += 1
                        print(f"⚠️ EarlyStopping patience: {early_patience_counter}/{hparams.early_stopping_patience}")
                        if early_patience_counter >= hparams.early_stopping_patience:
                            print("🛑 Early stopping triggered. Останавливаем обучение…")
                            stop_training = True

                # Распространяем флаг остановки на все процессы в распределенном режиме
                if hparams.distributed_run:
                    flag_tensor = torch.tensor([1 if stop_training else 0], dtype=torch.int, device=('cuda' if torch.cuda.is_available() else 'cpu'))
                    dist.broadcast(flag_tensor, src=0)
                    stop_training = bool(flag_tensor.item())

                if stop_training:
                    break

                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1

            if stop_training:
                break

        if stop_training:
            print("🏁 Обучение завершено по Early Stopping.")
            break

    # Завершаем MLflow run
    if MLFLOW_AVAILABLE and rank == 0 and mlflow.active_run() is not None:
        mlflow.end_run()

    logger.close()


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
