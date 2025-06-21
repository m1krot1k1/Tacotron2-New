from tools import HParams
from text import symbols
import logging


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500000,
        iters_per_checkpoint=500,
        save_interval=1000,          # Интервал сохранения чекпоинтов
        validate_interval=100,       # Интервал валидации
        warmup_steps=1000,          # Шаги прогрева learning rate
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=True,  # Включено FP16 для быстродействия
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],
        mmi_ignore_layers=["decoder.linear_projection.linear_layer.weight", "decoder.linear_projection.linear_layer.bias", "decoder.gate_layer.linear_layer.weight"],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        dataset_path='data/',  # Исправлен путь к данным
        training_files='data/dataset/train.csv',  # Исправлен путь к файлу обучения
        validation_files='data/dataset/val.csv',  # Исправлен путь к файлу валидации
        text_cleaners=['transliteration_cleaners_with_stress'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=2000,  # Увеличено для стабильности
        gate_threshold=0.5,
        p_attention_dropout=0.15,  # Увеличено с 0.1
        p_decoder_dropout=0.15,   # Увеличено с 0.1
        p_teacher_forcing=1.0,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        # GST ======================
        # Reference encoder
        use_gst=True,
        ref_enc_filters=[32, 32, 64, 64, 128, 128],
        ref_enc_size=[3, 3],
        ref_enc_strides=[2, 2],
        ref_enc_pad=[1, 1],
        ref_enc_gru_size=128,

        # Style Token Layer
        token_embedding_size=256,
        token_num=10,
        num_heads=8,

        no_dga=False,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=5e-4,  # Уменьшено с 1e-3 для стабильности
        learning_rate_decay=0.98,  # Более мягкое снижение
        learning_rate_decay_patience=3,  # Более агрессивное снижение
        min_learning_rate=1e-7,  # Более низкий минимум
        weight_decay=1e-5,  # Увеличена регуляризация
        grad_clip_thresh=1.0,
        batch_size=32,  # Уменьшено с 48 для стабильности
        mask_padding=True,  # set model's padded outputs to padded values

        ################################
        # FINE-TUNE #
        ################################
        use_mmi=True,  # Включаем MMI для лучшего качества

        drop_frame_rate=0.1,  # Увеличено для регуляризации
        use_gaf=True,  # Обучение с Guided Attention Force включено
        update_gaf_every_n_step=5,  # Более частые обновления
        max_gaf=0.8,  # Увеличено для лучшего выравнивания
        
        global_mean_npy='ruslan_global_mean.npy',
        
        ################################
        # Regularization & Stability  #
        ################################
        dropout_rate=0.5,             # Общий dropout для всех дропаут-слоев
        encoder_dropout_rate=0.1,
        postnet_dropout_rate=0.15,
        
        # Early stopping parameters
        early_stopping=True,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
    )

    if hparams_string:
        logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
