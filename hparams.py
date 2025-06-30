# Оптимизированная конфигурация hparams для Tacotron2
# Исправления проблем с толстыми полосами attention и повышение качества модели
# Совместимость с Smart Tuner V2 системой

from tools import HParams
from text import symbols
import logging

def create_hparams(hparams_string=None, verbose=False):
    """Создание оптимизированных гиперпараметров модели с исправлениями для attention alignment."""

    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500000,
        iters_per_checkpoint=1000,          # Увеличено для стабильности
        save_interval=2000,                 # Увеличено для экономии места
        validate_interval=200,              # Более частая валидация
        validation_freq=200,                # Частота валидации (iterations)
        warmup_steps=2000,                  # Увеличено для лучшей стабильности
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=True,                     # Отключено по умолчанию (требует NVIDIA Apex)
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],
        mmi_ignore_layers=["decoder.linear_projection.linear_layer.weight", 
                          "decoder.linear_projection.linear_layer.bias", 
                          "decoder.gate_layer.linear_layer.weight"],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        dataset_path='data/',
        training_files='data/dataset/train.csv',
        validation_files='data/dataset/val.csv',
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

        # Encoder parameters - оптимизированы для лучшего выравнивания
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters - исправления для attention alignment
        n_frames_per_step=1,                # Оставляем 1 для лучшего качества
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=2500,             # Увеличено для длинных последовательностей
        gate_threshold=0.5,
        p_attention_dropout=0.1,            # СНИЖЕНО с 0.15 до 0.1 для тонкого attention
        p_decoder_dropout=0.1,              # СНИЖЕНО с 0.15 до 0.1
        p_teacher_forcing=1.0,

        # Attention parameters - оптимизированы для тонких полос
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters - критически важно для attention
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        # GST - оптимизированные параметры Global Style Tokens
        use_gst=True,
        ref_enc_filters=[32, 32, 64, 64, 128, 128],
        ref_enc_size=[3, 3],
        ref_enc_strides=[2, 2],
        ref_enc_pad=[1, 1],
        ref_enc_gru_size=128,

        # Style Token Layer - улучшенные параметры
        token_embedding_size=256,
        token_num=8,                        # УМЕНЬШЕНО с 10 до 8 для лучшей стабильности
        num_heads=4,                        # УМЕНЬШЕНО с 8 до 4 для фокусировки

        no_dga=False,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,                 # ПОВЫШЕНО с 5e-4 до 1e-3 для быстрой сходимости
        learning_rate_decay=0.5,            # Более агрессивное снижение
        learning_rate_decay_patience=2000,  # Более частое снижение
        learning_rate_min=1e-5,             # Переименовано из min_learning_rate
        weight_decay=1e-6,                  # СНИЖЕНО с 1e-5 до 1e-6
        grad_clip_thresh=1.0,
        batch_size=48,                      # УВЕЛИЧЕНО с 32 до 48 для стабильности
        mask_padding=True,

        ################################
        # FINE-TUNE - Критические исправления #
        ################################
        use_mmi=True,
        mmi_map=None,                       # MMI карта (None для автоматического вычисления)
        mmi_weight=0.1,                     # Вес MMI loss

        # Guided Attention Force - исправления для тонких полос
        drop_frame_rate=0.05,               # СНИЖЕНО с 0.1 до 0.05
        use_gaf=True,
        update_gaf_every_n_step=3,          # УМЕНЬШЕНО с 5 до 3 для частых обновлений
        max_gaf=0.5,                        # СНИЖЕНО с 0.8 до 0.5 для мягкого воздействия
        
        # Динамический вес для Guide Loss - улучшенные параметры
        use_dynamic_guide_loss=True,
        guide_loss_initial_weight=0.5,      # СНИЖЕНО с 1.0 до 0.5
        guide_loss_decay_start=3000,        # УМЕНЬШЕНО с 5000 до 3000
        guide_loss_decay_steps=30000,       # УМЕНЬШЕНО с 50000 до 30000
        
        global_mean_npy='ruslan_global_mean.npy',
        
        ################################
        # Regularization & Stability  #
        ################################
        # Критические исправления dropout для тонкого attention
        dropout_rate=0.3,                   # СНИЖЕНО с 0.5 до 0.3
        encoder_dropout_rate=0.05,          # СНИЖЕНО с 0.1 до 0.05
        postnet_dropout_rate=0.1,           # СНИЖЕНО с 0.15 до 0.1
        
        # Early stopping parameters - оптимизированы
        early_stopping=True,
        early_stopping_patience=15,         # УВЕЛИЧЕНО с 10 до 15
        early_stopping_min_delta=0.0005,    # УМЕНЬШЕНО с 0.001 до 0.0005

        ################################
        # NEW: Attention Optimization  #
        ################################
        # Новые параметры для улучшения attention alignment
        attention_alignment_loss_weight=1.0,  # Вес loss для выравнивания
        use_attention_smoothing=True,         # Сглаживание attention
        attention_smoothing_factor=0.1,      # Коэффициент сглаживания
        
        # Teacher forcing scheduling - новые параметры
        use_scheduled_teacher_forcing=True,   # Включаем планированное teacher forcing
        teacher_forcing_start_ratio=1.0,     # Начальное значение
        teacher_forcing_end_ratio=0.5,       # Конечное значение
        teacher_forcing_decay_start=10000,   # Начало снижения
        teacher_forcing_decay_steps=50000,   # Шаги для снижения

        ################################
        # NEW: Advanced Training       #
        ################################
        # Дополнительные параметры для стабильности
        use_balanced_sampling=True,          # Балансированная выборка
        gradient_accumulation_steps=1,       # Накопление градиентов
        max_grad_norm=1.0,                  # Максимальная норма градиента
        
        # Loss scaling для FP16 - улучшенные настройки
        loss_scale_init=512.0,              # Начальный масштаб loss
        loss_scale_factor=2.0,              # Фактор изменения масштаба
        
        # Attention mechanism improvements
        use_forward_attention=False,         # Можно включить для монотонности
        use_location_sensitive_attention=True,  # Уже используется, явно указываем
        
        # Memory optimization
        use_memory_efficient_attention=True, # Оптимизация памяти для attention
        attention_chunk_size=None,          # Размер чанка для attention
        
        ################################
        # NEW: Smart Tuner V2 TTS Parameters #
        ################################
        # TTS-специфичные параметры для Smart Tuner V2 (только недостающие)
        guided_attention_enabled=True,       # Включение guided attention
        guide_loss_weight=1.0,              # Простой вес guided loss (дублирует guide_loss_initial_weight)
        
        # Примечание: остальные параметры уже определены выше:
        # p_attention_dropout (как attention_dropout), gate_threshold, 
        # dropout_rate/encoder_dropout_rate/postnet_dropout_rate (как prenet/postnet_dropout)
        
        ################################
        # NEW: Monitoring & Diagnostics #
        ################################
        # Новые параметры для мониторинга качества
        log_attention_weights=True,          # Логирование весов attention
        log_alignment_metrics=True,          # Логирование метрик выравнивания
        alignment_diagonal_threshold=0.5,    # Порог для диагональности
        
        # Validation enhancements
        validation_attention_analysis=True,  # Анализ attention на валидации
        save_attention_plots=True,          # Сохранение графиков attention
        attention_plot_frequency=1000,      # Частота сохранения графиков

        # Guided Attention
        use_guided_attn=True,
        guided_attn_weight=1.0,             # Вес guided attention loss
    )

    if hparams_string:
        logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        logging.info('Final parsed hparams: %s', hparams.values())

    # Валидация критических параметров
    _validate_attention_params(hparams)
    
    return hparams

def _validate_attention_params(hparams):
    """Валидация параметров для оптимального attention alignment."""
    warnings = []
    
    # Проверка dropout параметров
    if hparams.dropout_rate > 0.4:
        warnings.append("dropout_rate > 0.4 может ухудшить attention alignment")
    
    if hparams.p_attention_dropout > 0.15:
        warnings.append("p_attention_dropout > 0.15 может создать толстые полосы")
    
    # Проверка guided attention параметров
    if hparams.max_gaf > 0.7:
        warnings.append("max_gaf > 0.7 может размазать attention")
    
    # Проверка batch size
    if hparams.batch_size < 32:
        warnings.append("batch_size < 32 может ухудшить стабильность attention")
    
    # Проверка GST параметров
    if hparams.token_num > 10:
        warnings.append("token_num > 10 может ухудшить стабильность GST")
    
    if warnings:
        logging.warning("Предупреждения конфигурации:")
        for warning in warnings:
            logging.warning(f"  - {warning}")
    else:
        logging.info("Конфигурация прошла валидацию для оптимального attention alignment")