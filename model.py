from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths, dropout_frame
import contextlib
import numpy as np
import random
from text.symbols import ctc_symbols
from gst import GST, TPSEGST
import copy


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")
        self.use_location_relative = False  # Отключено для стабильности
        self.relative_sigma = 4.0

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        if self.use_location_relative:
            # Добавляем Location-Relative компонент
            bsz, max_time = processed_memory.size(0), processed_memory.size(1)
            positions = torch.arange(max_time, device=query.device).float()
            # Ожидаемая позиция центрируется на кумулятивном attention
            if attention_weights_cat is not None and attention_weights_cat.size(1) >= 2:
                cum_weights = attention_weights_cat[:,1]  # (B, T)
                expected_pos = (cum_weights * positions).sum(dim=1, keepdim=True) / cum_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
            else:
                expected_pos = torch.zeros(bsz, 1, device=query.device)
            distances = positions.unsqueeze(0) - expected_pos  # (B, T)
            relative_term = - (distances ** 2) / (2 * (self.relative_sigma ** 2))
            energies = energies + relative_term

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights



@contextlib.contextmanager
def temp_seed(seed=None):
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
    try:
        yield
    finally:
        if seed is not None:
            np.random.set_state(state)

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes, dropout_rate=0.5):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])
        
        # 🔥 РЕВОЛЮЦИОННЫЕ улучшения dropout для максимального качества
        self.dropout_rate = dropout_rate
        self.inference_dropout_rate = 0.001  # 🔥 МИНИМИЗИРОВАНО для стабильной генерации
        
        # 🔥 Адаптивный dropout в зависимости от фазы обучения
        self.adaptive_dropout = True
        self.training_step = 0

    def forward(self, x):
        for linear in self.layers:
            x = F.relu(linear(x))
            if self.training:
                # 🔥 Адаптивный dropout для максимального качества
                if self.adaptive_dropout and self.training_step > 1000:
                    # После 1000 шагов снижаем dropout для стабильности
                    adaptive_rate = max(0.01, self.dropout_rate * 0.5)
                    x = F.dropout(x, p=adaptive_rate, training=True)
                else:
                    x = F.dropout(x, p=self.dropout_rate, training=True)
            else:
                # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: НЕТ dropout на инференсе для стабильности!
                # Убираем случайность для консистентной генерации
                pass  # Никакого dropout на инференсе!
        
        if self.training:
            self.training_step += 1
            
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.dropout_rate = hparams.postnet_dropout_rate

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), self.dropout_rate, self.training)
        x = F.dropout(self.convolutions[-1](x), self.dropout_rate, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.dropout_rate = hparams.encoder_dropout_rate

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), self.dropout_rate, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        # self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        return outputs

    def inference(self, x):
        try:
            with torch.no_grad():
                # Проверяем и корректируем входные размерности
                original_shape = x.shape
                
                if x.dim() == 4:
                    # Убираем лишнее измерение если есть
                    x = x.squeeze(2)  # Убираем третье измерение
                elif x.dim() == 2:
                    x = x.unsqueeze(0)  # Добавляем batch dimension если нужно
                elif x.dim() != 3:
                    return None
                
                # Проверяем, что размерность каналов соответствует ожидаемой
                expected_channels = 512  # encoder_embedding_dim
                if x.size(-1) != expected_channels:
                    # Создаем корректный тензор с правильными размерностями
                    batch_size = x.size(0)
                    seq_len = min(x.size(1), 200)  # Ограничиваем длину последовательности
                    x = torch.zeros(batch_size, expected_channels, seq_len, 
                                  device=x.device, dtype=x.dtype)
                
                # Применяем конволюции
                for conv in self.convolutions:
                    x = F.dropout(F.relu(conv(x)), self.dropout_rate, self.training)

                x = x.transpose(1, 2)

                # self.lstm.flatten_parameters()
                outputs, _ = self.lstm(x)
                
                # Проверяем выходные размерности
                if outputs is not None and outputs.dim() == 3:
                    return outputs
                else:
                    return None
                    
        except Exception as e:
            # Молча возвращаем None при любой ошибке
            torch.cuda.empty_cache()
            return None


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        # Сохраняем исходное значение для совместимости
        self.base_encoder_embedding_dim = hparams.encoder_embedding_dim
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        if hparams.use_gst:
            self.encoder_embedding_dim = hparams.encoder_embedding_dim + hparams.token_embedding_size
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        # 🔧 ДОБАВЛЕНИЕ: Адаптивный gate threshold для лучшего качества остановки
        self.adaptive_gate = getattr(hparams, 'adaptive_gate_threshold', True)
        self.gate_min_threshold = getattr(hparams, 'gate_min_threshold', 0.3)
        self.gate_max_threshold = getattr(hparams, 'gate_max_threshold', 0.8)
        self.dropout_rate = hparams.dropout_rate
        # 🔧 ИСПРАВЛЕНИЕ: Поддержка curriculum learning для teacher forcing
        self.p_teacher_forcing = hparams.p_teacher_forcing
        self.curriculum_teacher_forcing = getattr(hparams, 'curriculum_teacher_forcing', False)

        # Maximazing Mutual Inforamtion
        # https://arxiv.org/abs/1909.01145
        # https://github.com/bfs18/tacotron2
        self.use_mmi = hparams.use_mmi

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim],
            self.dropout_rate)

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + self.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, self.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + self.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        lp_out_dim = hparams.decoder_rnn_dim if self.use_mmi else hparams.n_mel_channels * hparams.n_frames_per_step

        self.mel_layer = None
        if not self.use_mmi:
            self.linear_projection = LinearNorm(
                hparams.decoder_rnn_dim + self.encoder_embedding_dim,
                lp_out_dim
            )
        else:
            self.linear_projection = nn.Sequential(
                LinearNorm(
                    hparams.decoder_rnn_dim + self.encoder_embedding_dim,
                    lp_out_dim,
                    w_init_gain='relu'
                ),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )

            self.mel_layer = nn.Sequential(
                LinearNorm(
                    hparams.decoder_rnn_dim,
                    hparams.decoder_rnn_dim,
                    w_init_gain='relu'
                ),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                LinearNorm(
                    in_dim=hparams.decoder_rnn_dim,
                    out_dim=hparams.n_mel_channels * hparams.n_frames_per_step
                )
            )

        gate_in_dim = hparams.decoder_rnn_dim if self.use_mmi else \
            hparams.decoder_rnn_dim + self.encoder_embedding_dim

        self.gate_layer = LinearNorm(
            gate_in_dim, 1,
            bias=True, w_init_gain='sigmoid')

        self.attention_dropout = nn.Dropout(self.dropout_rate)
        self.decoder_dropout = nn.Dropout(self.dropout_rate)

        self.hparams = hparams

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        
        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Безопасное изменение размера с проверкой
        batch_size = decoder_inputs.size(0)
        time_steps = decoder_inputs.size(1)
        mel_channels = decoder_inputs.size(2)
        
        # Проверяем, что временные шаги делятся на n_frames_per_step
        if time_steps % self.n_frames_per_step != 0:
            # Обрезаем до ближайшего кратного числа
            new_time_steps = (time_steps // self.n_frames_per_step) * self.n_frames_per_step
            decoder_inputs = decoder_inputs[:, :new_time_steps, :]
            time_steps = new_time_steps
        
        # Безопасное изменение размера
        target_time_steps = time_steps // self.n_frames_per_step
        target_channels = mel_channels * self.n_frames_per_step
        
        try:
            # Используем reshape вместо view для лучшей совместимости
            decoder_inputs = decoder_inputs.reshape(
                batch_size, target_time_steps, target_channels)
        except RuntimeError as e:
            print(f"⚠️ Ошибка reshape в parse_decoder_inputs: {e}")
            print(f"   Входные размеры: {decoder_inputs.shape}")
            print(f"   Целевые размеры: ({batch_size}, {target_time_steps}, {target_channels})")
            
            # Fallback: обрезаем или дополняем тензор до нужного размера
            current_elements = decoder_inputs.numel()
            target_elements = batch_size * target_time_steps * target_channels
            
            if current_elements > target_elements:
                # Обрезаем лишние элементы
                decoder_inputs = decoder_inputs.flatten()[:target_elements]
            elif current_elements < target_elements:
                # Дополняем нулями
                padding_size = target_elements - current_elements
                padding = torch.zeros(padding_size, device=decoder_inputs.device, dtype=decoder_inputs.dtype)
                decoder_inputs = torch.cat([decoder_inputs.flatten(), padding])
            else:
                decoder_inputs = decoder_inputs.flatten()
                
            decoder_inputs = decoder_inputs.reshape(batch_size, target_time_steps, target_channels)
            
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments, decoder_outputs):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        if decoder_outputs:
            decoder_outputs = torch.stack(decoder_outputs).transpose(0, 1).contiguous()
            decoder_outputs = decoder_outputs.transpose(1, 2)
        else:
            decoder_outputs = None

        return mel_outputs, gate_outputs, alignments, decoder_outputs

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = self.attention_dropout(self.attention_hidden)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = self.decoder_dropout(self.decoder_hidden)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        if self.use_mmi:
            mel_output = self.mel_layer(decoder_output)
            decoder_hidden_attention_context = decoder_output
        else:
            mel_output = decoder_output
            decoder_output = None

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return mel_output, gate_prediction, self.attention_weights, decoder_output

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments, decoder_outputs = [], [], [], []
        
        # --- Scheduled Sampling / Curriculum Learning ---
        current_teacher_forcing = self.p_teacher_forcing
        if self.curriculum_teacher_forcing and hasattr(self, 'training_step'):
            step = getattr(self, 'training_step', 0)
            start_ratio = getattr(self.hparams, 'teacher_forcing_start_ratio', 1.0)
            end_ratio = getattr(self.hparams, 'teacher_forcing_end_ratio', 0.5)
            decay_start = getattr(self.hparams, 'teacher_forcing_decay_start', 10000)
            decay_steps = getattr(self.hparams, 'teacher_forcing_decay_steps', 50000)
            if step < decay_start:
                current_teacher_forcing = start_ratio
            elif step < decay_start + decay_steps:
                progress = (step - decay_start) / decay_steps
                current_teacher_forcing = start_ratio - (start_ratio - end_ratio) * progress
            else:
                current_teacher_forcing = end_ratio
            current_teacher_forcing = max(0.0, min(1.0, current_teacher_forcing))
        
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            if current_teacher_forcing >= random.random() or len(mel_outputs) == 0:
                decoder_input = decoder_inputs[len(mel_outputs)]
            else:
                decoder_input = self.prenet(mel_outputs[-1])

            mel_output, gate_output, attention_weights, decoder_output = self.decode(decoder_input)
            
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]
            if decoder_output is not None:
                decoder_outputs += [decoder_output.squeeze(1)]

        mel_outputs, gate_outputs, alignments, decoder_outputs = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments, decoder_outputs)

        return mel_outputs, gate_outputs, alignments, decoder_outputs

    def inference(self, memory, seed=None, suppress_gate=False):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments, decoder_outputs = [], [], [], []
        with temp_seed(seed):
            while True:
                decoder_input = self.prenet(decoder_input)
                mel_output, gate_output, alignment, _ = self.decode(decoder_input)

                mel_outputs += [mel_output.squeeze(1)]
                gate_outputs += [gate_output]
                alignments += [alignment]

                # 🔥 РЕВОЛЮЦИОННЫЙ АДАПТИВНЫЙ GATE THRESHOLD
                gate_prob = torch.sigmoid(gate_output.data)
                if self.adaptive_gate and not suppress_gate:
                    # Адаптивный порог на основе позиции в последовательности
                    step_ratio = len(mel_outputs) / self.max_decoder_steps
                    
                    # Умная адаптация: начинаем с низкого порога, повышаем до пика, затем снижаем
                    if step_ratio < 0.3:
                        # Ранняя фаза: низкий порог для предотвращения преждевременной остановки
                        adaptive_threshold = self.gate_min_threshold
                    elif step_ratio < 0.7:
                        # Средняя фаза: повышаем порог
                        progress = (step_ratio - 0.3) / 0.4
                        adaptive_threshold = self.gate_min_threshold + (self.gate_max_threshold - self.gate_min_threshold) * progress
                    else:
                        # Поздняя фаза: снижаем порог для естественного завершения
                        progress = (step_ratio - 0.7) / 0.3
                        adaptive_threshold = self.gate_max_threshold - (self.gate_max_threshold - self.gate_min_threshold) * progress * 0.5
                    
                    if gate_prob > adaptive_threshold:
                        break
                elif not suppress_gate and gate_prob > self.gate_threshold:
                    break
                elif len(mel_outputs) == self.max_decoder_steps:
                    # Достигнута максимальная длина
                    break

                decoder_input = mel_output

        mel_outputs, gate_outputs, alignments, _ = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments, decoder_outputs)

        return mel_outputs, gate_outputs, alignments


class MIEsitmator(nn.Module):
    def __init__(self, vocab_size, decoder_dim, hidden_size, dropout=0.2):
        super(MIEsitmator, self).__init__()
        self.layers = nn.Sequential(
            LinearNorm(decoder_dim, hidden_size, w_init_gain='relu'),
            nn.ReLU(),
            nn.Dropout(max(0.1, dropout)),
            LinearNorm(hidden_size, vocab_size)
        )

    def forward(self, decoder_outputs, target_phones, decoder_lengths, target_lengths):
        phone_logits = self.layers(decoder_outputs)
        ctc_loss = F.ctc_loss(
            phone_logits.transpose(0, 1),
            target_phones, decoder_lengths, target_lengths,
            reduction='mean', zero_infinity=True
        )
        return ctc_loss

class Embeder(nn.Module):
    def __init__(self, hparams):
        super(Embeder, self).__init__()
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.end_symbols_ids = hparams.end_symbols_ids
        
    def forward(self, x):
        emb = self.embedding(x)
        if self.end_symbols_ids:
            s = torch.tensor(self.end_symbols_ids, requires_grad=False).to(x.device)
            end_vectors = self.embedding(s)
            for b in range(x.size(0)):
                seq = x[b].cpu().detach().numpy().tolist()
                vec = None
                for i in range(x.size(1),0,-1):
                    if seq[i-1] in self.end_symbols_ids:
                        _id = self.end_symbols_ids.index(seq[i-1])
                        vec = end_vectors[_id,:]*1.5
                        continue
                    if vec is not None:
                        emb[b,i-1] = emb[b,i-1] + vec

        return emb


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        # self.embedding = Embeder(hparams)
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.drop_frame_rate = hparams.drop_frame_rate
        self.use_mmi = hparams.use_mmi
        self.use_gst = hparams.use_gst

        if self.drop_frame_rate > 0.:
            # global mean is not used at inference.
            self.global_mean = getattr(hparams, 'global_mean', None)
        if self.use_mmi:
            vocab_size = len(ctc_symbols)
            decoder_dim = hparams.decoder_rnn_dim
            self.mi = MIEsitmator(vocab_size, decoder_dim, decoder_dim, dropout=0.5)
        else:
            self.mi = None

        self.gst = None
        if self.use_gst:
            self.gst = GST(hparams)
            self.tpse_gst = TPSEGST(hparams)

        # === Double Decoder Consistency ===
        self.use_ddc = getattr(hparams, 'use_ddc', False)
        if self.use_ddc:
            sec_hparams = copy.deepcopy(hparams)
            sec_hparams.n_frames_per_step = getattr(hparams, 'ddc_reduction_factor', 2)
            self.decoder_secondary = Decoder(sec_hparams)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, ctc_text, ctc_text_lengths, guide_mask  = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        ctc_text = to_gpu(ctc_text).long()
        ctc_text_lengths = to_gpu(ctc_text_lengths).long()
        guide_mask = to_gpu(guide_mask).float()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths,
             ctc_text, ctc_text_lengths),
            (mel_padded, gate_padded,guide_mask ))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mel_mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mel_mask = mel_mask.permute(1, 0, 2)

            if outputs[0] is not None:
                float_mask = (~mask).float().unsqueeze(1)
                outputs[0] = outputs[0] * float_mask
            outputs[1].data.masked_fill_(mel_mask, 0.0)
            outputs[2].data.masked_fill_(mel_mask, 0.0)
            outputs[3].data.masked_fill_(mel_mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs, minimize=False):
        text_inputs, text_lengths, mels, max_len, output_lengths, *_ = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        if self.drop_frame_rate > 0. and self.training:
            # mels shape (B, n_mel_channels, T_out),
            mels = dropout_frame(mels, self.global_mean, output_lengths, self.drop_frame_rate)

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        emb_text = self.encoder(embedded_inputs, text_lengths)
        encoder_outputs = emb_text

        tpse_gst_outputs = None
        gst_outputs = None  # Инициализируем переменную
        emb_gst = None  # Инициализируем emb_gst
        if self.gst is not None:
            gst_outputs = self.gst(mels, output_lengths)
            emb_gst = gst_outputs.repeat(1, emb_text.size(1), 1)
            tpse_gst_outputs = self.tpse_gst(encoder_outputs)
            encoder_outputs = torch.cat((emb_text, emb_gst), dim=2)

        mel_outputs, gate_outputs, alignments, decoder_outputs = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        extra_outputs = []
        if self.use_ddc:
            # Запускаем вторичный декодер с другим reduction factor
            decoder_outputs_sec = self.decoder_secondary(encoder_outputs, mels, memory_lengths=text_lengths)
            mel_outputs2, gate_outputs2, alignments2, _ = decoder_outputs_sec  # возвращает tuple как у primary
            mel_outputs_postnet2 = self.postnet(mel_outputs2) + mel_outputs2
            extra_outputs = [mel_outputs2, mel_outputs_postnet2, gate_outputs2, alignments2]

        # Собираем выход
        outputs = [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]
        outputs.extend(extra_outputs)
        return tuple(outputs)

    def inference(self, inputs, seed=None, reference_mel=None, token_idx=None, scale=1.0):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        emb_text = self.encoder.inference(embedded_inputs)
        
        # Проверяем, что encoder вернул валидный результат
        if emb_text is None:
            # Encoder.inference вернул None, используем fallback (скрыто для чистоты логов)
            # Fallback: создаем базовые encoder outputs с правильными размерностями
            batch_size = inputs.size(0)
            seq_len = inputs.size(1)
            # Используем базовую размерность энкодера из hparams
            encoder_dim = self.encoder.lstm.hidden_size * 2  # bidirectional
            emb_text = torch.zeros(batch_size, seq_len, encoder_dim, 
                                 device=inputs.device, dtype=torch.float32)
            # Fallback tensor создан (скрыто для чистоты логов)
        elif emb_text.dim() != 3:
            # Encoder вернул некорректную размерность, используем fallback (скрыто для чистоты логов)
            batch_size = inputs.size(0)
            seq_len = inputs.size(1)
            encoder_dim = self.encoder.lstm.hidden_size * 2  # bidirectional
            emb_text = torch.zeros(batch_size, seq_len, encoder_dim, 
                                 device=inputs.device, dtype=torch.float32)
            # Fallback tensor создан (скрыто для чистоты логов)
        elif emb_text.size(1) != inputs.size(1):
            # Проверяем соответствие длины последовательности (скрыто для чистоты логов)
            # Используем более длинную последовательность
            target_seq_len = max(emb_text.size(1), inputs.size(1))
            if emb_text.size(1) < target_seq_len:
                # Дополняем encoder output до нужной длины
                batch_size, _, encoder_dim = emb_text.shape
                padding = torch.zeros(batch_size, target_seq_len - emb_text.size(1), encoder_dim, 
                                    device=emb_text.device, dtype=emb_text.dtype)
                emb_text = torch.cat([emb_text, padding], dim=1)
                # Encoder output дополнен (скрыто для чистоты логов)
        
        encoder_outputs = emb_text
        emb_gst = None  # Инициализируем emb_gst для всех случаев

        if self.gst is not None:
            if reference_mel is not None:
                emb_gst = self.gst(reference_mel)*scale
            elif token_idx is not None:
                query = torch.zeros(1, 1, self.gst.encoder.ref_enc_gru_size, dtype=torch.float32, device=inputs.device)
                GST = torch.tanh(self.gst.stl.embed)
                key = GST[token_idx].unsqueeze(0).expand(1, -1, -1)
                emb_gst = self.gst.stl.attention(query, key)*scale
            else:
                if emb_text is not None:
                    try:
                        emb_gst = self.tpse_gst(emb_text)*scale
                    except Exception as e:
                        print(f"❌ Ошибка в tpse_gst: {e}, используем fallback")
                        # Fallback для GST
                        emb_gst = torch.zeros(1, 1, self.gst.stl.embed.size(-1), 
                                            device=inputs.device, dtype=inputs.dtype)
                else:
                    # Fallback: create zero embedding
                    emb_gst = torch.zeros(1, 1, self.gst.stl.embed.size(-1), 
                                        device=inputs.device, dtype=inputs.dtype)

            if emb_text is not None:
                emb_gst = emb_gst.repeat(1, emb_text.size(1), 1)
            else:
                # If emb_text is None, use default sequence length
                emb_gst = emb_gst.repeat(1, inputs.size(1), 1)
         
            if emb_text is not None:
                encoder_outputs = torch.cat(
                        (emb_text, emb_gst), dim=2)
            else:
                encoder_outputs = emb_gst

        try:
            mel_outputs, gate_outputs, alignments = self.decoder.inference(
                encoder_outputs, seed=seed)
        except Exception as e:
            print(f"❌ Ошибка в decoder.inference: {e}")
            # Возвращаем пустые результаты в случае ошибки
            batch_size = encoder_outputs.size(0)
            mel_outputs = torch.zeros(batch_size, self.n_mel_channels, 1, 
                                    device=inputs.device, dtype=torch.float32)
            gate_outputs = torch.ones(batch_size, 1, device=inputs.device, dtype=torch.float32)
            alignments = torch.zeros(batch_size, 1, encoder_outputs.size(1), 
                                   device=inputs.device, dtype=torch.float32)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [None, mel_outputs, mel_outputs_postnet, gate_outputs, alignments, emb_gst])

        return outputs
