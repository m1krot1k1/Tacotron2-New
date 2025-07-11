# Based on mellotron https://github.com/NVIDIA/mellotron/blob/master/modules.py

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, hp):

        super().__init__()
        K = len(hp.ref_enc_filters)
        filters = [1] + hp.ref_enc_filters

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=hp.ref_enc_filters[i])
             for i in range(K)])

        out_channels = self.calculate_channels(hp.n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
                          hidden_size=hp.ref_enc_gru_size,
                          batch_first=True)
        self.n_mel_channels = hp.n_mel_channels
        self.ref_enc_gru_size = hp.ref_enc_gru_size

    def forward(self, inputs, input_lengths=None):
        out = inputs.view(inputs.size(0), 1, -1, self.n_mel_channels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)


        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]
        if input_lengths is not None:
            input_lengths = torch.ceil(input_lengths.float() / 2 ** len(self.convs))
            input_lengths = input_lengths.cpu().numpy().astype(int)            
            out = nn.utils.rnn.pack_padded_sequence(
                        out, input_lengths, batch_first=True, enforce_sorted=False)

        self.gru.flatten_parameters()
        _, out = self.gru(out)
        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    '''
    inputs --- [N, token_embedding_size//2]
    '''
    def __init__(self, hp):
        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(hp.token_num, hp.token_embedding_size // hp.num_heads))
        d_q = hp.ref_enc_gru_size
        d_k = hp.token_embedding_size // hp.num_heads
        self.attention = MultiHeadAttention(
            query_dim=d_q, key_dim=d_k, num_units=hp.token_embedding_size,
            num_heads=hp.num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, token_embedding_size // num_heads]
        style_embed = self.attention(query, keys)

        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out


class GST(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.encoder = ReferenceEncoder(hp)
        self.stl = STL(hp)

    def forward(self, inputs, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)
        style_embed = self.stl(enc_out)

        return style_embed

# TODO: poprobovat otrubit etot modul
class TPSEGST(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.dim = hparams.token_embedding_size
        self.gru = torch.nn.GRU(
            input_size=hparams.token_embedding_size,
            hidden_size=hparams.token_embedding_size//2,
            batch_first=True,
            bidirectional=True
        )
        self.inp = torch.nn.Sequential(
            nn.Conv1d(hparams.encoder_embedding_dim, hparams.token_embedding_size, 3, padding=1, bias=False),
            nn.BatchNorm1d( hparams.token_embedding_size),
            nn.ReLU()
        )

        self.linear =  torch.nn.Linear(hparams.token_embedding_size, hparams.token_embedding_size)
        
        # torch.nn.Sequential(
        #     torch.nn.Linear(hparams.token_embedding_size, hparams.encoder_embedding_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hparams.encoder_embedding_dim//2,hparams.token_embedding_size),
        # )

    def forward(self, x):
        # Detaching from main graph to not send gradient to the GST layer
        if x is None:
            return None
        
        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Сохраняем batch dimension
        batch_size = x.size(0)
        x = x.contiguous().detach()
        x = x.transpose(1, 2)
        x = self.inp(x)
        x = x.transpose(1, 2)
        self.gru.flatten_parameters()
        _, y = self.gru(x)
        
        # 🔥 ИСПРАВЛЕНИЕ: Правильная обработка размерностей с сохранением batch dimension
        # y имеет размерность [num_layers*num_directions, batch, hidden_size]
        # Для bidirectional GRU: [2, batch, hidden_size//2]
        y = y.transpose(0, 1)  # [batch, num_layers*num_directions, hidden_size//2]
        y = y.contiguous().view(batch_size, -1)  # [batch, hidden_size]
        y = y.unsqueeze(1)  # [batch, 1, hidden_size]
        y = torch.tanh(self.linear(y))
        
        # 🔥 ДОПОЛНИТЕЛЬНАЯ СТАБИЛИЗАЦИЯ: Клампинг для предотвращения экстремальных значений
        y = torch.clamp(y, min=-2.0, max=2.0)  # Ограничиваем диапазон
        
        # 🔥 МЯГКАЯ НОРМАЛИЗАЦИЯ для дополнительной стабильности
        y_norm = torch.norm(y, p=2, dim=-1, keepdim=True)
        y_norm_safe = torch.clamp(y_norm, min=1e-8, max=5.0)  # Предотвращаем слишком большие нормы
        y = y / y_norm_safe * torch.clamp(y_norm_safe, max=1.0)  # Нормализуем с ограничением

        return y