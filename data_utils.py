import random
import os
import numpy as np
import torch
import torch.utils.data

import layers
from text import text_to_sequence, sequence_to_ctc_sequence

try:
    from utils import load_wav_to_torch, load_filepaths_and_text, guide_attention_fast
except ImportError:
    import importlib.util
    import sys
    import os
    utils_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils.py')
    spec = importlib.util.spec_from_file_location('utils', utils_path)
    utils_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_mod)
    load_wav_to_torch = utils_mod.load_wav_to_torch
    load_filepaths_and_text = utils_mod.load_filepaths_and_text
    guide_attention_fast = utils_mod.guide_attention_fast


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.ds_path = hparams.dataset_path
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.clean_non_existent()
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)
        # Dataset files скрыто для чистоты логов

    def clean_non_existent(self):
        """Удаляю несуществующие файлы из датасета"""
        out = []
        for el in self.audiopaths_and_text:
            # Файлы в CSV уже содержат полные пути
            audio_path = el[0]
            if os.path.exists(audio_path):
                out.append(el)
            else:
                print(f"Файл не найден: {audio_path}")
        self.audiopaths_and_text = out

    def get_mel_text_pair(self, audiopath_and_text):
        # Получаю путь к файлу и текст
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1].strip()
        text, ctc_text = self.get_text(text)
        
        # Ограничиваем длину текста до 200 символов (маска 200x1000)
        if text.size(0) > 200:
            # Берём первые 200 символов, чтобы не выйти за пределы guide-маски
            text = text[:200]
            ctc_text = ctc_text[:200]
            # Text truncated warning скрыт для чистоты логов

        mel = self.get_mel(audiopath)  # Используем полный путь

        # Ограничиваем длину мел-спектрограммы до 1000 фреймов
        if mel.shape[-1] > 1000:
            mel = mel[:, :1000]
            # Mel truncated warning скрыт для чистоты логов

        guide_mask = torch.FloatTensor(guide_attention_fast(len(text), mel.shape[-1], 200, 1000))
        return (text, ctc_text, mel, guide_mask)

    def get_mel(self, filename):
        """Загружаю или вычисляю мел-спектрограмму"""
        if not os.path.exists(filename+'.npy'):
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
            np.save(filename+'.npy',melspec)
        else:
            melspec = torch.from_numpy(np.load(filename+'.npy'))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        sequence = text_to_sequence(text, self.text_cleaners)
        # Проверяем, что последовательность не пустая
        if len(sequence) == 0:
            # Пустой текст warning скрыт для чистоты логов
            sequence = [0]  # Добавляем минимальный символ
        text_norm = torch.IntTensor(sequence)
        ctc_text_norm = torch.IntTensor(sequence_to_ctc_sequence(sequence))
        return text_norm, ctc_text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # 🔥 ИСПРАВЛЕНИЕ: Ограничиваем максимальные размеры для стабильности
        MAX_TEXT_LEN = 200  # Максимальная длина текста
        MAX_MEL_LEN = 1000  # Максимальная длина mel
        
        # Right zero-pad all one-hot text sequences to max input length
        text_lengths = [min(len(x[0]), MAX_TEXT_LEN) for x in batch]  # Ограничиваем длину
        # Проверяем на нулевые длины и исправляем их
        text_lengths = [max(1, length) for length in text_lengths]  # Минимум 1
        
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor(text_lengths),
            dim=0, descending=True)
        max_input_len = min(input_lengths[0], MAX_TEXT_LEN)  # Ограничиваем

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            # 🔥 ИСПРАВЛЕНИЕ: Обрезаем текст если он длиннее максимума
            text_len = min(text.size(0), max_input_len, MAX_TEXT_LEN)
            text_padded[i, :text_len] = text[:text_len]

        max_ctc_txt_len = max([len(x[1]) for x in batch])
        ctc_text_paded = torch.LongTensor(len(batch), max_ctc_txt_len)
        ctc_text_paded .zero_()
        ctc_text_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            ctc_text = batch[ids_sorted_decreasing[i]][1]
            ctc_text_paded[i, :ctc_text.size(0)] = ctc_text
            ctc_text_lengths[i] = ctc_text.size(0)

        # 🔥 ИСПРАВЛЕНИЕ: Right zero-pad mel-spec с ограничением размера
        num_mels = batch[0][2].size(0)
        max_target_len = min(max([x[2].size(1) for x in batch]), MAX_MEL_LEN)  # Ограничиваем
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            max_target_len = min(max_target_len, MAX_MEL_LEN)  # Повторно ограничиваем после выравнивания
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][2]
            # 🔥 ИСПРАВЛЕНИЕ: Обрезаем mel если он длиннее максимума
            mel_len = min(mel.size(1), max_target_len, MAX_MEL_LEN)
            mel_padded[i, :, :mel_len] = mel[:, :mel_len]
            gate_padded[i, mel_len-1:] = 1
            output_lengths[i] = mel_len

        guide_padded = torch.FloatTensor(len(batch), 200, 1000)
        guide_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            guide = batch[ids_sorted_decreasing[i]][3]
            guide_padded[i, :, :] = guide

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, ctc_text_paded, ctc_text_lengths, guide_padded