import torch
import torch.nn.functional as F

class DynamicPaddingCollator:
    """
    Динамический collator для батчей переменной длины.
    Выравнивает тензоры только до максимальной длины в текущем батче.
    Совместим с форматом данных TextMelLoader.
    """
    def __init__(self, pad_value=0.0, n_frames_per_step=1):
        self.pad_value = pad_value
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, ctc_text, mel_normalized, guide_mask]
        """
        # Right zero-pad all one-hot text sequences to max input length
        text_lengths = [len(x[0]) for x in batch]
        # Проверяем на нулевые длины и исправляем их
        text_lengths = [max(1, length) for length in text_lengths]  # Минимум 1
        
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor(text_lengths),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        max_ctc_txt_len = max([len(x[1]) for x in batch])
        ctc_text_paded = torch.LongTensor(len(batch), max_ctc_txt_len)
        ctc_text_paded.zero_()
        ctc_text_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            ctc_text = batch[ids_sorted_decreasing[i]][1]
            ctc_text_paded[i, :ctc_text.size(0)] = ctc_text
            ctc_text_lengths[i] = ctc_text.size(0)

        # Right zero-pad mel-spec
        num_mels = batch[0][2].size(0)
        max_target_len = max([x[2].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][2]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        guide_padded = torch.FloatTensor(len(batch), 200, 1000)
        guide_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            guide = batch[ids_sorted_decreasing[i]][3]
            guide_padded[i, :, :] = guide

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, ctc_text_paded, ctc_text_lengths, guide_padded 