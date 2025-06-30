from torch import nn
import torch
import numpy as np


class Tacotron2Loss(nn.Module):
    def __init__(self, hparams, iteration=0):
        super(Tacotron2Loss, self).__init__()
        self.hparams = hparams
        self.guide_decay = 0.99999
        self.scale = 40.0 * (self.guide_decay**iteration)
        print('Guide scale:',self.scale)
        self.guide_lowbound = 1.0
        self.criterion_attention = nn.L1Loss()

    def forward(self, model_output, targets):
        _, mel_out, mel_out_postnet, gate_out, alignments_out, tpse_gst_pred,gst_target = model_output
        mel_target, gate_target, guide_target = targets[0], targets[1], targets[2]

        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        guide_target = guide_target.transpose(2,1)
        _,w,h = alignments_out.shape
        guide_target = guide_target[:,:w,:h]

        
        gate_out = gate_out.view(-1, 1)
        emb_loss = torch.tensor(0)
        if tpse_gst_pred is not None:
            emb_loss = nn.L1Loss()(tpse_gst_pred, gst_target.detach())
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = 1.3 * nn.BCEWithLogitsLoss()(gate_out, gate_target)

        # loss_atten = torch.tensor(0)
        # if not self.hparams.no_dga:
        attention_masks = torch.ones_like(alignments_out)

        loss_atten = self.criterion_attention(
                guide_target * alignments_out * attention_masks,
                torch.zeros_like(alignments_out)) * self.scale
    
        self.scale *= self.guide_decay
        # self.scale = 100
        if self.scale < self.guide_lowbound:
            self.scale = self.guide_lowbound

        return mel_loss, gate_loss, loss_atten, emb_loss


class GuidedAttentionLoss(nn.Module):
    """
    Guided Attention Loss для улучшения монотонности attention alignment в Tacotron2.
    
    Основан на статье "Efficiently Trainable Text-to-Speech System Based on Deep Convolutional 
    Networks with Guided Attention" (https://arxiv.org/abs/1710.08969)
    """
    
    def __init__(self, sigma=0.4, alpha=1.0, reset_always=True):
        """
        Args:
            sigma (float): Стандартное отклонение для гауссовой функции направляющей матрицы
            alpha (float): Начальный вес guided attention loss
            reset_always (bool): Сбрасывать ли вес на каждом forward pass
        """
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.reset_always = reset_always
        self.guided_attn_masks = {}
        
    def _make_guided_attention_mask(self, ilen, olen):
        """Создание направляющей маски attention."""
        grid_x, grid_y = torch.meshgrid(
            torch.arange(olen).to(torch.float32),
            torch.arange(ilen).to(torch.float32),
            indexing='ij'
        )
        
        # Нормализация координат
        grid_x = grid_x / (olen - 1) if olen > 1 else grid_x
        grid_y = grid_y / (ilen - 1) if ilen > 1 else grid_y
        
        # Гауссова функция для создания диагональной направляющей маски
        return 1.0 - torch.exp(-((grid_x - grid_y) ** 2) / (2 * (self.sigma ** 2)))
    
    def forward(self, model_output, input_lengths=None, output_lengths=None):
        """
        Args:
            model_output: Выход модели, содержащий attention weights
            input_lengths: Длины входных последовательностей
            output_lengths: Длины выходных последовательностей
            
        Returns:
            guided_attn_loss: Guided attention loss
        """
        # Извлекаем attention weights из model_output
        if isinstance(model_output, (list, tuple)):
            # Tacotron2 возвращает кортеж: (mel_outputs, mel_outputs_postnet, gate_outputs, alignments)
            if len(model_output) >= 4:
                attention_weights = model_output[4]  # alignments
            else:
                # Если alignments нет, возвращаем нулевой loss
                return torch.tensor(0.0, device=model_output[0].device, requires_grad=True)
        else:
            attention_weights = model_output
            
        if attention_weights is None:
            return torch.tensor(0.0, requires_grad=True)
            
        batch_size, max_target_len, max_input_len = attention_weights.size()
        
        # Если длины не предоставлены, используем максимальные
        if input_lengths is None:
            input_lengths = [max_input_len] * batch_size
        if output_lengths is None:
            output_lengths = [max_target_len] * batch_size
            
        guided_attn_loss = 0.0
        
        for b in range(batch_size):
            ilen = input_lengths[b]
            olen = output_lengths[b]
            
            # Создаем ключ для кэширования маски
            mask_key = (ilen, olen)
            
            if mask_key not in self.guided_attn_masks:
                # Создаем направляющую маску
                mask = self._make_guided_attention_mask(ilen, olen)
                self.guided_attn_masks[mask_key] = mask
            else:
                mask = self.guided_attn_masks[mask_key]
                
            # Перемещаем маску на нужное устройство
            mask = mask.to(attention_weights.device)
            
            # Обрезаем attention weights и маску до реальных длин
            attn = attention_weights[b, :olen, :ilen]
            mask = mask[:olen, :ilen]
            
            # Вычисляем guided attention loss для данного примера
            guided_attn_loss += torch.mean(attn * mask)
            
        # Усредняем по батчу
        guided_attn_loss /= batch_size
        
        return self.alpha * guided_attn_loss
    
    def get_weight(self):
        """Возвращает текущий вес guided attention loss."""
        return self.alpha
    
    def set_weight(self, alpha):
        """Устанавливает новый вес guided attention loss."""
        self.alpha = alpha
        
    def decay_weight(self, decay_factor=0.99):
        """Уменьшает вес guided attention loss."""
        self.alpha *= decay_factor
        
    def reset_weight(self, alpha=1.0):
        """Сбрасывает вес guided attention loss."""
        self.alpha = alpha
