
# Решение 1: Исправление интерполяции attention векторов
import torch
import torch.nn.functional as F

class SafeDDCLoss(nn.Module):
    def __init__(self, interpolation_mode='linear'):
        super(SafeDDCLoss, self).__init__()
        self.interpolation_mode = interpolation_mode

    def forward(self, coarse_attention, fine_attention):
        """
        Безопасная DDC loss с правильной интерполяцией
        Args:
            coarse_attention: [batch_size, coarse_time_steps, encoder_dim]
            fine_attention: [batch_size, fine_time_steps, encoder_dim]
        """
        batch_size = coarse_attention.size(0)
        coarse_steps = coarse_attention.size(1)
        fine_steps = fine_attention.size(1)
        encoder_dim = coarse_attention.size(2)

        if coarse_steps == fine_steps:
            # Размеры уже совпадают
            return F.mse_loss(coarse_attention, fine_attention)

        # Интерполяция coarse attention до размера fine attention
        coarse_resized = F.interpolate(
            coarse_attention.transpose(1, 2),  # [batch, encoder_dim, coarse_steps]
            size=fine_steps,
            mode=self.interpolation_mode,
            align_corners=False
        ).transpose(1, 2)  # [batch, fine_steps, encoder_dim]

        return F.mse_loss(coarse_resized, fine_attention)
