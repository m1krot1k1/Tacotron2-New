import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryEfficientDDC(nn.Module):
    """
    DDC loss с оптимизацией памяти (chunked вычисления для длинных последовательностей).
    """
    def __init__(self, max_sequence_length=1000, chunk_size=100):
        super(MemoryEfficientDDC, self).__init__()
        self.max_seq_len = max_sequence_length
        self.chunk_size = chunk_size

    def chunked_ddc_loss(self, coarse_attention, fine_attention):
        batch_size = coarse_attention.size(0)
        seq_len = min(coarse_attention.size(1), fine_attention.size(1))
        total_loss = 0.0
        num_chunks = 0
        
        for start_idx in range(0, seq_len, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, seq_len)
            coarse_chunk = coarse_attention[:, start_idx:end_idx, :]
            fine_chunk = fine_attention[:, start_idx:end_idx, :]
            chunk_loss = F.mse_loss(coarse_chunk, fine_chunk)
            total_loss += chunk_loss
            num_chunks += 1
        
        if num_chunks > 0:
            avg_loss = total_loss / num_chunks
        else:
            # Если нет чанков, создаем нулевой loss с правильными свойствами
            avg_loss = torch.tensor(0.0, requires_grad=True, device=coarse_attention.device)
            
        return avg_loss

    def forward(self, coarse_attention, fine_attention):
        coarse_len = coarse_attention.size(1)
        fine_len = fine_attention.size(1)
        
        # Если последовательности слишком длинные, используем chunked computation
        if max(coarse_len, fine_len) > self.max_seq_len:
            return self.chunked_ddc_loss(coarse_attention, fine_attention)
        
        # Обычное вычисление для коротких последовательностей
        min_len = min(coarse_len, fine_len)
        coarse_trimmed = coarse_attention[:, :min_len, :]
        fine_trimmed = fine_attention[:, :min_len, :]
        
        # Вычисляем loss и убеждаемся, что он требует градиенты
        loss = F.mse_loss(coarse_trimmed, fine_trimmed)
        if not loss.requires_grad:
            loss.requires_grad_(True)
            
        return loss 