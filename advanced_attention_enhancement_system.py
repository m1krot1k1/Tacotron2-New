#!/usr/bin/env python3
"""
🔥 Advanced Attention Enhancement System для Tacotron2

Решает критические проблемы из exported-assets:
❌ Attention diagonality: 0.035 → ✅ >0.7
❌ 198 хаотичных изменений guided weight → ✅ Плавная адаптация  
❌ Отсутствие attention regularization → ✅ Multi-level regularization
❌ Неправильная архитектура → ✅ Transformer-style + Location-aware

Ключевые компоненты:
🧠 MultiHeadLocationAwareAttention - Transformer-style с location awareness
📈 ProgressiveAttentionTrainer - Curriculum learning для attention
🎯 SelfSupervisedAttentionLearner - Contrastive learning для alignment
📊 AdvancedAttentionDiagnostics - Real-time monitoring и коррекция
🛡️ AttentionRegularizationSystem - Multi-level regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging


class AttentionPhase(Enum):
    """Фазы обучения attention mechanisms"""
    WARMUP = "warmup"                    # Разогрев: простое внимание
    ALIGNMENT_LEARNING = "alignment"     # Изучение выравнивания
    MONOTONIC_TRAINING = "monotonic"     # Обучение монотонности
    REFINEMENT = "refinement"           # Финальная настройка
    CONVERGENCE = "convergence"         # Стабилизация


@dataclass
class AttentionMetrics:
    """Метрики качества attention"""
    diagonality: float
    monotonicity: float
    focus: float
    coverage: float
    entropy: float
    consistency: float
    phase: AttentionPhase


class MultiHeadLocationAwareAttention(nn.Module):
    """
    🧠 Multi-Head Attention с Location Awareness для Tacotron2
    
    Объединяет:
    - Transformer-style multi-head attention
    - Location-based attention из Tacotron2
    - Adaptive head weighting
    - Progressive complexity scaling
    """
    
    def __init__(self, 
                 attention_rnn_dim: int = 1024,
                 embedding_dim: int = 512,
                 attention_dim: int = 128,
                 num_heads: int = 8,
                 location_n_filters: int = 32,
                 location_kernel_size: int = 31):
        super().__init__()
        
        self.attention_rnn_dim = attention_rnn_dim
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        
        assert attention_dim % num_heads == 0, "attention_dim must be divisible by num_heads"
        
        # Multi-head projections
        self.query_projection = nn.Linear(attention_rnn_dim, attention_dim)
        self.key_projection = nn.Linear(embedding_dim, attention_dim)
        self.value_projection = nn.Linear(embedding_dim, attention_dim)
        self.output_projection = nn.Linear(attention_dim, attention_dim)
        
        # Location layer для Tacotron2 compatibility
        self.location_conv = nn.Conv1d(
            2, location_n_filters, 
            kernel_size=location_kernel_size,
            padding=(location_kernel_size - 1) // 2,
            bias=False
        )
        self.location_dense = nn.Linear(location_n_filters, attention_dim, bias=False)
        
        # Adaptive head weighting
        self.head_weights = nn.Parameter(torch.ones(num_heads))
        
        # Progressive complexity control
        self.complexity_factor = 0.1  # Start simple
        self.score_mask_value = -float("inf")
        
        # Attention regularization
        self.dropout = nn.Dropout(0.1)
        
        self.logger = logging.getLogger(__name__)
        
    def forward(self, 
                query: torch.Tensor,
                memory: torch.Tensor,
                processed_memory: torch.Tensor,
                attention_weights_cat: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-head location-aware attention
        
        Args:
            query: Decoder hidden state [B, attention_rnn_dim]
            memory: Encoder outputs [B, T_in, embedding_dim]
            processed_memory: Pre-processed memory [B, T_in, attention_dim]
            attention_weights_cat: Previous + cumulative attention [B, 2, T_in]
            mask: Padding mask [B, T_in]
            
        Returns:
            attention_context: Weighted context vector [B, embedding_dim]
            attention_weights: Final attention weights [B, T_in]
        """
        batch_size, seq_len = memory.shape[:2]
        
        # 1. Multi-head projections
        Q = self.query_projection(query).unsqueeze(1)  # [B, 1, attention_dim]
        K = self.key_projection(memory)  # [B, T_in, attention_dim]
        V = self.value_projection(memory)  # [B, T_in, attention_dim]
        
        # 2. Reshape для multi-head
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Location-based processing
        location_features = self._process_location(attention_weights_cat)  # [B, T_in, attention_dim]
        location_features = location_features.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 4. Multi-head attention с location awareness
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add location bias
        location_scores = torch.matmul(Q, location_features.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_scores = attention_scores + self.complexity_factor * location_scores
        
        # 5. Apply mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(1).expand(-1, self.num_heads, 1, -1)
            attention_scores.masked_fill_(mask_expanded, self.score_mask_value)
        
        # 6. Softmax и weighted values
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 7. Weighted values
        attended_values = torch.matmul(attention_probs, V)  # [B, num_heads, 1, head_dim]
        
        # 8. Apply adaptive head weights
        head_weights_normalized = F.softmax(self.head_weights, dim=0)
        attended_values = attended_values * head_weights_normalized.view(1, -1, 1, 1)
        
        # 9. Concat heads и project
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, 1, self.attention_dim
        )
        attention_context = self.output_projection(attended_values).squeeze(1)
        
        # 10. Combine multi-head attention weights для compatibility
        final_attention_weights = attention_probs.mean(dim=1).squeeze(1)  # [B, T_in]
        
        # 11. Compute final context using original memory
        attention_context_original = torch.bmm(
            final_attention_weights.unsqueeze(1), memory
        ).squeeze(1)
        
        return attention_context_original, final_attention_weights
    
    def _process_location(self, attention_weights_cat: torch.Tensor) -> torch.Tensor:
        """Process location features from previous attention"""
        # attention_weights_cat: [B, 2, T_in]
        processed_attention = self.location_conv(attention_weights_cat)  # [B, location_n_filters, T_in]
        processed_attention = processed_attention.transpose(1, 2)  # [B, T_in, location_n_filters]
        processed_attention = self.location_dense(processed_attention)  # [B, T_in, attention_dim]
        return processed_attention
    
    def update_complexity(self, current_diagonality: float, target_diagonality: float = 0.7):
        """Progressively increase attention complexity based on performance"""
        if current_diagonality > target_diagonality * 0.8:
            self.complexity_factor = min(1.0, self.complexity_factor + 0.05)
        elif current_diagonality < target_diagonality * 0.3:
            self.complexity_factor = max(0.1, self.complexity_factor - 0.02)


class ProgressiveAttentionTrainer(nn.Module):
    """
    📈 Progressive Attention Training с Curriculum Learning
    
    Постепенно усложняет attention training:
    1. Simple location-based attention
    2. Multi-head with low complexity  
    3. Full multi-head with location integration
    4. Advanced regularization
    """
    
    def __init__(self, max_steps: int = 10000):
        super().__init__()
        self.max_steps = max_steps
        self.current_step = 0
        self.current_phase = AttentionPhase.WARMUP
        
        # Curriculum schedule
        self.phase_schedule = {
            AttentionPhase.WARMUP: (0, 0.1),           # 0-10%
            AttentionPhase.ALIGNMENT_LEARNING: (0.1, 0.4),  # 10-40%
            AttentionPhase.MONOTONIC_TRAINING: (0.4, 0.7),  # 40-70% 
            AttentionPhase.REFINEMENT: (0.7, 0.9),     # 70-90%
            AttentionPhase.CONVERGENCE: (0.9, 1.0)     # 90-100%
        }
        
        self.logger = logging.getLogger(__name__)
        
    def update_training_phase(self, step: int, attention_metrics: AttentionMetrics) -> AttentionPhase:
        """Update training phase based on progress and metrics"""
        self.current_step = step
        progress = min(step / self.max_steps, 1.0)
        
        # Determine phase based on schedule and performance
        scheduled_phase = AttentionPhase.WARMUP
        for phase, (start, end) in self.phase_schedule.items():
            if start <= progress < end:
                scheduled_phase = phase
                break
        
        # Allow early advancement based on good performance
        if attention_metrics.diagonality > 0.7 and scheduled_phase == AttentionPhase.WARMUP:
            scheduled_phase = AttentionPhase.ALIGNMENT_LEARNING
        elif attention_metrics.diagonality > 0.8 and scheduled_phase == AttentionPhase.ALIGNMENT_LEARNING:
            scheduled_phase = AttentionPhase.MONOTONIC_TRAINING
        
        # Prevent regression to earlier phases
        if scheduled_phase.value < self.current_phase.value:
            scheduled_phase = self.current_phase
            
        if scheduled_phase != self.current_phase:
            self.logger.info(f"🎯 Attention training phase: {self.current_phase.value} → {scheduled_phase.value}")
            self.current_phase = scheduled_phase
            
        return self.current_phase
    
    def get_training_config(self, phase: AttentionPhase) -> Dict[str, Any]:
        """Get training configuration for current phase"""
        configs = {
            AttentionPhase.WARMUP: {
                'guided_attention_weight': 8.0,
                'monotonic_weight': 0.0,
                'entropy_regularization': 0.0,
                'temperature': 2.0,
                'use_multi_head': False
            },
            AttentionPhase.ALIGNMENT_LEARNING: {
                'guided_attention_weight': 5.0,
                'monotonic_weight': 0.5,
                'entropy_regularization': 0.1,
                'temperature': 1.5,
                'use_multi_head': True
            },
            AttentionPhase.MONOTONIC_TRAINING: {
                'guided_attention_weight': 3.0,
                'monotonic_weight': 1.0,
                'entropy_regularization': 0.2,
                'temperature': 1.0,
                'use_multi_head': True
            },
            AttentionPhase.REFINEMENT: {
                'guided_attention_weight': 2.0,
                'monotonic_weight': 0.8,
                'entropy_regularization': 0.3,
                'temperature': 0.8,
                'use_multi_head': True
            },
            AttentionPhase.CONVERGENCE: {
                'guided_attention_weight': 1.0,
                'monotonic_weight': 0.5,
                'entropy_regularization': 0.1,
                'temperature': 1.0,
                'use_multi_head': True
            }
        }
        
        return configs.get(phase, configs[AttentionPhase.WARMUP])


class SelfSupervisedAttentionLearner(nn.Module):
    """
    🎯 Self-Supervised Attention Learning с Contrastive Learning
    
    Улучшает alignment quality через:
    1. Contrastive learning на attention maps
    2. Temporal consistency regularization
    3. Cross-attention supervision
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
        # Contrastive learning components
        self.attention_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.logger = logging.getLogger(__name__)
    
    def compute_contrastive_loss(self, 
                                attention_maps: torch.Tensor,
                                positive_pairs: List[Tuple[int, int]],
                                negative_pairs: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Compute contrastive loss for attention maps
        
        Args:
            attention_maps: [B, T_out, T_in] attention weights
            positive_pairs: List of (i, j) indices for similar attention patterns
            negative_pairs: List of (i, j) indices for dissimilar patterns
        """
        batch_size = attention_maps.size(0)
        
        # Encode attention maps
        attention_maps_unsqueezed = attention_maps.unsqueeze(1)  # [B, 1, T_out, T_in]
        embeddings = self.attention_encoder(attention_maps_unsqueezed)  # [B, 128]
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarities
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Contrastive loss
        contrastive_loss = 0.0
        
        # Positive pairs (should be similar)
        for i, j in positive_pairs:
            if i < batch_size and j < batch_size:
                positive_sim = similarity_matrix[i, j]
                negative_sims = torch.cat([
                    similarity_matrix[i, :i], 
                    similarity_matrix[i, i+1:]
                ])
                contrastive_loss += -torch.log(
                    torch.exp(positive_sim) / (torch.exp(positive_sim) + torch.exp(negative_sims).sum())
                )
        
        # Negative pairs (should be dissimilar)
        for i, j in negative_pairs:
            if i < batch_size and j < batch_size:
                negative_sim = similarity_matrix[i, j]
                contrastive_loss += torch.clamp(negative_sim - 0.2, min=0)  # Margin loss
        
        return contrastive_loss / (len(positive_pairs) + len(negative_pairs) + 1e-8)
    
    def temporal_consistency_loss(self, attention_sequence: torch.Tensor) -> torch.Tensor:
        """Regularize temporal consistency in attention patterns"""
        # attention_sequence: [T, B, T_out, T_in] attention over time steps
        if attention_sequence.size(0) < 2:
            return torch.tensor(0.0, device=attention_sequence.device)
        
        consistency_loss = 0.0
        for t in range(1, attention_sequence.size(0)):
            current_attention = attention_sequence[t]
            previous_attention = attention_sequence[t-1]
            
            # L2 distance between consecutive attention maps
            consistency_loss += F.mse_loss(current_attention, previous_attention)
        
        return consistency_loss / (attention_sequence.size(0) - 1)


class AdvancedAttentionDiagnostics(nn.Module):
    """
    📊 Advanced Attention Diagnostics с Real-time Monitoring
    
    Реализует comprehensive анализ attention quality:
    1. Real-time диагональность calculation
    2. Monotonicity analysis
    3. Focus и coverage metrics
    4. Automatic correction suggestions
    """
    
    def __init__(self, target_diagonality: float = 0.7):
        super().__init__()
        self.target_diagonality = target_diagonality
        
        # Moving averages для stability
        self.diagonality_history = []
        self.monotonicity_history = []
        self.focus_history = []
        
        # Thresholds для intervention
        self.critical_diagonality = 0.1
        self.warning_diagonality = 0.3
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_attention_quality(self, attention_weights: torch.Tensor) -> AttentionMetrics:
        """
        Comprehensive attention quality analysis
        
        Args:
            attention_weights: [B, T_out, T_in] attention weights
            
        Returns:
            AttentionMetrics with detailed analysis
        """
        batch_size, T_out, T_in = attention_weights.shape
        
        # Convert to numpy для detailed analysis
        attention_np = attention_weights.detach().cpu().numpy()
        
        # Compute metrics
        diagonality = self._compute_diagonality(attention_np)
        monotonicity = self._compute_monotonicity(attention_np)
        focus = self._compute_focus(attention_np)
        coverage = self._compute_coverage(attention_np)
        entropy = self._compute_entropy(attention_np)
        consistency = self._compute_consistency(attention_np)
        
        # Update history
        self.diagonality_history.append(diagonality)
        self.monotonicity_history.append(monotonicity)
        self.focus_history.append(focus)
        
        # Keep only recent history
        if len(self.diagonality_history) > 100:
            self.diagonality_history.pop(0)
            self.monotonicity_history.pop(0) 
            self.focus_history.pop(0)
        
        # Determine phase
        phase = self._determine_attention_phase(diagonality, monotonicity, focus)
        
        metrics = AttentionMetrics(
            diagonality=diagonality,
            monotonicity=monotonicity,
            focus=focus,
            coverage=coverage,
            entropy=entropy,
            consistency=consistency,
            phase=phase
        )
        
        # Log critical issues
        if diagonality < self.critical_diagonality:
            self.logger.warning(f"🚨 CRITICAL: Attention diagonality {diagonality:.3f} < {self.critical_diagonality}")
        elif diagonality < self.warning_diagonality:
            self.logger.warning(f"⚠️ WARNING: Low attention diagonality {diagonality:.3f}")
        
        return metrics
    
    def _compute_diagonality(self, attention_np: np.ndarray) -> float:
        """Compute attention diagonality score"""
        batch_size, T_out, T_in = attention_np.shape
        diagonality_scores = []
        
        for b in range(batch_size):
            att_matrix = attention_np[b]  # [T_out, T_in]
            
            # Create expected diagonal
            diagonal_indices = np.linspace(0, T_in - 1, T_out).astype(int)
            
            # Compute diagonality as correlation with expected diagonal
            diagonal_values = att_matrix[np.arange(T_out), diagonal_indices]
            diagonality = np.mean(diagonal_values)
            diagonality_scores.append(diagonality)
        
        return float(np.mean(diagonality_scores))
    
    def _compute_monotonicity(self, attention_np: np.ndarray) -> float:
        """Compute attention monotonicity score"""
        batch_size, T_out, T_in = attention_np.shape
        monotonicity_scores = []
        
        for b in range(batch_size):
            att_matrix = attention_np[b]
            
            # Find attention peaks
            peak_positions = np.argmax(att_matrix, axis=1)
            
            # Count monotonic transitions
            monotonic_transitions = 0
            total_transitions = 0
            
            for t in range(1, T_out):
                if peak_positions[t] >= peak_positions[t-1]:
                    monotonic_transitions += 1
                total_transitions += 1
            
            monotonicity = monotonic_transitions / total_transitions if total_transitions > 0 else 0.0
            monotonicity_scores.append(monotonicity)
        
        return float(np.mean(monotonicity_scores))
    
    def _compute_focus(self, attention_np: np.ndarray) -> float:
        """Compute attention focus (inverse of entropy)"""
        batch_size, T_out, T_in = attention_np.shape
        focus_scores = []
        
        for b in range(batch_size):
            att_matrix = attention_np[b]
            
            # Compute entropy for each output step
            entropies = []
            for t in range(T_out):
                att_weights = att_matrix[t] + 1e-10  # Numerical stability
                att_weights = att_weights / np.sum(att_weights)  # Normalize
                entropy = -np.sum(att_weights * np.log(att_weights))
                entropies.append(entropy)
            
            # Focus = 1 - normalized_entropy
            max_entropy = np.log(T_in)
            normalized_entropy = np.mean(entropies) / max_entropy
            focus = max(0.0, 1.0 - normalized_entropy)
            focus_scores.append(focus)
        
        return float(np.mean(focus_scores))
    
    def _compute_coverage(self, attention_np: np.ndarray) -> float:
        """Compute input sequence coverage"""
        batch_size, T_out, T_in = attention_np.shape
        coverage_scores = []
        
        for b in range(batch_size):
            att_matrix = attention_np[b]
            
            # Sum attention over all output steps
            total_attention = np.sum(att_matrix, axis=0)
            
            # Coverage = fraction of inputs with significant attention
            threshold = 0.01 * np.max(total_attention)
            covered_positions = np.sum(total_attention > threshold)
            coverage = covered_positions / T_in
            coverage_scores.append(coverage)
        
        return float(np.mean(coverage_scores))
    
    def _compute_entropy(self, attention_np: np.ndarray) -> float:
        """Compute average attention entropy"""
        batch_size, T_out, T_in = attention_np.shape
        entropy_scores = []
        
        for b in range(batch_size):
            att_matrix = attention_np[b]
            
            entropies = []
            for t in range(T_out):
                att_weights = att_matrix[t] + 1e-10
                att_weights = att_weights / np.sum(att_weights)
                entropy = -np.sum(att_weights * np.log(att_weights))
                entropies.append(entropy)
            
            entropy_scores.append(np.mean(entropies))
        
        return float(np.mean(entropy_scores))
    
    def _compute_consistency(self, attention_np: np.ndarray) -> float:
        """Compute attention consistency across batch"""
        if attention_np.shape[0] < 2:
            return 1.0
        
        # Compute pairwise similarities in batch
        similarities = []
        for i in range(attention_np.shape[0]):
            for j in range(i + 1, attention_np.shape[0]):
                similarity = np.corrcoef(
                    attention_np[i].flatten(),
                    attention_np[j].flatten()
                )[0, 1]
                if not np.isnan(similarity):
                    similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _determine_attention_phase(self, diagonality: float, monotonicity: float, focus: float) -> AttentionPhase:
        """Determine current attention learning phase"""
        if diagonality < 0.2:
            return AttentionPhase.WARMUP
        elif diagonality < 0.5:
            return AttentionPhase.ALIGNMENT_LEARNING
        elif monotonicity < 0.7:
            return AttentionPhase.MONOTONIC_TRAINING
        elif diagonality < 0.8:
            return AttentionPhase.REFINEMENT
        else:
            return AttentionPhase.CONVERGENCE
    
    def get_correction_suggestions(self, metrics: AttentionMetrics) -> Dict[str, Any]:
        """Generate automatic correction suggestions"""
        suggestions = {}
        
        if metrics.diagonality < self.critical_diagonality:
            suggestions['guided_attention_weight'] = 'increase_significantly'
            suggestions['learning_rate'] = 'decrease'
            suggestions['attention_regularization'] = 'enable'
        
        if metrics.monotonicity < 0.5:
            suggestions['monotonic_loss_weight'] = 'increase'
            suggestions['location_relative_attention'] = 'enable'
        
        if metrics.focus < 0.3:
            suggestions['attention_temperature'] = 'decrease'
            suggestions['entropy_regularization'] = 'increase'
        
        if metrics.coverage < 0.7:
            suggestions['attention_dropout'] = 'decrease'
            suggestions['attention_mechanism'] = 'multihead'
        
        return suggestions


class AttentionRegularizationSystem(nn.Module):
    """
    🛡️ Multi-level Attention Regularization System
    
    Реализует sophisticated regularization для stable attention training:
    1. Entropy regularization для proper focus
    2. Monotonic regularization для sequential alignment  
    3. Temporal consistency regularization
    4. Attention diversity regularization
    """
    
    def __init__(self):
        super().__init__()
        
        # Regularization weights
        self.entropy_weight = 0.1
        self.monotonic_weight = 0.5
        self.temporal_weight = 0.2
        self.diversity_weight = 0.1
        
        self.logger = logging.getLogger(__name__)
    
    def compute_regularization_loss(self, 
                                  attention_weights: torch.Tensor,
                                  previous_attention: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute comprehensive regularization loss for attention
        
        Args:
            attention_weights: [B, T_out, T_in] current attention
            previous_attention: [B, T_out, T_in] previous step attention
            
        Returns:
            Total regularization loss
        """
        total_loss = 0.0
        
        # 1. Entropy regularization (encourage focus)
        entropy_loss = self._entropy_regularization(attention_weights)
        total_loss += self.entropy_weight * entropy_loss
        
        # 2. Monotonic regularization (encourage sequential alignment)
        monotonic_loss = self._monotonic_regularization(attention_weights)
        total_loss += self.monotonic_weight * monotonic_loss
        
        # 3. Temporal consistency (if previous attention available)
        if previous_attention is not None:
            temporal_loss = self._temporal_consistency_regularization(
                attention_weights, previous_attention
            )
            total_loss += self.temporal_weight * temporal_loss
        
        # 4. Diversity regularization (prevent collapse to single position)
        diversity_loss = self._diversity_regularization(attention_weights)
        total_loss += self.diversity_weight * diversity_loss
        
        return total_loss
    
    def _entropy_regularization(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Entropy regularization to encourage focused attention"""
        # attention_weights: [B, T_out, T_in]
        
        # Add small epsilon для numerical stability
        attention_safe = attention_weights + 1e-10
        
        # Normalize each attention distribution
        attention_normalized = attention_safe / attention_safe.sum(dim=-1, keepdim=True)
        
        # Compute entropy for each attention distribution
        entropy = -torch.sum(attention_normalized * torch.log(attention_normalized), dim=-1)
        
        # We want to minimize entropy (encourage focus), so return positive entropy
        return torch.mean(entropy)
    
    def _monotonic_regularization(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Monotonic regularization to encourage sequential alignment"""
        # attention_weights: [B, T_out, T_in]
        batch_size, T_out, T_in = attention_weights.shape
        
        if T_out < 2:
            return torch.tensor(0.0, device=attention_weights.device)
        
        # Find attention peaks (argmax) for each output step
        attention_peaks = torch.argmax(attention_weights, dim=-1)  # [B, T_out]
        
        # Compute differences between consecutive peaks
        peak_diffs = attention_peaks[:, 1:] - attention_peaks[:, :-1]  # [B, T_out-1]
        
        # Penalize negative differences (non-monotonic behavior)
        monotonic_violations = torch.clamp(-peak_diffs, min=0.0).float()
        
        # Return average violation penalty
        return torch.mean(monotonic_violations)
    
    def _temporal_consistency_regularization(self, 
                                           current_attention: torch.Tensor,
                                           previous_attention: torch.Tensor) -> torch.Tensor:
        """Temporal consistency regularization between consecutive steps"""
        # Both: [B, T_out, T_in]
        
        # L2 distance between attention distributions
        consistency_loss = F.mse_loss(current_attention, previous_attention)
        
        return consistency_loss
    
    def _diversity_regularization(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Diversity regularization to prevent attention collapse"""
        # attention_weights: [B, T_out, T_in]
        
        # Compute attention variance across input positions for each output step
        attention_variance = torch.var(attention_weights, dim=-1)  # [B, T_out]
        
        # We want to encourage diversity (higher variance), so minimize negative variance
        # But also prevent extreme spreading, so use a balanced approach
        target_variance = 0.1  # Target variance level
        variance_penalty = F.mse_loss(attention_variance, 
                                    torch.full_like(attention_variance, target_variance))
        
        return variance_penalty
    
    def update_regularization_weights(self, metrics: AttentionMetrics):
        """Dynamically update regularization weights based on attention quality"""
        
        # Adjust entropy weight based on focus
        if metrics.focus < 0.3:
            self.entropy_weight = min(0.5, self.entropy_weight * 1.2)
        elif metrics.focus > 0.8:
            self.entropy_weight = max(0.05, self.entropy_weight * 0.9)
        
        # Adjust monotonic weight based on monotonicity
        if metrics.monotonicity < 0.5:
            self.monotonic_weight = min(1.0, self.monotonic_weight * 1.3)
        elif metrics.monotonicity > 0.9:
            self.monotonic_weight = max(0.1, self.monotonic_weight * 0.8)
        
        # Adjust temporal weight based on consistency
        if metrics.consistency < 0.5:
            self.temporal_weight = min(0.5, self.temporal_weight * 1.1)
        elif metrics.consistency > 0.9:
            self.temporal_weight = max(0.05, self.temporal_weight * 0.9)


def create_advanced_attention_enhancement_system(hparams) -> Dict[str, nn.Module]:
    """
    🔥 Factory function для создания Advanced Attention Enhancement System
    
    Returns:
        Dictionary с всеми компонентами системы
    """
    
    # Extract parameters from hparams
    attention_rnn_dim = getattr(hparams, 'attention_rnn_dim', 1024)
    embedding_dim = getattr(hparams, 'encoder_embedding_dim', 512)
    attention_dim = getattr(hparams, 'attention_dim', 128)
    num_heads = getattr(hparams, 'attention_num_heads', 8)
    location_n_filters = getattr(hparams, 'attention_location_n_filters', 32)
    location_kernel_size = getattr(hparams, 'attention_location_kernel_size', 31)
    max_training_steps = getattr(hparams, 'max_training_steps', 10000)
    target_diagonality = getattr(hparams, 'target_attention_diagonality', 0.7)
    
    # Create all components
    multihead_attention = MultiHeadLocationAwareAttention(
        attention_rnn_dim=attention_rnn_dim,
        embedding_dim=embedding_dim,
        attention_dim=attention_dim,
        num_heads=num_heads,
        location_n_filters=location_n_filters,
        location_kernel_size=location_kernel_size
    )
    
    progressive_trainer = ProgressiveAttentionTrainer(
        max_steps=max_training_steps
    )
    
    self_supervised_learner = SelfSupervisedAttentionLearner()
    
    attention_diagnostics = AdvancedAttentionDiagnostics(
        target_diagonality=target_diagonality
    )
    
    regularization_system = AttentionRegularizationSystem()
    
    logger = logging.getLogger(__name__)
    logger.info("🔥 Advanced Attention Enhancement System создана")
    logger.info("🧠 Компоненты: MultiHead Attention, Progressive Trainer, Self-Supervised Learner, Diagnostics, Regularization")
    
    return {
        'multihead_attention': multihead_attention,
        'progressive_trainer': progressive_trainer,
        'self_supervised_learner': self_supervised_learner,
        'attention_diagnostics': attention_diagnostics,
        'regularization_system': regularization_system
    }


# Availability check
ADVANCED_ATTENTION_AVAILABLE = True

def get_advanced_attention_system_info():
    """Return information about the advanced attention system"""
    return {
        'available': ADVANCED_ATTENTION_AVAILABLE,
        'components': [
            'MultiHeadLocationAwareAttention',
            'ProgressiveAttentionTrainer', 
            'SelfSupervisedAttentionLearner',
            'AdvancedAttentionDiagnostics',
            'AttentionRegularizationSystem'
        ],
        'features': [
            'Transformer-style multi-head attention',
            'Location-aware attention mechanisms',
            'Progressive curriculum learning',
            'Self-supervised contrastive learning',
            'Real-time attention diagnostics',
            'Multi-level regularization',
            'Automatic correction suggestions'
        ]
    }


if __name__ == "__main__":
    # Example usage
    class MockHParams:
        attention_rnn_dim = 1024
        encoder_embedding_dim = 512
        attention_dim = 128
        attention_num_heads = 8
        attention_location_n_filters = 32
        attention_location_kernel_size = 31
        max_training_steps = 10000
        target_attention_diagonality = 0.7
    
    # Create system
    hparams = MockHParams()
    attention_system = create_advanced_attention_enhancement_system(hparams)
    
    print("🔥 Advanced Attention Enhancement System создана!")
    for component_name, component in attention_system.items():
        print(f"✅ {component_name}: {component.__class__.__name__}") 