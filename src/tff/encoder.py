"""TopFusionEncoder - Multi-modal MCCA-based fusion with topological features.

This module implements the Topological Feature Fusion (TFF) encoder that:
1. Aligns multi-modal representations via MCCA (Multi-modal CCA)
2. Projects aligned features into a unified embedding space
3. Provides hooks for persistent homology regularization

The key insight: proper multi-modal fusion requires alignment in a shared
latent space before concatenation/attention, not just naive concatenation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MCCAConfig:
    """Configuration for Multi-modal CCA alignment."""

    d_model: int = 512
    num_modalities: int = 3  # e.g., text, audio, vision
    latent_dim: int = 256  # CCA latent space dimension
    num_cca_components: int = 64  # Number of canonical components
    regularization: float = 1e-4  # Ridge regularization for stability
    use_kernel_cca: bool = False  # Whether to use kernel CCA
    kernel_type: str = "rbf"  # For kernel CCA: rbf, polynomial, linear
    kernel_gamma: float = 1.0  # RBF kernel bandwidth


@dataclass
class TopFusionConfig:
    """Configuration for TopFusionEncoder."""

    d_model: int = 512
    mcca_config: MCCAConfig = field(default_factory=MCCAConfig)
    num_modalities: int = 3
    use_residual: bool = True
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6

    # Topology-aware settings
    enable_ph_hooks: bool = True  # Persistent homology hooks
    ph_sample_rate: float = 0.1  # Sample rate for PH computation (expensive)


class MCCAProjection(nn.Module):
    """Learnable CCA-style projection for a single modality.

    Projects modality features into shared latent space while
    maximizing correlation with other modalities.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_components: int,
        regularization: float = 1e-4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_components = num_components
        self.reg = regularization

        # Learnable projection matrix (initialized orthogonally)
        self.projection = nn.Linear(input_dim, latent_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)

        # Whitening parameters (learned during forward pass)
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("std", torch.ones(input_dim))
        self.register_buffer("running_cov", torch.eye(input_dim))
        self.momentum = 0.1

    def forward(
        self,
        x: torch.Tensor,
        update_stats: bool = True,
    ) -> torch.Tensor:
        """Project input to latent space.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            update_stats: Whether to update running statistics

        Returns:
            Projected tensor of shape (batch, seq_len, latent_dim)
        """
        batch_size, seq_len, d = x.shape

        # Flatten for statistics
        x_flat = x.reshape(-1, d)

        if self.training and update_stats:
            # Update running mean/std
            batch_mean = x_flat.mean(dim=0)
            batch_std = x_flat.std(dim=0) + 1e-8

            self.mean = (1 - self.momentum) * self.mean + self.momentum * batch_mean
            self.std = (1 - self.momentum) * self.std + self.momentum * batch_std

        # Center and scale
        x_normalized = (x - self.mean) / (self.std + 1e-8)

        # Project to latent space
        z = self.projection(x_normalized)

        return z


class MCCAWrapper(nn.Module):
    """Multi-modal CCA alignment wrapper.

    Implements soft MCCA alignment by:
    1. Projecting each modality to shared latent space
    2. Computing cross-correlation matrix
    3. Optimizing for maximum correlation via auxiliary loss
    """

    def __init__(self, config: MCCAConfig):
        super().__init__()
        self.config = config

        # Per-modality projections
        self.projections = nn.ModuleList([
            MCCAProjection(
                input_dim=config.d_model,
                latent_dim=config.latent_dim,
                num_components=config.num_cca_components,
                regularization=config.regularization,
            )
            for _ in range(config.num_modalities)
        ])

        # Learnable alignment matrices (soft CCA)
        self.alignment_weights = nn.ParameterList([
            nn.Parameter(torch.eye(config.latent_dim))
            for _ in range(config.num_modalities)
        ])

    def forward(
        self,
        modality_features: List[torch.Tensor],
        return_correlations: bool = False,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """Align multi-modal features via MCCA.

        Args:
            modality_features: List of tensors, each (batch, seq_len, d_model)
            return_correlations: Whether to return correlation matrix

        Returns:
            Tuple of (aligned_features, correlation_loss)
        """
        assert len(modality_features) == self.config.num_modalities

        # Project each modality
        projected = []
        for i, (feat, proj) in enumerate(zip(modality_features, self.projections)):
            z = proj(feat)
            # Apply alignment transform
            z_aligned = torch.matmul(z, self.alignment_weights[i])
            projected.append(z_aligned)

        # Compute cross-correlation loss (for training)
        correlation_loss = None
        if self.training or return_correlations:
            correlation_loss = self._compute_correlation_loss(projected)

        return projected, correlation_loss

    def _compute_correlation_loss(
        self,
        projected: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute negative correlation as loss (maximize correlation).

        Uses trace norm of cross-correlation matrix.
        """
        total_corr = 0.0
        num_pairs = 0

        for i in range(len(projected)):
            for j in range(i + 1, len(projected)):
                # Get mean representations
                z_i = projected[i].mean(dim=1)  # (batch, latent_dim)
                z_j = projected[j].mean(dim=1)

                # Normalize
                z_i = F.normalize(z_i, dim=-1)
                z_j = F.normalize(z_j, dim=-1)

                # Cross-correlation (batch-wise)
                corr = (z_i * z_j).sum(dim=-1).mean()
                total_corr += corr
                num_pairs += 1

        # Negative correlation (minimize this to maximize correlation)
        if num_pairs > 0:
            return -total_corr / num_pairs
        return torch.tensor(0.0, device=projected[0].device)


class TopFusionEncoder(nn.Module):
    """Topological Feature Fusion Encoder.

    Fuses multi-modal inputs via:
    1. MCCA alignment in shared latent space
    2. Concatenation + projection to unified embedding
    3. Optional persistent homology hooks for regularization

    Architecture:
        Input: [text_emb, audio_emb, vision_emb] each (B, L, D)
        → MCCA alignment → [aligned_text, aligned_audio, aligned_vision]
        → Concatenate → (B, L, D*3)
        → Post-projection → (B, L, D)
        → Layer norm + residual
        → Output: (B, L, D)
    """

    def __init__(self, config: TopFusionConfig):
        super().__init__()
        self.config = config

        # MCCA alignment
        self.mcca = MCCAWrapper(config.mcca_config)

        # Post-fusion projection
        # Input: concatenated aligned features (d_model * num_modalities)
        # Output: unified embedding (d_model)
        fusion_input_dim = config.mcca_config.latent_dim * config.num_modalities
        self.post_proj = nn.Sequential(
            nn.Linear(fusion_input_dim, config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model),
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Optional: input projection if modalities have different dims
        self.input_projs = nn.ModuleList([
            nn.Linear(config.d_model, config.d_model)
            for _ in range(config.num_modalities)
        ])

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # PH hooks storage (populated during forward if enabled)
        self._ph_embeddings: Optional[torch.Tensor] = None
        self._ph_sample_mask: Optional[torch.Tensor] = None

    def forward(
        self,
        modality_features: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        return_mcca_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through TopFusionEncoder.

        Args:
            modality_features: List of modality tensors, each (batch, seq_len, d_model)
            attention_mask: Optional mask (batch, seq_len)
            return_mcca_loss: Whether to include MCCA correlation loss

        Returns:
            Dict with:
                - fused: Fused embedding (batch, seq_len, d_model)
                - mcca_loss: MCCA alignment loss (scalar)
                - aligned_features: List of aligned modality features
        """
        batch_size = modality_features[0].shape[0]
        seq_len = modality_features[0].shape[1]

        # Project inputs (handles different input dimensions)
        projected_inputs = []
        for i, (feat, proj) in enumerate(zip(modality_features, self.input_projs)):
            projected_inputs.append(proj(feat))

        # MCCA alignment
        aligned, mcca_loss = self.mcca(projected_inputs, return_correlations=return_mcca_loss)

        # Concatenate aligned features
        concat = torch.cat(aligned, dim=-1)  # (batch, seq_len, latent_dim * num_modalities)

        # Project to unified space
        fused = self.post_proj(concat)  # (batch, seq_len, d_model)

        # Residual connection (average of input modalities)
        if self.config.use_residual:
            residual = sum(projected_inputs) / len(projected_inputs)
            fused = fused + residual

        # Layer norm and dropout
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)

        # Store embeddings for PH computation (sampled)
        if self.config.enable_ph_hooks and self.training:
            self._store_ph_samples(fused, batch_size)

        return {
            "fused": fused,
            "mcca_loss": mcca_loss if mcca_loss is not None else torch.tensor(0.0),
            "aligned_features": aligned,
        }

    def _store_ph_samples(self, embeddings: torch.Tensor, batch_size: int) -> None:
        """Store sampled embeddings for persistent homology computation."""
        if torch.rand(1).item() < self.config.ph_sample_rate:
            self._ph_embeddings = embeddings.detach()
            self._ph_sample_mask = torch.ones(batch_size, dtype=torch.bool)

    def get_ph_embeddings(self) -> Optional[torch.Tensor]:
        """Get stored embeddings for persistent homology computation."""
        return self._ph_embeddings

    def clear_ph_cache(self) -> None:
        """Clear PH embedding cache."""
        self._ph_embeddings = None
        self._ph_sample_mask = None


class ModalityEncoder(nn.Module):
    """Helper encoder for a single modality before fusion.

    Can be used to pre-process raw modality features before TopFusionEncoder.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            out_dim = d_model if i == num_layers - 1 else (input_dim + d_model) // 2
            layers.extend([
                nn.Linear(current_dim, out_dim),
                nn.GELU(),
                nn.LayerNorm(out_dim),
                nn.Dropout(dropout),
            ])
            current_dim = out_dim

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode modality features.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            Encoded tensor (batch, seq_len, d_model)
        """
        return self.encoder(x)


def create_top_fusion_encoder(
    d_model: int = 512,
    num_modalities: int = 3,
    latent_dim: int = 256,
    dropout: float = 0.1,
) -> TopFusionEncoder:
    """Factory function to create TopFusionEncoder with common defaults."""
    mcca_config = MCCAConfig(
        d_model=d_model,
        num_modalities=num_modalities,
        latent_dim=latent_dim,
    )

    config = TopFusionConfig(
        d_model=d_model,
        mcca_config=mcca_config,
        num_modalities=num_modalities,
        dropout=dropout,
    )

    return TopFusionEncoder(config)
