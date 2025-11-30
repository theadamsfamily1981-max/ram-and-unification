"""TF-A-N 7B: Transformer with Formal Alignment Network.

A 7.122B parameter decoder-only transformer with:
- Selective Sparse Attention (SSA) - O(N log N) complexity
- RoPE (Rotary Positional Embeddings)
- RMSNorm normalization
- SwiGLU MLP activation
- GQA (Grouped Query Attention) with 8 KV heads
- Topological regularization hooks
- Emotion head for valence/arousal prediction
- FDT (Fluctuation-Dissipation Theorem) controller

Architecture (PROFILE-A):
- 34 layers
- 4096 hidden dimension
- 32 attention heads (8 KV heads for GQA)
- 13312 intermediate size (ffn_mult=3.25)
- 32768 vocabulary
- 32768 max context length
"""

from .config import TFANConfig, count_parameters, calculate_params_from_config
from .modeling_tfan import TFANModel, TFANForCausalLM, TFANDecoderLayer
from .norm import RMSNorm, RMSNormFused
from .rope import RotaryEmbedding, apply_rotary_pos_emb, apply_rotary_pos_emb_single
from .attention_sparse import SSAAttention, ssa_attention, ssa_attention_fast
from .mlp_glu import SwiGLUMLP, SwiGLUMLPFused, GatedMLP
from .fdt_controller import FDTController, FDTConfig, FDTMetrics, GradientClipper, LearningRateScheduler

__all__ = [
    # Config
    "TFANConfig",
    "count_parameters",
    "calculate_params_from_config",
    # Models
    "TFANModel",
    "TFANForCausalLM",
    "TFANDecoderLayer",
    # Normalization
    "RMSNorm",
    "RMSNormFused",
    # Positional Embeddings
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "apply_rotary_pos_emb_single",
    # Attention
    "SSAAttention",
    "ssa_attention",
    "ssa_attention_fast",
    # MLP
    "SwiGLUMLP",
    "SwiGLUMLPFused",
    "GatedMLP",
    # Training
    "FDTController",
    "FDTConfig",
    "FDTMetrics",
    "GradientClipper",
    "LearningRateScheduler",
]
