"""TF-A-N 7B Configuration.

Defines the TFANConfig class for model hyperparameters.
Compatible with HuggingFace-style configuration patterns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Literal


@dataclass
class TFANConfig:
    """Configuration for TF-A-N 7B model.

    PROFILE-A (7.122B parameters):
    - 34 layers, 4096 hidden, 32 heads, 8 KV heads (GQA)
    - SwiGLU MLP with ffn_mult=3.25
    - RoPE, RMSNorm, tied embeddings
    - SSA with keep_ratio=0.33, local=128, hops=2

    Args:
        vocab_size: Vocabulary size (default: 32768)
        hidden_size: Hidden dimension (default: 4096)
        num_hidden_layers: Number of transformer layers (default: 34)
        num_attention_heads: Number of attention heads (default: 32)
        num_kv_heads: Number of KV heads for GQA (default: 8)
        intermediate_size: FFN intermediate dimension (default: 13312)
        ffn_mult: FFN multiplier (default: 3.25)
        max_position_embeddings: Maximum context length (default: 32768)
        rms_norm_eps: RMSNorm epsilon (default: 1e-6)
        rope_theta: RoPE base frequency (default: 10000.0)
        rope_scaling: RoPE scaling config (default: None)
        tie_word_embeddings: Whether to tie input/output embeddings (default: True)
        use_bias: Whether to use bias in linear layers (default: False)
        attention_impl: Attention implementation (default: "ssa_radial_v1")
        ssa_keep_ratio: SSA sparsity keep ratio (default: 0.33)
        ssa_local: SSA local window size (default: 128)
        ssa_hops: SSA radial hops (default: 2)
        tls_alpha: TLS landmark selection alpha (default: 0.7)
        dropout: Dropout probability (default: 0.0)
        attention_dropout: Attention dropout probability (default: 0.0)
        initializer_range: Weight initialization std (default: 0.02)
        use_cache: Whether to use KV cache (default: True)
        bos_token_id: Beginning of sequence token ID (default: 1)
        eos_token_id: End of sequence token ID (default: 2)
        pad_token_id: Padding token ID (default: None)
        torch_dtype: Model dtype (default: "bfloat16")
    """

    # Model architecture
    model_type: str = "tfan7b"
    architectures: list = field(default_factory=lambda: ["TFANForCausalLM"])
    vocab_size: int = 32768
    hidden_size: int = 4096
    num_hidden_layers: int = 34
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    intermediate_size: int = 13312
    ffn_mult: float = 3.25
    max_position_embeddings: int = 32768

    # Normalization
    rms_norm_eps: float = 1e-6

    # Positional embeddings
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None

    # Embeddings
    tie_word_embeddings: bool = True
    use_bias: bool = False

    # Activation
    activation: str = "swiglu"

    # Attention
    attention_impl: Literal["ssa_radial_v1", "flash", "sdpa", "eager"] = "ssa_radial_v1"
    ssa_keep_ratio: float = 0.33
    ssa_local: int = 128
    ssa_hops: int = 2
    tls_alpha: float = 0.7

    # Regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # Initialization
    initializer_range: float = 0.02

    # Inference
    use_cache: bool = True

    # Special tokens
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: Optional[int] = None

    # Precision
    torch_dtype: str = "bfloat16"

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate GQA heads
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible "
                f"by num_kv_heads ({self.num_kv_heads})"
            )

        # Validate hidden size
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible "
                f"by num_attention_heads ({self.num_attention_heads})"
            )

        # Set default rope_scaling if not provided
        if self.rope_scaling is None:
            self.rope_scaling = {"type": "linear", "factor": 1.0}

    @property
    def head_dim(self) -> int:
        """Get dimension per attention head."""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        """Get number of query heads per KV head (for GQA)."""
        return self.num_attention_heads // self.num_kv_heads

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save config to directory."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        config_path = save_directory / "config.json"
        with open(config_path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TFANConfig":
        """Create config from dictionary."""
        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)

    @classmethod
    def from_json(cls, json_str: str) -> "TFANConfig":
        """Create config from JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json_file(cls, json_path: str | Path) -> "TFANConfig":
        """Create config from JSON file."""
        with open(json_path, "r") as f:
            return cls.from_json(f.read())

    @classmethod
    def from_pretrained(cls, pretrained_path: str | Path) -> "TFANConfig":
        """Load config from pretrained directory."""
        pretrained_path = Path(pretrained_path)

        if pretrained_path.is_file():
            return cls.from_json_file(pretrained_path)

        config_path = pretrained_path / "config.json"
        if config_path.exists():
            return cls.from_json_file(config_path)

        raise FileNotFoundError(f"Config not found at {pretrained_path}")


def count_parameters(model) -> Dict[str, Any]:
    """Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "total_millions": total / 1e6,
        "total_billions": total / 1e9,
    }


def calculate_params_from_config(config: TFANConfig) -> Dict[str, Any]:
    """Calculate expected parameter count from config (no PyTorch required).

    Args:
        config: TFANConfig instance

    Returns:
        Dictionary with expected parameter counts
    """
    L = config.num_hidden_layers
    d = config.hidden_size
    H = config.num_attention_heads
    H_kv = config.num_kv_heads
    ffn_dim = config.intermediate_size
    V = config.vocab_size
    head_dim = d // H

    # Embeddings (tied with output)
    emb_params = V * d

    # Per-layer parameters
    # Attention: Q, K, V projections + output projection
    attn_qkv = d * d  # Q projection
    attn_qkv += d * (H_kv * head_dim)  # K projection (GQA)
    attn_qkv += d * (H_kv * head_dim)  # V projection (GQA)
    attn_out = d * d  # Output projection
    attn_params_per_layer = attn_qkv + attn_out

    # SwiGLU MLP: gate + value + output
    mlp_gate = d * ffn_dim
    mlp_value = d * ffn_dim
    mlp_out = ffn_dim * d
    mlp_params_per_layer = mlp_gate + mlp_value + mlp_out

    # RMSNorm: 2 per layer (pre-attn, pre-mlp)
    norm_params_per_layer = d * 2

    # Total per layer
    params_per_layer = attn_params_per_layer + mlp_params_per_layer + norm_params_per_layer

    # Total model
    layer_params = L * params_per_layer
    final_norm = d
    total_params = emb_params + layer_params + final_norm

    return {
        "embeddings": emb_params,
        "per_layer": params_per_layer,
        "total_layers": layer_params,
        "final_norm": final_norm,
        "total": total_params,
        "total_millions": total_params / 1e6,
        "total_billions": total_params / 1e9,
    }


__all__ = [
    "TFANConfig",
    "count_parameters",
    "calculate_params_from_config",
]
