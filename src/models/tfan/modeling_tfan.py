"""TF-A-N 7B Model Implementation.

Transformer with Formal Alignment Network - a 7.122B parameter
decoder-only transformer with SSA attention, RoPE, RMSNorm, and SwiGLU.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Union

from .config import TFANConfig
from .norm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary_pos_emb
from .attention_sparse import SSAAttention
from .mlp_glu import SwiGLUMLP


@dataclass
class BaseModelOutput:
    """Base output for TFANModel."""
    last_hidden_state: torch.Tensor
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


@dataclass
class CausalLMOutput:
    """Output for TFANForCausalLM."""
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class TFANDecoderLayer(nn.Module):
    """Single transformer decoder layer for TF-A-N.

    Architecture:
        x -> RMSNorm -> SSA Attention -> residual
        x -> RMSNorm -> SwiGLU MLP -> residual

    Args:
        config: TFANConfig instance
        layer_idx: Layer index for debugging/profiling
    """

    def __init__(self, config: TFANConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Pre-attention normalization
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Selective Sparse Attention
        self.self_attn = SSAAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            num_landmarks=max(16, config.max_position_embeddings // 512),
            local_window=config.ssa_local,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            use_fast=True,
        )

        # Pre-MLP normalization
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # SwiGLU MLP
        self.mlp = SwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            mlp_bias=config.use_bias,
        )

        # Optional dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through decoder layer.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_value: Optional cached KV states
            output_attentions: Whether to return attention weights
            use_cache: Whether to return updated KV cache

        Returns:
            hidden_states: Output tensor [batch, seq_len, hidden_size]
            attn_weights: Attention weights if output_attentions=True
            past_key_value: Updated KV cache if use_cache=True
        """
        residual = hidden_states

        # Pre-attention norm
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        attn_output, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # Residual connection + dropout
        hidden_states = residual + self.dropout(attn_output)

        # Pre-MLP norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        hidden_states = self.mlp(hidden_states)

        # Residual connection + dropout
        hidden_states = residual + self.dropout(hidden_states)

        return hidden_states, attn_weights, present_key_value


class TFANModel(nn.Module):
    """TF-A-N base model (decoder-only transformer).

    Outputs hidden states without language modeling head.

    Args:
        config: TFANConfig instance
    """

    def __init__(self, config: TFANConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Decoder layers
        self.layers = nn.ModuleList([
            TFANDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize module weights."""
        std = self.config.initializer_range

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embeddings."""
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        """Set input embeddings."""
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """Forward pass through TF-A-N model.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position indices [batch, seq_len]
            past_key_values: Cached KV states for generation
            inputs_embeds: Pre-computed input embeddings
            use_cache: Whether to return KV cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return BaseModelOutput

        Returns:
            BaseModelOutput or tuple of tensors
        """
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True

        # Get input embeddings
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len = inputs_embeds.shape[:2]

        # Initialize past_key_values
        if past_key_values is None:
            past_key_values = [None] * self.num_layers
        past_length = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0

        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + seq_len,
                device=inputs_embeds.device
            ).unsqueeze(0).expand(batch_size, -1)

        # Prepare attention mask (convert to 4D causal mask)
        if attention_mask is not None:
            # [batch, seq_len] -> [batch, 1, seq_len, past_len + seq_len]
            attention_mask = self._prepare_attention_mask(
                attention_mask, batch_size, seq_len, past_length,
                inputs_embeds.device, inputs_embeds.dtype
            )

        hidden_states = inputs_embeds

        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = () if use_cache else None

        # Forward through layers
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_kv = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                hidden_states, attn_weights, present_kv = self._gradient_checkpointing_forward(
                    layer, hidden_states, attention_mask, position_ids,
                    past_kv, output_attentions, use_cache
                )
            else:
                hidden_states, attn_weights, present_kv = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_kv,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            if use_cache:
                next_cache = next_cache + (present_kv,)

            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        seq_len: int,
        past_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare 4D causal attention mask."""
        total_len = past_length + seq_len

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, total_len, device=device, dtype=torch.bool),
            diagonal=past_length + 1
        )

        # Combine with padding mask
        if attention_mask is not None:
            # Expand padding mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            padding_mask = attention_mask[:, None, None, :total_len].to(torch.bool)
            # Invert: 1 = attend, 0 = mask
            padding_mask = ~padding_mask.bool()
        else:
            padding_mask = torch.zeros(batch_size, 1, 1, total_len, device=device, dtype=torch.bool)

        # Combine masks
        combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) | padding_mask

        # Convert to attention bias
        attention_bias = torch.zeros_like(combined_mask, dtype=dtype)
        attention_bias = attention_bias.masked_fill(combined_mask, float('-inf'))

        return attention_bias

    def _gradient_checkpointing_forward(
        self,
        layer: TFANDecoderLayer,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
        output_attentions: bool,
        use_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward with gradient checkpointing."""
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        return torch.utils.checkpoint.checkpoint(
            create_custom_forward(layer),
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            use_reentrant=False,
        )


class TFANForCausalLM(nn.Module):
    """TF-A-N model with language modeling head.

    Args:
        config: TFANConfig instance
    """

    def __init__(self, config: TFANConfig):
        super().__init__()
        self.config = config

        # Base model
        self.model = TFANModel(config)

        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # Initialize lm_head if not tied
        if not config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=config.initializer_range)

    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embeddings."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        """Set input embeddings."""
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        """Get output embeddings (lm_head)."""
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear):
        """Set output embeddings."""
        self.lm_head = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        """Forward pass for causal language modeling.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position indices [batch, seq_len]
            past_key_values: Cached KV states for generation
            inputs_embeds: Pre-computed input embeddings
            labels: Target labels for loss computation [batch, seq_len]
            use_cache: Whether to return KV cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return CausalLMOutput

        Returns:
            CausalLMOutput or tuple of tensors
        """
        return_dict = return_dict if return_dict is not None else True

        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state

        # Compute logits
        logits = self.lm_head(hidden_states)
        logits = logits.float()  # Compute loss in FP32

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for cross-entropy
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            # Compute loss
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare inputs for generation step.

        Args:
            input_ids: Input token IDs
            past_key_values: Cached KV states
            attention_mask: Attention mask
            inputs_embeds: Pre-computed embeddings

        Returns:
            Dictionary of model inputs
        """
        # If past_key_values is provided, only use the last token
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            # Only keep last token for input_ids
            if input_ids.shape[1] > past_length:
                input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # Create position_ids on the fly
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

        return model_inputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """Simple greedy/sampling generation.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample (vs greedy)
            eos_token_id: End of sequence token
            pad_token_id: Padding token

        Returns:
            Generated token IDs [batch, seq_len + generated]
        """
        eos_token_id = eos_token_id or self.config.eos_token_id
        pad_token_id = pad_token_id or self.config.pad_token_id or eos_token_id

        batch_size = input_ids.shape[0]
        generated = input_ids
        past_key_values = None

        for _ in range(max_new_tokens):
            # Prepare inputs
            model_inputs = self.prepare_inputs_for_generation(
                generated, past_key_values=past_key_values, **kwargs
            )

            # Forward pass
            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample or greedy
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated
            generated = torch.cat([generated, next_tokens], dim=-1)

            # Check for EOS
            if eos_token_id is not None:
                if (next_tokens == eos_token_id).all():
                    break

        return generated


__all__ = [
    "TFANModel",
    "TFANForCausalLM",
    "TFANDecoderLayer",
    "BaseModelOutput",
    "CausalLMOutput",
]
