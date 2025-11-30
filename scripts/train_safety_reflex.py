#!/usr/bin/env python3
"""Train Tier-1 Safety Reflex SNN from TF-A-N safety head.

This script implements the teacher-student distillation pipeline:
1. Load TF-A-N 7B with safety head (or use synthetic teacher)
2. Generate training data: (telemetry → action) pairs
3. Train SNN fabric to imitate the safety head
4. Export to Kitten FPGA format

Usage:
    # With TF-A-N checkpoint
    python scripts/train_safety_reflex.py \
        --teacher-checkpoint path/to/tfan7b \
        --fabric-config configs/snn_fabric/tier1_safety_reflex.yaml \
        --output-dir artifacts/safety_reflex_v1

    # With synthetic teacher (for testing)
    python scripts/train_safety_reflex.py \
        --synthetic-teacher \
        --fabric-config configs/snn_fabric/tier1_safety_reflex.yaml \
        --output-dir artifacts/safety_reflex_v1
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

# Add src to path
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

import yaml


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TelemetrySnapshot:
    """Single snapshot of system telemetry."""
    gpu_util: np.ndarray        # [8]
    gpu_mem: np.ndarray         # [8]
    temp_delta: np.ndarray      # [16]
    power_rail: np.ndarray      # [8]
    net_error: np.ndarray       # [4]
    job_queue: np.ndarray       # [4]
    watchdog: np.ndarray        # [8]
    failure_flags: np.ndarray   # [8]

    def to_tensor(self) -> torch.Tensor:
        """Flatten to [64] tensor."""
        return torch.tensor(np.concatenate([
            self.gpu_util,
            self.gpu_mem,
            self.temp_delta,
            self.power_rail,
            self.net_error,
            self.job_queue,
            self.watchdog,
            self.failure_flags,
        ]), dtype=torch.float32)


@dataclass
class SafetyLabel:
    """Safety action label."""
    action: int  # 0=SAFE, 1=WARN, 2=BRAKE, 3=CUT
    confidence: float = 1.0

    ACTION_NAMES = ["SAFE", "WARN", "BRAKE", "CUT"]

    @property
    def name(self) -> str:
        return self.ACTION_NAMES[self.action]


# =============================================================================
# Synthetic Teacher
# =============================================================================

class SyntheticSafetyTeacher:
    """Rule-based synthetic teacher for generating training data.

    This mimics what a TF-A-N safety head would learn, using simple rules.
    Useful for testing the pipeline before having a trained TF-A-N model.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_telemetry(self, scenario: str = "random") -> TelemetrySnapshot:
        """Generate synthetic telemetry snapshot."""
        if scenario == "healthy":
            return TelemetrySnapshot(
                gpu_util=self.rng.uniform(0.3, 0.7, 8),
                gpu_mem=self.rng.uniform(0.2, 0.6, 8),
                temp_delta=self.rng.uniform(-5, 10, 16),
                power_rail=self.rng.uniform(0.8, 1.0, 8),
                net_error=self.rng.uniform(0.0, 0.05, 4),
                job_queue=self.rng.uniform(0, 20, 4),
                watchdog=self.rng.uniform(0, 1, 8),
                failure_flags=np.zeros(8),
            )
        elif scenario == "thermal_warning":
            return TelemetrySnapshot(
                gpu_util=self.rng.uniform(0.8, 0.95, 8),
                gpu_mem=self.rng.uniform(0.6, 0.8, 8),
                temp_delta=self.rng.uniform(15, 30, 16),
                power_rail=self.rng.uniform(0.7, 0.9, 8),
                net_error=self.rng.uniform(0.0, 0.1, 4),
                job_queue=self.rng.uniform(10, 40, 4),
                watchdog=self.rng.uniform(0, 2, 8),
                failure_flags=np.zeros(8),
            )
        elif scenario == "memory_pressure":
            return TelemetrySnapshot(
                gpu_util=self.rng.uniform(0.7, 0.9, 8),
                gpu_mem=self.rng.uniform(0.85, 0.98, 8),
                temp_delta=self.rng.uniform(10, 25, 16),
                power_rail=self.rng.uniform(0.75, 0.95, 8),
                net_error=self.rng.uniform(0.0, 0.1, 4),
                job_queue=self.rng.uniform(20, 60, 4),
                watchdog=self.rng.uniform(1, 3, 8),
                failure_flags=np.zeros(8),
            )
        elif scenario == "error_cascade":
            return TelemetrySnapshot(
                gpu_util=self.rng.uniform(0.5, 0.9, 8),
                gpu_mem=self.rng.uniform(0.5, 0.9, 8),
                temp_delta=self.rng.uniform(5, 20, 16),
                power_rail=self.rng.uniform(0.6, 0.85, 8),
                net_error=self.rng.uniform(0.2, 0.5, 4),
                job_queue=self.rng.uniform(30, 80, 4),
                watchdog=self.rng.uniform(3, 6, 8),
                failure_flags=(self.rng.random(8) > 0.5).astype(np.float32),
            )
        elif scenario == "critical":
            return TelemetrySnapshot(
                gpu_util=self.rng.uniform(0.9, 1.0, 8),
                gpu_mem=self.rng.uniform(0.95, 1.0, 8),
                temp_delta=self.rng.uniform(35, 50, 16),
                power_rail=self.rng.uniform(0.4, 0.7, 8),
                net_error=self.rng.uniform(0.3, 0.8, 4),
                job_queue=self.rng.uniform(50, 100, 4),
                watchdog=self.rng.uniform(5, 10, 8),
                failure_flags=(self.rng.random(8) > 0.3).astype(np.float32),
            )
        else:  # random
            scenario = self.rng.choice([
                "healthy", "healthy", "healthy",  # 60% healthy
                "thermal_warning", "memory_pressure",  # 20% warning
                "error_cascade",  # 10% brake
                "critical",  # 10% critical
            ])
            return self.generate_telemetry(scenario)

    def predict(self, telemetry: TelemetrySnapshot) -> SafetyLabel:
        """Predict safety action based on telemetry (rule-based)."""
        # Compute risk scores
        thermal_risk = np.mean(telemetry.temp_delta > 25)
        memory_risk = np.mean(telemetry.gpu_mem > 0.9)
        error_risk = np.mean(telemetry.net_error > 0.2)
        watchdog_risk = np.mean(telemetry.watchdog > 4)
        failure_risk = np.mean(telemetry.failure_flags)
        power_risk = np.mean(telemetry.power_rail < 0.7)

        # Aggregate risk
        total_risk = (
            0.25 * thermal_risk +
            0.20 * memory_risk +
            0.15 * error_risk +
            0.15 * watchdog_risk +
            0.15 * failure_risk +
            0.10 * power_risk
        )

        # Map to action
        if total_risk > 0.6 or failure_risk > 0.5:
            return SafetyLabel(action=3, confidence=min(1.0, total_risk + 0.2))  # CUT
        elif total_risk > 0.35 or memory_risk > 0.7:
            return SafetyLabel(action=2, confidence=min(1.0, total_risk + 0.1))  # BRAKE
        elif total_risk > 0.15:
            return SafetyLabel(action=1, confidence=min(1.0, total_risk + 0.2))  # WARN
        else:
            return SafetyLabel(action=0, confidence=1.0 - total_risk)  # SAFE

    def generate_dataset(self, n_samples: int) -> List[Tuple[TelemetrySnapshot, SafetyLabel]]:
        """Generate training dataset."""
        data = []
        for _ in range(n_samples):
            telemetry = self.generate_telemetry("random")
            label = self.predict(telemetry)
            data.append((telemetry, label))
        return data


# =============================================================================
# TF-A-N Safety Head Teacher
# =============================================================================

class TFANSafetyTeacher:
    """Teacher using TF-A-N model's safety head."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Try to load TF-A-N model
        try:
            from models.tfan import TFANConfig, TFANForCausalLM

            print(f"[Teacher] Loading TF-A-N from {checkpoint_path}...")
            self.config = TFANConfig.from_pretrained(checkpoint_path)
            self.model = TFANForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            self.model.to(device)
            self.model.eval()

            # Check for safety head
            if not hasattr(self.model, "safety_head"):
                print("[Teacher] WARNING: No safety_head found, using fallback")
                self._use_fallback = True
            else:
                self._use_fallback = False

        except Exception as e:
            print(f"[Teacher] WARNING: Could not load TF-A-N: {e}")
            print("[Teacher] Using synthetic teacher fallback")
            self._use_fallback = True
            self._fallback = SyntheticSafetyTeacher()

    def predict(self, telemetry: TelemetrySnapshot) -> SafetyLabel:
        """Predict safety action."""
        if self._use_fallback:
            return self._fallback.predict(telemetry)

        with torch.no_grad():
            x = telemetry.to_tensor().unsqueeze(0).to(self.device)
            logits = self.model.safety_head(x)
            probs = F.softmax(logits, dim=-1)
            action = probs.argmax(dim=-1).item()
            confidence = probs[0, action].item()
            return SafetyLabel(action=action, confidence=confidence)

    def generate_dataset(self, n_samples: int) -> List[Tuple[TelemetrySnapshot, SafetyLabel]]:
        """Generate training dataset."""
        if self._use_fallback:
            return self._fallback.generate_dataset(n_samples)

        # For real TF-A-N, we'd need actual telemetry data
        # For now, generate synthetic telemetry and label with TF-A-N
        synth = SyntheticSafetyTeacher()
        data = []
        for _ in range(n_samples):
            telemetry = synth.generate_telemetry("random")
            label = self.predict(telemetry)
            data.append((telemetry, label))
        return data


# =============================================================================
# Dataset
# =============================================================================

class SafetyReflexDataset(Dataset):
    """PyTorch dataset for safety reflex training."""

    def __init__(
        self,
        data: List[Tuple[TelemetrySnapshot, SafetyLabel]],
        time_steps: int = 64,
        encoding_rate: float = 100.0,
    ):
        self.data = data
        self.time_steps = time_steps
        self.encoding_rate = encoding_rate
        self.dt = 1.0  # ms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        telemetry, label = self.data[idx]

        # Convert telemetry to spike trains [time_steps, 64]
        values = telemetry.to_tensor().numpy()

        # Rate coding: value → spike probability per timestep
        # Higher value → higher probability
        spike_probs = values * (self.encoding_rate * self.dt / 1000.0)
        spike_probs = np.clip(spike_probs, 0, 1)

        # Generate spikes
        spikes = (np.random.random((self.time_steps, 64)) < spike_probs).astype(np.float32)

        return {
            "spikes": torch.tensor(spikes, dtype=torch.float32),
            "label": torch.tensor(label.action, dtype=torch.long),
            "confidence": torch.tensor(label.confidence, dtype=torch.float32),
        }


# =============================================================================
# SNN Model (PyTorch surrogate gradient training)
# =============================================================================

class SurrogateSpikeFunction(torch.autograd.Function):
    """Surrogate gradient for spike function."""

    @staticmethod
    def forward(ctx, x, threshold):
        ctx.save_for_backward(x, torch.tensor(threshold))
        return (x >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        # Fast sigmoid surrogate
        grad_input = grad_output * torch.sigmoid(4.0 * (x - threshold)) * (1 - torch.sigmoid(4.0 * (x - threshold)))
        return grad_input, None


class LIFLayer(nn.Module):
    """Leaky Integrate-and-Fire layer with surrogate gradient."""

    def __init__(self, N: int, alpha: float = 0.9, v_th: float = 0.5):
        super().__init__()
        self.N = N
        self.alpha = alpha
        self.v_th = v_th

    def forward(self, current: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single timestep update.

        Args:
            current: Input current [batch, N]
            state: Membrane potential [batch, N]

        Returns:
            spikes: Output spikes [batch, N]
            new_state: Updated membrane potential [batch, N]
        """
        # Leak
        v = self.alpha * state + current

        # Spike
        spikes = SurrogateSpikeFunction.apply(v, self.v_th)

        # Reset
        new_state = v * (1.0 - spikes)

        return spikes, new_state


class SafetyReflexSNN(nn.Module):
    """SNN model for safety reflex (surrogate gradient trainable)."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Parse population sizes
        pops = config["populations"]
        self.N_input = pops["input"]["N"]
        self.N_h0 = pops["hidden_0"]["N"]
        self.N_h1 = pops["hidden_1"]["N"]
        self.N_h2 = pops["hidden_2"]["N"]
        self.N_h3 = pops["hidden_3"]["N"]
        self.N_output = pops["output"]["N"]

        # Projection weights (dense for training, will be sparsified for export)
        self.W_input_h0 = nn.Linear(self.N_input, self.N_h0, bias=False)
        self.W_h0_h1 = nn.Linear(self.N_h0, self.N_h1, bias=False)
        self.W_h1_h2 = nn.Linear(self.N_h1, self.N_h2, bias=False)
        self.W_h2_h2 = nn.Linear(self.N_h2, self.N_h2, bias=False)  # Recurrent
        self.W_h2_h3 = nn.Linear(self.N_h2, self.N_h3, bias=False)
        self.W_h3_out = nn.Linear(self.N_h3, self.N_output, bias=False)

        # LIF layers
        self.lif_h0 = LIFLayer(self.N_h0, alpha=0.9, v_th=0.5)
        self.lif_h1 = LIFLayer(self.N_h1, alpha=0.95, v_th=0.5)
        self.lif_h2 = LIFLayer(self.N_h2, alpha=0.93, v_th=0.5)
        self.lif_h3 = LIFLayer(self.N_h3, alpha=0.9, v_th=0.5)
        self.lif_out = LIFLayer(self.N_output, alpha=0.82, v_th=0.8)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        for module in [self.W_input_h0, self.W_h0_h1, self.W_h1_h2,
                       self.W_h2_h2, self.W_h2_h3, self.W_h3_out]:
            nn.init.normal_(module.weight, mean=0.0, std=0.05)

    def forward(
        self,
        input_spikes: torch.Tensor,  # [batch, time, N_input]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over all timesteps.

        Returns:
            output_spikes: [batch, time, N_output]
            spike_counts: [batch, N_output] total spike counts
        """
        batch, T, _ = input_spikes.shape
        device = input_spikes.device

        # Initialize states
        v_h0 = torch.zeros(batch, self.N_h0, device=device)
        v_h1 = torch.zeros(batch, self.N_h1, device=device)
        v_h2 = torch.zeros(batch, self.N_h2, device=device)
        v_h3 = torch.zeros(batch, self.N_h3, device=device)
        v_out = torch.zeros(batch, self.N_output, device=device)

        output_spikes = []
        s_h2 = torch.zeros(batch, self.N_h2, device=device)  # For recurrence

        for t in range(T):
            # Input → H0
            i_h0 = self.W_input_h0(input_spikes[:, t, :])
            s_h0, v_h0 = self.lif_h0(i_h0, v_h0)

            # H0 → H1
            i_h1 = self.W_h0_h1(s_h0)
            s_h1, v_h1 = self.lif_h1(i_h1, v_h1)

            # H1 → H2 + H2 → H2 (recurrent)
            i_h2 = self.W_h1_h2(s_h1) + self.W_h2_h2(s_h2)
            s_h2, v_h2 = self.lif_h2(i_h2, v_h2)

            # H2 → H3
            i_h3 = self.W_h2_h3(s_h2)
            s_h3, v_h3 = self.lif_h3(i_h3, v_h3)

            # H3 → Output
            i_out = self.W_h3_out(s_h3)
            s_out, v_out = self.lif_out(i_out, v_out)

            output_spikes.append(s_out)

        output_spikes = torch.stack(output_spikes, dim=1)  # [batch, T, N_output]

        # Sum spike counts (for classification)
        spike_counts = output_spikes.sum(dim=1)  # [batch, N_output]

        return output_spikes, spike_counts


# =============================================================================
# Training Loop
# =============================================================================

def train_safety_reflex(
    model: SafetyReflexSNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = "cpu",
    output_dir: Path = None,
):
    """Train the safety reflex SNN."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            spikes = batch["spikes"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            _, spike_counts = model(spikes)

            # Reshape spike counts for voting: [batch, 4, 4] then sum over voters
            votes = spike_counts.view(-1, 4, 4).sum(dim=2)  # [batch, 4]
            loss = F.cross_entropy(votes, labels)

            # Spike regularization (encourage sparsity)
            spike_rate = spike_counts.mean()
            loss += 0.01 * (spike_rate - 0.05).abs()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        history["train_loss"].append(avg_loss)

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                spikes = batch["spikes"].to(device)
                labels = batch["label"].to(device)

                _, spike_counts = model(spikes)
                votes = spike_counts.view(-1, 4, 4).sum(dim=2)
                preds = votes.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_acc and output_dir:
            best_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    return history


# =============================================================================
# Export to Kitten Format
# =============================================================================

def export_to_kitten(
    model: SafetyReflexSNN,
    config: dict,
    output_dir: Path,
    sparsity_threshold: float = 0.01,
):
    """Export trained SNN to Kitten FPGA format."""
    from models.tfan.snn.fabric import (
        KittenFabricData,
        export_kitten_fabric,
        quantize_q5_10,
        quantize_q1_6,
        dense_to_csr,
    )

    print("[Export] Converting to Kitten format...")

    # Total neurons
    num_neurons = (
        model.N_input + model.N_h0 + model.N_h1 +
        model.N_h2 + model.N_h3 + model.N_output
    )

    # Build neuron parameters
    v_init = torch.zeros(num_neurons)
    v_th = torch.zeros(num_neurons)
    alpha = torch.zeros(num_neurons)

    offset = 0
    for pop_name, pop_cfg in config["populations"].items():
        N = pop_cfg["N"]
        params = pop_cfg.get("params", {})
        v_th[offset:offset+N] = params.get("v_th", 0.5)
        alpha[offset:offset+N] = params.get("alpha", 0.9)
        offset += N

    # Quantize
    v_init_fp = quantize_q5_10(v_init).numpy()
    v_th_fp = quantize_q5_10(v_th).numpy()
    alpha_fp = quantize_q5_10(alpha).numpy()  # Store alpha too

    fabric_data = KittenFabricData(
        name=config["fabric"]["name"],
        num_neurons=num_neurons,
        v_init_fp=v_init_fp,
        v_th_fp=v_th_fp,
        alpha_fp=alpha_fp,
    )

    # Population offsets
    offsets = {}
    offset = 0
    for pop_name, pop_cfg in config["populations"].items():
        offsets[pop_name] = offset
        offset += pop_cfg["N"]

    # Export projections
    projections = [
        ("input_to_h0", "input", "hidden_0", model.W_input_h0),
        ("h0_to_h1", "hidden_0", "hidden_1", model.W_h0_h1),
        ("h1_to_h2", "hidden_1", "hidden_2", model.W_h1_h2),
        ("h2_recurrent", "hidden_2", "hidden_2", model.W_h2_h2),
        ("h2_to_h3", "hidden_2", "hidden_3", model.W_h2_h3),
        ("h3_to_output", "hidden_3", "output", model.W_h3_out),
    ]

    for name, pre, post, layer in projections:
        W = layer.weight.detach().cpu().T  # Transpose to [N_pre, N_post]
        row_ptr, col_idx, w_fp = dense_to_csr(W, threshold=sparsity_threshold)

        pre_start = offsets[pre]
        pre_end = pre_start + config["populations"][pre]["N"]
        post_start = offsets[post]
        post_end = post_start + config["populations"][post]["N"]

        fabric_data.add_projection(
            name=name,
            pre_start=pre_start,
            pre_end=pre_end,
            post_start=post_start,
            post_end=post_end,
            row_ptr=row_ptr,
            col_idx=col_idx,
            weights_fp=w_fp,
        )

    # Export
    files = export_kitten_fabric(fabric_data, output_dir / "fabric")

    print(f"[Export] Exported to {output_dir / 'fabric'}")
    return files


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train Tier-1 Safety Reflex SNN")

    # Teacher
    p.add_argument("--teacher-checkpoint", type=str, default=None,
                   help="Path to TF-A-N checkpoint for teacher")
    p.add_argument("--synthetic-teacher", action="store_true",
                   help="Use synthetic rule-based teacher")

    # Config
    p.add_argument("--fabric-config", type=str, required=True,
                   help="Path to fabric YAML config")

    # Output
    p.add_argument("--output-dir", type=str, required=True,
                   help="Output directory")

    # Training
    p.add_argument("--train-samples", type=int, default=50000,
                   help="Number of training samples")
    p.add_argument("--val-samples", type=int, default=5000,
                   help="Number of validation samples")
    p.add_argument("--epochs", type=int, default=50,
                   help="Training epochs")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size")
    p.add_argument("--lr", type=float, default=0.001,
                   help="Learning rate")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device (cpu/cuda)")

    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Tier-1 Safety Reflex SNN Training")
    print("=" * 80)

    # Load config
    print(f"\n[1] Loading fabric config from {args.fabric_config}...")
    with open(args.fabric_config, "r") as f:
        config = yaml.safe_load(f)
    print(f"    Fabric: {config['fabric']['name']}")

    # Initialize teacher
    print("\n[2] Initializing teacher...")
    if args.synthetic_teacher or args.teacher_checkpoint is None:
        teacher = SyntheticSafetyTeacher()
        print("    Using synthetic rule-based teacher")
    else:
        teacher = TFANSafetyTeacher(args.teacher_checkpoint, args.device)

    # Generate datasets
    print(f"\n[3] Generating datasets...")
    print(f"    Training samples: {args.train_samples}")
    print(f"    Validation samples: {args.val_samples}")

    train_data = teacher.generate_dataset(args.train_samples)
    val_data = teacher.generate_dataset(args.val_samples)

    # Count class distribution
    train_counts = [0, 0, 0, 0]
    for _, label in train_data:
        train_counts[label.action] += 1
    print(f"    Class distribution: SAFE={train_counts[0]}, WARN={train_counts[1]}, "
          f"BRAKE={train_counts[2]}, CUT={train_counts[3]}")

    train_dataset = SafetyReflexDataset(train_data, time_steps=config["fabric"]["time_steps"])
    val_dataset = SafetyReflexDataset(val_data, time_steps=config["fabric"]["time_steps"])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model
    print("\n[4] Initializing SNN model...")
    model = SafetyReflexSNN(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Total parameters: {total_params:,}")

    # Train
    print(f"\n[5] Training for {args.epochs} epochs...")
    history = train_safety_reflex(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        output_dir=output_dir,
    )

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Export to Kitten format
    print("\n[6] Exporting to Kitten FPGA format...")
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    export_to_kitten(model, config, output_dir)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nArtifacts in: {output_dir}/")
    print("  - best_model.pt       (trained PyTorch model)")
    print("  - history.json        (training metrics)")
    print("  - fabric/             (Kitten FPGA files)")
    print("    - neurons.bin")
    print("    - weights.bin")
    print("    - fabric_topology.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
