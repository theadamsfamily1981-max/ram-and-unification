#!/usr/bin/env python3
# experiments/train_tfan.py
# T-FAN Training Loop with Live Telemetry
#
# Trains the T-FAN model on synthetic data and emits metrics to ~/.tfan/metrics.json
# for the GNOME HUD to visualize.

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfan import (
    TFanCore,
    UDKController,
    CognitiveOffloadingSubsystem,
    TelemetryEmitter,
    create_tfan_core,
    create_udk_controller,
    create_cos,
)


# ============================================================================
#  Synthetic Data Generator
# ============================================================================

class SyntheticDataset:
    """
    Generates synthetic multimodal data for T-FAN training.

    In real use, replace with actual data loaders (medical imaging, robotics, etc.)
    """

    def __init__(
        self,
        num_samples: int = 1000,
        d_model: int = 512,
        num_modalities: int = 2,
        num_classes: int = 10,
        batch_size: int = 32,
    ):
        self.num_samples = num_samples
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.batch_size = batch_size

        # Pre-generate all data
        self.data = [
            [torch.randn(num_samples, d_model) for _ in range(num_modalities)],
            torch.randint(0, num_classes, (num_samples,)),
        ]

        self.num_batches = (num_samples + batch_size - 1) // batch_size
        self.current_batch = 0

    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        if self.current_batch >= self.num_batches:
            raise StopIteration

        start = self.current_batch * self.batch_size
        end = min(start + self.batch_size, self.num_samples)

        batch = {
            "modal_embeddings": [m[start:end] for m in self.data[0]],
            "labels": self.data[1][start:end],
        }

        self.current_batch += 1
        return batch

    def __len__(self):
        return self.num_batches


# ============================================================================
#  Training Loop
# ============================================================================

def train_epoch(
    model: TFanCore,
    udk: UDKController,
    cos: CognitiveOffloadingSubsystem,
    dataset: SyntheticDataset,
    optimizer: optim.Optimizer,
    telemetry: TelemetryEmitter,
    epoch: int,
    global_step: int,
    device: torch.device,
    log_every: int = 10,
    nce_check_every: int = 100,
) -> int:
    """
    Train for one epoch.

    Returns:
        Updated global_step
    """
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch in dataset:
        # Move to device
        batch["modal_embeddings"] = [m.to(device) for m in batch["modal_embeddings"]]
        batch["labels"] = batch["labels"].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch)
        losses = model.compute_losses(outputs, batch["labels"], udk)

        # Backward pass
        loss = losses["loss"]
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1
        global_step += 1

        # Emit telemetry
        if global_step % log_every == 0:
            udk_state = {
                "sigma_proxy": udk.state.sigma_proxy,
                "epsilon_proxy": udk.state.epsilon_proxy,
                "L_topo": udk.state.L_topo,
                "kappa_proxy": udk.state.kappa_proxy,
            }

            extra = {
                "epoch": epoch,
                "l_utility": losses["l_utility"].item(),
                "l_topo": losses["l_topo"].item(),
                "lambda_topo": losses["lambda_topo"],
                "utcf": udk.utcf_metrics_cost(),
            }

            telemetry.log_step(
                step=global_step,
                loss=loss.item(),
                udk_state=udk_state,
                extra=extra,
            )

        # Periodic NCE check
        if global_step % nce_check_every == 0:
            sigma = udk.state.sigma_proxy
            best_action = cos.get_best_action(sigma)

            if best_action and best_action["should_act"]:
                telemetry.log_nce_action(
                    step=global_step,
                    action_type=best_action["action_type"],
                    benefit=best_action["benefit"],
                    cost_ext=best_action["cost_ext"],
                    executed=True,
                )
                print(f"  [Step {global_step}] NCE: {best_action['action_type']} "
                      f"(benefit={best_action['benefit']:.1f} > cost={best_action['cost_ext']:.1f})")

    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    return global_step


def train(
    num_epochs: int = 100,
    d_model: int = 512,
    num_modalities: int = 2,
    num_classes: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_samples: int = 1000,
    device_name: str = "cuda",
    log_every: int = 10,
    nce_check_every: int = 100,
):
    """
    Main training function.
    """
    # Device setup
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # Create components
    print("Creating T-FAN model...")
    model = create_tfan_core(
        d_model=d_model,
        num_modalities=num_modalities,
        num_classes=num_classes,
    ).to(device)

    udk = create_udk_controller()
    cos = create_cos(horizon_steps=5000)
    telemetry = TelemetryEmitter()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Dataset
    print("Creating synthetic dataset...")
    dataset = SyntheticDataset(
        num_samples=num_samples,
        d_model=d_model,
        num_modalities=num_modalities,
        num_classes=num_classes,
        batch_size=batch_size,
    )

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Telemetry → {telemetry.metrics_file}")
    print("=" * 60)

    global_step = 0

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        global_step = train_epoch(
            model=model,
            udk=udk,
            cos=cos,
            dataset=dataset,
            optimizer=optimizer,
            telemetry=telemetry,
            epoch=epoch,
            global_step=global_step,
            device=device,
            log_every=log_every,
            nce_check_every=nce_check_every,
        )

        epoch_time = time.time() - epoch_start

        # Get current metrics
        diag = udk.get_diagnostics()

        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Step {global_step:5d} | "
              f"UTCF {diag['utcf']:.4f} | "
              f"σ {diag['sigma_proxy']:.4f} | "
              f"ε {diag['epsilon_proxy']:.4f} | "
              f"λ_topo {diag['lambda_topo']:.4f} | "
              f"{epoch_time:.1f}s")

    print("=" * 60)
    print("Training complete!")
    print(f"Final metrics saved to: {telemetry.metrics_file}")

    # Final summary
    print("\nFinal UDK State:")
    for k, v in udk.get_diagnostics().items():
        print(f"  {k}: {v:.6f}")

    return model, udk, cos


# ============================================================================
#  CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train T-FAN with live telemetry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num-modalities", type=int, default=2, help="Number of modalities")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-samples", type=int, default=1000, help="Training samples")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--log-every", type=int, default=10, help="Log telemetry every N steps")
    parser.add_argument("--nce-every", type=int, default=100, help="NCE check every N steps")

    args = parser.parse_args()

    train(
        num_epochs=args.epochs,
        d_model=args.d_model,
        num_modalities=args.num_modalities,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_samples=args.num_samples,
        device_name=args.device,
        log_every=args.log_every,
        nce_check_every=args.nce_every,
    )


if __name__ == "__main__":
    main()
