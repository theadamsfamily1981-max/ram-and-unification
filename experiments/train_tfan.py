#!/usr/bin/env python3
# experiments/train_tfan.py
# T-FAN Training Loop with Live Telemetry
#
# Trains the T-FAN model on synthetic data and emits metrics to
# XDG_RUNTIME_DIR/tfan_hud_metrics.json for the GNOME Cockpit to visualize.
#
# Usage:
#   python experiments/train_tfan.py --epochs 100 --device cuda
#
# Then in another terminal:
#   python hud/tfan_gnome.py

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
    TFANHudMetricsClient,
    TfanBrainSnapshot,
    TelemetryExporter,
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
    hud: TFANHudMetricsClient,
    brain_exporter: TelemetryExporter,
    nce_actions: List[Dict[str, Any]],
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

        # Compute gradient norm before step
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            preds = outputs["logits"].argmax(dim=-1)
            accuracy = (preds == batch["labels"]).float().mean().item()

        epoch_loss += loss.item()
        num_batches += 1
        global_step += 1

        # Extract Betti features
        beta_k = outputs["topology"]["beta_k"]
        beta0 = float(beta_k[0].item()) if len(beta_k) > 0 else 0.0
        beta1 = float(beta_k[1].item()) if len(beta_k) > 1 else 0.0

        # Get current UTCF and update homeostatic core
        utcf = float(udk.utcf_metrics_cost())
        udk.update_homeostatic_core(
            loss=loss.item(),
            accuracy=accuracy,
            grad_norm=grad_norm,
        )

        # Get homeostatic state
        homeo_state = udk.homeostatic_core.get_state()

        # Emit telemetry to GNOME cockpit (simple HUD)
        if global_step % log_every == 0:
            hud.update(
                # Basic training metrics
                step=global_step,
                epoch=epoch,
                loss=loss.item(),
                accuracy=accuracy,
                # UDK proxies (T-FAN brain state)
                sigma_proxy=float(udk.state.sigma_proxy),
                epsilon_proxy=float(udk.state.epsilon_proxy),
                kappa_proxy=float(udk.state.kappa_proxy),
                L_topo=float(udk.state.L_topo),
                # Betti numbers
                beta0=beta0,
                beta1=beta1,
                # Costs and weights
                utcf=utcf,
                lambda_topo=float(losses["lambda_topo"]),
                l_utility=float(losses["l_utility"].item()),
                l_topo=float(losses["l_topo"].item()),
                # Homeostatic core (Layer 1)
                drive_total=homeo_state["drive_total"],
                n_energy=homeo_state["n_energy"],
                n_integrity=homeo_state["n_integrity"],
                n_cogload=homeo_state["n_cogload"],
                n_social=homeo_state["n_social"],
                n_novelty=homeo_state["n_novelty"],
                n_safety=homeo_state["n_safety"],
                valence=homeo_state["valence"],
                # NCE actions (pass the whole list)
                nce_actions=nce_actions[-10:],  # Keep last 10
            )

            # Also push full brain snapshot for detailed telemetry
            snap = TfanBrainSnapshot(
                step=global_step,
                epoch=epoch,
                wall_time=time.time(),
                phase="train",
                # Losses
                loss_utility=float(losses["l_utility"].item()),
                loss_topo=float(losses["l_topo"].item()),
                loss_total=loss.item(),
                accuracy=accuracy,
                # Layer 1: Homeostatic Core
                drive_total=homeo_state["drive_total"],
                n_energy=homeo_state["n_energy"],
                n_integrity=homeo_state["n_integrity"],
                n_cogload=homeo_state["n_cogload"],
                n_social=homeo_state["n_social"],
                n_novelty=homeo_state["n_novelty"],
                n_safety=homeo_state["n_safety"],
                # Layer 2: TFF Topology
                beta0=beta0,
                beta1=beta1,
                kappa_proxy=float(udk.state.kappa_proxy),
                lambda_topo=float(losses["lambda_topo"]),
                # Layer 3: UDK Thermodynamics
                sigma_proxy=float(udk.state.sigma_proxy),
                epsilon_proxy=float(udk.state.epsilon_proxy),
                utcf=utcf,
                # Layer 4: NCE/COS
                offload_actions_total=len(nce_actions),
                offload_actions_executed=sum(1 for a in nce_actions if a.get("executed")),
                last_offload_action=nce_actions[-1]["action_type"] if nce_actions else "none",
                # Affective state
                valence=homeo_state["valence"],
                arousal=abs(utcf),
                # Training metadata
                lr=optimizer.param_groups[0]["lr"],
                grad_norm=grad_norm,
                batch_size=batch["labels"].size(0),
            )
            brain_exporter.push(snap)

        # Periodic NCE check
        if global_step % nce_check_every == 0:
            sigma = udk.state.sigma_proxy
            best_action = cos.get_best_action(sigma)

            if best_action:
                action_record = {
                    "step": global_step,
                    "action_type": best_action["action_type"],
                    "benefit": best_action["benefit"],
                    "cost_ext": best_action["cost_ext"],
                    "should_act": best_action["should_act"],
                    "executed": best_action["should_act"],
                }
                nce_actions.append(action_record)

                if best_action["should_act"]:
                    print(f"  [Step {global_step}] NCE: {best_action['action_type']} "
                          f"(benefit={best_action['benefit']:.1f} > cost={best_action['cost_ext']:.1f})")

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

    # HUD metrics client (writes to XDG_RUNTIME_DIR/tfan_hud_metrics.json)
    hud = TFANHudMetricsClient(model_id="tfan_main")

    # Full brain telemetry exporter (writes to XDG_RUNTIME_DIR/tfan_brain_metrics.json)
    brain_exporter = TelemetryExporter()

    # NCE action history (shared list that gets updated)
    nce_actions: List[Dict[str, Any]] = []

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
    print(f"HUD Metrics  → {hud.path}")
    print(f"Brain Telemetry → {brain_exporter.path}")
    print("Launch the cockpit: python hud/tfan_gnome.py")
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
            hud=hud,
            brain_exporter=brain_exporter,
            nce_actions=nce_actions,
            epoch=epoch,
            global_step=global_step,
            device=device,
            log_every=log_every,
            nce_check_every=nce_check_every,
        )

        epoch_time = time.time() - epoch_start

        # Get current metrics (includes Layer 1 homeostatic state)
        diag = udk.get_diagnostics()

        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Step {global_step:5d} | "
              f"UTCF {diag['utcf']:.4f} | "
              f"Drive {diag['drive_total']:.3f} | "
              f"σ {diag['sigma_proxy']:.4f} | "
              f"ε {diag['epsilon_proxy']:.4f} | "
              f"{epoch_time:.1f}s")

    print("=" * 60)
    print("Training complete!")

    # Final summary
    print("\nFinal UDK State:")
    for k, v in udk.get_diagnostics().items():
        print(f"  {k}: {v:.6f}")

    print(f"\nNCE Actions taken: {sum(1 for a in nce_actions if a.get('executed', False))}/{len(nce_actions)}")

    # Cleanup
    hud.close()

    return model, udk, cos


# ============================================================================
#  CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train T-FAN with live telemetry to GNOME Cockpit",
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
