#!/usr/bin/env python3
"""
train_tgsfn_mvp.py
Minimal viable TGSFN training script

Trains a TGSFN network on a simple task (pattern classification)
while maintaining criticality through Pi_q regularization.

Usage:
    python train_tgsfn_mvp.py --config configs/tgsfn_mvp.yaml
    python train_tgsfn_mvp.py --n_neurons 256 --epochs 100
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tgsfn_core import LIFLayer, TGSFNNetwork, tgsfn_loss, compute_piq
from tgsfn_core.metrics import CriticalityMonitor, compute_branching_ratio


def create_synthetic_dataset(
    n_samples: int = 1000,
    seq_len: int = 100,
    input_dim: int = 64,
    n_classes: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic temporal classification dataset.

    Each class has a distinct temporal pattern.
    """
    X = torch.zeros(n_samples, seq_len, input_dim)
    y = torch.zeros(n_samples, dtype=torch.long)

    for i in range(n_samples):
        label = i % n_classes
        y[i] = label

        # Create class-specific patterns
        if label == 0:
            # Increasing activity
            for t in range(seq_len):
                n_active = min(input_dim, 5 + t // 10)
                X[i, t, :n_active] = 1.0
        elif label == 1:
            # Decreasing activity
            for t in range(seq_len):
                n_active = max(1, input_dim - 5 - t // 10)
                X[i, t, :n_active] = 1.0
        elif label == 2:
            # Oscillating activity
            for t in range(seq_len):
                n_active = 10 + int(20 * torch.sin(torch.tensor(t * 0.1)))
                X[i, t, :max(1, n_active)] = 1.0
        else:
            # Random bursts
            burst_times = torch.randint(0, seq_len, (10,))
            for bt in burst_times:
                if bt < seq_len:
                    X[i, bt, :input_dim//2] = 1.0

    return X, y


def train_epoch(
    model: TGSFNNetwork,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    criticality_monitor: CriticalityMonitor,
    lambda_homeo: float = 0.1,
    lambda_diss: float = 0.01,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Train for one epoch.

    Returns:
        Dict with loss, accuracy, and criticality metrics
    """
    model.train()

    total_loss = 0.0
    total_task_loss = 0.0
    total_piq = 0.0
    correct = 0
    total = 0

    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # Reset network state
        model.reset_state()

        # Forward pass
        outputs = []
        spikes_list = []

        for t in range(X.shape[1]):
            out, spikes = model(X[:, t, :])
            outputs.append(out)
            spikes_list.append(spikes)

            # Update criticality monitor
            R = spikes.sum(dim=-1)
            criticality_monitor.update(R)

        # Stack outputs and compute logits
        outputs = torch.stack(outputs, dim=1)  # (B, T, n_classes)
        logits = outputs.mean(dim=1)  # Average over time

        # Task loss
        task_loss = criterion(logits, y)

        # Stack spikes for Piq computation
        spikes_tensor = torch.stack(spikes_list, dim=1)  # (B, T, N)

        # Get network state for free energy
        rate_history = spikes_tensor.float().mean(dim=1)  # (B, N)
        target_rate = 0.1 * torch.ones_like(rate_history)

        # Compute Piq
        J_proxy = model.compute_jacobian_proxy()
        piq = compute_piq(spikes_tensor, J_proxy)

        # Homeostatic term
        homeo_loss = ((rate_history - target_rate) ** 2).sum(dim=-1).mean()

        # Total loss
        loss = task_loss + lambda_homeo * homeo_loss + lambda_diss * piq

        # Backward and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_piq += piq.item()

        _, predicted = logits.max(dim=1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    n_batches = len(dataloader)

    return {
        "loss": total_loss / n_batches,
        "task_loss": total_task_loss / n_batches,
        "piq": total_piq / n_batches,
        "accuracy": 100.0 * correct / total,
        "branching_ratio": criticality_monitor.get_branching_ratio(),
    }


def evaluate(
    model: TGSFNNetwork,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate model on dataset."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            model.reset_state()

            outputs = []
            for t in range(X.shape[1]):
                out, _ = model(X[:, t, :])
                outputs.append(out)

            outputs = torch.stack(outputs, dim=1)
            logits = outputs.mean(dim=1)

            loss = criterion(logits, y)
            total_loss += loss.item()

            _, predicted = logits.max(dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": 100.0 * correct / total,
    }


def main():
    parser = argparse.ArgumentParser(description="Train TGSFN MVP")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--n_neurons", type=int, default=256)
    parser.add_argument("--input_dim", type=int, default=64)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_homeo", type=float, default=0.1)
    parser.add_argument("--lambda_diss", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("TGSFN MVP Training")
    print("=" * 60)
    print(f"Neurons: {args.n_neurons}")
    print(f"Input dim: {args.input_dim}")
    print(f"Classes: {args.n_classes}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Create dataset
    print("\nCreating synthetic dataset...")
    X_train, y_train = create_synthetic_dataset(
        n_samples=800,
        seq_len=args.seq_len,
        input_dim=args.input_dim,
        n_classes=args.n_classes,
    )
    X_val, y_val = create_synthetic_dataset(
        n_samples=200,
        seq_len=args.seq_len,
        input_dim=args.input_dim,
        n_classes=args.n_classes,
    )

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create model
    print("\nCreating TGSFN network...")
    model = TGSFNNetwork(
        input_dim=args.input_dim,
        n_neurons=args.n_neurons,
        output_dim=args.n_classes,
        ei_ratio=0.8,  # 80% excitatory
        connectivity=0.1,
    ).to(args.device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Create optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Criticality monitor
    criticality_monitor = CriticalityMonitor(
        buffer_size=10000,
        update_interval=1000,
    )

    # Training loop
    print("\nStarting training...")
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            criticality_monitor=criticality_monitor,
            lambda_homeo=args.lambda_homeo,
            lambda_diss=args.lambda_diss,
            device=args.device,
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, args.device)

        # Print progress
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.1f}% | "
              f"Val Acc: {val_metrics['accuracy']:.1f}% | "
              f"Piq: {train_metrics['piq']:.4f} | "
              f"m: {train_metrics['branching_ratio']:.3f}")

        # Track best
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']

    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Final branching ratio: {train_metrics['branching_ratio']:.4f}")

    # Get criticality metrics
    crit_metrics = criticality_monitor.get_metrics()
    print(f"Criticality metrics:")
    print(f"  tau: {crit_metrics.tau:.3f}")
    print(f"  alpha: {crit_metrics.alpha:.3f}")
    print(f"  gamma_sT: {crit_metrics.gamma_sT:.3f}")
    print(f"  is_critical: {crit_metrics.is_critical}")
    print("=" * 60)


if __name__ == "__main__":
    main()
