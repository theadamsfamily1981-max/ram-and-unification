# TGSFN-Ara v0.1

**Thermodynamic-Geometric Spiking Field Network with Autonomous Reasoning Agent**

A minimal viable implementation combining:
- **TGSFN**: Spiking neural network operating near criticality
- **Ara**: Autonomous agent with hyperbolic identity and thermodynamic control

## Overview

This package implements the core components for a biologically-inspired cognitive architecture that maintains stability through thermodynamic constraints while operating at the edge of chaos (criticality).

### Key Features

- **LIF Spiking Network** with E:I balance (4:1 ratio)
- **Π_q Regularization** (entropy production proxy)
- **Avalanche Detection** and C-DP universality validation
- **Hyperbolic Identity Manifold** (Poincaré ball embedding)
- **L5 Control Law** (thermodynamically-constrained actions)
- **Homeostatic Regulation** (internal free energy minimization)
- **Dynamic Axiom Updater** (antifragility through self-repair)

## Installation

```bash
# Clone and install
cd tgsfn-ara
pip install -e .

# Or install dependencies directly
pip install torch numpy scipy geoopt  # geoopt optional
```

## Quick Start

### Train TGSFN on Synthetic Task

```bash
python experiments/train_tgsfn_mvp.py --epochs 50 --n_neurons 256
```

### Measure Avalanche Statistics

```bash
python experiments/measure_avalanches.py --timesteps 50000
```

### Run Ara Agent Loop

```bash
python experiments/ara_loop_toy_env.py --episodes 10 --verbose
```

## Package Structure

```
tgsfn-ara/
├── tgsfn_core/           # Core SNN components
│   ├── snn_model.py      # LIF layer and TGSFN network
│   ├── piq_loss.py       # Π_q and thermodynamic losses
│   ├── avalanches.py     # Avalanche detection
│   └── metrics.py        # Criticality metrics
├── ara_agent/            # Agent-level components
│   ├── identity_manifold.py  # Poincaré ball identity
│   ├── control_law.py    # L5 controller
│   ├── homeostasis.py    # Needs and regulation
│   └── dau.py            # Dynamic Axiom Updater
├── experiments/          # Runnable experiments
├── configs/              # YAML configurations
└── scripts/              # Shell scripts
```

## Core Concepts

### Criticality (C-DP Universality)

The network operates near the critical point of the Conserved Directed Percolation universality class, characterized by:

| Exponent | Symbol | Target Value |
|----------|--------|--------------|
| Size     | τ      | 1.5          |
| Duration | α      | 2.0          |
| Scaling  | γ_sT   | 2.0          |

Scaling relation: (τ-1)/(α-1) ≈ γ_sT

### Thermodynamic Regularization (Π_q)

Entropy production proxy:
```
Π_q = Σ (1 - r_i) * r_i / τ_mem + λ_J * ||J||
```

Where `||J||` is the Jacobian norm proxy for stability.

### L5 Control Law

Actions are constrained by identity manifold geometry and throttled by dissipation:
```
v*(t) = proj_{T_z M}(F_action) * min(1, Π_max / Π_q)
```

### Identity Manifold

Agent identity is embedded in a Poincaré ball (hyperbolic space):
- Distance from origin indicates identity coherence
- Geodesics define smooth identity transitions
- Hierarchical structure naturally encoded

## Configuration

See `configs/tgsfn_mvp.yaml` for full configuration options:

```yaml
network:
  n_neurons: 256
  ei_ratio: 0.8      # E:I = 4:1
  connectivity: 0.1

loss:
  lambda_homeo: 0.1  # Homeostatic weight
  lambda_diss: 0.01  # Dissipation weight

criticality:
  tau_target: 1.5
  alpha_target: 2.0
```

## Experiments

### 1. TGSFN Training (`train_tgsfn_mvp.py`)

Trains the network on temporal pattern classification while maintaining criticality through Π_q regularization.

**Key outputs:**
- Classification accuracy
- Branching ratio m
- Criticality exponents (τ, α, γ_sT)

### 2. Avalanche Measurement (`measure_avalanches.py`)

Runs spontaneous activity and validates criticality through avalanche statistics.

**Expected results:**
- τ ≈ 1.5 (with finite-size correction)
- α ≈ 2.0
- Scaling relation satisfied

### 3. Ara Loop (`ara_loop_toy_env.py`)

Full agent loop in toy grid world:
1. Observe environment
2. Update identity via hyperbolic encoding
3. Compute homeostatic needs
4. Generate L5-controlled action
5. Apply DAU corrections if needed

## Scientific References

- **Criticality**: Beggs & Plenz (2003), Friedman et al. (2012)
- **Thermodynamics**: Friston (2019), England (2013)
- **Hyperbolic embeddings**: Nickel & Kiela (2017)
- **Antifragility**: Taleb (2012)

## License

MIT License - See LICENSE file.

## Citation

```bibtex
@software{tgsfn_ara_2024,
  title={TGSFN-Ara: Thermodynamic-Geometric Spiking Field Network},
  author={TGSFN Team},
  year={2024},
  version={0.1.0}
}
```
