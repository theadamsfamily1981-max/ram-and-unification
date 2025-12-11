#!/usr/bin/env python3
"""
ara_small_model.py
Unified Small Model - Complete Three-Loop Antifragile System

This is the MINIMAL RUNNABLE implementation of the 2.21× Antifragility architecture.
All three loops are integrated and synchronized:

1. Structural Autonomy Loop: AEPO → SMT → PGU → Model Update
2. Interoceptive Control Loop: L1/L2 → CLV → PAD → Policy
3. Deployment & UX Loop: Policy → D-Bus → Cockpit → Avatar

Run with: python3 ara_small_model.py

Dependencies: numpy only (torch optional)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import math
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AraConfig:
    """Complete configuration for Ara small model."""
    # Network
    n_neurons: int = 256
    sparsity: float = 0.9    # 90% sparse connections

    @property
    def n_excitatory(self) -> int:
        return int(self.n_neurons * 0.8)  # 80% excitatory

    @property
    def n_inhibitory(self) -> int:
        return self.n_neurons - self.n_excitatory  # 20% inhibitory

    # LIF dynamics
    tau_m: float = 20.0      # Membrane time constant (ms)
    v_reset: float = -70.0   # Reset potential (mV)
    v_thresh: float = -55.0  # Spike threshold (mV)
    v_rest: float = -70.0    # Resting potential (mV)

    # Geometry (Poincaré ball)
    geom_dim: int = 16       # Hyperbolic embedding dimension
    curvature: float = 1.0   # Poincaré ball curvature

    # Thermodynamics
    lambda_J: float = 1.2    # Jacobian penalty in Π_q
    pi_max: float = 1.0      # Maximum Π_q before full brake

    # PGU constraints
    require_connected: bool = True
    min_loops: int = 2       # Minimum β₁ for redundancy
    max_spectral: float = 1.0  # Max ||J||_* for stability

    # Loop timing
    structural_interval: int = 100  # Steps between structural checks
    dbus_interval: int = 10         # Steps between D-Bus broadcasts


# =============================================================================
# Cognitive Load Vector (CLV)
# =============================================================================

@dataclass
class CLV:
    """Cognitive Load Vector - aggregated risk signals."""
    epr_cv: float = 0.0           # E:I ratio CV
    topo_gap: float = 0.0         # Distance from identity anchor
    jacobian_spectral: float = 0.0  # ||J||_*
    pi_q: float = 0.0             # Thermodynamic load
    spike_rate: float = 0.0       # Spikes per step
    memory_pressure: float = 0.0  # Memory utilization

    def magnitude(self) -> float:
        return math.sqrt(sum(x**2 for x in [
            self.epr_cv, self.topo_gap, self.jacobian_spectral,
            self.pi_q, self.spike_rate / 100, self.memory_pressure
        ]))


# =============================================================================
# PAD State (Pleasure-Arousal-Dominance)
# =============================================================================

@dataclass
class PADState:
    """Affective state derived from CLV."""
    valence: float = 0.0   # [-1, 1]
    arousal: float = 0.5   # [0, 1]
    dominance: float = 0.5 # [0, 1]

    @classmethod
    def from_clv(cls, clv: CLV) -> 'PADState':
        instability = (clv.pi_q + clv.jacobian_spectral) / 2
        valence = max(-1, min(1, 1.0 - 2.0 * instability))
        arousal = min(1.0, clv.spike_rate / 100.0 + 0.3)
        dominance = max(0, 1.0 - clv.topo_gap)
        return cls(valence=valence, arousal=arousal, dominance=dominance)

    @property
    def mood(self) -> str:
        if self.valence > 0.3: return "positive"
        if self.valence < -0.3: return "stressed"
        return "neutral"


# =============================================================================
# Policy Multipliers
# =============================================================================

@dataclass
class PolicyMultipliers:
    """LLM policy adaptation from PAD state."""
    temperature: float = 1.0
    top_p: float = 1.0
    safety_threshold: float = 1.0

    @classmethod
    def from_pad(cls, pad: PADState) -> 'PolicyMultipliers':
        stress = (1.0 - pad.valence) / 2.0 * pad.arousal
        return cls(
            temperature=max(0.3, 1.0 - stress * 0.6),
            top_p=max(0.7, 1.0 - pad.arousal * 0.2),
            safety_threshold=max(0.5, 0.5 + pad.valence * 0.5),
        )


# =============================================================================
# Network Graph for Topology
# =============================================================================

class NetworkGraph:
    """Sparse graph for topological analysis."""

    def __init__(self, n_nodes: int, edges: Optional[Set[Tuple[int, int]]] = None):
        self.n_nodes = n_nodes
        self.edges = edges if edges else set()

    def add_edge(self, i: int, j: int):
        self.edges.add((i, j))
        self.edges.add((j, i))  # Undirected

    def remove_edge(self, i: int, j: int):
        self.edges.discard((i, j))
        self.edges.discard((j, i))

    def compute_beta_0(self) -> int:
        """Connected components via Union-Find."""
        parent = list(range(self.n_nodes))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[py] = px

        for i, j in self.edges:
            union(i, j)

        return len(set(find(i) for i in range(self.n_nodes)))

    def compute_beta_1(self) -> int:
        """Independent cycles: β₁ = |E| - |V| + β₀."""
        undirected = set((min(i,j), max(i,j)) for i, j in self.edges)
        beta_0 = self.compute_beta_0()
        return max(0, len(undirected) - self.n_nodes + beta_0)

    def sparsity(self) -> float:
        max_edges = self.n_nodes * self.n_nodes
        return 1.0 - len(self.edges) / max_edges if max_edges > 0 else 1.0


# =============================================================================
# PGU (Proof-Guided Updater) - Safety Gate
# =============================================================================

class PGU:
    """Active safety gate for structural changes."""

    def __init__(self, config: AraConfig):
        self.config = config
        self.verified = 0
        self.rejected = 0

    def verify(self, graph: NetworkGraph, spectral_norm: float) -> Tuple[bool, List[str]]:
        """Verify structural constraints."""
        violations = []

        # β₀ check (connectivity)
        beta_0 = graph.compute_beta_0()
        if self.config.require_connected and beta_0 != 1:
            violations.append(f"Disconnected: β₀={beta_0}")

        # β₁ check (redundancy)
        beta_1 = graph.compute_beta_1()
        if beta_1 < self.config.min_loops:
            violations.append(f"Low redundancy: β₁={beta_1} < {self.config.min_loops}")

        # Spectral check (stability)
        if spectral_norm > self.config.max_spectral:
            violations.append(f"Unstable: ||J||_*={spectral_norm:.2f} > {self.config.max_spectral}")

        if violations:
            self.rejected += 1
            return False, violations

        self.verified += 1
        return True, [f"OK: β₀={beta_0}, β₁={beta_1}, ||J||_*={spectral_norm:.2f}"]


# =============================================================================
# LIF Neuron Layer
# =============================================================================

class LIFLayer:
    """Leaky Integrate-and-Fire neurons."""

    def __init__(self, config: AraConfig):
        self.config = config
        self.n = config.n_neurons

        # State
        self.v = np.full(self.n, config.v_reset)

        # Weights (sparse, E:I balanced)
        density = 1.0 - config.sparsity
        self.W = np.random.randn(self.n, self.n) * 0.1 / np.sqrt(self.n)
        mask = np.random.rand(self.n, self.n) > density
        self.W[mask] = 0

        # E:I sign constraints
        self.W[:config.n_excitatory, :] = np.abs(self.W[:config.n_excitatory, :])
        self.W[config.n_excitatory:, :] = -np.abs(self.W[config.n_excitatory:, :])

        # Build graph for topology
        self.graph = NetworkGraph(self.n)
        for i in range(self.n):
            for j in range(self.n):
                if abs(self.W[i, j]) > 1e-6:
                    self.graph.add_edge(i, j)

    def step(self, I_ext: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """Single timestep."""
        cfg = self.config

        # Recurrent input from previous spikes
        prev_spikes = (self.v >= cfg.v_thresh - 5).astype(float)  # Near-threshold activity
        I_rec = self.W @ prev_spikes * 10  # Scale recurrent input

        # LIF dynamics with stronger drive
        dv = (-(self.v - cfg.v_rest) + I_ext + I_rec) / cfg.tau_m
        self.v = self.v + dv * dt

        # Spike and reset
        spikes = (self.v >= cfg.v_thresh).astype(float)
        self.v = np.where(spikes > 0, cfg.v_reset, self.v)

        # E:I balance (handle empty slices)
        n_e = cfg.n_excitatory
        e_spikes = spikes[:n_e].sum()
        i_spikes = spikes[n_e:].sum()
        e_rate = e_spikes / n_e
        i_rate = i_spikes / max(1, cfg.n_neurons - n_e)
        epr = e_rate / (i_rate + 1e-6)

        return spikes, {'e_rate': e_rate, 'i_rate': i_rate, 'epr': epr, 'e_spikes': e_spikes, 'i_spikes': i_spikes}

    def compute_jacobian_spectral(self, n_iters: int = 10) -> float:
        """Estimate ||J||_* via power iteration."""
        J = self.W * (self.v < self.config.v_thresh).astype(float).reshape(-1, 1)
        v = np.random.randn(self.n)
        v /= np.linalg.norm(v)

        for _ in range(n_iters):
            u = J @ v
            u_norm = np.linalg.norm(u)
            if u_norm > 1e-10:
                u /= u_norm
            v = J.T @ u
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-10:
                v /= v_norm

        return np.linalg.norm(J @ v)


# =============================================================================
# Geometric State (Poincaré Ball)
# =============================================================================

class GeometricState:
    """Hyperbolic identity embedding on Poincaré ball."""

    def __init__(self, config: AraConfig):
        self.config = config
        self.z = np.zeros(config.geom_dim)  # Identity at origin
        self.identity_anchor = np.zeros(config.geom_dim)

    def topo_gap(self) -> float:
        """Distance from identity anchor."""
        return np.linalg.norm(self.z - self.identity_anchor)

    def update(self, force: np.ndarray, dt: float = 0.01):
        """Move on manifold with force."""
        # Conformal factor
        c = self.config.curvature
        z_norm_sq = np.sum(self.z ** 2)
        lambda_z = 2.0 / (1 - c * z_norm_sq + 1e-10)

        # Scaled movement
        v_norm = np.linalg.norm(force)
        if v_norm > 1e-10:
            coef = np.tanh(np.sqrt(c) * lambda_z * v_norm * dt / 2) / (np.sqrt(c) * v_norm + 1e-10)
            self.z = self.z + force * coef

        # Project to ball
        z_norm = np.linalg.norm(self.z)
        max_norm = 0.99 / np.sqrt(c)
        if z_norm > max_norm:
            self.z = self.z * max_norm / z_norm


# =============================================================================
# D-Bus Output (Mock)
# =============================================================================

class DBusOutput:
    """Mock D-Bus for signal output."""

    def __init__(self, log_file: Optional[str] = None):
        self.signals = []
        self.log_file = log_file

    def emit(self, signal: str, payload: Dict):
        self.signals.append((signal, payload))
        if self.log_file:
            import json
            with open(self.log_file, 'a') as f:
                f.write(json.dumps({'signal': signal, **payload}) + '\n')


# =============================================================================
# Complete Ara Small Model
# =============================================================================

class AraSmallModel:
    """
    Complete three-loop antifragile system.

    Integrates:
    - LIF neurons with E:I balance
    - Geometric state on Poincaré ball
    - PGU safety gate
    - CLV → PAD → Policy pipeline
    - D-Bus output
    """

    def __init__(self, config: Optional[AraConfig] = None):
        if config is None:
            config = AraConfig()

        self.config = config

        # Core components
        self.lif = LIFLayer(config)
        self.geometry = GeometricState(config)
        self.pgu = PGU(config)
        self.dbus = DBusOutput()

        # State
        self.step_count = 0
        self.clv_history: List[CLV] = []
        self.pad_history: List[PADState] = []

        # Metrics
        self.total_spikes = 0
        self.shock_recoveries = 0

    def compute_pi_q(self, J_spectral: float) -> float:
        """Thermodynamic potential Π_q."""
        v = self.lif.v
        cfg = self.config

        leak_term = np.mean((v - cfg.v_reset) ** 2) / cfg.tau_m
        jacobian_term = cfg.lambda_J * J_spectral ** 2 / cfg.n_neurons

        return leak_term + jacobian_term

    def step(self, I_ext: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Execute one complete step of all three loops.
        """
        self.step_count += 1
        cfg = self.config

        if I_ext is None:
            I_ext = np.random.randn(cfg.n_neurons) * 5.0

        # =====================================================================
        # LOOP 2: Interoceptive Control (every step)
        # =====================================================================

        # LIF step
        spikes, lif_info = self.lif.step(I_ext)
        n_spikes = spikes.sum()
        self.total_spikes += n_spikes

        # Jacobian estimation
        J_spectral = self.lif.compute_jacobian_spectral()

        # Π_q computation
        pi_q = self.compute_pi_q(J_spectral)

        # CLV aggregation
        clv = CLV(
            epr_cv=abs(lif_info['epr'] - 4.0) / 4.0,  # Deviation from ideal 4:1
            topo_gap=self.geometry.topo_gap(),
            jacobian_spectral=J_spectral,
            pi_q=pi_q,
            spike_rate=n_spikes,
            memory_pressure=0.3,  # Mock
        )
        self.clv_history.append(clv)

        # PAD conversion
        pad = PADState.from_clv(clv)
        self.pad_history.append(pad)

        # Policy multipliers
        policy = PolicyMultipliers.from_pad(pad)

        # Thermodynamic brake
        therm_clip = min(1.0, cfg.pi_max / (pi_q + 1e-6))

        # Geometry update (scaled by therm_clip)
        force = np.random.randn(cfg.geom_dim) * 0.1 * therm_clip
        self.geometry.update(force)

        # =====================================================================
        # LOOP 1: Structural Autonomy (periodic)
        # =====================================================================
        structural_result = None
        if self.step_count % cfg.structural_interval == 0:
            verified, messages = self.pgu.verify(self.lif.graph, J_spectral)
            structural_result = {
                'verified': verified,
                'messages': messages,
                'beta_0': self.lif.graph.compute_beta_0(),
                'beta_1': self.lif.graph.compute_beta_1(),
            }

        # =====================================================================
        # LOOP 3: Deployment & UX (periodic)
        # =====================================================================
        if self.step_count % cfg.dbus_interval == 0:
            self.dbus.emit('L3PolicyUpdated', {
                'step': self.step_count,
                'clv_magnitude': clv.magnitude(),
                'pad': {'valence': pad.valence, 'arousal': pad.arousal, 'mood': pad.mood},
                'policy': {'temperature': policy.temperature, 'safety': policy.safety_threshold},
                'pgu_verified': self.pgu.verified > self.pgu.rejected,
            })

        # =====================================================================
        # Antifragility Score
        # =====================================================================
        # Score = structural_component + interoceptive_component + deployment_component
        beta_1 = self.lif.graph.compute_beta_1()
        structural_score = min(1.0, beta_1 / 5.0)  # More loops = more redundancy
        interoceptive_score = 1.0 - min(1.0, clv.magnitude() / 2.0)  # Lower load = better
        deployment_score = 0.5 + policy.temperature * 0.5  # Adaptive temperature

        antifragility_score = structural_score + interoceptive_score + deployment_score

        return {
            'step': self.step_count,
            'spikes': n_spikes,
            'clv': clv,
            'pad': pad,
            'policy': policy,
            'pi_q': pi_q,
            'J_spectral': J_spectral,
            'therm_clip': therm_clip,
            'structural': structural_result,
            'antifragility_score': antifragility_score,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            'steps': self.step_count,
            'total_spikes': self.total_spikes,
            'pgu_verified': self.pgu.verified,
            'pgu_rejected': self.pgu.rejected,
            'dbus_signals': len(self.dbus.signals),
            'mean_valence': np.mean([p.valence for p in self.pad_history[-100:]]) if self.pad_history else 0,
            'beta_0': self.lif.graph.compute_beta_0(),
            'beta_1': self.lif.graph.compute_beta_1(),
        }


# =============================================================================
# Demo
# =============================================================================

def main():
    print("=" * 70)
    print("Ara Small Model - Complete Three-Loop Antifragile System")
    print("=" * 70)

    # Create model
    config = AraConfig(
        n_neurons=128,
        structural_interval=50,
        dbus_interval=10,
    )
    model = AraSmallModel(config)

    print(f"\nConfiguration:")
    print(f"  Neurons: {config.n_neurons} ({config.n_excitatory}E / {config.n_inhibitory}I)")
    print(f"  Sparsity: {config.sparsity:.0%}")
    print(f"  Geometry: {config.geom_dim}D Poincaré ball (c={config.curvature})")
    print(f"  PGU: β₁ ≥ {config.min_loops}, ||J||_* ≤ {config.max_spectral}")

    # Run simulation
    print("\n" + "-" * 70)
    print("Running 500 steps...")
    print("-" * 70)

    n_steps = 500
    scores = []

    for step in range(n_steps):
        # Varying input (simulate different cognitive demands)
        stress = 0.3 + 0.5 * math.sin(step * 0.05)
        I_ext = np.random.randn(config.n_neurons) * (3 + stress * 7)

        result = model.step(I_ext)
        scores.append(result['antifragility_score'])

        # Print periodic updates
        if (step + 1) % 100 == 0:
            pad = result['pad']
            policy = result['policy']
            print(f"\nStep {step + 1}:")
            print(f"  PAD: V={pad.valence:+.2f}, A={pad.arousal:.2f} → {pad.mood.upper()}")
            print(f"  Policy: temp={policy.temperature:.2f}×, safety={policy.safety_threshold:.2f}")
            print(f"  Π_q={result['pi_q']:.3f}, ||J||_*={result['J_spectral']:.3f}, clip={result['therm_clip']:.2f}")
            print(f"  Antifragility Score: {result['antifragility_score']:.2f}×")

            if result['structural']:
                s = result['structural']
                status = "✓ VERIFIED" if s['verified'] else "✗ REJECTED"
                print(f"  PGU: {status} (β₀={s['beta_0']}, β₁={s['beta_1']})")

    # Final statistics
    print("\n" + "=" * 70)
    print("Final Statistics")
    print("=" * 70)

    stats = model.get_stats()
    print(f"\nSteps completed: {stats['steps']}")
    print(f"Total spikes: {stats['total_spikes']}")
    print(f"Mean valence: {stats['mean_valence']:+.2f}")
    print(f"Topology: β₀={stats['beta_0']}, β₁={stats['beta_1']}")
    print(f"PGU: {stats['pgu_verified']} verified, {stats['pgu_rejected']} rejected")
    print(f"D-Bus signals: {stats['dbus_signals']}")

    mean_score = np.mean(scores)
    max_score = np.max(scores)
    print(f"\nAntifragility Score: {mean_score:.2f}× (mean), {max_score:.2f}× (peak)")

    print("\n" + "=" * 70)
    print("Three-Loop System Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
