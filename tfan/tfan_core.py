# tfan_core.py
# Consolidated T-FAN Core Skeleton (TFF + UDK + NCE/COS)
# Drop-in single file for Kitten/3090 bootstrap
#
# This is the "clean slate" version with clear TODO hooks.
# Can run standalone or alongside the detailed src/tff/ modules.

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
#  TFF: TopFusion Encoder
# ============================================================================

class TopFusionEncoder(nn.Module):
    """
    Transforms multi-modal embeddings (Text, Image, Sensor) into a single
    structurally coherent latent feature Z_fused using an MCCA-like alignment.

    TODO:
      - Replace the simple Linear aligners with a differentiable MCCA block.
      - Optionally plug in existing CCA/MCCA libs and wrap as a torch module.
    """

    def __init__(
        self,
        d_model: int,
        num_modalities: int,
        mcca_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.mcca_config = mcca_config or {}

        # Placeholder for learned "alignment" per modality
        self.mcca_aligners = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_modalities)
        ])

        # Simple fusion head: concat -> projection
        self.post_proj = nn.Sequential(
            nn.Linear(d_model * num_modalities, d_model),
            nn.GELU(),
        )

    def forward(self, modal_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modal_embeddings: list of tensors [B x D_i], assumed D_i ≈ d_model

        Returns:
            z_fused: [B x d_model]
        """
        assert len(modal_embeddings) == self.num_modalities, \
            f"Expected {self.num_modalities} modalities, got {len(modal_embeddings)}"

        aligned = [
            aligner(emb) for aligner, emb in zip(self.mcca_aligners, modal_embeddings)
        ]  # list of [B x d_model]

        concat = torch.cat(aligned, dim=-1)  # [B x (d_model * num_modalities)]
        z_fused = self.post_proj(concat)     # [B x d_model]
        return z_fused


# ============================================================================
#  TFF: Topology Head
# ============================================================================

class TopologyHead(nn.Module):
    """
    Computes topological invariants (Betti numbers etc.) and a topological
    embedding from the latent space.

    In this skeleton:
      - Topology is *stubbed* with simple statistics over z_fused.
      - Shape and API are correct so a real PH backend (giotto-tda, ripser, etc.)
        can be dropped in later.

    TODO:
      - Replace `_compute_betti_features_stub` with real persistent homology.
      - Replace topo_embedding MLP if you want true persistence images.
    """

    def __init__(
        self,
        d_model: int,
        ph_config: Optional[Dict[str, Any]] = None,
        betti_dims: Tuple[int, ...] = (0, 1),
        topo_embed_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.ph_config = ph_config or {}
        self.betti_dims = betti_dims
        self.topo_embed_dim = topo_embed_dim

        # Simple MLP over z_fused as a placeholder "persistence image" embedding
        self.topo_mlp = nn.Sequential(
            nn.Linear(d_model, topo_embed_dim),
            nn.GELU(),
            nn.Linear(topo_embed_dim, topo_embed_dim),
        )

        # Placeholder for external PH engine
        # TODO: plug in giotto-tda / ripser / your PH engine
        self.ph_engine = None

    def _compute_persistence_diagrams(self, z_np: np.ndarray) -> Any:
        """
        TODO: Real PH engine: take z_np [B x D] and return persistence diagrams.
        For now, we just return None as a stub.
        """
        _ = z_np
        return None

    def _compute_betti_features_stub(self, z_np: np.ndarray) -> np.ndarray:
        """
        Stub: use simple stats as cheap Betti-like features per batch.
        Real version should derive Betti numbers from persistence diagrams.
        """
        # z_np: [B x D]
        K = len(self.betti_dims)

        # For now: use mean |z| per batch as a scalar, broadcast to K dims
        mean_abs = np.abs(z_np).mean()
        beta_vec = np.full((K,), mean_abs, dtype=np.float32)
        return beta_vec  # [K]

    def forward(self, z_fused: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            z_fused: [B x d_model]

        Returns:
            {
              "beta_k":        [K]       (batch-level Betti-like summary),
              "topo_embedding":[B x E],  (per-sample topo embedding),
              "diagrams":      Any or None
            }
        """
        z_np = z_fused.detach().cpu().numpy()

        # 1) Compute persistence diagrams (stubbed)
        diagrams = self._compute_persistence_diagrams(z_np)

        # 2) Compute Betti features (stubbed)
        beta_k_vec = self._compute_betti_features_stub(z_np)  # [K]
        beta_k = torch.from_numpy(beta_k_vec).to(z_fused.device)  # [K]

        # 3) Compute topo embedding from z_fused (placeholder for PI encoder)
        topo_embedding = self.topo_mlp(z_fused)  # [B x topo_embed_dim]

        return {
            "beta_k": beta_k,
            "topo_embedding": topo_embedding,
            "diagrams": diagrams,
        }


# ============================================================================
#  TFF: Topological Regularizer
# ============================================================================

class TopologicalRegularizer(nn.Module):
    """
    Implements the Stability Constraint Function L_topo based on Betti number drift.

    L_topo = || mean(Beta_k_batch) - EMA(Beta_k_mean) ||^2

    NOTE:
      In this skeleton, beta_k is a *vector* [K], not [B x K].
      We treat it as a batch summary.
    """

    def __init__(self, ema_momentum: float = 0.99):
        super().__init__()
        self.ema_momentum = ema_momentum
        self.register_buffer("ema_beta_mean", None)

    def forward(self, beta_k: torch.Tensor) -> torch.Tensor:
        """
        Args:
            beta_k: [K] batch-level Betti features vector

        Returns:
            L_topo: scalar torch.Tensor
        """
        if self.ema_beta_mean is None:
            self.ema_beta_mean = beta_k.detach().clone()
            return torch.tensor(0.0, device=beta_k.device)

        diff = beta_k - self.ema_beta_mean
        l_topo = (diff ** 2).sum()

        # EMA update (no grad)
        with torch.no_grad():
            self.ema_beta_mean = (
                self.ema_momentum * self.ema_beta_mean
                + (1.0 - self.ema_momentum) * beta_k
            )

        return l_topo


# ============================================================================
#  UDK: Dissipation Monitor (epsilon_proxy)
# ============================================================================

class DissipationMonitor:
    """
    Tracks the Temporal Weight Variance (Epsilon Proxy) of a weight tensor.

    epsilon_proxy = Var_t(||Delta W_t||_2)
    """

    def __init__(self, window: int = 100):
        self.window = window
        self.delta_norms: deque = deque(maxlen=window)
        self.prev_weights: Optional[torch.Tensor] = None

    def update(self, current_weights: torch.Tensor) -> None:
        if self.prev_weights is not None:
            delta = current_weights - self.prev_weights
            delta_norm = delta.norm().item()
            self.delta_norms.append(delta_norm)

        self.prev_weights = current_weights.detach().clone()

    def epsilon_proxy(self) -> float:
        if len(self.delta_norms) < 2:
            return 0.0
        arr = np.array(list(self.delta_norms), dtype=np.float32)
        return float(arr.var())

    def mean_delta(self) -> float:
        if len(self.delta_norms) < 1:
            return 0.0
        return float(sum(self.delta_norms) / len(self.delta_norms))


# ============================================================================
#  UDK: Controller
# ============================================================================

@dataclass
class UDKState:
    """Current state of all UDK proxies."""
    sigma_proxy: float = 0.0
    epsilon_proxy: float = 0.0
    L_topo: float = 0.0
    kappa_proxy: float = 0.0


class UDKController:
    """
    Governs T-FAN's global cost landscape via the Unified Topo-Thermodynamic
    Cost Function (UTCF).

    UTCF = α_σ·σ + α_ε·ε + α_topo·L_topo + α_κ·κ

    Tracks:
        - sigma_proxy: entropy production / mismatch cost
        - epsilon_proxy: dissipation rate / weight turbulence
        - L_topo: topological stability penalty
        - kappa_proxy: manifold curvature proxy (FIM max eigenvalue)

    Governance policies:
        - Dynamic λ_topo based on instability
        - Optimizer dampening when ε is high
    """

    def __init__(
        self,
        alpha_sigma: float = 1.0,
        alpha_eps: float = 1.0,
        alpha_topo: float = 1.0,
        alpha_kappa: float = 1.0,
        lambda_topo_base: float = 0.1,
        epsilon_limit: float = 0.5,
    ):
        self.alpha_sigma = alpha_sigma
        self.alpha_eps = alpha_eps
        self.alpha_topo = alpha_topo
        self.alpha_kappa = alpha_kappa
        self.lambda_topo_base = lambda_topo_base
        self.epsilon_limit = epsilon_limit
        self.eps = 1e-6

        self.state = UDKState()
        self.dissipation_monitor = DissipationMonitor(window=100)

        # History for diagnostics
        self.utcf_history: deque = deque(maxlen=100)

    # ----- sigma (entropy production) ------------------------

    def compute_sigma_proxy(self, cost_R: float, precision_gain: float) -> float:
        """
        sigma_proxy = log(1 + Cost(R)) / (eps + Precision_Gain)

        Args:
            cost_R: FLOPs/time or other complexity measure of DAC update
            precision_gain: e.g., reduction in DAC mismatch loss
        """
        return math.log1p(cost_R) / (self.eps + precision_gain)

    def update_dac_metrics(self, cost_R: float, precision_gain: float) -> None:
        """Update sigma from DAC belief revision."""
        sigma = self.compute_sigma_proxy(cost_R, precision_gain)
        self.state.sigma_proxy = sigma

    # ----- epsilon / topo / kappa ----------------------------

    def update_tff_metrics(
        self,
        model: nn.Module,
        l_topo: float,
        kappa_proxy: Optional[float] = None,
    ) -> None:
        """
        Called by TFF training loop each step.

        Args:
            model: model containing top_fusion_encoder.post_proj weights
            l_topo: scalar topological loss
            kappa_proxy: optional curvature estimate
        """
        # 1) epsilon proxy - track fusion layer weights
        if hasattr(model, "top_fusion_encoder"):
            tff_weights = model.top_fusion_encoder.post_proj[0].weight
            self.dissipation_monitor.update(tff_weights)
            self.state.epsilon_proxy = self.dissipation_monitor.epsilon_proxy()

        # 2) L_topo
        self.state.L_topo = float(l_topo)

        # 3) kappa
        if kappa_proxy is not None:
            self.state.kappa_proxy = float(kappa_proxy)

    # ----- UTCF core + policies ------------------------------

    def utcf_metrics_cost(self) -> float:
        """Compute UTCF = α_σ·σ + α_ε·ε + α_topo·L_topo + α_κ·κ"""
        s = self.state
        cost = (
            self.alpha_sigma * s.sigma_proxy
            + self.alpha_eps * s.epsilon_proxy
            + self.alpha_topo * s.L_topo
            + self.alpha_kappa * s.kappa_proxy
        )
        self.utcf_history.append(cost)
        return cost

    def get_lambda_topo(self) -> float:
        """
        Dynamically scale the topological regularization coefficient based
        on structural instability (L_topo and kappa_proxy).

        λ_topo = base * (1 + L_topo * 0.1 + κ * 0.5)
        """
        s = self.state
        instability_factor = 1.0 + (s.L_topo * 0.1) + (s.kappa_proxy * 0.5)
        return max(0.01, self.lambda_topo_base * instability_factor)

    def adjust_optimizer_config(self, optimizer_config: Dict[str, float]) -> Dict[str, float]:
        """
        Dampen optimization when epsilon (turbulence) is high.
        """
        epsilon = self.state.epsilon_proxy

        if epsilon > self.epsilon_limit:
            optimizer_config["lr"] = optimizer_config.get("lr", 1e-3) * 0.8
            momentum = optimizer_config.get("momentum", 0.9)
            optimizer_config["momentum"] = min(0.95, momentum + 0.05)

        return optimizer_config

    def get_diagnostics(self) -> Dict[str, float]:
        """Return current state and metrics for logging."""
        return {
            "sigma_proxy": self.state.sigma_proxy,
            "epsilon_proxy": self.state.epsilon_proxy,
            "L_topo": self.state.L_topo,
            "kappa_proxy": self.state.kappa_proxy,
            "lambda_topo": self.get_lambda_topo(),
            "utcf": self.utcf_metrics_cost(),
            "utcf_mean": sum(self.utcf_history) / len(self.utcf_history) if self.utcf_history else 0.0,
        }


# ============================================================================
#  UDK: Curvature Approximation (kappa_proxy)
# ============================================================================

def estimate_fim_eigenvalue(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    n_power_iter: int = 3,
) -> float:
    """
    Estimates the largest eigenvalue of the Fisher Information Matrix (FIM)
    via a crude power iteration proxy tied to the current loss scale.

    TODO:
      - Replace with a true FIM-based power iteration (grad of log_prob).
    """
    # Simple example: compute CE loss and tie curvature proxy to it.
    if "logits" in batch:
        logits = batch["logits"]
    else:
        out = model(batch)
        logits = out.get("logits", out.get("z_fused"))

    labels = batch.get("labels")
    if labels is None:
        return 0.1  # Fallback

    loss = F.cross_entropy(logits, labels, reduction="mean")

    # Real implementation: would do v <- FIM v with power iteration
    # and compute Rayleigh quotient. This is a crude proxy.
    return float(loss.item() * 0.1)


# ============================================================================
#  NCE / COS: Cognitive Offloading Subsystem
# ============================================================================

@dataclass
class NicheAction:
    """A candidate niche construction action."""
    action_type: str
    cost_ext: float = 10.0
    sigma_reduction_factor: float = 1.0  # 0.7 = 30% reduction


DEFAULT_NICHE_ACTIONS = {
    "deploy_sensor": NicheAction("deploy_sensor", cost_ext=50.0, sigma_reduction_factor=0.7),
    "sidecar_logging": NicheAction("sidecar_logging", cost_ext=5.0, sigma_reduction_factor=0.9),
    "spawn_process": NicheAction("spawn_process", cost_ext=10.0, sigma_reduction_factor=0.85),
    "external_storage": NicheAction("external_storage", cost_ext=3.0, sigma_reduction_factor=0.95),
    "api_call": NicheAction("api_call", cost_ext=2.0, sigma_reduction_factor=0.92),
}


class CognitiveOffloadingSubsystem:
    """
    Implements the Niche Construction decision policy:

        Do action a  <=>  Δσ * H  >  Cost_ext(a)

    Where:
      - H is a planning horizon (number of future steps we care about)
      - Δσ = current_sigma - predicted_sigma_after (sigma reduction)
    """

    def __init__(
        self,
        horizon_steps: int = 5000,
        niche_actions: Optional[Dict[str, NicheAction]] = None,
    ):
        self.horizon_steps = horizon_steps
        self.niche_actions = niche_actions or DEFAULT_NICHE_ACTIONS
        self.action_log: deque = deque(maxlen=100)

    def estimate_external_cost(self, action_type: str) -> float:
        action = self.niche_actions.get(action_type)
        return action.cost_ext if action else 10.0

    def predict_sigma_after(self, action_type: str, current_sigma: float) -> float:
        """
        Predict expected σ after executing action.

        TODO: Replace heuristic with a learned regressor over past offload episodes.
        """
        action = self.niche_actions.get(action_type)
        if action:
            return current_sigma * action.sigma_reduction_factor
        return current_sigma

    def evaluate_action(
        self,
        action_type: str,
        current_sigma: float,
    ) -> Dict[str, Any]:
        """
        Evaluate whether an action is worth taking.

        Returns:
            Dict with should_act, benefit, cost_ext, etc.
        """
        predicted_sigma_after = self.predict_sigma_after(action_type, current_sigma)
        delta_sigma = current_sigma - predicted_sigma_after
        cost_ext = self.estimate_external_cost(action_type)

        benefit = delta_sigma * self.horizon_steps
        should_act = benefit > cost_ext

        return {
            "action_type": action_type,
            "current_sigma": current_sigma,
            "predicted_sigma_after": predicted_sigma_after,
            "delta_sigma": delta_sigma,
            "benefit": benefit,
            "cost_ext": cost_ext,
            "should_act": should_act,
        }

    def evaluate_and_act(
        self,
        action_type: str,
        current_sigma: float,
        execute: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate and optionally execute a niche action.

        Returns:
            (should_act, evaluation_dict)
        """
        result = self.evaluate_action(action_type, current_sigma)
        should_act = result["should_act"]

        if should_act and execute:
            # Log the action
            self.action_log.append({
                "action_type": action_type,
                "benefit": result["benefit"],
                "cost_ext": result["cost_ext"],
            })
            # TODO: integrate with robotics / low-level control
            print(f"[NCE] Executing: {action_type} "
                  f"(Benefit {result['benefit']:.2f} > Cost {result['cost_ext']:.2f})")
        elif not should_act:
            print(f"[NCE] Skipping: {action_type} "
                  f"(Benefit {result['benefit']:.2f} <= Cost {result['cost_ext']:.2f})")

        return should_act, result

    def get_best_action(self, current_sigma: float) -> Optional[Dict[str, Any]]:
        """Find best niche action given current σ_proxy."""
        best_action = None
        best_net_benefit = 0.0

        for action_type in self.niche_actions:
            result = self.evaluate_action(action_type, current_sigma)
            net_benefit = result["benefit"] - result["cost_ext"]

            if net_benefit > best_net_benefit:
                best_net_benefit = net_benefit
                best_action = result

        return best_action


# ============================================================================
#  High-Level TFAN Core Model
# ============================================================================

class TFanCore(nn.Module):
    """
    Minimal "core" model that wires:
      - TFF (TopFusionEncoder + TopologyHead + TopologicalRegularizer)
      - A simple classifier head (for experiments)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_modalities: int = 2,
        num_classes: int = 10,
        mcca_config: Optional[Dict[str, Any]] = None,
        ph_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.num_classes = num_classes

        self.top_fusion_encoder = TopFusionEncoder(d_model, num_modalities, mcca_config)
        self.topology_head = TopologyHead(d_model, ph_config)
        self.topo_reg = TopologicalRegularizer()
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Expects:
            batch["modal_embeddings"]: List[Tensor] (len = num_modalities)
            batch["labels"]:           [B] (optional, for training)

        Returns:
            {
              "logits": [B x num_classes],
              "z_fused": [B x d_model],
              "topology": {...}
            }
        """
        modal_embs = batch.get("modal_embeddings", batch.get("modalities"))
        z_fused = self.top_fusion_encoder(modal_embs)  # [B x d_model]
        topo_out = self.topology_head(z_fused)
        logits = self.classifier(z_fused)

        return {
            "logits": logits,
            "z_fused": z_fused,
            "topology": topo_out,
        }

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        udk: UDKController,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute:
          L_utility  = cross entropy
          L_topo     = topological regularization
          L_total    = L_utility + lambda_topo * L_topo

        Returns dict with all loss components.
        """
        logits = outputs["logits"]
        topo = outputs["topology"]

        # Utility loss
        l_utility = F.cross_entropy(logits, labels, reduction="mean")

        # Topological loss
        beta_k = topo["beta_k"]  # [K]
        l_topo = self.topo_reg(beta_k)

        # Ask UDK what lambda_topo should be given current state
        lambda_topo = udk.get_lambda_topo()
        l_total = l_utility + lambda_topo * l_topo

        # Update UDK metrics for this step
        udk.update_tff_metrics(model=self, l_topo=float(l_topo.detach()))

        return {
            "loss": l_total,
            "l_utility": l_utility,
            "l_topo": l_topo,
            "lambda_topo": lambda_topo,
        }


# ============================================================================
#  Factory Functions
# ============================================================================

def create_tfan_core(
    d_model: int = 512,
    num_modalities: int = 2,
    num_classes: int = 10,
) -> TFanCore:
    """Factory function for TFanCore with common defaults."""
    return TFanCore(
        d_model=d_model,
        num_modalities=num_modalities,
        num_classes=num_classes,
    )


def create_udk_controller() -> UDKController:
    """Factory function for UDKController with defaults."""
    return UDKController()


def create_cos(horizon_steps: int = 5000) -> CognitiveOffloadingSubsystem:
    """Factory function for COS with defaults."""
    return CognitiveOffloadingSubsystem(horizon_steps=horizon_steps)


# ============================================================================
#  Quick Sanity Check
# ============================================================================

if __name__ == "__main__":
    print("T-FAN Core Skeleton - Sanity Check")
    print("=" * 50)

    # Create model and controllers
    model = create_tfan_core(d_model=256, num_modalities=2, num_classes=10)
    udk = create_udk_controller()
    cos = create_cos()

    # Fake batch
    B = 8
    batch = {
        "modal_embeddings": [
            torch.randn(B, 256),  # "text"
            torch.randn(B, 256),  # "image"
        ],
        "labels": torch.randint(0, 10, (B,)),
    }

    # Forward pass
    outputs = model(batch)
    losses = model.compute_losses(outputs, batch["labels"], udk)

    print(f"z_fused shape: {outputs['z_fused'].shape}")
    print(f"logits shape:  {outputs['logits'].shape}")
    print(f"beta_k shape:  {outputs['topology']['beta_k'].shape}")
    print()
    print(f"L_utility: {losses['l_utility'].item():.4f}")
    print(f"L_topo:    {losses['l_topo'].item():.4f}")
    print(f"λ_topo:    {losses['lambda_topo']:.4f}")
    print(f"L_total:   {losses['loss'].item():.4f}")
    print()

    # UDK diagnostics
    diag = udk.get_diagnostics()
    print("UDK State:")
    for k, v in diag.items():
        print(f"  {k}: {v:.4f}")
    print()

    # NCE evaluation
    print("NCE Evaluation (σ=0.5):")
    for action_type in ["sidecar_logging", "deploy_sensor"]:
        result = cos.evaluate_action(action_type, current_sigma=0.5)
        print(f"  {action_type}: should_act={result['should_act']}, "
              f"benefit={result['benefit']:.1f}, cost={result['cost_ext']:.1f}")

    print()
    print("✓ Sanity check passed!")
