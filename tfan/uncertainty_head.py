# tfan/uncertainty_head.py
# Layer 3: Uncertainty Decomposition and Adaptive Policy Controller
#
# Implements the adaptive policy layer from the Generative Affective Cognition
# framework. Decomposes uncertainty into aleatoric (irreducible data noise) and
# epistemic (reducible model uncertainty) components.
#
# Key concepts:
# - Aleatoric uncertainty: inherent randomness in data (cannot be reduced)
# - Epistemic uncertainty: model's lack of knowledge (can be reduced via learning)
# - Policy gates: adaptive behavior modulation based on uncertainty type
# - LC (Locus Coeruleus) integration: arousal-modulated attention/exploration

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math


# ============================================================================
#  Uncertainty Decomposition Result
# ============================================================================

@dataclass
class UncertaintyEstimate:
    """Result of uncertainty decomposition."""
    total: float           # Total predictive uncertainty
    aleatoric: float       # Irreducible (data noise)
    epistemic: float       # Reducible (model uncertainty)
    confidence: float      # 1 - total (clamped)
    entropy: float         # Shannon entropy of predictions
    calibration_error: float  # Expected calibration error
    raw_logits: Optional[Any] = None  # Original logits for debugging


# ============================================================================
#  Policy Gate Decisions
# ============================================================================

@dataclass
class PolicyGateDecision:
    """Adaptive policy gate decision based on uncertainty."""
    gate_name: str
    should_activate: bool
    activation_strength: float  # 0-1, how strongly to activate
    reason: str
    uncertainty_type: str  # "aleatoric", "epistemic", or "both"


@dataclass
class PolicyDecision:
    """Full policy decision from Layer 3."""
    explore_vs_exploit: float   # -1 (exploit) to +1 (explore)
    confidence_threshold: float  # Dynamic threshold for action
    attention_gain: float        # LC-modulated attention multiplier
    learning_rate_mod: float     # Suggested LR modifier
    should_defer: bool           # Defer to human/external system?
    defer_reason: str
    gates: List[PolicyGateDecision] = field(default_factory=list)
    raw_uncertainty: Optional[UncertaintyEstimate] = None


# ============================================================================
#  Uncertainty Head (Computational)
# ============================================================================

class UncertaintyHead:
    """
    Layer 3: Uncertainty decomposition and adaptive policy controller.

    Estimates aleatoric vs epistemic uncertainty from model outputs
    and drives adaptive behavior (exploration, deferral, attention).

    Uses ensemble disagreement or MC Dropout proxy for epistemic,
    and predictive entropy for aleatoric.
    """

    def __init__(
        self,
        base_confidence_threshold: float = 0.7,
        explore_epistemic_threshold: float = 0.3,
        defer_threshold: float = 0.6,
        lc_gain_base: float = 1.0,
        lc_gain_max: float = 2.5,
        history_size: int = 100,
    ):
        """
        Args:
            base_confidence_threshold: Default confidence for action
            explore_epistemic_threshold: Epistemic level that triggers exploration
            defer_threshold: Total uncertainty that triggers deferral
            lc_gain_base: Baseline LC attention gain
            lc_gain_max: Maximum LC attention gain under high uncertainty
            history_size: Rolling history size for calibration
        """
        self.base_confidence_threshold = base_confidence_threshold
        self.explore_epistemic_threshold = explore_epistemic_threshold
        self.defer_threshold = defer_threshold
        self.lc_gain_base = lc_gain_base
        self.lc_gain_max = lc_gain_max
        self.history_size = history_size

        self.history: List[UncertaintyEstimate] = []
        self.calibration_history: List[Tuple[float, bool]] = []  # (confidence, correct)

    def estimate_uncertainty(
        self,
        logits: Any,
        ensemble_logits: Optional[List[Any]] = None,
        mc_samples: Optional[List[Any]] = None,
        true_label: Optional[int] = None,
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty from model outputs.

        Args:
            logits: Primary model logits [batch, classes] or [classes]
            ensemble_logits: Optional list of logits from ensemble members
            mc_samples: Optional list of logits from MC Dropout samples
            true_label: Optional ground truth for calibration tracking

        Returns:
            UncertaintyEstimate with decomposed uncertainties
        """
        import torch

        # Ensure tensor
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32)

        # Handle batched input - take mean across batch for summary
        if logits.dim() > 1:
            logits = logits.mean(dim=0)

        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=-1)

        # === Total uncertainty: predictive entropy ===
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        max_entropy = math.log(probs.size(-1))  # log(num_classes)
        total = entropy / max_entropy if max_entropy > 0 else 0.0

        # === Epistemic uncertainty: ensemble/MC disagreement ===
        epistemic = 0.0

        if ensemble_logits and len(ensemble_logits) > 1:
            # Ensemble method: variance of predictions
            ensemble_probs = [torch.softmax(l.mean(dim=0) if l.dim() > 1 else l, dim=-1)
                            for l in ensemble_logits]
            stacked = torch.stack(ensemble_probs)
            variance = stacked.var(dim=0).mean().item()
            epistemic = min(1.0, variance * 4)  # Scale to [0, 1]

        elif mc_samples and len(mc_samples) > 1:
            # MC Dropout method: similar variance approach
            mc_probs = [torch.softmax(l.mean(dim=0) if l.dim() > 1 else l, dim=-1)
                       for l in mc_samples]
            stacked = torch.stack(mc_probs)
            variance = stacked.var(dim=0).mean().item()
            epistemic = min(1.0, variance * 4)

        else:
            # Fallback: heuristic based on entropy and confidence
            max_prob = probs.max().item()
            # High entropy + low confidence suggests epistemic uncertainty
            epistemic = max(0.0, total - (1.0 - max_prob)) * 0.5

        # === Aleatoric uncertainty: residual ===
        aleatoric = max(0.0, total - epistemic)

        # === Confidence ===
        confidence = max(0.0, min(1.0, 1.0 - total))

        # === Calibration tracking ===
        calibration_error = self._compute_calibration_error()

        if true_label is not None:
            pred = probs.argmax().item()
            correct = (pred == true_label)
            self.calibration_history.append((confidence, correct))
            if len(self.calibration_history) > self.history_size:
                self.calibration_history.pop(0)

        result = UncertaintyEstimate(
            total=total,
            aleatoric=aleatoric,
            epistemic=epistemic,
            confidence=confidence,
            entropy=entropy,
            calibration_error=calibration_error,
            raw_logits=logits,
        )

        self.history.append(result)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        return result

    def _compute_calibration_error(self) -> float:
        """Compute expected calibration error from history."""
        if len(self.calibration_history) < 10:
            return 0.0

        # Bin predictions by confidence
        bins = [[] for _ in range(10)]
        for conf, correct in self.calibration_history:
            bin_idx = min(9, int(conf * 10))
            bins[bin_idx].append((conf, correct))

        # ECE: weighted average of |accuracy - confidence| per bin
        ece = 0.0
        total_samples = len(self.calibration_history)

        for bin_items in bins:
            if bin_items:
                avg_conf = sum(c for c, _ in bin_items) / len(bin_items)
                accuracy = sum(1 for _, c in bin_items if c) / len(bin_items)
                ece += len(bin_items) / total_samples * abs(accuracy - avg_conf)

        return ece

    def compute_policy(
        self,
        uncertainty: UncertaintyEstimate,
        arousal: float = 0.5,
        homeostatic_drive: float = 0.5,
    ) -> PolicyDecision:
        """
        Compute adaptive policy decision based on uncertainty.

        Args:
            uncertainty: Decomposed uncertainty estimate
            arousal: Current arousal level (from affective state)
            homeostatic_drive: Current homeostatic drive total

        Returns:
            PolicyDecision with exploration/exploitation balance, gates, etc.
        """
        gates: List[PolicyGateDecision] = []

        # === Explore vs Exploit ===
        # High epistemic → explore (can reduce uncertainty)
        # High aleatoric → exploit (noise won't reduce)
        explore_vs_exploit = 0.0

        if uncertainty.epistemic > self.explore_epistemic_threshold:
            explore_vs_exploit += uncertainty.epistemic * 0.8
            gates.append(PolicyGateDecision(
                gate_name="exploration",
                should_activate=True,
                activation_strength=uncertainty.epistemic,
                reason=f"High epistemic uncertainty ({uncertainty.epistemic:.2f})",
                uncertainty_type="epistemic",
            ))

        if uncertainty.aleatoric > 0.5:
            explore_vs_exploit -= uncertainty.aleatoric * 0.5  # Favor exploitation
            gates.append(PolicyGateDecision(
                gate_name="robust_action",
                should_activate=True,
                activation_strength=uncertainty.aleatoric,
                reason=f"High aleatoric uncertainty ({uncertainty.aleatoric:.2f})",
                uncertainty_type="aleatoric",
            ))

        explore_vs_exploit = max(-1.0, min(1.0, explore_vs_exploit))

        # === Dynamic confidence threshold ===
        # Raise threshold when homeostatic drive is high (be more careful)
        confidence_threshold = self.base_confidence_threshold
        confidence_threshold += homeostatic_drive * 0.2
        confidence_threshold = min(0.95, confidence_threshold)

        # === LC attention gain (arousal-modulated) ===
        # Higher arousal and uncertainty → higher attention gain
        lc_factor = 0.5 * arousal + 0.5 * uncertainty.total
        attention_gain = self.lc_gain_base + (self.lc_gain_max - self.lc_gain_base) * lc_factor

        # === Learning rate modulation ===
        # High epistemic → increase LR (more to learn)
        # Low epistemic → decrease LR (already know this)
        lr_mod = 1.0 + 0.5 * (uncertainty.epistemic - 0.5)
        lr_mod = max(0.5, min(2.0, lr_mod))

        # === Deferral decision ===
        should_defer = uncertainty.total > self.defer_threshold
        defer_reason = ""

        if should_defer:
            if uncertainty.epistemic > uncertainty.aleatoric:
                defer_reason = f"High epistemic uncertainty ({uncertainty.epistemic:.2f}) - need more data"
            else:
                defer_reason = f"High aleatoric uncertainty ({uncertainty.aleatoric:.2f}) - inherently noisy"

            gates.append(PolicyGateDecision(
                gate_name="deferral",
                should_activate=True,
                activation_strength=uncertainty.total,
                reason=defer_reason,
                uncertainty_type="both",
            ))

        # === Safety gate (uncertainty + high drive) ===
        if uncertainty.total > 0.4 and homeostatic_drive > 0.6:
            gates.append(PolicyGateDecision(
                gate_name="safety_check",
                should_activate=True,
                activation_strength=uncertainty.total * homeostatic_drive,
                reason="Uncertain state with high homeostatic drive",
                uncertainty_type="both",
            ))

        return PolicyDecision(
            explore_vs_exploit=explore_vs_exploit,
            confidence_threshold=confidence_threshold,
            attention_gain=attention_gain,
            learning_rate_mod=lr_mod,
            should_defer=should_defer,
            defer_reason=defer_reason,
            gates=gates,
            raw_uncertainty=uncertainty,
        )

    def get_state(self) -> Dict[str, Any]:
        """Get current state for telemetry."""
        if self.history:
            last = self.history[-1]
            return {
                "uncertainty_total": last.total,
                "uncertainty_aleatoric": last.aleatoric,
                "uncertainty_epistemic": last.epistemic,
                "confidence": last.confidence,
                "entropy": last.entropy,
                "calibration_error": last.calibration_error,
            }
        return {
            "uncertainty_total": 0.0,
            "uncertainty_aleatoric": 0.0,
            "uncertainty_epistemic": 0.0,
            "confidence": 1.0,
            "entropy": 0.0,
            "calibration_error": 0.0,
        }

    def get_rolling_stats(self) -> Dict[str, float]:
        """Get rolling statistics over history window."""
        if not self.history:
            return {
                "mean_total": 0.0,
                "mean_epistemic": 0.0,
                "mean_aleatoric": 0.0,
                "trend_epistemic": 0.0,
            }

        totals = [h.total for h in self.history]
        epistemics = [h.epistemic for h in self.history]
        aleatorics = [h.aleatoric for h in self.history]

        # Trend: difference between recent and older half
        n = len(epistemics)
        if n >= 4:
            recent = sum(epistemics[n//2:]) / (n - n//2)
            older = sum(epistemics[:n//2]) / (n//2)
            trend = recent - older
        else:
            trend = 0.0

        return {
            "mean_total": sum(totals) / len(totals),
            "mean_epistemic": sum(epistemics) / len(epistemics),
            "mean_aleatoric": sum(aleatorics) / len(aleatorics),
            "trend_epistemic": trend,
        }


# ============================================================================
#  Factory Function
# ============================================================================

def create_uncertainty_head(
    base_confidence_threshold: float = 0.7,
    explore_epistemic_threshold: float = 0.3,
    defer_threshold: float = 0.6,
) -> UncertaintyHead:
    """Create an uncertainty head with given thresholds."""
    return UncertaintyHead(
        base_confidence_threshold=base_confidence_threshold,
        explore_epistemic_threshold=explore_epistemic_threshold,
        defer_threshold=defer_threshold,
    )
