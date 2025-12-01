# TGSFN / Antifragile Agent - Developer Guide

*Hands-on implementation guide with code snippets (December 2025)*

This document explains **how to implement and run** the Thermodynamic-Geometric Spiking Field Network (TGSFN) and its antifragile control loop.

It is written for developers and coding assistants. The goal: a **working codebase** that:

- Simulates a balanced spiking network with thermodynamic regularization (Π_q)
- Exhibits near-critical avalanche statistics in the **C-DP mean-field** regime
- Supports a closed-form **L5 control law**
- Comes with a complete **analysis pipeline** for avalanche statistics and finite-size scaling

> **Scientific honesty rule:**
> All asymptotic claims use rigorously known results (critical branching α = 3/2, etc.).
> Finite-size scaling fits (like α(N) ≈ 3/2 + c/√N) are **phenomenological**, not theorems.

---

## 0. Project Layout

Recommended repo structure:

```text
tgsfn/
  __init__.py
  config/
    base.yaml
    tgsfn8b.yaml
  core/
    neurons.py          # Spiking cell models
    network.py          # Recurrent SNN + E/I balance
    geometry.py         # Hyperbolic + Euclidean latent ops
    pi_q.py             # Thermodynamic regularizer Π_q
    control_law.py      # L5 policy implementation
    dau.py              # Dynamic Axiomatic Update (DAU) skeleton
  train/
    trainer.py          # Training / simulation loop
    homeostasis.py      # Internal state, Σ^{-1} n(t), F_int
  analysis/
    avalanches.py       # Extract avalanche size/duration
    criticality.py      # Exponents, scaling, universality checks
    plotting.py         # Figures and diagnostics
  experiments/
    run_tgsfn8b.py      # Main canonical experiment
    sweep_lambda_diss.py
    finite_size_scaling.py
  hardware/
    vitis_hls_stub/
      README.md         # Mapping to HLS / FPGA
  docs/
    implementation_guide.md
    theory_notes.md
```

**Current implementation:** See `hrrl_agent/` for the existing implementation.

**Python minimum:** 3.10+

---

## 1. Dependencies and Setup

### 1.1 Core Libraries

| Library | Purpose |
|---------|---------|
| `torch` | Main tensor + autograd |
| `snnTorch` | Spiking neuron primitives (optional) |
| `geoopt` | Hyperbolic / Riemannian geometry |
| `numpy`, `scipy` | Numerical computation |
| `matplotlib` | Visualization |

### 1.2 Example `requirements.txt`

```text
torch>=2.2
numpy>=1.24
scipy>=1.11
matplotlib>=3.8
snnTorch>=0.7
geoopt>=0.5
scikit-learn>=1.4
powerlaw>=1.5
```

---

## 2. Core Modules

### 2.1 Spiking Neuron Model (`core/neurons.py`)

Discrete-time LIF-like neurons.

**State per neuron:**
- Membrane potential V_i(t)
- Spike S_i(t) ∈ {0,1}
- Time constant τ_m^i
- Threshold V_th, reset V_reset

**Implementation:**

```python
import torch
import torch.nn as nn

class LIFNeuronLayer(nn.Module):
    def __init__(self, N, tau_m, v_th, v_reset):
        super().__init__()
        self.N = N
        self.register_buffer("tau_m", torch.full((N,), tau_m))
        self.v_th = v_th
        self.v_reset = v_reset

    def forward(self, V, I):
        """
        V: (batch, N) membrane potentials at t
        I: (batch, N) synaptic/input current at t
        Returns: V_next, S (spikes)
        """
        # Simple Euler update: V <- V + (-V/tau_m + I) * dt, with dt=1
        dV = (-V / self.tau_m) + I
        V_next = V + dV

        S = (V_next >= self.v_th).float()
        V_next = torch.where(S > 0, torch.full_like(V_next, self.v_reset), V_next)
        return V_next, S
```

**Existing implementation:** `hrrl_agent/tgsfn.py`

---

### 2.2 Network + E/I Balance (`core/network.py`)

Balanced network:
- N neurons, fraction `p_exc` excitatory, `1-p_exc` inhibitory
- Weight matrix `W` with E/I structure
- External input `I_ext(t)` (random drive)

```python
class TGSFNCore(nn.Module):
    def __init__(self, N, p_exc, tau_m, v_th, v_reset, w_scale, device="cuda"):
        super().__init__()
        self.N = N
        self.device = device

        self.neurons = LIFNeuronLayer(N, tau_m, v_th, v_reset)

        # E/I mask
        n_exc = int(p_exc * N)
        exc_ids = torch.arange(n_exc)
        inh_ids = torch.arange(n_exc, N)

        W = torch.randn(N, N) * w_scale
        W[:, inh_ids] *= -1.0  # inhibitory columns
        self.W = nn.Parameter(W)

    def forward_step(self, V, S_prev, I_ext):
        """
        One time step of network dynamics.
        V: (batch, N)
        S_prev: (batch, N)
        I_ext: (batch, N)
        """
        I_rec = S_prev @ self.W.T   # (batch, N)
        I_tot = I_rec + I_ext
        V_next, S = self.neurons(V, I_tot)
        return V_next, S, I_tot
```

**Existing implementation:** `hrrl_agent/tgsfn.py` - `TGSFNSubstrate` class

---

### 2.3 Geometry Layer (`core/geometry.py`)

Use **Geoopt** for hyperbolic and Euclidean parts of the latent state.

```python
import geoopt
import torch.nn as nn

class LatentGeometry:
    def __init__(self, dim_hyper, dim_euclid):
        self.hyper = geoopt.PoincareBall(c=1.0)
        self.dim_hyper = dim_hyper
        self.dim_euclid = dim_euclid

    def split(self, z):
        z_h = z[..., :self.dim_hyper]
        z_e = z[..., self.dim_hyper:]
        return z_h, z_e

    def proj_to_tangent(self, z_h, v_h):
        return self.hyper.proju(z_h, v_h)

    def retract(self, z_h, v_h):
        return self.hyper.retr(z_h, v_h)
```

**Existing implementation:** `hrrl_agent/l2_hyperbolic.py`

---

### 2.4 Thermodynamic Regularizer Π_q (`core/pi_q.py`)

**Definition:**

```
Π_q = Σ_i (V_i - V_reset)² / (τ_m^i · σ_i²) + λ_J ||J_θ||_F²
```

**Implementation:**

```python
def compute_pi_q(V, V_reset, tau_m, sigma_sq, jacobian_norm_sq, lambda_J):
    """
    V:        (batch, N)
    tau_m:    (N,) or scalar
    sigma_sq: (N,) or scalar
    jacobian_norm_sq: scalar (approx ||J||_F^2)
    lambda_J: scalar
    """
    leak_term = ((V - V_reset) ** 2 / (tau_m * sigma_sq)).sum(dim=-1).mean()
    return leak_term + lambda_J * jacobian_norm_sq
```

**Jacobian approximation (Hutchinson estimator):**

```python
def approx_jacobian_norm_sq(dynamics_fn, state, n_samples=1):
    # Placeholder: implement power iteration/Hutchinson
    return state.new_tensor(0.0)
```

**Existing implementation:** `hrrl_agent/criticality.py:316-343`

---

### 2.5 L5 Control Law (`core/control_law.py`)

**Closed-form policy:**

```
v*(t) = proj_{T_z M}(Σ^{-1} n(t) + W_mot a(t) + g_τ(t) · ξ_epi(t)) · min(1, Π_max/(Π_q + ε))
```

**Implementation:**

```python
class L5Controller(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Σ^{-1} and W_mot as learnable or fixed parameters
        self.sigma_inv = nn.Parameter(torch.eye(latent_dim))
        self.W_mot = nn.Parameter(torch.randn(latent_dim, action_dim) * 0.01)

        self.manifold = geoopt.PoincareBall()

    def forward(self, z, n_t, a_t, g_tau_t, xi_epi_t, pi_q, pi_max, eps=1e-6):
        """
        z: latent state (batch, latent_dim), on hyperbolic manifold
        n_t: homeostatic drive (batch, latent_dim)
        a_t: action input (batch, action_dim)
        g_tau_t: scalar or (batch, 1)
        xi_epi_t: exploration noise (batch, latent_dim)
        pi_q: scalar Π_q
        pi_max: scalar Π_max
        """
        term_homeo = (self.sigma_inv @ n_t.T).T
        term_mot = (self.W_mot @ a_t.T).T
        base_vec = term_homeo + term_mot + g_tau_t * xi_epi_t

        v_tangent = self.manifold.proju(z, base_vec)
        therm_factor = torch.clamp(pi_max / (pi_q + eps), max=1.0)
        return v_tangent * therm_factor
```

---

### 2.6 DAU Skeleton (`core/dau.py`)

DAU: when Jacobian norm exceeds critical margin, adjust high-level "axioms" to reduce instability.

```python
class DAU:
    def __init__(self, lambda_crit, step_size):
        self.lambda_crit = lambda_crit
        self.step_size = step_size

    def estimate_spectral_norm(self, dynamics_fn, state):
        # TODO: implement power iteration around state
        return state.new_tensor(0.0)

    def update_axioms(self, axiom_params, J_norm):
        if J_norm > self.lambda_crit:
            for p in axiom_params:
                p.data.mul_(1.0 - self.step_size)
```

**Existing implementation:** `hrrl_agent/antifragile.py`

---

## 3. Training / Simulation Loop (`train/trainer.py`)

### 3.1 Objective

Total loss:

```
L_total = L_task + λ_diss · Π_q
```

For initial experiments, set L_task = 0 and just run under Π_q control.

### 3.2 Minimal Simulation Loop

```python
class TGSFNTrainer:
    def __init__(self, model, config):
        self.model = model
        self.cfg = config

    def run_simulation(self, T_steps, batch_size=1):
        N = self.model.N
        device = next(self.model.parameters()).device

        V = torch.zeros(batch_size, N, device=device)
        S = torch.zeros(batch_size, N, device=device)

        R_series = []

        for t in range(T_steps):
            I_ext = self.sample_external_input(batch_size, N, device)
            V, S, I_tot = self.model.forward_step(V, S, I_ext)

            # record population activity
            R = S.sum(dim=-1)  # (batch,)
            R_series.append(R.detach().cpu().numpy())

        R_series = np.stack(R_series, axis=0)  # (T_steps, batch)
        return R_series

    def sample_external_input(self, batch_size, N, device):
        return torch.randn(batch_size, N, device=device) * self.cfg["input_noise_scale"]
```

---

## 4. Avalanche Analysis

### 4.1 Extraction (`analysis/avalanches.py`)

```python
def extract_avalanches(R):
    """
    R: 1D array of population spike counts: shape (T,)
    Returns: sizes, durations
    """
    sizes, durs = [], []
    current_size = 0
    current_dur = 0
    for r in R:
        if r > 0:
            current_size += r
            current_dur += 1
        else:
            if current_dur > 0:
                sizes.append(current_size)
                durs.append(current_dur)
                current_size = 0
                current_dur = 0
    if current_dur > 0:
        sizes.append(current_size)
        durs.append(current_dur)
    return np.array(sizes), np.array(durs)
```

**Existing implementation:** `hrrl_agent/criticality.py:159-175`

### 4.2 Power-Law Fit (Simple MLE)

```python
def fit_power_law(data, x_min, x_max):
    mask = (data >= x_min) & (data <= x_max)
    x = data[mask].astype(float)
    n = len(x)
    if n == 0:
        return np.nan
    tau_hat = 1.0 + n / np.sum(np.log(x / x_min))
    return tau_hat
```

**Existing implementation:** `hrrl_agent/criticality.py:177-202`

### 4.3 Size-Duration Scaling

```python
def compute_size_duration_scaling(sizes, durs):
    # compute mean size per duration
    unique_T = np.unique(durs)
    mean_s = []
    T_vals = []
    for T in unique_T:
        mask = (durs == T)
        if mask.sum() < 5:
            continue
        mean_s.append(sizes[mask].mean())
        T_vals.append(T)

    T_vals = np.array(T_vals, dtype=float)
    mean_s = np.array(mean_s, dtype=float)

    # log-log regression
    logT = np.log10(T_vals)
    logs = np.log10(mean_s)
    coeffs = np.polyfit(logT, logs, 1)
    gamma_sT = coeffs[0]
    return gamma_sT
```

### 4.4 Universality Summary

```python
def universality_summary(sizes, durs, cfg):
    tau = fit_power_law(
        sizes,
        cfg["avalanche"]["s_min"],
        cfg["avalanche"]["s_max"]
    )
    alpha = fit_power_law(
        durs,
        cfg["avalanche"]["t_min"],
        cfg["avalanche"]["t_max"]
    )
    gamma_sT = compute_size_duration_scaling(sizes, durs)

    crackling_rhs = (tau - 1) / (alpha - 1) if alpha > 1 and tau > 1 else np.nan

    return {
        "tau": float(tau),
        "alpha": float(alpha),
        "gamma_sT": float(gamma_sT),
        "crackling_rhs": float(crackling_rhs),
    }
```

---

## 5. First Canonical Experiment: TGSFN-8B

Use `config/tgsfn8b.yaml` and `experiments/run_tgsfn8b.py`:

1. Initialize TGSFNCore with N=8192, λ_diss ≈ 1.2
2. Run T ~ 2×10⁶ time steps
3. Extract avalanches
4. Compute τ, α, γ_sT, crackling_rhs, m_eff
5. Save results and plots

**Target qualitative outcomes (not pass/fail):**

| Metric | Expected Value |
|--------|----------------|
| τ_eff | ~1.6-1.7 in size window [5,300] |
| α | ~2.0 in duration window |
| γ_sT | ~2.0 |
| m_eff | ~0.99 (slightly subcritical) |

---

## 6. λ_diss Sweep

`experiments/sweep_lambda_diss.py`:

- For λ_diss in [0.0, 0.5, 1.2, 2.0, 5.0]:
  - Run shorter simulation (T = 5×10⁵)
  - Compute avalanche exponents and m_eff

**Expected behavior:**

| λ_diss | Regime | m_eff | τ_eff |
|--------|--------|-------|-------|
| 0.0 | Supercritical | > 1 | << 1.5 |
| ~1.0-1.5 | Near-critical | ≈ 1^- | ≈ 1.6 |
| Large | Subcritical | << 1 | >> 1.5 |

---

## 7. Finite-Size Scaling

`experiments/finite_size_scaling.py`:

- N in [1024, 2048, 4096, 8192]
- For each N:
  - Use same λ_diss (e.g. 1.2)
  - Run simulation
  - Compute τ_eff(N)

**Analysis:**

1. Fit τ_eff(N) ≈ 3/2 + c/√N → get `c` and R²
2. Optionally compare with τ_eff(N) ≈ 3/2 + 1/(ln N + C)

**Important:** 1/√N is **effective** scaling over this range, not a proven asymptotic law.

See: `docs/FINITE_SIZE_SCALING.md`

---

## 8. Hardware Mapping Stubs

### 8.1 Numeric Format

Target: 16-bit fixed point in core ops.

```python
def quantize_fixed16(x, scale):
    return torch.round(x * scale).clamp(-32768, 32767) / scale
```

### 8.2 FTP (Fully Temporal-Parallel) Dataflow

Design each `forward_step` to map to an HLS kernel:
- Input: previous spikes / membrane
- Output: new spikes / membrane

### 8.3 Host Interface

- Use asynchronous OpenCL/SYCL
- Overlap data transfer and computation

**Existing implementation:** `hrrl_agent/hardware.py`

---

## 9. Logging and Reproducibility

Every experiment should:

1. **Set seeds:** `torch.manual_seed`, `np.random.seed`
2. **Save:**
   - Config YAML
   - Metadata JSON: N, λ_diss, T_steps, τ, α, γ_sT, crackling_rhs, m_eff
3. **Generate plots** in `experiments/results/`

---

## 10. Scientific Honesty Checklist

When writing documentation or papers:

| Claim | Status |
|-------|--------|
| α = 3/2 at N→∞ | **Exact** (mean-field universality) |
| Π_q drives toward criticality | **Proven** in mean-field limit |
| Finite-size deviations (m < 1, τ > 3/2) | **Expected** from theory |
| α(N) ≈ 3/2 + c/√N with c ≈ 6.6 | **Empirical fit** (not derived) |

**Rule:** No hype, no overclaiming.

---

## Cross-References

| Document | Purpose |
|----------|---------|
| `docs/TGSFN_IMPLEMENTATION_GUIDE.md` | Master blueprint/spec |
| `docs/FINITE_SIZE_SCALING.md` | Theoretical derivations |
| `config/tgsfn8b.yaml` | Canonical experiment config |
| `hrrl_agent/` | Existing implementation |
