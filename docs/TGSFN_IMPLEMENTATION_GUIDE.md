# TGSFN / Antifragile Agent Implementation Guide

**Master Implementation Blueprint**
*Referee-safe, December 2025*

---

## 0. Mission and Global Constraints

### Goal

Implement a *Thermodynamic-Geometric Spiking Field Network (TGSFN)* that:

1. Runs a balanced spiking neural network driven toward **near-criticality** by minimizing a thermodynamic regularizer (Π_q)
2. Exhibits **neuronal avalanche statistics** consistent with the **mean-field Conserved Directed Percolation (C-DP)** universality class, in the hardware-relevant regime (N ~ 10³-10⁵)
3. Supports an **L5 control law** (closed-form policy) combining homeostatic drive, geometric consistency, and thermodynamic clipping
4. Comes with a full **analysis pipeline** for avalanche exponents, finite-size scaling, and universality checks
5. Is designed to be portable to an **FPGA/FPGA+GPU** stack later (Vitis HLS etc.), but initial implementation is in **PyTorch (+ snnTorch + Geoopt)**

### Global Scientific Honesty Constraints

**These constraints must not be violated in any documentation or publication:**

* True asymptotic branching exponent is **α = 3/2** (critical branching) in N → ∞, and TGSFN aims to approach that universality class
* For finite N, any non-zero dissipation/regularization (including Π_q) slightly pushes the system **subcritical**: m*(N) = 1 - Δm(N), Δm(N) > 0, typically Δm(N) = O(N^{-1/2})
* The observed **effective size exponent** τ_eff(N) (from fits over a finite window) is > 3/2 and drifts **downward** toward 3/2 as N increases
* The empirical fit τ_eff(N) ≈ 3/2 + c/√N with c ≈ 6.6 is a **good phenomenological description** in the accessible N-range, but **not** an exact asymptotic law. Do *not* present c as derived analytically
* Universality class: TGSFN belongs (empirically) to **C-DP mean-field**: τ ≈ 1.5-1.65, α ≈ 2.0, scaling relations consistent with C-DP. Don't claim more than that

**Referee-safe wording must respect these bullets.**

---

## 1. Software Stack and Project Structure

### 1.1 Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Core | PyTorch | Tensors + autograd |
| Spiking | snnTorch | Spiking neuron primitives (optional) |
| Geometry | Geoopt | Hyperbolic / Riemannian operations |
| Analysis | NumPy, SciPy | Numerical computation |
| Visualization | matplotlib | Plotting |

### 1.2 Current Repository Layout

The existing implementation is in `hrrl_agent/`:

```text
hrrl_agent/
  __init__.py
  criticality.py      # Edge of Chaos control, Π_q, avalanche analysis
  thermodynamics.py   # Entropy production monitoring
  tgsfn.py           # Main TGSFN substrate (Loss, Layer, Substrate classes)
  antifragile.py     # Stability monitoring via Jacobian spectral norm
  hardware.py        # Fixed-point arithmetic, manifold recentering, K-FAC
  l1_homeostat.py    # L1: Homeostatic core (free energy F_int)
  l2_hyperbolic.py   # L2: Hyperbolic appraisal on Poincaré manifold
  l3_gating.py       # L3: Explicit gating equations for τ, lr, memory write
  l4_memory.py       # L4: Memory, replay distribution, LoRA personalization
  loops.py           # Online and sleep training loops
```

See also: `grok_tgsfn/` for cleaner mathematical implementations.

---

## 2. Core Mathematical Objects

### 2.1 TGSFN State and Dynamics

A discrete-time spiking network with:

| Symbol | Description |
|--------|-------------|
| N | Number of neurons |
| V_i(t) | Membrane potential of neuron i |
| V_reset | Reset potential |
| τ_m^i | Time constant for neuron i |
| S_i(t) ∈ {0,1} | Spike indicator |
| W | Synaptic weight matrix (E/I structured) |
| z_i(t) | Latent representation in H^m × R^k |

**Update equations (schematic):**

```
V(t+1) = f_θ(V(t), S(t), x(t))
S_i(t+1) = Θ(V_i(t+1) - V_th)
V_i(t+1) ← V_reset  if S_i(t+1) = 1
```

Where f_θ encodes leak, recurrent input (W·S(t)), external drive, and geometric constraints.

**Implementation:** `hrrl_agent/tgsfn.py` - `TGSFNSubstrate` class

### 2.2 Thermodynamic Regularizer Π_q

**Definition:**

```
Π_q = Σ_i (V_i - V_reset)² / (τ_m^i · σ_i²) + λ_J ||J_θ||_F²
```

| Term | Purpose |
|------|---------|
| First term | Membrane deviation (E/I balance) |
| Second term | Jacobian stress regularization |

**Implementation:** `hrrl_agent/criticality.py:316-343`

```python
def compute_pi_q(
    membrane_potentials: torch.Tensor,
    v_reset: float,
    tau_m: torch.Tensor,
    sigma: torch.Tensor,
    jacobian: Optional[torch.Tensor] = None,
    lambda_j: float = 0.01
) -> torch.Tensor:
    """
    Compute entropy production proxy Π_q.

    Π_q = Σ_i (V_m^i - V_reset)² / (τ_m^i · σ_i²) + λ_J ||J_θ||_F²
    """
    deviation = membrane_potentials - v_reset
    leak_term = torch.sum(deviation ** 2 / (tau_m * sigma ** 2 + 1e-8))

    jacobian_term = torch.tensor(0.0, device=membrane_potentials.device)
    if jacobian is not None:
        jacobian_term = lambda_j * torch.sum(jacobian ** 2)

    return leak_term + jacobian_term
```

### 2.3 Full TGSFN Objective

```
L_total = L_task + λ_diss · Π_q
```

| Component | Purpose |
|-----------|---------|
| L_task | Task-specific loss (classification, RL, etc.) |
| λ_diss | Critical dial for criticality tuning |

**Implementation:** `hrrl_agent/tgsfn.py:93-144` - `TGSFNLoss` class

---

## 3. L5 Control Law

### Closed-Form Optimal Policy

```
v*(t) = proj_{T_z M}(Σ^{-1} n(t) + W_mot a(t) + g_τ(t) · ξ_epi(t)) · min(1, Π_max(t)/(Π_q + ε))
```

| Component | Interpretation |
|-----------|----------------|
| Σ^{-1} n(t) | Allostatic drive (homeostatic gradient) |
| W_mot a(t) | Motor/action drive (policy contribution) |
| g_τ(t) · ξ_epi(t) | Exploration/epistemic noise |
| proj_{T_z M}(...) | Riemannian projection to tangent space |
| min(1, Π_max/(Π_q+ε)) | Thermodynamic brake |

### Implementation Plan

1. **Geometry wrappers** (`hrrl_agent/l2_hyperbolic.py`):
   - Poincaré ball manifold for H^m
   - Euclidean manifold for R^k
   - Functions: `proj_to_tangent(z, v)`, `retract(z, v)`

2. **L5 Controller:**

```python
class L5Controller(nn.Module):
    def __init__(self, dim_hyper, dim_euclid, W_mot, sigma_inv):
        super().__init__()
        self.manifold = geoopt.PoincareBall()
        self.W_mot = nn.Parameter(W_mot)
        self.sigma_inv = nn.Parameter(sigma_inv)

    def forward(self, z, n_t, a_t, g_tau_t, xi_epi_t, pi_q, pi_max, eps=1e-6):
        base_vec = (self.sigma_inv @ n_t.T).T + (self.W_mot @ a_t.T).T + g_tau_t * xi_epi_t
        v_tangent = self.manifold.proju(z, base_vec)
        therm_factor = torch.clamp(pi_max / (pi_q + eps), max=1.0)
        return v_tangent * therm_factor
```

---

## 4. Avalanche Measurement and Criticality Analysis

### 4.1 Avalanche Definition

Given spike count per time bin R(t) = Σ_i S_i(t):

| Term | Definition |
|------|------------|
| Avalanche | Maximal contiguous sequence with R(t) > 0 |
| Size (s) | Total spikes in avalanche: s = Σ R(t) |
| Duration (T) | Number of timesteps in avalanche |

**Implementation:** `hrrl_agent/criticality.py:159-175`

```python
def extract_avalanches(R):
    """
    R: 1D array of population spike counts per time bin.
    Returns: (sizes, durations)
    """
    sizes, durs = [], []
    current_size, current_dur = 0, 0

    for r in R:
        if r > 0:
            current_size += r
            current_dur += 1
        else:
            if current_dur > 0:
                sizes.append(current_size)
                durs.append(current_dur)
                current_size, current_dur = 0, 0

    if current_dur > 0:
        sizes.append(current_size)
        durs.append(current_dur)

    return np.array(sizes), np.array(durs)
```

### 4.2 Power-Law Fitting

**Canonical fitting windows (N ≈ 8k):**
- Size: s ∈ [5, 300]
- Duration: T ∈ [3, 300]

**MLE Estimator:**

```python
def fit_power_law(data, s_min, s_max):
    mask = (data >= s_min) & (data <= s_max)
    x = data[mask]
    n = len(x)
    tau_hat = 1 + n / np.sum(np.log(x / s_min))
    return tau_hat
```

**Implementation:** `hrrl_agent/criticality.py:177-202`

### 4.3 Effective Branching Ratio m_eff

```python
def estimate_branching_ratio(R):
    R_t = R[:-1]
    R_tp1 = R[1:]
    cov = np.cov(R_t, R_tp1, bias=True)[0,1]
    var = np.var(R_t)
    return cov / (var + 1e-8)
```

**Target:** m_eff ≈ 0.99-1.0 (slightly subcritical)

**Implementation:** `hrrl_agent/criticality.py:120-138`

### 4.4 Universality Checks (C-DP)

| Metric | Expected (C-DP Mean-Field) | TGSFN Observed |
|--------|---------------------------|----------------|
| τ (size exponent) | ~1.5 | 1.6 ± 0.04 |
| α (duration exponent) | ~2.0 | ~2.0 |
| γ_sT (size-duration) | ~2 | ~2 |

**Scaling relations to verify:**

1. **Crackling noise relation:**
   ```
   1/(σνz) = (τ - 1)/(α - 1)
   ```

2. **Size-duration relation:**
   ```
   γ_sT = (τ - 1)/(α - 1)
   ```

---

## 5. Finite-Size Scaling Protocol

See also: `docs/FINITE_SIZE_SCALING.md`

### 5.1 Network Size Sweep

| N | Purpose |
|---|---------|
| 1024 | Baseline |
| 2048 | Intermediate |
| 4096 | Standard |
| 8192 | Primary target |

For each N:
1. Fix λ_diss (e.g., 1.2)
2. Run T = 2×10⁶ timesteps
3. Extract avalanches
4. Estimate τ_eff(N) over fixed window

### 5.2 Output Metrics

For each N record:
- τ_eff(N) - effective size exponent
- α_eff(N) - effective duration exponent
- m_eff(N) - branching ratio
- σ² - offspring variance
- s_c - avalanche cutoff scale

### 5.3 Scaling Fits

**Phenomenological (operational range):**
```
τ_eff(N) = 3/2 + c/√N,  c ≈ 6.6
```

**Asymptotic (theoretical):**
```
τ_eff(N) = 3/2 + 1/(ln N + C)
```

**Implementation:** `hrrl_agent/criticality.py:260-295` - `predict_finite_size_alpha()`

**Critical:** Both forms fit well in accessible N-range. Do NOT claim 1/√N is the true asymptotic law.

---

## 6. λ_diss Control Sweep

### Protocol

1. Fix N (e.g., 8192)
2. Sweep λ_diss ∈ {0.0, 0.5, 1.2, 2.0, 5.0}
3. For each λ_diss:
   - Run TGSFN
   - Extract avalanches
   - Compute m_eff, τ_eff, α_eff, firing rate

### Expected Behavior

| λ_diss | Regime | m_eff | τ_eff |
|--------|--------|-------|-------|
| 0.0 | Supercritical | > 1 | << 1.5 |
| ~1.0-1.5 | Near-critical | ≈ 1^- | ≈ 1.6 |
| High | Subcritical | << 1 | >> 1.5 |

### Outputs

- Per-λ_diss summary JSON
- τ_eff vs λ_diss plot
- m_eff vs λ_diss plot

---

## 7. DAU (Dynamic Axiomatic Update)

### Purpose

Antifragility mechanism: when dynamics become unstable, adjust high-level "axioms" in hyperbolic latent space to restore critical margin.

### Implementation

**Implementation:** `hrrl_agent/antifragile.py`

1. **Spectral norm estimation:** Power iteration on linearized dynamics
2. **Target spectral radius:** λ_crit corresponding to near-criticality
3. **Trigger condition:** ||J||_* > λ_crit + δ

```python
class DAU:
    def __init__(self, lambda_crit, step_size):
        self.lambda_crit = lambda_crit
        self.step_size = step_size

    def estimate_spectral_norm(self, dynamics_fn, state):
        # Power iteration on linearized dynamics
        pass

    def update_axioms(self, axiom_params, J_norm):
        if J_norm > self.lambda_crit:
            for p in axiom_params:
                p.data.mul_(1.0 - self.step_size)
```

---

## 8. Geometry Layer and Hyperbolic Identity

### Manifold Structure

Identity and core "values" live in low-distortion region of hyperbolic space:
- Use Geoopt's PoincaréBall for H^m embeddings
- Special `identity_embeds` parameters
- Homeostatic constraint on updates

**Implementation:** `hrrl_agent/l2_hyperbolic.py`

### Identity Regularization

```
L_identity = λ_id · |Δz_I|²
```

Where Δz_I is hyperbolic distance of identity embedding change per update.

---

## 9. Hardware Co-Design

### 9.1 Numeric Format

**Target:** 16-bit fixed-point for FPGA hyperbolic ops

```python
def quantize_fixed16(x, scale):
    return torch.round(x * scale).clamp(-32768, 32767) / scale
```

**Implementation:** `hrrl_agent/hardware.py`

### 9.2 Hardware Requirements

From `docs/FINITE_SIZE_SCALING.md`:

1. **16-bit fixed-point** for hyperbolic operations
2. **Periodic recentering** every 10⁶ cycles
3. **Validation checklist:**
   - α ∈ [1.5, 1.7]
   - g ∈ [0.9, 1.1]
   - E/I ratio ∈ [0.8, 1.2]
   - ||J||_* < 1.1
   - No fixed-point overflows

---

## 10. Implementation Order

When implementing from scratch:

1. **Core spiking dynamics** (`neurons.py`, `network.py`)
2. **Π_q regularizer** (`pi_q.py`)
3. **Avalanche analysis pipeline** (`avalanches.py`, `criticality.py`)
4. **Verify pipeline** with toy branching process first
5. **Integrate Π_q** into training, tune λ_diss
6. **L5 Controller** with stable numerics
7. **DAU skeleton**
8. **Hyperbolic identity manifold**
9. **Quantization stubs** for hardware

---

## 11. Output Requirements

### Per-Experiment Outputs

1. **Config:** YAML with all hyperparameters
2. **Results JSON:**
   ```json
   {
     "N": 8192,
     "lambda_diss": 1.2,
     "tau_eff": 1.63,
     "alpha_eff": 2.01,
     "gamma_sT": 1.98,
     "m_eff": 0.995,
     "firing_rate": 0.05,
     "sigma_sq": 1.02
   }
   ```
3. **Plots:**
   - Log-log P(s) and P(T)
   - ⟨s⟩_T vs T
   - τ_eff vs N
   - m_eff vs λ_diss

### Reproducibility

- Seeded PRNGs (`torch.manual_seed`, `np.random.seed`)
- Full config logging
- Git commit hash in outputs

---

## 12. Cross-References

| Topic | Document |
|-------|----------|
| Finite-size scaling theory | `docs/FINITE_SIZE_SCALING.md` |
| Criticality implementation | `hrrl_agent/criticality.py` |
| Hardware constraints | `hrrl_agent/hardware.py` |
| Thermodynamics | `hrrl_agent/thermodynamics.py` |
| TGSFN substrate | `hrrl_agent/tgsfn.py` |
| Antifragility | `hrrl_agent/antifragile.py` |

---

*This document serves as the authoritative implementation guide for TGSFN development. All claims about scaling behavior and universality class must adhere to the scientific honesty constraints specified in Section 0.*
