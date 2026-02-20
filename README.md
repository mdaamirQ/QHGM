# Quantum Hamiltonian-Based Gene Expression Model (QHGM)


## Overview

This codebase infers a gene regulatory network (GRN) by fitting a parameterized quantum Hamiltonian to time-series single-cell measurement data. The pipeline has two stages:

1. **Pseudotime calculation** (`pseudotime_with_via.ipynb`) — computes pseudotime ordering of cells from raw scRNA-seq data using the VIA trajectory inference algorithm, producing the time-indexed sample files used downstream.
2. **Quantum optimization runs** (`final_real_1.py` - `final_real_10.py`) — fits a 14-qubit parameterized quantum circuit to the pseudotime-ordered data via negative log-likelihood minimization over POVM measurement outcomes.

---

## Background

Each qubit represents a gene (14 genes total). The Hamiltonian encodes directed regulatory interactions via terms of the form |1⟩⟨1|_i ⊗ Y_j, meaning gene i (when active) drives gene j. Cell states at each pseudotime point are treated as samples from a POVM measurement of the evolving quantum state. The optimizer learns the Hamiltonian interaction weights, effectively the regulatory adjacency matrix that best explains the observed single-cell distributions over pseudotime.

---


### Stage 1 — `pseudotime_with_via.ipynb`
Computes pseudotime from raw single-cell data using the [VIA](https://github.com/ShobiStassen/VIA) algorithm. 

### Stage 2 — Optimization Runs

All 10 runs share the same quantum circuit architecture and optimizer. They differ in **initial state parameterization** and **random seed**:

| Runs | θ (thetas) | φ (phis) | Weights | Description |
|---|---|---|---|---|
| 1–5 | Fixed at π/2 | Fixed at 0 | Random (seeded) | Only Hamiltonian weights are learned |
| 6–10 | Random (seeded) | Random (seeded) | Random (seeded) | Full parameter learning: θ, φ, and weights |

Each run uses a **distinct random seed** for initialization so that the 5 runs within each group provide independent restarts, enabling assessment of convergence consistency and landscape exploration.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pennylane` | Quantum circuit simulation |
| `jax` / `jaxlib` | Autodiff + JIT compilation |
| `optax` | Adam optimizer |
| `numpy` / `pandas` | Data I/O |
| `via` | Pseudotime trajectory inference (Stage 1) |
| `matplotlib` | Plotting |

Install with:
```bash
pip install pennylane jax jaxlib optax pandas matplotlib pyVIA
```


---

## Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `n_qubits` | 14 | Number of qubits / genes |
| `Nt` | 50 | Number of pseudotime points |
| `Nc` | 2499 | Number of cell samples per time point |
| `M` | 20 | Mini-batch size per time step |
| `num_steps` | 3000 | Total optimization steps |
| `lr` | `0.085 / sqrt(step/4 + 1)` | Decaying Adam learning rate |
| `n_weights` | 182 | Hamiltonian interaction weights (n × (n−1)) |
| `n_thetas` | 14 | Initial state polar angles (learned in runs 6–10) |
| `n_phis` | 14 | Initial state azimuthal angles (learned in runs 6–10) |

---

## Quantum Circuit Architecture

### `graph_inf_network(params, time)`
1. Apply RY(π/2) and RZ(φ_i) to each qubit to prepare the initial state |ψ₀⟩ (φ fixed at 0 in runs 1–5; learned in runs 6–10).
2. Evolve under the parameterized Hamiltonian `H` for pseudotime `t` using `qml.evolve`.
3. Return the full statevector |ψ(t)⟩.

### POVM
A 4-element POVM {E0, E1, E2, E3} is applied per qubit. The probability of a full system outcome is computed by tensor contraction of the per-qubit operators against |ψ(t)⟩.

### Loss Function
Negative log-likelihood:
```
L = -1/(Nt × M) × Σ_t Σ_j log( p(outcome_j | ψ(t)) + ε )
```


 

| File | Contents |
|---|---|
| `initial_weights.npy` | Initial Hamiltonian weight parameters |
| `initial_thetas.npy` | Initial θ angles |
| `initial_phis.npy` | Initial φ angles |
| `learned_params.npy` | Final optimized parameters |
| `weights_history.npy` | Weight parameters at every step |
| `thetas_history.npy` | θ history (constant across steps in runs 1–5) |
| `phis_history.npy` | φ history (constant across steps in runs 1–5) |
| `loss_history.npy` | Loss value at every step |

> Files are saved every step inside the training loop, so results are preserved even if a run is interrupted.

---

## Usage

**Step 1 — Compute pseudotime:**
```bash
python pseudotime_with_via.ipynb
```

**Step 2 — Run optimization (example for a single run):**
```bash
python final_real_1.py   # fixed θ/φ, seed 1
python final_real_6.py   # learned θ/φ, seed 1
```
All 10 runs can be submitted in parallel as independent HPC jobs.

---

## Notes

- `jax_enable_x64` is required for numerical precision in the quantum evolution.
- Hamiltonian weight parameters are passed through `tanh` scaling to bound them to `[−π/2, π/2]`.
- The Adam optimizer is re-initialized each step to implement a decaying learning rate. This resets moment estimates, so momentum does not accumulate across steps. To preserve Adam momentum while using a schedule, replace with `optax.adam(optax.schedules.cosine_decay_schedule(...))` initialized once before the loop.
- Data used in the paper can be found here: [link](https://drive.google.com/drive/folders/15B-uuOt3Nx-HbIBkYkX-vERSclwRqYhD?usp=sharing)
