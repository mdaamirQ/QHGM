#!/usr/bin/env python
# coding: utf-8

# ## Installing packages

# ## Importing packages

# In[4]:

import os
import pennylane as qml
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np
import time
import optax
import matplotlib.pyplot as plt
from functools import reduce
from pathlib import Path
import pandas as pd

jax.config.update("jax_enable_x64", True)
# Force PennyLane to recognize JAX
import pennylane.pulse.parametrized_evolution
pennylane.pulse.parametrized_evolution.has_jax = True
 

print("JAX version:", jax.__version__)
print("PennyLane version:", qml.__version__)



import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
from pathlib import Path


# set these
Nt = 50
Nc = 2499

data=pd.read_csv(f"/nfs/turbo/umms-ukarvind/rsudhars/quantum/final_simulations/split_files/output_df_{Nt}_rest.csv", index_col=0)

times_sampled = jnp.array(data.index, dtype=jnp.float64)
#samples = jnp.array(data.values, dtype=jnp.int64)
 

# Ensure it goes through numpy first, then to JAX with correct dtype
samples = jnp.array(np.array(data.values, dtype=np.int64))
# optional sanity checks
assert times_sampled.shape[0] == Nt, f"times length {times_sampled.shape[0]} != Nt={Nt}"
assert samples.shape[:2] == (Nt, Nc), f"samples shape {samples.shape} != (Nt={Nt}, Nc={Nc}, ...)"

# dtype checks (JAX arrays compare fine to NumPy dtypes)
assert samples.dtype == jnp.int64,   f"samples dtype {samples.dtype} != int64"
assert times_sampled.dtype == jnp.float64, f"times_sampled dtype {times_sampled.dtype} != float64"

# --- NaN check in samples ---
has_nan_samples = bool(jnp.any(jnp.isnan(samples.astype(jnp.float64))))
assert not has_nan_samples, "samples contains NaN(s)"

# --- Zero check for times_sampled ---
has_zero_times = bool(jnp.any(times_sampled == 0.0))
assert not has_zero_times, "times_sampled contains one or more zeros"

### Simulation Setup
n_qubits = 14

n_thetas = n_qubits
n_phis = n_qubits
n_weights = n_qubits*(n_qubits-1)
n_params = n_weights + n_thetas + n_phis

ns_povm = 4
nPOVMS = ns_povm**n_qubits

# Define your ranges
theta_min, theta_max = 0, jnp.pi      # typical Bloch sphere polar angle range
phi_min, phi_max = 0, 2 * jnp.pi      # azimuthal phase range
w_min, w_max = -jnp.pi/2, jnp.pi/2    # example range for weights

### POVMS construction
I = jnp.eye(2)
X = jnp.array([[0, 1],
              [1, 0]])
Y = jnp.array([[0, -1j],
              [1j, 0]])
Z = jnp.array([[1, 0],
              [0, -1]])

E0 = (I + X / 2 + jnp.sqrt(3) * Z / 2) / 4
E1 = (I - X / 2 - jnp.sqrt(2) * Y / 2 + Z / 2) / 4
E2 = (I - X / 2 + jnp.sqrt(2) * Y / 2 - Z / 2) / 4
E3 = (I + X / 2 - jnp.sqrt(3) * Z / 2) / 4
E = jnp.array([E0,E1,E2,E3])

### Supplementary Functions
def idx2base4(num, base = ns_povm):
    """Convert integer `num` to n-digit base-4 JAX array (zero-padded)."""
    digits = []
    for _ in range(n_qubits):
        digits.append(num % base)
        num //= base
    return jnp.array(digits[::-1]) ### Eg. 15 -> jnp.array([0,3,3]) ### Eg. jnp.array([0,3,3]) -> E0 ⊗ E3 ⊗ E3


### Regulatory Network Hamiltonian
Hij = [qml.Projector([0,1], wires = [i]) @ qml.PauliY(j) for i in range(n_qubits) for j in range(n_qubits) if i != j]
# print(Hij) ### Structure: |1><1|_0 ⊗ Y_1 + |1><1|_0 ⊗ Y_2 +..... (row wise weights in adjacency matrix. Control Row and Target Column)

### Parameterized Hamiltonian Construction
coeffs = [lambda p, t: p for _ in range(n_weights)]
H = qml.dot(coeffs, Hij)

### Parameterized Regulatory Network
dev = qml.device("default.qubit", wires=range(n_qubits))

@jax.jit
@qml.qnode(dev, interface="jax")
def graph_inf_network(params,time):

  thetas = params[:n_thetas]
  phis = params[n_thetas:n_thetas+n_phis]
  weights = w_max*jnp.tanh(params[n_thetas+n_phis:])

  for i in range(n_qubits):
        qml.RY(thetas[i], wires=i) 
        qml.RZ(phis[i], wires=i)

  qml.evolve(H)(weights, t=time)

  return qml.state()

@jax.jit
def pred_prob_i(i, psi):
    seq = idx2base4(i)                         # [n_qubits]
    v = psi.reshape((2,) * n_qubits)           # (2,2,...,2)
    # apply E[seq[k]] along axis k, keep tensor rank constant
    for k in range(n_qubits):
        v = jnp.tensordot(E[seq[k]], v, axes=([1],[k]))   # -> (2, ...drop k...)
        v = jnp.moveaxis(v, 0, k)                          # put axis back to position k
    v = v.reshape(psi.shape)
    # return jnp.real(jnp.vdot(psi, v)).astype(jnp.float32)
    return jnp.real(jnp.vdot(psi, v))
@jax.jit
def povm_prediction(psi, indices):
    K = indices.shape[0]
    prob = jnp.zeros((K,))

    def body(i, prob):
        idx = indices[i]
        p = pred_prob_i(idx, psi)
        prob = prob.at[i].set(p)
        return prob

    return lax.fori_loop(0, K, body, prob)

def get_batch_indices(sample_bank, batch_idx, random, rng_key):
    """
    Return (Nt, M) indices either by deterministic batching or random subsampling.
    """
    def true_fun(_):
        keys = jax.random.split(rng_key, Nt)
        return jax.vmap(lambda k, row: jax.random.choice(k, row, shape=(M,), replace=False))(keys, sample_bank)

    def false_fun(_):
        start = batch_idx * M
        return lax.dynamic_slice(sample_bank, (0, start), (Nt, M))

    return lax.cond(random, true_fun, false_fun, None)

### Negative Log
@jax.jit
def cost(params, batch_idx, random=False, rng_key=None):
    batch_indices = get_batch_indices(samples, batch_idx, random, rng_key)

    def body(i, total_cost):
        t = times_sampled[i]
        psi_t = graph_inf_network(params, t)
        idxs = batch_indices[i]
        pred_probs = povm_prediction(psi_t, idxs)
        return total_cost - jnp.sum(jnp.log(pred_probs + 1e-12))

    total = lax.fori_loop(0, Nt, body, 0.0)
    return total / (Nt * M)
### Jitting Loss and Grad
loss_grad = jax.jit(jax.value_and_grad(cost))

#### Optimization:
import optax

# ------------------------------------
# Optimization loop setup
# ------------------------------------
# key = jax.random.PRNGKey(0)

exp_num= 12
key = jax.random.key(exp_num)

M = 20 # Block size



exp=f"Experiment_{exp_num}"

### Initialize parameters
key1, key2 = jax.random.split(jax.random.key(exp_num))

weights_init = jax.random.uniform(key1, shape=(n_weights,), minval= w_min, maxval= w_max)
thetas_init = jax.random.uniform(key2, shape=(n_thetas,), minval= theta_min, maxval= theta_max)
phis_init = jax.random.uniform(key2, shape=(n_phis,), minval= phi_min, maxval= phi_max)

# weights_init = ((w_min+w_max)/2)*jnp.ones(n_weights)
# thetas_init = ((theta_min+theta_max)/2)*jnp.ones(n_thetas)
# phis_init = ((phi_min+phi_max)/2)*jnp.ones(n_phis)

learned_params = jnp.concatenate([thetas_init, phis_init, weights_init])  # shape = (n_thetas,n_weights +)

initial_params = learned_params

# Adam optimizer setup
lr = 0.001
opt = optax.adam(lr)
opt_state = opt.init(learned_params)

# Training loop
num_steps = 3000
loss_history = []
thetas_history = []
phis_history = []
weights_history = []

start = time.time()
init_cost, init_grad = loss_grad(initial_params, 0, random=False, rng_key=key)
end = time.time()
print("Time: ", end-start," secs")

batch_idx = 0
num_sample_batches = Nc/M
random_start_step = num_sample_batches  # start randomness after evaluating all batches

start_time = time.time()
for step in range(num_steps):
    key = jax.random.fold_in(key, step)

    # Variable learning rate (example: 0.1 / sqrt(step/4+1))
    lr = 0.085 / jnp.sqrt(step/4 + 1)
    opt = optax.adam(lr)

    # Toggle between phases
    if step < random_start_step:
        # loss_val = cost(learned_params, batch_idx, random=False)
        # loss_val, grads = jax.value_and_grad(cost)(learned_params, batch_idx, random=False, rng_key=key)
        loss_val, grads = loss_grad(learned_params, batch_idx, random=False, rng_key=key)

        batch_idx = int((batch_idx + 1) % num_sample_batches)
    else:
        # loss_val = cost(learned_params, batch_idx=0, random=True, rng_key=key)
        # loss_val, grads = jax.value_and_grad(cost)(learned_params,batch_idx=0,random=True, rng_key=key)
        loss_val, grads = loss_grad(learned_params,batch_idx=0,random=True, rng_key=key)

    updates, opt_state = opt.update(grads, opt_state)
    learned_params = optax.apply_updates(learned_params, updates)

    loss_history.append(loss_val.item())
    thetas_history.append(learned_params[:n_thetas])
    phis_history.append(learned_params[n_thetas:n_thetas+n_phis])
    weights_history.append(learned_params[n_thetas+n_phis:])
    # Build folder path like ./runs/14Q_Nt45_Ns1250
    stem = Path(f"/nfs/turbo/umms-ukarvind/rsudhars/quantum/final_simulations/real_data/{exp}")

    stem.mkdir(parents=True, exist_ok=True)
     
    print("Saving to:", stem.resolve())
    np.save(stem / "initial_weights.npy", weights_init)
    np.save(stem / "initial_thetas.npy", thetas_init)
    np.save(stem / "initial_phis.npy", phis_init)
    np.save(stem / "thetas_history.npy", thetas_history)
    np.save(stem / "phis_history.npy", phis_history)
    np.save(stem / "weights_history.npy", weights_history)
    np.save(stem / "loss_history.npy", loss_history)
    np.save(stem / "learned_params.npy", learned_params)
 
    if step % 1 == 0:
        print(f"Step {step}: Loss = {loss_val:.6f}")
	

end_time = time.time()
print("Training Time: ", (end_time-start_time)/3600, " hrs")

from pathlib import Path
import numpy as np


stem = Path(f"/nfs/turbo/umms-ukarvind/rsudhars/quantum/final_simulations/real_data/{exp}")

# Create directory
stem.mkdir(parents=True, exist_ok=True)

print("Saving to:", stem.resolve())

# Save arrays
np.save(stem / "initial_weights.npy", weights_init)
np.save(stem / "initial_thetas.npy", thetas_init)
np.save(stem / "initial_phis.npy", phis_init)
np.save(stem / "thetas_history.npy", thetas_history)
np.save(stem / "phis_history.npy", phis_history)
np.save(stem / "weights_history.npy", weights_history)
np.save(stem / "loss_history.npy", loss_history)
np.save(stem / "learned_params.npy", learned_params)

