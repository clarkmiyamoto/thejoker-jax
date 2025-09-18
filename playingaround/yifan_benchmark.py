import jax
jax.config.update("jax_enable_x64", True) # IMPORTANT: This is necessary for highly skewed distributions.

import jax.numpy as jnp
import jax.random as random
import numpy as np
import time

# Import custom modules
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)


def compare_jax_numpy():
    """Simple comparison of JAX vs NumPy for 50D case"""

    # Setup
    seed = 100
    dim = 20
    n_samples = 1000
    burn_in = 4000
    total_samples = n_samples + burn_in

    # Create problem
    np.random.seed(seed)
    cond_number = 1000
    eigenvals = 0.1 * np.linspace(1, cond_number, dim)
    H = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(H)
    precision = Q @ np.diag(eigenvals) @ Q.T
    precision = 0.5 * (precision + precision.T)

    true_mean = np.ones(dim)
    initial = np.zeros(dim)

    # Parameters
    params = {"n_chains_per_group": max(dim,20), "epsilon": 0.2, "n_leapfrog": 5, "beta": 1.0}

    print(f"{dim}D Gaussian test: {params}")

    # NumPy functions
    def gradient_np(x):
        if x.ndim == 1: x = x.reshape(1, -1)
        return np.einsum('jk,ij->ik', precision, x - true_mean)

    def potential_np(x):
        if x.ndim == 1: x = x.reshape(1, -1)
        centered = x - true_mean
        return 0.5 * np.einsum('ij,jk,ik->i', centered, precision, centered)

    # JAX functions
    precision_jax = jnp.array(precision)
    true_mean_jax = jnp.array(true_mean)

    def potential_jax(x):
        centered = x - true_mean_jax
        return 0.5 * jnp.dot(centered, precision_jax @ centered)

    def gradient_jax(x):
        return precision_jax @ (x - true_mean_jax)

    # Run NumPy
    try:
        from src.sampler_numpy import hamiltonian_side_move as hwm_np
        start = time.time()
        samples_np, acc_np = hwm_np(gradient_np, potential_np, initial, total_samples, **params)
        time_np = time.time() - start

        flat_np = samples_np[:, burn_in:, :].reshape(-1, dim)
        mean_np = np.mean(flat_np, axis=0)
        cov_np = np.cov(flat_np, rowvar=False)
        error_mean_np = np.linalg.norm(mean_np - true_mean)
        error_cov_np = np.linalg.norm(cov_np - precision)

        print(f"NumPy: accept={np.mean(acc_np):.3f}, mean error={error_mean_np:.3f}, cov error={error_cov_np:.3f}, time={time_np:.1f}s")
    except Exception as e:
        print(f"NumPy failed: {e}")

    # Run JAX
    try:
        from src.sampler_jax import hamiltonian_side_move as hwm_jax
        key = random.PRNGKey(seed)
        start = time.time()
        samples_jax, acc_jax = hwm_jax(
            potential_jax, jnp.array(initial), total_samples, gradient_jax,
            params["n_chains_per_group"], params["epsilon"], params["n_leapfrog"],
            params["beta"], 1, key
        )
        time_jax = time.time() - start

        # Handle array format
        if samples_jax.shape[0] == total_samples:
            flat_jax = samples_jax[burn_in:, :, :].reshape(-1, dim)
        else:
            flat_jax = samples_jax[:, burn_in:, :].reshape(-1, dim)

        mean_jax = jnp.mean(flat_jax, axis=0)
        cov_jax = jnp.cov(flat_jax, rowvar=False)
        error_mean_jax = jnp.linalg.norm(mean_jax - true_mean_jax)
        error_cov_jax = jnp.linalg.norm(cov_jax - precision_jax)

        print(f"JAX:   accept={jnp.mean(acc_jax):.3f}, mean error={error_mean_jax:.3f}, cov error={error_cov_jax:.3f}, time={time_jax:.1f}s")
    except Exception as e:
        print(f"JAX failed: {e}")

if __name__ == "__main__":
    compare_jax_numpy()