import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from typing import NamedTuple

from src.velocity import velocity
from src.likelihood import LogLikelihood

import hemcee

import jax
import jax.numpy as jnp

import corner
import matplotlib.pyplot as plt

def generate_poisson_observations(key: jax.random.PRNGKey, 
                                  N: int, 
                                  T: float):
    """
    Generate observation times with Poisson-like distribution.
    First observation at t=0, last at t=T, remaining N-2 Poisson distributed.
    
    Parameters:
    - key: PRNG Key
    - N: Number of observations
    - T: Total time span
    
    Returns:
    - time_obs: Array of observation times
    """
    if N <= 2:
        return jnp.array([0.0, T])
    
    # Generate N-1 intervals from exponential distribution
    # Mean interval should be T/(N-1) to fill the total time
    mean_interval = T / (N - 1)
    intervals = jax.random.exponential(key, shape=(N - 1,)) * mean_interval
    
    # Normalize intervals so their sum equals T
    intervals = intervals * T / jnp.sum(intervals)
    
    # Create observation times as cumulative sum
    time_obs = jnp.cumsum(jnp.concatenate([jnp.array([0]), intervals]))
    
    # Ensure last observation is exactly at T
    time_obs = time_obs.at[-1].set(T)
    
    return time_obs

def generate_rv_data(key: jax.random.PRNGKey, 
                     time_obs: jnp.ndarray, 
                     period: float, 
                     eccentricity: float, 
                     omega: float, 
                     phi0: float, 
                     K: float, 
                     v0: float, 
                     sigma: float):
    """
    Generate radial velocity data with noise.
    
    Parameters:
    - time_obs: Array of observation times
    - period: Orbital period (days)
    - eccentricity: Orbital eccentricity
    - omega: Argument of periastron (radians)
    - phi0: Phase offset (radians)
    - K: Semi-amplitude of velocity (m/s)
    - v0: Systemic velocity offset (m/s)
    - sigma: RV uncertainty (m/s)
    - seed: Random seed for reproducibility
    
    Returns:
    - time_obs: Observation times
    - rv_obs: Observed radial velocities with noise
    - rv_err: RV uncertainties
    - rv_true: True radial velocities (no noise)
    """
    if period < 0:
        raise ValueError()
    if (0 > eccentricity) and (eccentricity > 1):
        raise ValueError()
    if (0 > omega) and (omega > 2 * jnp.pi):
        raise ValueError()
    if (0 > phi0) and (phi0 > 2 * jnp.pi):
        raise ValueError()
    if sigma <= 0:
        raise ValueError()

    # Generate true RV curve
    rv_true = velocity(time_obs, period, eccentricity, omega, phi0, K, v0)
    
    # Add Gaussian noise
    n_observations = len(time_obs)
    rv_obs = rv_true + (sigma ** 2) * jax.random.normal(key=key, shape=(n_observations,))
    rv_err = jnp.full_like(rv_obs, sigma)
    
    return time_obs, rv_obs, rv_err, rv_true

def generate_fake_data(
        key: jax.random.PRNGKey,
        n_observations: int,
        total_time: float,
        period: float,
        eccentricity: float,
        omega: float,
        phi0: float,
        K: float,
        v0: float,
        sigma: float
):
    '''
    Generates fake data

    Args:
        key: RNG Key
        n_observations: How many observations do we have access to?
        total_time: 
    '''
    keys = jax.random.split(key, 2)
    # Generate clustered observation times
    time_obs = generate_poisson_observations(keys[0], n_observations, total_time)
        
    # Generate RV data
    time_obs, rv_obs, rv_err, rv_true = generate_rv_data(
        keys[1],
        time_obs, 
        period, eccentricity, omega, phi0, K, v0, 
        sigma
    )
    return time_obs, rv_obs, rv_err

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 3)

    n_observations = 50
    total_time = 100. # days
    period = 365. # days
    eccentricity = 0.3
    omega = jnp.pi # rads
    phi0 = jnp.pi # rads
    K = 45. # meters/sec
    v0 = 100. # meters/sec
    sigma = 5. # sqrt(meters/sec)


    times, rv_obs, rv_err = generate_fake_data(keys[0], 
                                               n_observations, 
                                               total_time,
                                               period, eccentricity, omega, phi0, K, v0,
                                               sigma)
    log_prob = LogLikelihood(times, rv_obs, rv_err)

    total_chains = 30
    dim = 6
    num_samples = 10**4
    warmup = 10**4
    thin_by = 3

    sampler = hemcee.HamiltonianEnsembleSampler(total_chains, dim, log_prob, L=1, step_size=0.001, adapt_length=False, adapt_step_size=False)
    
    # Initialize random starting positions for each chain
    init_period = jax.random.uniform(keys[2], shape=(total_chains, 1), minval=300, maxval=400)
    init_eccentricity = jax.random.uniform(jax.random.PRNGKey(1), shape=(total_chains, 1), minval=0.0, maxval=0.9)
    init_omega = jax.random.uniform(jax.random.PRNGKey(2), shape=(total_chains, 1), minval=0.0, maxval=2*jnp.pi)
    init_phi0 = jax.random.uniform(jax.random.PRNGKey(3), shape=(total_chains, 1), minval=0.0, maxval=2*jnp.pi)
    init_K = jax.random.uniform(jax.random.PRNGKey(4), shape=(total_chains, 1), minval=10.0, maxval=80.0)
    init_v0 = jax.random.uniform(jax.random.PRNGKey(5), shape=(total_chains, 1), minval=50.0, maxval=150.0)
    
    initial_state = jnp.concatenate([init_period, init_eccentricity, init_omega, init_phi0, init_K, init_v0], axis=1)
    
    samples = sampler.run_mcmc(keys[2], initial_state, 
                     num_samples=num_samples, 
                     warmup=warmup,
                     thin_by=thin_by,
                     show_progress=True)
    
    print('Adaptation:')
    final_StepSize, final_IntegrationLength = sampler.adapter.finalize(sampler.adapter_state)
    print(f'  Step Size: {final_StepSize}')
    print(f'  Integration Length: {final_IntegrationLength}')

    print('Acceptance Rates:')
    warmup_acceptance_rate = sampler.diagnostics_warmup['acceptance_rate']
    main_acceptance_rate = sampler.diagnostics_main['acceptance_rate']
    print(f'  Warmup: {warmup_acceptance_rate}')
    print(f'  Main: {main_acceptance_rate}')

    _ = corner.corner(samples.reshape(-1, dim).__array__())
    plt.show()