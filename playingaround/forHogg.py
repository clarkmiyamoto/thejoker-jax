import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)
import argparse

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

import corner
import matplotlib.pyplot as plt

from src.velocity import velocity
from src.likelihood import LogLikelihood

import hemcee

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

def generate_multi_planet_rv_data(key: jax.random.PRNGKey,
                                  time_obs: jnp.ndarray,
                                  params_list: list,
                                  sigma: float):
    """
    Generate radial velocity data for multiple planets.
    
    Parameters:
    - key: PRNG Key
    - time_obs: Array of observation times
    - params_list: List of planet parameters, each as [period, eccentricity, omega, phi0, K]
    - sigma: RV uncertainty (m/s)
    
    Returns:
    - time_obs: Observation times
    - rv_obs: Observed radial velocities with noise
    - rv_err: RV uncertainties
    - rv_true: True radial velocities (no noise)
    """
    if sigma <= 0:
        raise ValueError()
    
    # Sum contributions from all planets (no v0 yet)
    rv_true = jnp.zeros_like(time_obs)
    for params in params_list:
        period, eccentricity, omega, phi0, K = params
        rv_true += velocity(time_obs, period, eccentricity, omega, phi0, K, 0.0)
    
    # Add v0 at the end
    # v0 is included in the last planet's params or could be added separately
    # For now, we'll assume it's 0 and add it in the main code
    
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

def calculate_init_bounds(true_params: dict, coverage_fraction: float):
    """
    Calculate initialization bounds as a fraction centered on true parameter values.
    
    Parameters:
    - true_params: Dictionary with true parameter values:
        {'period': float, 'eccentricity': float, 'omega': float, 
         'phi0': float, 'K': float, 'v0': float}
    - coverage_fraction: Fraction of parameter space to cover (e.g., 0.2 = ±20%)
    
    Returns:
    - Dictionary with 'minval' and 'maxval' arrays for each parameter
    """
    bounds = {}
    
    # Period: symmetric bounds around true value
    period_range = true_params['period'] * coverage_fraction
    bounds['period'] = {
        'minval': max(0.0, true_params['period'] - period_range),
        'maxval': true_params['period'] + period_range
    }
    
    # Eccentricity: symmetric bounds, clamped to [0, 1]
    ecc_range = true_params['eccentricity'] * coverage_fraction
    bounds['eccentricity'] = {
        'minval': max(0.0, true_params['eccentricity'] - ecc_range),
        'maxval': min(1.0, true_params['eccentricity'] + ecc_range)
    }
    
    # Omega: symmetric bounds centered on true value, clipped to [0, 2π]
    omega_range = true_params['omega'] * coverage_fraction
    bounds['omega'] = {
        'minval': max(0.0, true_params['omega'] - omega_range),
        'maxval': min(2 * jnp.pi, true_params['omega'] + omega_range)
    }
    
    # Phi0: symmetric bounds centered on true value, clipped to [0, 2π]
    phi0_range = true_params['phi0'] * coverage_fraction
    bounds['phi0'] = {
        'minval': max(0.0, true_params['phi0'] - phi0_range),
        'maxval': min(2 * jnp.pi, true_params['phi0'] + phi0_range)
    }
    
    # K: symmetric bounds around true value
    K_range = true_params['K'] * coverage_fraction
    bounds['K'] = {
        'minval': max(0.0, true_params['K'] - K_range),
        'maxval': true_params['K'] + K_range
    }
    
    # v0: symmetric bounds around true value
    v0_range = abs(true_params['v0']) * coverage_fraction
    bounds['v0'] = {
        'minval': true_params['v0'] - v0_range,
        'maxval': true_params['v0'] + v0_range
    }
    
    return bounds

def plots(samples: jnp.ndarray, log_probs: jnp.ndarray, times: jnp.ndarray, rv_obs: jnp.ndarray, rv_err: jnp.ndarray, n_planets: int = 2):
    # Reshape samples for analysis
    flat_samples = samples.reshape(-1, samples.shape[-1])
    flat_log_probs = log_probs.reshape(-1)

    
    # Find MAP estimate using sampler's built-in log probability method
    print('\nFinding MAP estimate...')
    map_idx = jnp.argmax(flat_log_probs)
    map_estimate = flat_samples[map_idx]
    
    print(f'MAP estimate:')
    for i in range(n_planets):
        idx = i * 5
        print(f'  Planet {i+1}:')
        print(f'    Period: {map_estimate[idx]:.3f} days')
        print(f'    Eccentricity: {map_estimate[idx+1]:.3f}')
        print(f'    Omega: {map_estimate[idx+2]:.3f} rad')
        print(f'    Phi0: {map_estimate[idx+3]:.3f} rad')
        print(f'    K: {map_estimate[idx+4]:.3f} m/s')
    print(f'  v0: {map_estimate[-1]:.3f} m/s')
    
    # Generate smooth time grid for RV curves
    time_grid = jnp.linspace(times.min(), times.max(), 500)
    
    # Compute multi-planet RV using LogLikelihood's velocity_model
    from src.likelihood import LogLikelihood
    log_prob_temp = LogLikelihood(time_grid, jnp.zeros_like(time_grid), jnp.ones_like(time_grid), n_planets=n_planets)
    rv_map = log_prob_temp.velocity_model(map_estimate)
    
    # Create labels for corner plot
    labels = []
    for i in range(n_planets):
        labels.extend([f'P{i+1}', f'e{i+1}', f'ω{i+1}', f'φ{i+1}', f'K{i+1}'])
    labels.append('v0')
    
    # Plot corner plot
    _ = corner.corner(flat_samples.__array__(), labels=labels)
    plt.savefig('hogg_corner.png')
    plt.show()
    
    # Plot RV curve with sampled curves and MAP estimate
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sample a subset of posterior samples to plot
    n_sampled_curves = 100
    n_total_samples = flat_samples.shape[0]
    if n_sampled_curves > n_total_samples:
        n_sampled_curves = n_total_samples
    
    # Randomly select samples (excluding MAP to avoid duplicate)
    sample_indices = jnp.arange(n_total_samples)
    sample_indices = sample_indices[sample_indices != map_idx]
    rng_key_plot = jax.random.PRNGKey(42)
    selected_indices = jax.random.choice(rng_key_plot, sample_indices, shape=(n_sampled_curves,), replace=False)
    
    # Plot sampled RV curves in blue with varying alpha
    for i, idx in enumerate(selected_indices):
        sample = flat_samples[idx]
        rv_sample = log_prob_temp.velocity_model(sample)
        ax.plot(time_grid, rv_sample, '-', color='blue', linewidth=1, alpha=0.1)
    
    # Plot MAP estimate with higher alpha and thicker line
    ax.plot(time_grid, rv_map, '-', color='red', linewidth=2.5, 
            label='MAP Estimate', alpha=0.9)
    
    # Plot observed data
    ax.errorbar(times, rv_obs, yerr=rv_err, fmt='o', color='black', 
                label='Observed RV', capsize=3, markersize=6, alpha=0.7)
    
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Radial Velocity (m/s)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Radial Velocity: Observed vs MAP Estimate', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('hogg_rv.png')
    plt.show()

def initalize_sampler(method: str, total_chains: int, dim: int, log_prob: callable):
    if method == 'hmc':
        sampler = hemcee.HamiltonianSampler(total_chains, dim, log_prob, L=15, step_size=0.05, adapt_length=True, adapt_step_size=True)
    elif method == 'hmc_walk':
        from hemcee.moves.hamiltonian.hmc_walk import hmc_walk_move
        sampler = hemcee.HamiltonianEnsembleSampler(total_chains, dim, log_prob, L=15, step_size=0.05, move=hmc_walk_move, adapt_length=True, adapt_step_size=True)
    elif method == 'hmc_side':
        from hemcee.moves.hamiltonian.hmc_side import hmc_side_move
        sampler = hemcee.HamiltonianEnsembleSampler(total_chains, dim, log_prob, L=15, step_size=0.05, move=hmc_side_move, adapt_length=True, adapt_step_size=True)
    elif method == 'stretch':
        from hemcee.moves.vanilla.stretch import stretch_move
        sampler = hemcee.EnsembleSampler(total_chains, dim, log_prob, move=stretch_move)
    elif method == 'walk':
        from hemcee.moves.vanilla.walk import walk_move
        sampler = hemcee.EnsembleSampler(total_chains, dim, log_prob, move=walk_move)
    elif method == 'side':
        from hemcee.moves.vanilla.side import side_move
        sampler = hemcee.EnsembleSampler(total_chains, dim, log_prob, move=side_move)
    return sampler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['hmc', 'hmc_walk', 'stretch', 'walk'], required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4)

    n_observations = 75
    observation_coverage = 0.7

    total_time = 365 * observation_coverage # days
    
    # Two-planet system parameters
    # Planet 1
    period1 = 365  # days
    eccentricity1 = 0.3
    omega1 = jnp.pi # rads
    phi0_1 = jnp.pi # rads
    K1 = 45. # meters/sec
    
    # Planet 2
    period2 = 180. # days
    eccentricity2 = 0.5
    omega2 = 0.5 * jnp.pi # rads
    phi0_2 = 0.5 * jnp.pi # rads
    K2 = 30. # meters/sec
    
    # Shared parameters
    v0 = 100. # meters/sec
    sigma = 2. # sqrt(meters/sec)
    
    # Generate two-planet data
    keys_split = jax.random.split(keys[0], 2)
    time_obs = generate_poisson_observations(keys_split[0], n_observations, total_time)
    
    times, rv_obs, rv_err, rv_true = generate_multi_planet_rv_data(
        keys_split[1],
        time_obs,
        [[period1, eccentricity1, omega1, phi0_1, K1],
         [period2, eccentricity2, omega2, phi0_2, K2]],
        sigma
    )
    # Add v0 to the true velocities
    rv_true = rv_true + v0
    rv_obs = rv_obs + v0
    
    log_prob = LogLikelihood(times, rv_obs, rv_err, n_planets=2)

    total_chains = 40
    dim = 11  # 5 parameters * 2 planets + 1 v0
    warmup = 700
    num_samples = 1000
    thin_by = 5
    
    sampler = initalize_sampler(args.method, total_chains, dim, log_prob)

    # Calculate initialization bounds as a fraction centered on true values
    coverage_fraction = 0.05 # sets fract of bounds that the inital samples are drawn from
    
    # Planet 1 bounds
    true_params1 = {
        'period': period1,
        'eccentricity': eccentricity1,
        'omega': omega1,
        'phi0': phi0_1,
        'K': K1,
        'v0': v0
    }
    bounds1 = calculate_init_bounds(true_params1, coverage_fraction)
    
    # Planet 2 bounds
    true_params2 = {
        'period': period2,
        'eccentricity': eccentricity2,
        'omega': omega2,
        'phi0': phi0_2,
        'K': K2,
        'v0': v0
    }
    bounds2 = calculate_init_bounds(true_params2, coverage_fraction)
    
    # Initialize random starting positions for each chain using calculated bounds
    init_keys = jax.random.split(keys[2], 11)  # 2 planets * 5 params + 1 v0
    
    # Planet 1 initialization
    init_period1 = jax.random.uniform(init_keys[0], shape=(total_chains, 1), 
                                      minval=bounds1['period']['minval'], 
                                      maxval=bounds1['period']['maxval'])
    init_eccentricity1 = jax.random.uniform(init_keys[1], shape=(total_chains, 1), 
                                            minval=bounds1['eccentricity']['minval'], 
                                            maxval=bounds1['eccentricity']['maxval'])
    init_omega1 = jax.random.uniform(init_keys[2], shape=(total_chains, 1), 
                                     minval=bounds1['omega']['minval'], 
                                     maxval=bounds1['omega']['maxval'])
    init_phi0_1 = jax.random.uniform(init_keys[3], shape=(total_chains, 1), 
                                     minval=bounds1['phi0']['minval'], 
                                     maxval=bounds1['phi0']['maxval'])
    init_K1 = jax.random.uniform(init_keys[4], shape=(total_chains, 1), 
                                 minval=bounds1['K']['minval'], 
                                 maxval=bounds1['K']['maxval'])
    
    # Planet 2 initialization
    init_period2 = jax.random.uniform(init_keys[5], shape=(total_chains, 1), 
                                      minval=bounds2['period']['minval'], 
                                      maxval=bounds2['period']['maxval'])
    init_eccentricity2 = jax.random.uniform(init_keys[6], shape=(total_chains, 1), 
                                            minval=bounds2['eccentricity']['minval'], 
                                            maxval=bounds2['eccentricity']['maxval'])
    init_omega2 = jax.random.uniform(init_keys[7], shape=(total_chains, 1), 
                                     minval=bounds2['omega']['minval'], 
                                     maxval=bounds2['omega']['maxval'])
    init_phi0_2 = jax.random.uniform(init_keys[8], shape=(total_chains, 1), 
                                     minval=bounds2['phi0']['minval'], 
                                     maxval=bounds2['phi0']['maxval'])
    init_K2 = jax.random.uniform(init_keys[9], shape=(total_chains, 1), 
                                 minval=bounds2['K']['minval'], 
                                 maxval=bounds2['K']['maxval'])
    
    # v0 initialization
    init_v0 = jax.random.uniform(init_keys[10], shape=(total_chains, 1), 
                                 minval=bounds1['v0']['minval'], 
                                 maxval=bounds1['v0']['maxval'])
    
    initial_state = jnp.concatenate([init_period1, init_eccentricity1, init_omega1, init_phi0_1, init_K1,
                                     init_period2, init_eccentricity2, init_omega2, init_phi0_2, init_K2,
                                     init_v0], axis=1)
    
    samples = sampler.run_mcmc(keys[3], initial_state, 
                     num_samples=num_samples, 
                     warmup=warmup,
                     thin_by=thin_by,
                     show_progress=True)
    
    print('Diagnostics:')
    
    print('Autocorrelation Time:')
    try:
        tau = hemcee.autocorr.integrated_time(samples)
        print(f'  Integrated time: {tau}')
    except Exception as e:
        print(f'  Error calculating autocorrelation time: {e}')
        tau = None

    print('Adaptation:')
    try: 
        final_StepSize, final_IntegrationLength = sampler.adapter.finalize(sampler.adapter_state)
        print(f'  Step Size: {final_StepSize}')
        print(f'  Integration Length: {final_IntegrationLength}')
    except Exception as e:
        print('  No adapter for this method')

    print('Acceptance Rates:')
    try:
        warmup_acceptance_rate = sampler.diagnostics_warmup['acceptance_rate']
        print(f'  Warmup: {warmup_acceptance_rate}')
    except Exception as e:
        print('  No acceptance rates for this method')
    main_acceptance_rate = sampler.diagnostics_main['acceptance_rate']
    print(f'  Main: {main_acceptance_rate}')

    print('Plotting...')
    log_probs = sampler.get_logprob(discard=warmup, thin=thin_by)
    plots(samples, log_probs, times, rv_obs, rv_err, n_planets=2)