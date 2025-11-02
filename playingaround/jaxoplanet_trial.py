import numpy as np
from jax import numpy as jnp
from jaxoplanet.orbits import keplerian
import jax
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)

import hemcee
import corner
import matplotlib.pyplot as plt



def rv_logprob(theta, t, v_obs, sigma, jitter=0.0):
    """
    Log probability for emcee.

    Parameters
    ----------
    theta : array-like
        [P, e, omega, t_peri, K, gamma]
        P, t_peri in days
        omega in radians
        K, gamma in same RV units (e.g. m/s)
    t : array-like
        Observation times (days)
    v_obs : array-like
        Observed RVs
    sigma : array-like
        Measurement uncertainties (same shape as v_obs)
    jitter : float, optional
        Additional noise term (default: 0.0)

    Returns
    -------
    lnprob : float
    """
    P, e, omega, t_peri, K, gamma = theta

    # Enforce basic physical bounds using JAX-compatible operations
    # Return -inf if any bounds are violated
    # Note: Use jnp.logical_and for JAX compatibility
    valid = jnp.logical_and(
        jnp.logical_and(P > 0, e >= 0),
        jnp.logical_and(e < 1, K >= 0)
    )
    log_prob_value = jnp.where(
        valid,
        _compute_logprob(P, e, omega, t_peri, K, gamma, jitter, t, v_obs, sigma),
        -jnp.inf
    )
    
    return log_prob_value


def _compute_logprob(P, e, omega, t_peri, K, gamma, jitter, t, v_obs, sigma):
    """Helper function to compute log probability without bounds checking."""
    # Build system
    # Central requires exactly two of mass, radius, density
    star = keplerian.Central(mass=1.0, radius=1.0)  # Solar mass and radius
    system = keplerian.System(star).add_body(
        period=P,
        eccentricity=e,
        omega_peri=omega,
        time_peri=t_peri,
        radial_velocity_semiamplitude=K,
    )

    # Compute model
    v_model = system.radial_velocity(jnp.array(t))[0] + gamma

    # Gaussian log-likelihood
    var = sigma**2 + jitter**2
    resid = v_obs - v_model
    lnlike = -0.5 * (jnp.sum(resid**2 / var) + jnp.sum(jnp.log(2.0 * jnp.pi * var)))

    # (Optional) add log-priors here (currently flat)
    lnprior = 0.0

    return lnprior + lnlike


class JaxoplanetLogLikelihood:
    """
    Log likelihood class compatible with hemcee sampler.
    Similar interface to LogLikelihood from src.likelihood.
    """
    
    def __init__(self, 
                 times: jnp.ndarray,
                 observed_data: jnp.ndarray,
                 uncertainty: jnp.ndarray,
                 jitter: float = 0.0):
        """
        Args:
            times: Observation times (days)
            observed_data: Observed radial velocities
            uncertainty: Measurement uncertainties (same shape as observed_data)
            jitter: Additional noise term (default: 0.0)
        """
        self.times = times
        self.observed_data = observed_data
        self.uncertainty = uncertainty
        self.jitter = jitter
    
    def __call__(self, params: jnp.ndarray) -> float:
        """
        Compute log probability for given parameters.
        
        Args:
            params: Array of shape (6,) containing [period, eccentricity, omega, t_peri, K, gamma]
                - period: Orbital period (days)
                - eccentricity: Orbital eccentricity (0 <= e < 1)
                - omega: Argument of periastron (radians)
                - t_peri: Time of periastron passage (days)
                - K: Radial velocity semi-amplitude (m/s)
                - gamma: Systemic velocity offset (m/s)
        
        Returns:
            Log probability (float)
        """
        return rv_logprob(params, self.times, self.observed_data, self.uncertainty, self.jitter)


def generate_poisson_observations(key: jax.random.PRNGKey, N: int, T: float):
    """
    Generate observation times with Poisson-like distribution.
    
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
    mean_interval = T / (N - 1)
    intervals = jax.random.exponential(key, shape=(N - 1,)) * mean_interval
    
    # Normalize intervals so their sum equals T
    intervals = intervals * T / jnp.sum(intervals)
    
    # Create observation times as cumulative sum
    time_obs = jnp.cumsum(jnp.concatenate([jnp.array([0]), intervals]))
    
    # Ensure last observation is exactly at T
    time_obs = time_obs.at[-1].set(T)
    
    return time_obs


def generate_fake_rv_data(key: jax.random.PRNGKey,
                         time_obs: jnp.ndarray,
                         period: float,
                         eccentricity: float,
                         omega: float,
                         t_peri: float,
                         K: float,
                         gamma: float,
                         sigma: float):
    """
    Generate radial velocity data using jaxoplanet with noise.
    
    Parameters:
    - key: PRNG Key
    - time_obs: Array of observation times (days)
    - period: Orbital period (days)
    - eccentricity: Orbital eccentricity
    - omega: Argument of periastron (radians)
    - t_peri: Time of periastron passage (days)
    - K: Semi-amplitude of velocity (m/s)
    - gamma: Systemic velocity offset (m/s)
    - sigma: RV uncertainty (m/s)
    
    Returns:
    - time_obs: Observation times
    - rv_obs: Observed radial velocities with noise
    - rv_err: RV uncertainties
    - rv_true: True radial velocities (no noise)
    """
    # Build system using jaxoplanet
    # Central requires exactly two of mass, radius, density
    star = keplerian.Central(mass=1.0, radius=1.0)  # Solar mass and radius
    system = keplerian.System(star).add_body(
        period=period,
        eccentricity=eccentricity,
        omega_peri=omega,
        time_peri=t_peri,
        radial_velocity_semiamplitude=K,
    )
    
    # Compute true RV curve
    rv_true = system.radial_velocity(jnp.array(time_obs))[0] + gamma
    
    # Add Gaussian noise
    n_observations = len(time_obs)
    rv_obs = rv_true + sigma * jax.random.normal(key=key, shape=(n_observations,))
    rv_err = jnp.full_like(rv_obs, sigma)
    
    return time_obs, rv_obs, rv_err, rv_true


def generate_fake_data(key: jax.random.PRNGKey,
                      n_observations: int,
                      total_time: float,
                      period: float,
                      eccentricity: float,
                      omega: float,
                      t_peri: float,
                      K: float,
                      gamma: float,
                      sigma: float):
    """
    Generate fake RV data for testing.
    
    Args:
        key: RNG Key
        n_observations: Number of observations
        total_time: Total time span (days)
        period: Orbital period (days)
        eccentricity: Orbital eccentricity
        omega: Argument of periastron (radians)
        t_peri: Time of periastron passage (days)
        K: Semi-amplitude (m/s)
        gamma: Systemic velocity (m/s)
        sigma: Measurement uncertainty (m/s)
    
    Returns:
        times, rv_obs, rv_err
    """
    keys = jax.random.split(key, 2)
    
    # Generate observation times
    time_obs = generate_poisson_observations(keys[0], n_observations, total_time)
    
    # Generate RV data
    time_obs, rv_obs, rv_err, _ = generate_fake_rv_data(
        keys[1],
        time_obs,
        period, eccentricity, omega, t_peri, K, gamma, sigma
    )
    
    return time_obs, rv_obs, rv_err


if __name__ == '__main__':
    # Set up RNG keys
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)
    
    # Parameters for fake data generation
    n_observations = 50
    total_time = 100.0  # days
    period = 365.0  # days
    eccentricity = 0.3
    omega = jnp.pi  # radians
    t_peri = 0.0  # time of periastron (days)
    K = 45.0  # m/s
    gamma = 100.0  # m/s (systemic velocity)
    sigma = 5.0  # m/s (measurement uncertainty)
    
    # Generate fake data
    print("Generating fake data...")
    times, rv_obs, rv_err = generate_fake_data(
        keys[0],
        n_observations,
        total_time,
        period, eccentricity, omega, t_peri, K, gamma, sigma
    )
    
    print(f"Generated {len(times)} observations")
    print(f"Time span: {times.min():.1f} to {times.max():.1f} days")
    
    # Create log probability function
    log_prob = JaxoplanetLogLikelihood(times, rv_obs, rv_err)
    
    # Set up sampler
    total_chains = 100
    dim = 6  # [period, eccentricity, omega, t_peri, K, gamma]
    num_samples = 50
    warmup = 2
    thin_by = 1
    
    print(f"\nSetting up hemcee sampler...")
    print(f"  Chains: {total_chains}")
    print(f"  Dimensions: {dim}")
    print(f"  Samples: {num_samples}")
    print(f"  Warmup: {warmup}")
    
    sampler = hemcee.HamiltonianSampler(total_chains, 
                                        dim, log_prob,
                                        step_size=0.001,
                                        L=1,
                                        adapt_length=False,
                                        adapt_step_size=False)
    
    # Initialize random starting positions for each chain
    init_period = jax.random.uniform(keys[1], shape=(total_chains, 1), minval=300.0, maxval=400.0)
    init_eccentricity = jax.random.uniform(jax.random.PRNGKey(1), shape=(total_chains, 1), minval=0.0, maxval=0.9)
    init_omega = jax.random.uniform(jax.random.PRNGKey(2), shape=(total_chains, 1), minval=0.0, maxval=2*jnp.pi)
    init_t_peri = jax.random.uniform(jax.random.PRNGKey(3), shape=(total_chains, 1), minval=-50.0, maxval=50.0)
    init_K = jax.random.uniform(jax.random.PRNGKey(4), shape=(total_chains, 1), minval=10.0, maxval=80.0)
    init_gamma = jax.random.uniform(jax.random.PRNGKey(5), shape=(total_chains, 1), minval=50.0, maxval=150.0)
    
    initial_state = jnp.concatenate([
        init_period, init_eccentricity, init_omega, 
        init_t_peri, init_K, init_gamma
    ], axis=1)
    
    # Debug: Check initial state log probabilities
    print('\nDebug: Checking initial positions...')
    initial_lps = jnp.array([log_prob(initial_state[i]) for i in range(min(5, total_chains))])
    print(f'  Initial log_probs (first 5 chains): {initial_lps}')
    print(f'  All finite: {jnp.all(jnp.isfinite(initial_lps))}')
    
    # Debug: Check gradient computation
    print('Debug: Testing gradient computation...')
    test_grad = jax.grad(log_prob)(initial_state[0])
    print(f'  Gradient at first chain: {test_grad}')
    print(f'  Gradient finite: {jnp.all(jnp.isfinite(test_grad))}')
    print(f'  Gradient norm: {jnp.linalg.norm(test_grad):.3f}')
    
    # Run MCMC
    print("\nRunning MCMC...")
    samples = sampler.run_mcmc(
        keys[2], 
        initial_state, 
        num_samples=num_samples,
        warmup=warmup,
        thin_by=thin_by,
        batch_size=1,
        show_progress=True
    )
    print(f'\nSampling complete!')
    
    # Debug: Check sample properties
    print('\nDebug: Checking samples...')
    print(f'  Sample shape: {samples.shape}')
    flat_samples_check = samples.reshape(-1, dim)
    print(f'  Samples contain NaN: {jnp.any(jnp.isnan(flat_samples_check))}')
    print(f'  Samples contain Inf: {jnp.any(jnp.isinf(flat_samples_check))}')
    print(f'  Sample range (first param - period):')
    print(f'    Min: {jnp.min(flat_samples_check[:, 0]):.3f}, Max: {jnp.max(flat_samples_check[:, 0]):.3f}')
    print(f'  Sample variation (std of first 10 samples, first param): {jnp.std(flat_samples_check[:10, 0]):.6f}')
    
    # Check if samples are all identical (stuck chains)
    first_sample = flat_samples_check[0]
    samples_different = jnp.any(jnp.abs(flat_samples_check - first_sample) > 1e-6, axis=1)
    print(f'  Number of samples different from first: {jnp.sum(samples_different)} / {len(flat_samples_check)}')
    
    # Debug: Check log probabilities of recovered samples
    sample_lps = jnp.array([log_prob(flat_samples_check[i]) for i in range(min(10, len(flat_samples_check)))])
    print(f'  Log_probs of first 10 samples: {sample_lps}')
    print(f'  All finite: {jnp.all(jnp.isfinite(sample_lps))}')

    try: 
        tau = hemcee.autocorr.integrated_time(samples)
        print(f'Integrated time: {tau}')
    except Exception as e:
        print(f'Error calculating integrated time: {e}')
        tau = None

    print('\nAdaptation:')
    final_StepSize, final_IntegrationLength = sampler.adapter.finalize(sampler.adapter_state)
    print(f'  Step Size: {final_StepSize}')
    print(f'  Integration Length: {final_IntegrationLength}')
    
    print('\nAcceptance Rates:')
    warmup_acceptance_rate = sampler.diagnostics_warmup['acceptance_rate']
    main_acceptance_rate = sampler.diagnostics_main['acceptance_rate']
    print(f'  Warmup: {warmup_acceptance_rate}')
    print(f'  Main: {main_acceptance_rate}')
    
    
    
    # Print summary statistics
    flat_samples = samples.reshape(-1, dim).__array__()
    param_names = ['period', 'eccentricity', 'omega', 't_peri', 'K', 'gamma']
    true_values = [period, eccentricity, omega, t_peri, K, gamma]
    
    print('\nParameter Summary (median ± std):')
    for i, (name, true_val) in enumerate(zip(param_names, true_values)):
        median = np.median(flat_samples[:, i])
        std = np.std(flat_samples[:, i])
        print(f'  {name:12s}: {median:8.3f} ± {std:6.3f} (true: {true_val:8.3f})')

    _ = corner.corner(flat_samples.__array__(), labels=param_names, truths=true_values)
    plt.show()
