import jax
import jax.numpy as jnp
from typing import Callable

from .velocity import velocity

class LogLikelihood:
    '''
    Implementation of the unmarginalized likelihood.
    Equation (10) in https://arxiv.org/pdf/1610.07602
    '''

    def __init__(self, 
                 times: jnp.ndarray,
                 observed_data: jnp.ndarray,
                 uncertainity: jnp.ndarray,
                 jitter: float = 0.0,
                 n_planets: int = 1):
        '''
        Args:
            observed_data: Shape is (batch_size,).
            uncertainity: Shape is (batch_size,).
            jitter: Additioinal noise.
            n_planets: Number of non-interacting planets in the system.
        '''
        self.times = times
        self.observed_data = observed_data
        self.uncertainity = uncertainity
        self.jitter = jitter
        self.n_planets = n_planets

        # Prepares model for batching
        def velocity_model(params):
            # Parse flattened params array
            # Structure: [P1, e1, ω1, φ1, K1, P2, e2, ω2, φ2, K2, ..., v0]
            total_v = jnp.zeros_like(times)
            
            # Use JAX fori_loop to sum planet contributions
            def add_planet_contribution(i, total_v):
                idx = i * 5
                period = params[idx]
                eccentricity = params[idx + 1]
                omega = params[idx + 2]
                phi0 = params[idx + 3]
                K = params[idx + 4]
                # Get orbital contribution (without v0)
                orbital_v = velocity(times, period, eccentricity, omega, phi0, K, 0.0)
                return total_v + orbital_v
            
            total_v = jax.lax.fori_loop(0, n_planets, add_planet_contribution, total_v)
            
            # Add shared systemic velocity
            v0 = params[-1]
            return total_v + v0
        
        self.velocity_model = velocity_model

    def __call__(self, params: jnp.ndarray) -> float:
        '''
        Args
            - params: flattened array of parameters
              For n_planets: [P1, e1, ω1, φ1, K1, P2, e2, ω2, φ2, K2, ..., Pn, en, ωn, φn, Kn, v0]
              Total length: 5 * n_planets + 1
        '''
        # Check for invalid parameter values for all planets
        def check_planet_validity(i, invalid):
            idx = i * 5
            period = params[idx]
            eccentricity = params[idx + 1]
            return invalid | (period < 0) | (eccentricity < 0) | (eccentricity > 1)
        
        invalid = jax.lax.fori_loop(0, self.n_planets, check_planet_validity, jnp.array(False))
        
        # Compute velocity model (kepler function now handles invalid ecc safely)
        velocity_pred = self.velocity_model(params)
        
        # Compute the likelihood
        log_likelihood = (
            -0.5 * jnp.sum((self.observed_data - velocity_pred) ** 2 / (self.uncertainity ** 2 + self.jitter ** 2))
            -0.5 * jnp.sum(jnp.log(2 * jnp.pi * (self.uncertainity ** 2 + self.jitter ** 2)))
        )
        
        # Return -inf if invalid, otherwise return the computed likelihood
        return jnp.where(invalid, -jnp.inf, log_likelihood)