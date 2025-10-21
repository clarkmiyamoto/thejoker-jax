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
                 jitter: float = 0.0):
        '''
        Args:
            observed_data: Shape is (batch_size,).
            uncertainity: Shape is (batch_size,).
            jitter: Additioinal noise.
        '''
        self.times = times
        self.observed_data = observed_data
        self.uncertainity = uncertainity
        self.jitter = jitter

        # Prepares model for batching
        def velocity_model(params):
            period, eccentricity, omega, phi0, K, v0 = params
            t = times
            return velocity(t, period, eccentricity, omega, phi0, K, v0)
        self.velocity_model = velocity_model

    def __call__(self, params: jnp.ndarray) -> float:
        '''
        Args
            - params (tuple): in th order (period, eccentricity, omega, phi0, K, v0)
        '''
        return (
        -0.5 * jnp.sum((self.observed_data - self.velocity_model(params)) ** 2 / (self.uncertainity ** 2 + self.jitter ** 2))
        -0.5 * jnp.sum(jnp.log(2 * jnp.pi * (self.uncertainity ** 2 + self.jitter ** 2)))
    )