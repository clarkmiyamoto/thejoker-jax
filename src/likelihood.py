from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp

class Likelihood(ABC):

    @abstractmethod
    def unmarginalized_likelihood(self,
                                  v_observed: jnp.ndarray, 
                                  v_simulated: jnp.ndarray,
                                  uncertainity: jnp.ndarray,
                                  jitter: float = 0.0) -> float:
        """
        Implementation of the unmarginalized likelihood.
        Equation (10) in https://arxiv.org/pdf/1610.07602

        Args:
        - v_observed: observed radial velocities. Shape is (batch_size,)
        - v_simulated: simulated radial velocities. Shape is (batch_size,)
        - uncertainity: measurement uncertainity. Shape is (batch_size,)
        - jitter: jitter term.

        Returns:
        - unmarginalized likelihood [unitless]
        """
        return (
            -0.5 * jnp.sum((v_observed - v_simulated) ** 2 / (uncertainity ** 2 + jitter ** 2))
            -0.5 * jnp.sum(jnp.log(2 * jnp.pi * (uncertainity ** 2 + jitter ** 2)))
        )