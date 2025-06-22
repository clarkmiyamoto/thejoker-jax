import jax
import jax.numpy as jnp
from jaxoplanet.core import kepler
################################################################################
# Solving Kepler's Equation smartly by implementing `jaxoplanets`
# See https://jax.exoplanet.codes/en/latest/tutorials/core-from-scratch/.
################################################################################

# In astro-literature, they use double precision.
jax.config.update("jax_enable_x64", True)

################################################################################
# State Equation
# See https://arxiv.org/pdf/1610.07602
################################################################################

@jax.jit
def velocity(t: jnp.ndarray, 
             period: float, 
             eccentricity: float, 
             omega: float, 
             phi0: float, 
             K: float,
             v0: float) -> jnp.ndarray:
    """
    Calculate velocity at times `t` for a given period, eccentricity, omega, phi0, K, and v0.
    Using equation (1-4) in https://arxiv.org/pdf/1610.07602

    Args:
        - t: time. Units (days). Shape is (batch_size,).
        - period: period of the orbit. Units (days).
        - eccentricity: eccentricity of the orbit.
        - omega: argument of periastron. Units (radians).
        - phi0: phase offset. Units (radians).
        - K: semi-amplitude of the velocity. Units (m/s).
        - v0: velocity offset. Units (m/s).

    Returns:
        - velocity at time `t`. Shape is (batch_size,).
    """
    return _velocity(t, period, eccentricity, omega, phi0, K, v0)


def _velocity(t: jnp.ndarray, 
             period: float, 
             eccentricity: float, 
             omega: float, 
             phi0: float, 
             K: float,
             v0: float) -> float:
    """
    For documentation, see `velocity`.
    """
    # Equation (3)
    mean_anom = (2 * jnp.pi * t / period) - phi0 

    # Equation (4)
    sin_f, cos_f = kepler(mean_anom, eccentricity)

    # Equation (1)
    return (
        v0 + 
        K * (
            jnp.cos(omega) * cos_f 
            - jnp.sin(omega) * sin_f 
            + eccentricity * jnp.sin(omega)
            )
    )