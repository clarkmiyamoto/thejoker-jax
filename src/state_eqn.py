import jax
import jax.numpy as jnp

# In astro-literature, they use double precision.
jax.config.update("jax_enable_x64", True)

def kepler_starter(mean_anom, ecc):
    ome = 1 - ecc
    M2 = jnp.square(mean_anom)
    alpha = 3 * jnp.pi / (jnp.pi - 6 / jnp.pi)
    alpha += 1.6 / (jnp.pi - 6 / jnp.pi) * (jnp.pi - mean_anom) / (1 + ecc)
    d = 3 * ome + alpha * ecc
    alphad = alpha * d
    r = (3 * alphad * (d - ome) + M2) * mean_anom
    q = 2 * alphad * ome - M2
    q2 = jnp.square(q)
    w = jnp.square(jnp.cbrt(jnp.abs(r) + jnp.sqrt(q2 * q + r * r)))
    return (2 * r * w / (jnp.square(w) + w * q + q2) + mean_anom) / d

def kepler_refiner(mean_anom, ecc, ecc_anom):
    ome = 1 - ecc
    sE = ecc_anom - jnp.sin(ecc_anom)
    cE = 1 - jnp.cos(ecc_anom)

    f_0 = ecc * sE + ecc_anom * ome - mean_anom
    f_1 = ecc * cE + ome
    f_2 = ecc * (ecc_anom - sE)
    f_3 = 1 - f_1
    d_3 = -f_0 / (f_1 - 0.5 * f_0 * f_2 / f_1)
    d_4 = -f_0 / (f_1 + 0.5 * d_3 * f_2 + (d_3 * d_3) * f_3 / 6)
    d_42 = d_4 * d_4
    dE = -f_0 / (f_1 + 0.5 * d_4 * f_2 + d_4 * d_4 * f_3 / 6 - d_42 * d_4 * f_2 / 24)

    return ecc_anom + dE

@jax.jit
@jnp.vectorize
def kepler_solver_impl(mean_anom, ecc):
    """
    Calculates eccentric anomaly `E` from mean anomaly `M` and eccentricity `e`.
    """
    mean_anom = mean_anom % (2 * jnp.pi)

    # We restrict to the range [0, pi)
    high = mean_anom > jnp.pi
    mean_anom = jnp.where(high, 2 * jnp.pi - mean_anom, mean_anom)

    # Solve
    ecc_anom = kepler_starter(mean_anom, ecc)
    ecc_anom = kepler_refiner(mean_anom, ecc, ecc_anom)

    # Re-wrap back into the full range
    ecc_anom = jnp.where(high, 2 * jnp.pi - ecc_anom, ecc_anom)

    return ecc_anom

def velocity(t, period, eccentricity, omega, phi0, K, v0):
    """
    """
    mean_anom = (2 * jnp.pi * t / period) + phi0
    E = kepler_solver_impl(mean_anom, eccentricity)

    f = jnp.arccos((jnp.cos(E) - eccentricity) / (1 - eccentricity * jnp.cos(E)))

    return (
        v0 + 
        K * (jnp.cos(omega + f) +eccentricity * jnp.sin(omega))
    )



