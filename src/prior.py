import jax
import jax.numpy as jnp
from jax import random

'''
Implementaiton of priors. See equations (5-9) in https://arxiv.org/pdf/1610.07602
'''

def sampleLogPeriod(key: jax.random.PRNGKey, 
               batch_size: int,
               logP_min: float,
               logP_max: float) -> jnp.ndarray:
    '''
    Sample log period from a uniform distribution.

    Args:
        - key: random number generator key
        - batch_size: number of samples to draw
        - logP_min: minimum log period
        - logP_max: maximum log period

    Returns:
        - Shape (batch_size,).
    '''
    return random.uniform(key, shape=(batch_size,), minval=logP_min, maxval=logP_max)

def sampleEccentricity(key: jax.random.PRNGKey, batch_size: int,) -> jnp.ndarray:
    '''
    Sample eccentricity from a beta distribution.

    Args:
        - key: random number generator key
        - batch_size: number of samples to draw

    Returns:
        - Shape (batch_size,).
    '''
    a = 0.867
    b = 3.03
    return random.beta(key, shape=(batch_size,), a=a, b=b)

def sampleAngularRad(key: jax.random.PRNGKey, batch_size: int,) -> jnp.ndarray:
    '''
    Sample angular radian from a uniform distribution.

    Args:
        - key: random number generator key
        - batch_size: number of samples to draw

    Returns:
        - Shape (batch_size,).
    '''
    return random.uniform(key, shape=(batch_size,), minval=0.0, maxval=2 * jnp.pi)

def samplePhase(key: jax.random.PRNGKey, batch_size: int,) -> jnp.ndarray:
    '''
    Sample phase from a uniform distribution.

    Args:
        - key: random number generator key
        - batch_size: number of samples to draw

    Returns:
        - Shape (batch_size,).
    '''
    return random.uniform(key, shape=(batch_size,), minval=0.0, maxval=2 * jnp.pi)

def sampleLogJitter(key: jax.random.PRNGKey, 
                    batch_size: int,
                    mean: float,
                    std: float) -> jnp.ndarray:
    '''
    Sample log jitter from a normal distribution.

    Args:
        - key: random number generator key
        - batch_size: number of samples to draw
        - mean: mean of the normal distribution
        - std: standard deviation of the normal distribution

    Returns:
        - Shape (batch_size,).
    '''
    return random.normal(key, shape=(batch_size,), mean=mean, stddev=std)
    







