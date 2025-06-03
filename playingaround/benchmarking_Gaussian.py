import sys
import os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import time 
from tqdm import tqdm
import logging

def setup_logger(name="my_logger", log_file="default_name.txt", level=logging.DEBUG):
    """Set up a logger that prints to console and writes to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if this function is called multiple times
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

from src import sampler_numpy
from src import sampler_jax


### Benchmarking
def benchmark_jax(sampler, key):
    start_time = time.time()
    _, _ = sampler(key)
    _.block_until_ready()
    end_time = time.time()
    time_taken = end_time - start_time
    return time_taken

def benchmark_numpy(sampler):
    start_time = time.time()
    _, _ = sampler()
    end_time = time.time()
    time_taken = end_time - start_time
    return time_taken


### Distribution to test on
dim = 3
cov_jax = jnp.eye(dim)
cov_np = np.array(cov_jax)

def log_prob_np(x):
    return 0.5 * np.einsum('ij,jk,ik->i', x, cov_np, x)

def grad_log_prob_np(x):
    return np.einsum('jk,ij->ik', cov_np, x)

def log_prob_jax(x):
    return -0.5 * jnp.einsum('i,ij,j->', x, cov_jax, x)

def grad_log_prob_jax(x):
    return -jnp.einsum('i,ij->i', x, cov_jax)

initial_params_jax = jnp.zeros(dim)
initial_params_np = np.array(initial_params_jax)


### Run Test
if __name__ == "__main__":
    logger = setup_logger(log_file='benchmarking_Gaussian.txt')
    logger.info(f"########################################################")
    logger.info(f"Begin Experiment")
    logger.info(f"########################################################")


    ### Settings
    # Experiment Setup
    n_trials = 2
    keys = [jax.random.PRNGKey(i) for i in range(n_trials)]

    # Sampling Setup
    n_samples = 10 ** 5
    n_chains = 10; assert n_samples % 2 == 0
    n_chains_per_group = int(n_chains / 2)

    ### Code
    hmc_np = lambda: sampler_numpy.hmc(log_prob_np, initial_params_np, n_samples, grad_log_prob_np, n_chains=n_chains)
    hmc_side_np = lambda: sampler_numpy.hamiltonian_side_move(grad_log_prob_np, log_prob_np, initial_params_np, n_samples, n_chains_per_group=n_chains_per_group)
    hmc_walk_np = lambda: sampler_numpy.hamiltonian_walk_move(grad_log_prob_np, log_prob_np, initial_params_np, n_samples, n_chains_per_group=n_chains_per_group)

    hmc_jax = lambda key: sampler_jax.hmc(log_prob_jax, initial_params_jax, n_samples, grad_log_prob_jax, n_chains=n_chains, key=key)
    hmc_side_jax = lambda key: sampler_jax.hamiltonian_side_move(log_prob_jax, initial_params_jax, n_samples, grad_log_prob_jax, n_chains_per_group=n_chains_per_group, key=key)
    hmc_walk_jax = lambda key: sampler_jax.hamiltonian_walk_move(log_prob_jax, initial_params_jax, n_samples, grad_log_prob_jax, n_chains_per_group=n_chains_per_group, key=key)

    names = ['HMC', 'HMC Side', 'HMC Walk']
    samplers_np = [hmc_np, hmc_side_np, hmc_walk_np]
    samplers_jax = [hmc_jax, hmc_side_jax, hmc_walk_jax]


    # Run experiment
    for sampler_i_np, sampler_i_jax, name in zip(samplers_np, samplers_jax, names):
        # Get times
        times_np = [float(benchmark_numpy(sampler_i_np)) for _ in tqdm(keys)]
        times_jax = [float(benchmark_jax(sampler_i_jax, key)) for key in tqdm(keys)]

        # Statistics
        mean_time_np = np.mean(times_np)
        mean_time_jax = np.mean(times_jax)
        std_time_np = np.std(times_np)
        std_time_jax = np.std(times_jax)

        mean_time_jax_wo_jit = np.mean(times_jax[1:])
        std_time_jax_wo_jit = np.std(times_jax[1:])

        # Logging
        logger.info(f"--------------------------------")
        logger.info(f"Method: {name}")
        logger.info(f"--------------------------------")
        logger.info(f"Raw Times")
        logger.info(f"Numpy: {times_np}")
        logger.info(f"Jax: {times_jax}")
        logger.info(f"--------------------------------")
        logger.info(f"Statistics")
        logger.info(f"Numpy: {mean_time_np} ± {std_time_np}") 
        logger.info(f"Jax: {mean_time_jax} ± {std_time_jax}") 
        logger.info(f"Jax (removed JIT'd time): {mean_time_jax_wo_jit} ± {std_time_jax_wo_jit}")
        logger.info(f"")