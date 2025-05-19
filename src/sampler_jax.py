import jax 
import jax.numpy as jnp
from tqdm import tqdm
def hmc(log_prob,
        initial: jnp.ndarray,
        n_samples,
        epsilon=0.1,
        L=10,
        n_chains=1,
        n_thin=1,
        key=jax.random.PRNGKey(0)):
    """Hamiltonian Monte Carlo (HMC) sampler implementation using JAX.
    
    This function implements the HMC algorithm, which uses Hamiltonian dynamics to propose
    new states in the Markov chain. It supports multiple chains and thinning of samples.
    
    Args:
        log_prob: Function that computes the log probability density of the target distribution.
                 Should take a single argument (parameters) (shaped (dim,)) and return a scalar.
        initial: Initial parameter values for the Markov chains. Shape (dim,)
        n_samples: Number of samples to generate per chain
        epsilon: Step size for the leapfrog integrator. Controls the discretization of
                Hamiltonian dynamics. Default: 0.1
        L: Number of leapfrog steps per proposal. Controls how far each proposal can move.
           Default: 10
        n_chains: Number of independent Markov chains to run in parallel. Default: 1
        n_thin: Thinning interval for the samples. Only every nth sample is stored.
                Default: 1 (no thinning)
        key: JAX random key for reproducibility. Default: jax.random.PRNGKey(0)
    
    Returns:
        tuple: A tuple containing:
            - samples: Array of shape (n_chains, n_samples, dim) containing the MCMC samples
            - acceptance_rates: Array of shape (n_chains,) containing the acceptance rate
                              for each chain
    
    Notes:
        - The algorithm uses the leapfrog integrator to simulate Hamiltonian dynamics
        - Metropolis-Hastings acceptance is used to ensure detailed balance
        - NaN gradients are handled by replacing them with zeros
        - The implementation is vectorized to run multiple chains in parallel
    """

    ### Setup
    # JIT access to functions
    log_prob_fn = jax.jit(jax.vmap(log_prob))
    grad_fn = jax.jit(jax.vmap(jax.grad(log_prob)))

    # Integers
    dim = len(initial)
    total_iterations = 1 + (n_samples - 1) * n_thin
    sample_idx = 1

    # Random Numbers
    subkey_momentum, subkey_acceptance = jax.random.split(key, 2)
    p_rng = jax.random.normal(subkey_momentum, shape=(total_iterations + 1, n_chains, dim))
    acceptance_rng = jax.random.uniform(subkey_acceptance, shape=(total_iterations, n_chains,))
    

    chains = jnp.tile(initial, (n_chains, 1)) + 0.1 * p_rng[0]
    chain_log_probs = log_prob_fn(chains) # Shape (n_chains,)
    
    samples = jnp.zeros((n_chains, n_samples, dim))
    accepts = jnp.zeros(n_chains)

    samples = samples.at[:, 0, :].set(chains) 

   

    for i in tqdm(range(1, total_iterations)):
        # Generate momentum variables for all chains at once
        p = p_rng[i]
        current_p = p.copy()

        # Leapfrog integration
        x = chains.copy()

        # Get gradients for all chains
        x_grad = grad_fn(x)
        x_grad = jnp.nan_to_num(x_grad, nan=0.0)

        # Half step for momentum
        p -= 0.5 * epsilon * x_grad

        # Full steps for position and momentum
        for j in range(L):
            x += epsilon * p
            
            if (j < L - 1):
                x_grad = grad_fn(x)
                x_grad = jnp.nan_to_num(x_grad, nan=0.0)

                p -= epsilon * x_grad
            
        x_grad = grad_fn(x)
        x_grad = jnp.nan_to_num(x_grad, nan=0.0)
       

        p -= 0.5 * epsilon * x_grad

        # Flip momentum for reversibility
        p *= -1

        # Metropolis acceptance (vectorized)
        proposal_log_probs = log_prob_fn(x)

        # Compute log acceptance ratio directly in log space - avoiding exp
        current_K = 0.5 * jnp.sum(current_p**2, axis=1)
        proposal_K = 0.5 * jnp.sum(p**2, axis=1)

        # Calculate log acceptance probability directly to avoid overflow
        log_accept_prob = jnp.minimum(0, proposal_log_probs - chain_log_probs - proposal_K + current_K)

        # Generate uniform random numbers for acceptance decision
        log_u = jnp.log(acceptance_rng[i]) # Shape (n_chains,)
        
        # Create mask for accepted proposals
        accept_mask = log_u < log_accept_prob
        
        # Update chains and log probabilities where accepted
        chains          = jnp.where(accept_mask[:, None], x, chains)
        chain_log_probs = jnp.where(accept_mask, proposal_log_probs, chain_log_probs)
        
        # Track acceptances
        accepts += accept_mask
        
        # Store current state for all chains (only every n_thin iterations after the first)
        if (i - 1) % n_thin == 0 and sample_idx < n_samples:
            samples = samples.at[:, sample_idx, :].set(chains)
            sample_idx += 1

    # Calculate acceptance rates for all chains
    acceptance_rates = accepts / (total_iterations - 1)
    
    return samples, acceptance_rates

