import jax 
import jax.numpy as jnp
from tqdm import tqdm

def leapfrog(x, p, grad_fn, epsilon, L):
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

    return x, p
    
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
    # Shape (total_iterations + 1, ...), the +1 is for the initial momentum
    p_rng = jax.random.normal(subkey_momentum, shape=(total_iterations + 1, n_chains, dim))
    acceptance_rng = jax.random.uniform(subkey_acceptance, shape=(total_iterations, n_chains,))
    

    chains = jnp.tile(initial, (n_chains, 1)) + 0.1 * p_rng[0]
    chain_log_probs = log_prob_fn(chains) # Shape (n_chains,)
    
    samples = jnp.zeros((n_chains, n_samples, dim))
    accepts = jnp.zeros(n_chains)

    samples = samples.at[:, 0, :].set(chains) 

   

    for i in tqdm(range(1, total_iterations)):
        p = p_rng[i]

        # Leapfrog integration
        current_p = p.copy()
        x = chains.copy()
        
        x, p = leapfrog(x, p, grad_fn, epsilon, L)

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

########################################################
# Hamiltonian Side Move (HSM)
########################################################

def leapfrog_side_move(q1, 
                       p1_current, 
                       grad_fn, 
                       beta_eps, 
                       L,
                       n_chains_per_group,
                       diff_particles_group2):
    '''
    Args
        - q1: position of first group of chains. Shape (n_chains_per_group, dim)
        - p1: momentum of first group of chains. Shape (n_chains_per_group,)
        - grad_fn: gradient of the potential function. Function of shape (n_chains, dim) -> (n_chains, dim)
        - beta_eps: half of the step size. Scalar
        - L: number of leapfrog steps. Scalar
        - L: number of leapfrog steps
    
    Returns
        - q1: position of first group of chains. Shape (n_chains_per_group, dim)
        - p1_current: momentum of first group of chains. Shape (n_chains_per_group,)
    '''
    # Initial half-step for momentum - VECTORIZED
    grad1 = grad_fn(q1) # Shape (n_chains_per_group, dim)
    grad1 = jnp.nan_to_num(grad1, nan=0.0)
    
    # Compute dot products between gradients and difference particles - VECTORIZED
    gradient_projections = jnp.sum(grad1 * diff_particles_group2, axis=1) # Shape (n_chains_per_group,)
    p1_current -= 0.5 * beta_eps * gradient_projections # Shape (n_chains_per_group,)

    # Full leapfrog steps
    for step in range(L):
        q1 += beta_eps * (jnp.expand_dims(p1_current, axis=1) * diff_particles_group2) # Shape: (n_chains_per_group, dim)
        
        if (step < L - 1):
            grad1 = grad_fn(q1) # Shape (n_chains_per_group, dim)
            grad1 = jnp.nan_to_num(grad1, nan=0.0)

            gradient_projections = jnp.sum(grad1 * diff_particles_group2, axis=1) # Shape (n_chains_per_group,)
            p1_current -= beta_eps * gradient_projections # Shape (n_chains_per_group,)
    
    # Final half-step for momentum - VECTORIZED 
    grad1 = grad_fn(q1)
    grad1 = jnp.nan_to_num(grad1, nan=0.0)

    gradient_projections = jnp.sum(grad1 * diff_particles_group2, axis=1)
    p1_current -= 0.5 * beta_eps * gradient_projections

    return q1, p1_current # Shape (n_chains_per_group, dim), (n_chains_per_group,)

def hamiltonian_side_move(potential_func, 
                          initial, 
                          n_samples, 
                          n_chains_per_group=5, 
                          epsilon=0.01, 
                          L=10, 
                          beta=1.0,
                          n_thin = 1,
                          key=jax.random.PRNGKey(0)):
    """Hamiltonian Side Move (HSM) sampler implementation using JAX.
    
    This function implements the HSM algorithm, which uses Hamiltonian dynamics with side moves
    to propose new states in the Markov chain. It maintains two groups of chains that interact
    with each other through Hamiltonian dynamics, allowing for better exploration of the target
    distribution. It supports multiple chains per group and thinning of samples.
    
    Args:
        potential_func: Function that computes the potential energy (negative log probability)
                       of the target distribution. Should take a single argument (parameters)
                       and return a scalar.
        initial: Initial parameter values for the Markov chains. Can be any shape, will be
                flattened internally.
        n_samples: Number of samples to generate per chain
        n_chains_per_group: Number of chains in each of the two groups. Total number of chains
                           will be 2 * n_chains_per_group. Default: 5
        epsilon: Step size for the leapfrog integrator. Controls the discretization of
                Hamiltonian dynamics. Default: 0.01
        L: Number of leapfrog steps per proposal. Controls how far each proposal
                   can move. Default: 10
        beta: Temperature parameter that controls the strength of the Hamiltonian dynamics.
              Higher values lead to more aggressive exploration. Default: 1.0
        n_thin: Thinning interval for the samples. Only every nth sample is stored.
                Default: 1 (no thinning)
        key: JAX random key for reproducibility. Default: jax.random.PRNGKey(0)
    
    Returns:
        tuple: A tuple containing:
            - samples: Array of shape (2*n_chains_per_group, n_samples, *initial.shape)
                      containing the MCMC samples
            - acceptance_rates: Array of shape (2*n_chains_per_group,) containing the
                              acceptance rate for each chain
    
    Notes:
        - The algorithm uses two groups of chains that interact through Hamiltonian dynamics
        - Each chain in one group interacts with two randomly selected chains from the other group
        - The interaction is mediated through the difference between the selected chains
        - Metropolis-Hastings acceptance is used to ensure detailed balance
        - NaN gradients are handled by replacing them with zeros
        - The implementation is vectorized to run multiple chains in parallel
    """

    ### ERROR CHECKING
    if (n_chains_per_group <= 1):
        raise ValueError("n_chains_per_group must be greater than 1")
    
    # Initialize
    dim = len(initial)
    total_chains = 2 * n_chains_per_group

    potential_func_vmap = jax.jit(jax.vmap(potential_func))           # F: (n_chains, dim) -> (n_chains,)
    grad_fn_vmap        = jax.jit(jax.vmap(jax.grad(potential_func))) # F: (n_chains, dim) -> (n_chains, dim)

    # Create initial states with small random perturbations

    states = jnp.tile(initial.flatten(), (total_chains, 1)) + 0.1 * jax.random.normal(key, shape=(total_chains, dim)) # Shape (total_chains, dim)
    
    # Split into two groups
    group1_indices = jnp.arange(n_chains_per_group)               # Shape (n_chains_per_group,)
    group2_indices = jnp.arange(n_chains_per_group, total_chains) # Shape (n_chains_per_group,)

    states_group1 = states[:n_chains_per_group] # Shape (n_chains_per_group, dim)
    states_group2 = states[n_chains_per_group:] # Shape (n_chains_per_group, dim)

    # Calculate total iterations needed based on thinning factor
    total_iterations = n_samples * n_thin

    # Storage for samples and acceptance tracking
    accepts = jnp.zeros(total_chains)

    # Precompute some constants for efficiency
    beta_eps = beta * epsilon

    keys_per_iter = 6
    all_keys = jax.random.split(key, total_iterations * keys_per_iter).reshape(total_iterations, keys_per_iter, 2)
    
    
    def main_loop(carry, keys):
        #---------------------------------------------
        # Unpack input
        #---------------------------------------------
        states, accepts = carry # Array shapped (total_chains, dim), integer
        
        # Store current state from all chains (only every n_thin iterations)
        # if (i % n_thin == 0) and (sample_idx < n_samples):
        #     samples = samples.at[:, sample_idx, :].set(states)
        #     sample_idx += 1

        #---------------------------------------------
        # First group update - VECTORIZED
        #---------------------------------------------

        # For each particle in group 1, randomly select TWO particles from group 2
        keys_choices= jax.random.split(keys[0], n_chains_per_group)
        choices = jax.vmap(lambda k: jax.random.choice(k, group2_indices, shape=(2,), replace=False))(keys_choices) # Shape (n_chains_per_group, 2)
        random_indices1_from_group2 = choices[:, 0]
        random_indices2_from_group2 = choices[:, 1]

        # Get the two sets of selected particles from group 2. Shape: (n_chains_per_group, dim)
        selected_particles1_group2 = states[random_indices1_from_group2] # Shape (n_chains_per_group, dim)
        selected_particles2_group2 = states[random_indices2_from_group2] # Shape (n_chains_per_group, dim)
        
        # Use the difference between the two particles. Shape (n_chains_per_group, dim)
        diff_particles_group2 = (selected_particles1_group2 - selected_particles2_group2) / jnp.sqrt(2*n_chains_per_group)
        
        # Generate momentum - one scalar per chain (shape: n_chains_per_group)
        p1 = jax.random.normal(keys[1], shape=(n_chains_per_group,))

        # Store current state and energy
        current_q1 = states[group1_indices].copy() # Shape (n_chains_per_group, dim)
        current_U1 = potential_func_vmap(current_q1) # Shape (n_chains_per_group, dim) -> (n_chains_per_group,)

        current_K1 = jnp.clip(0.5 * p1**2, 0, 1000)

        q1, p1_current = leapfrog_side_move(
            current_q1, 
            p1, 
            grad_fn_vmap, 
            beta_eps, 
            L, 
            n_chains_per_group,
            diff_particles_group2
        ) # Shape (n_chains_per_group, dim), (n_chains_per_group,)
        
        # Compute proposed energy
        proposed_U1 = potential_func_vmap(q1) # Shape (n_chains_per_group,)
        proposed_K1 = jnp.clip(0.5 * p1_current**2, 0, 1000) # Shape (n_chains_per_group,)
        
        # Metropolis acceptance - IMPROVED FOR NUMERICAL STABILITY
        dH1 = (proposed_U1 + proposed_K1) - (current_U1 + current_K1) # Shape (n_chains_per_group,)

        # Calculate acceptance probabilities with numerical safeguards for exp
        accept_probs1 = jnp.ones_like(dH1)  # Shape (n_chains_per_group,)
        
        # Only calculate exp for positive dH (negative cases auto-accept with prob 1.0)
        exp_needed = dH1 > 0
        safe_dH = jnp.clip(dH1, a_min=None, a_max=100.0)   # Shape (n_chains_per_group,)
        accept_probs1 = jnp.where(
            exp_needed,                # boolean mask
            jnp.exp(-safe_dH),         # “true” branch (vectorised)
            accept_probs1              # “false” branch (original values)
        ) # Shape (n_chains_per_group,)

        accepts1 = jax.random.uniform(keys[2], shape=(n_chains_per_group,)) < accept_probs1 # Shape (n_chains_per_group,)
        
        # Update states - VECTORIZED
        updated_group1_states = jnp.where(accepts1[:, None], q1, states[group1_indices])
        states = states.at[group1_indices].set(updated_group1_states)
        
        # Track acceptance for first chains
        accepts += jnp.sum(jnp.count_nonzero(accepts1))
        #---------------------------------------------
        # Second group update - VECTORIZED similarly
        #---------------------------------------------
        
        # For each particle in group 2, randomly select TWO particles from group 1
        keys_choices= jax.random.split(keys[3], n_chains_per_group)
        choices = jax.vmap(lambda k: jax.random.choice(k, group1_indices, shape=(2,), replace=False))(keys_choices) # Shape (n_chains_per_group, 2)
        random_indices1_from_group1 = choices[:, 0]
        random_indices2_from_group1 = choices[:, 1]

        # Get the two sets of selected particles from group 1. Shape (n_chains_per_group, dim)
        selected_particles1_group1 = states[random_indices1_from_group1]
        selected_particles2_group1 = states[random_indices2_from_group1]

        # Use the difference between the two particles (shape: n_chains_per_group x dim)
        diff_particles_group1 = (selected_particles1_group1 - selected_particles2_group1) / jnp.sqrt(2*n_chains_per_group)

        # Generate momentum - one scalar per chain (shape: n_chains_per_group)
        p2 = jax.random.normal(keys[4], shape=(n_chains_per_group,))   
        
        # Store current state and energy
        current_q2 = states[group2_indices].copy()
        current_U2 = potential_func_vmap(current_q2)
        current_K2 = jnp.clip(0.5 * p2**2, 0, 1000)

        # Leapfrog integration with preconditioning
        # q2 = current_q2.copy()
        # p2_current = p2.copy()

        q2, p2_current = leapfrog_side_move(
            current_q2, 
            p2, 
            grad_fn_vmap, 
            beta_eps, 
            L, 
            n_chains_per_group, 
            diff_particles_group1
        )

        # Compute proposed energy
        proposed_U2 = potential_func_vmap(q2)
        proposed_K2 = jnp.clip(0.5 * p2_current**2, 0, 1000)

        # Metropolis acceptance - IMPROVED FOR NUMERICAL STABILITY
        dH2 = (proposed_U2 + proposed_K2) - (current_U2 + current_K2)

        # Calculate acceptance probabilities with numerical safeguards for exp
        accept_probs2 = jnp.ones_like(dH2)  # Default to accept
        # Only calculate exp for positive dH (negative cases auto-accept with prob 1.0)
        exp_needed = dH2 > 0
        safe_dH = jnp.clip(dH2, a_min=None, a_max=100.0)   # Shape (n_chains_per_group,)
        accept_probs2 = jnp.where(
            exp_needed,                # boolean mask
            jnp.exp(-safe_dH),         # “true” branch (vectorised)
            accept_probs2              # “false” branch (original values)
        ) # Shape (n_chains_per_group,)
        
        accepts2 = jax.random.uniform(keys[5], shape=(n_chains_per_group,)) < accept_probs2

        # Update states - VECTORIZED
        updated_group2_states = jnp.where(accepts2[:, None], q2, states[group2_indices])
        states = states.at[group2_indices].set(updated_group2_states)

        # Track acceptance for second chains
        accepts += jnp.sum(jnp.count_nonzero(accepts2))

        return (states, accepts), states

    carry, previous_states = jax.lax.scan(main_loop, init=(states, 0), xs=all_keys)
    current_states, accepts = carry
    
    # Compute acceptance rates for all chains
    acceptance_rates = accepts / (total_iterations * n_chains_per_group * 2)
    
    return previous_states, acceptance_rates