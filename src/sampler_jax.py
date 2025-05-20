import jax 
import jax.numpy as jnp
from tqdm import tqdm

def leapfrog(p, x, grad_fn, epsilon, L):
    '''
    '''
    p = p.copy()
    x = x.copy()

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

def hamiltonian_side_move(potential_func, 
                          initial, 
                          n_samples, 
                          n_chains_per_group=5, 
                          epsilon=0.01, 
                          n_leapfrog=10, 
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
        n_leapfrog: Number of leapfrog steps per proposal. Controls how far each proposal
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

    # Initialize
    orig_dim = initial.shape
    flat_dim = jnp.prod(orig_dim)
    total_chains = 2 * n_chains_per_group

    potential_func = jax.jit(jax.vmap(potential_func))
    grad_fn        = jax.jit(jax.vmap(jax.grad(potential_func)))

    # Create initial states with small random perturbations

    states = jnp.tile(initial.flatten(), (total_chains, 1)) + 0.1 * jax.random.normal(key, shape=(total_chains, flat_dim))
    
    # Split into two groups
    group1_indices = jnp.arange(n_chains_per_group)
    group2_indices = jnp.arange(n_chains_per_group, total_chains)

    # Calculate total iterations needed based on thinning factor
    total_iterations = n_samples * n_thin

    # Storage for samples and acceptance tracking
    samples = jnp.zeros((total_chains, n_samples, flat_dim))
    accepts = jnp.zeros(total_chains)
    
    # Sample index to track where to store thinned samples
    sample_idx = 0

    # Precompute some constants for efficiency
    beta_eps = beta * epsilon
    beta_eps_half = beta_eps / 2

    for i in tqdm(range(total_iterations)):
        keys = jax.random.split(key, 3)

        # Store current state from all chains (only every n_thin iterations)
        if (i % n_thin == 0) and (sample_idx < n_samples):
            samples = samples.at[:, sample_idx, :].set(states)
            sample_idx += 1

        #---------------------------------------------
        # First group update - VECTORIZED
        #---------------------------------------------

        # For each particle in group 1, randomly select TWO particles from group 2
        random_indices1_from_group2 = jax.random.choice(keys[0], group2_indices, shape=(n_chains_per_group,))
        random_indices2_from_group2 = jax.random.choice(keys[1], group2_indices, shape=(n_chains_per_group,))

        # Ensure the two selected particles are different
        for j in range(n_chains_per_group):
            while random_indices1_from_group2[j] == random_indices2_from_group2[j]:
                random_indices2_from_group2 = jax.random.choice(keys[2], group2_indices, shape=(n_chains_per_group,))

        # Get the two sets of selected particles from group 2 (shape: n_chains_per_group x flat_dim)
        selected_particles1_group2 = states[random_indices1_from_group2]
        selected_particles2_group2 = states[random_indices2_from_group2]    
        
        # Use the difference between the two particles (shape: n_chains_per_group x flat_dim)
        diff_particles_group2 = (selected_particles1_group2 - selected_particles2_group2) / jnp.sqrt(2*n_chains_per_group)
        
        # Generate momentum - one scalar per chain (shape: n_chains_per_group)
        p1 = jax.random.normal(keys[3], shape=(n_chains_per_group,))

        # Store current state and energy
        current_q1 = states[group1_indices].copy()
        current_q1_reshaped = current_q1.reshape(n_chains_per_group, *orig_dim)
        current_U1 = potential_func(current_q1_reshaped)

        current_K1 = jnp.clip(0.5 * p1**2, 0, 1000)

        # Leapfrog integration with preconditioning
        q1 = current_q1.copy()
        p1_current = p1.copy()

        # Initial half-step for momentum - VECTORIZED
        grad1 = grad_fn(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        
        # Compute dot products between gradients and difference particles - VECTORIZED
        gradient_projections = jnp.sum(grad1 * diff_particles_group2, axis=1)
        p1_current -= beta_eps_half * gradient_projections
        
        # Full leapfrog steps
        n_steps = n_leapfrog
        for step in range(n_steps):
            # Full position step - VECTORIZED with broadcasting
            q1 += beta_eps * (jnp.expand_dims(p1_current, axis=1) * diff_particles_group2)
            
            if step < n_steps - 1:
                # Full momentum step (except at last iteration) - VECTORIZED
                grad1 = grad_fn(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
                # Handle NaNs safely
                grad1 = jnp.nan_to_num(grad1, nan=0.0)

                gradient_projections = jnp.sum(grad1 * diff_particles_group2, axis=1)
                p1_current -= beta_eps * gradient_projections
        
        # Final half-step for momentum - VECTORIZED 
        grad1 = grad_fn(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad1 = jnp.nan_to_num(grad1, nan=0.0)

        gradient_projections = jnp.sum(grad1 * diff_particles_group2, axis=1)
        p1_current -= beta_eps_half * gradient_projections
        
        # Compute proposed energy
        proposed_U1 = potential_func(q1.reshape(n_chains_per_group, *orig_dim))
        proposed_K1 = jnp.clip(0.5 * p1_current**2, 0, 1000)
        
        # Metropolis acceptance - IMPROVED FOR NUMERICAL STABILITY
        dH1 = (proposed_U1 + proposed_K1) - (current_U1 + current_K1)

        # Calculate acceptance probabilities with numerical safeguards for exp
        accept_probs1 = jnp.ones_like(dH1)  # Default to accept
        # Only calculate exp for positive dH (negative cases auto-accept with prob 1.0)
        exp_needed = dH1 > 0
        if jnp.any(exp_needed):
            # Clip extremely high values before exponentiating
            safe_dH = jnp.clip(dH1[exp_needed], None, 100)  # No lower bound, upper bound of 100
            accept_probs1 = accept_probs1.at[exp_needed].set(jnp.exp(-safe_dH))
        
        accepts1 = jnp.random.random(n_chains_per_group) < accept_probs1
        
        # Update states - VECTORIZED
        group1_indices_accepts1 = group1_indices[accepts1]
        states = states.at[group1_indices_accepts1].set(q1[accepts1])
        # Track acceptance for first chains
        accepts = accepts.at[group1_indices].set(accepts[group1_indices] + accepts1)

        #---------------------------------------------
        # Second group update - VECTORIZED similarly
        #---------------------------------------------
        
        # For each particle in group 2, randomly select TWO particles from group 1
        random_indices1_from_group1 = jax.random.choice(keys[4], group1_indices, shape=(n_chains_per_group,))
        random_indices2_from_group1 = jax.random.choice(keys[5], group1_indices, shape=(n_chains_per_group,))
        
        # Ensure the two selected particles are different
        for j in range(n_chains_per_group):
            while (random_indices1_from_group1[j] == random_indices2_from_group1[j]):
                random_indices2_from_group1 = jax.random.choice(keys[6], group1_indices, shape=(n_chains_per_group,))
        
        # Get the two sets of selected particles from group 1 (shape: n_chains_per_group x flat_dim)
        selected_particles1_group1 = states[random_indices1_from_group1]
        selected_particles2_group1 = states[random_indices2_from_group1]

        # Use the difference between the two particles (shape: n_chains_per_group x flat_dim)
        diff_particles_group1 = (selected_particles1_group1 - selected_particles2_group1) / jnp.sqrt(2*n_chains_per_group)

        # Generate momentum - one scalar per chain (shape: n_chains_per_group)
        p2 = jax.random.normal(keys[7], shape=(n_chains_per_group,))   
        
        # Store current state and energy
        current_q2 = states[group2_indices].copy()
        current_q2_reshaped = current_q2.reshape(n_chains_per_group, *orig_dim)
        current_U2 = potential_func(current_q2_reshaped)
        current_K2 = jnp.clip(0.5 * p2**2, 0, 1000)

        # Leapfrog integration with preconditioning
        q2 = current_q2.copy()
        p2_current = p2.copy()

        # Initial half-step for momentum - VECTORIZED
        grad2 = grad_fn(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)

        # Compute dot products between gradients and difference particles - VECTORIZED
        gradient_projections = jnp.sum(grad2 * diff_particles_group1, axis=1)
        p2_current -= beta_eps_half * gradient_projections

        # Full leapfrog steps
        n_steps = n_leapfrog
        for step in range(n_steps):
            # Full position step - VECTORIZED with broadcasting
            q2 += beta_eps * (jnp.expand_dims(p2_current, axis=1) * diff_particles_group1)

            if step < n_steps - 1:
                # Full momentum step (except at last iteration) - VECTORIZED
                grad2 = grad_fn(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
                # Handle NaNs safely
                grad2 = jnp.nan_to_num(grad2, nan=0.0)

                gradient_projections = jnp.sum(grad2 * diff_particles_group1, axis=1)
                p2_current -= beta_eps * gradient_projections
        
        # Final half-step for momentum - VECTORIZED
        grad2 = grad_fn(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad2 = jnp.nan_to_num(grad2, nan=0.0)

        gradient_projections = jnp.sum(grad2 * diff_particles_group1, axis=1)
        p2_current -= beta_eps_half * gradient_projections

        # Compute proposed energy
        proposed_U2 = potential_func(q2.reshape(n_chains_per_group, *orig_dim))
        proposed_K2 = jnp.clip(0.5 * p2_current**2, 0, 1000)

        # Metropolis acceptance - IMPROVED FOR NUMERICAL STABILITY
        dH2 = (proposed_U2 + proposed_K2) - (current_U2 + current_K2)

        # Calculate acceptance probabilities with numerical safeguards for exp
        accept_probs2 = jnp.ones_like(dH2)  # Default to accept
        # Only calculate exp for positive dH (negative cases auto-accept with prob 1.0)
        exp_needed = dH2 > 0
        if jnp.any(exp_needed):
            # Clip extremely high values before exponentiating 
            safe_dH = jnp.clip(dH2[exp_needed], None, 100)  # No lower bound, upper bound of 100
            accept_probs2 = accept_probs2.at[exp_needed].set(jnp.exp(-safe_dH))
        
        accepts2 = jnp.random.random(n_chains_per_group) < accept_probs2

        # Update states - VECTORIZED
        group2_indices_accepts2 = group2_indices[accepts2]
        states = states.at[group2_indices_accepts2].set(q2[accepts2])
        # Track acceptance for second chains
        accepts = accepts.at[group2_indices].set(accepts[group2_indices] + accepts2)
        
    # Reshape final samples to original dimensions
    samples = samples.reshape((total_chains, n_samples) + orig_dim)
    
    # Compute acceptance rates for all chains
    acceptance_rates = accepts / total_iterations
    
    return samples, acceptance_rates
        
        

        
        
        
        
        
        

        
        
        
        
        
        
    
    
    
    
    