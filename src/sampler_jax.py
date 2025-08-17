import jax 
import jax.numpy as jnp
from tqdm import tqdm

import jax
import jax.numpy as jnp

from typing import Callable


def leapfrog(x: jnp.ndarray, p: jnp.ndarray, grad_fn: callable, epsilon: float, L: int):
    """
    Vectorised leap-frog integrator without Python-side conditionals.

    Args:
        - x: current position. Shape (n_chains, dim)
        - p: current momentum. Shape (n_chains, dim)
        - grad_fn: callable - ∇log π(x). Function of shape (n_chains, dim) -> (n_chains, dim)
        - epsilon: Step size.
        - L: Number of leap-frog steps

    Returns:
        - x: proposed position. Shape (n_chains, dim)
        - p: proposed momentum. Shape (n_chains, dim)
    """
    # initial half step for momentum
    x_grad = jnp.nan_to_num(grad_fn(x), nan=0.0)
    p      = p - 0.5 * epsilon * x_grad

    # body for the first L-1 full steps
    def body(_, state):
        x, p = state
        x = x + epsilon * p                        # full position step
        x_grad = jnp.nan_to_num(grad_fn(x), nan=0.0)
        p = p - epsilon * x_grad                   # full momentum step
        return (x, p)

    # iterate body L-1 times
    x, p = jax.lax.fori_loop(0, L - 1, body, (x, p))

    # final position update (last half step for x)
    x = x + epsilon * p

    # final half step for momentum
    x_grad = jnp.nan_to_num(grad_fn(x), nan=0.0)
    p = p - 0.5 * epsilon * x_grad

    # flip momentum for reversibility
    return x, -p

    
def hmc(log_prob,
        initial: jnp.ndarray,
        n_samples,
        grad_fn = None,
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

    if grad_fn is None:
        grad_fn = jax.jit(jax.vmap(jax.grad(log_prob))) # F: (n_chains, dim) -> (n_chains, dim)
    else:
        grad_fn = jax.jit(jax.vmap(grad_fn)) # F: (n_chains, dim) -> (n_chains, dim)

    # Integers
    dim = len(initial)
    total_iterations = n_samples * n_thin

    # Random Numbers
    subkey_momentum, subkey_acceptance = jax.random.split(key, 2)
    
    # Shape (total_iterations + 1, ...), the +1 is for the initial momentum
    p_rngs = jax.random.normal(subkey_momentum, shape=(total_iterations + 1, n_chains, dim))
    acceptance_rngs = jnp.log(jax.random.uniform(subkey_acceptance, shape=(total_iterations, n_chains,), minval=1e-6, maxval=1))
    
    spread: float = 0.1
    chains_init = initial[None, :] + spread * p_rngs[0] # shape (n_chains, dim)
    accepts_init = jnp.zeros(n_chains) # shape (n_chains,)
    
    def main_loop(carry, lst_i):
        x, accepts = carry
        p, log_u = lst_i

        current_x = x.copy()
        current_p = p.copy() # Shape (n_chains, dim)

        ### Leapfrog integration
        # Proposed state
        x, p = leapfrog(x, p, grad_fn, epsilon, L) # Shape (n_chains, dim), (n_chains, dim)

        ### Metropolis acceptance
        current_log_probs = log_prob_fn(current_x)      # Shape (n_chains,)
        proposal_log_probs = log_prob_fn(x)             # Shape (n_chains,)
        current_K = 0.5 * jnp.sum(current_p**2, axis=1) # Shape (n_chains,)
        proposal_K = 0.5 * jnp.sum(p**2, axis=1)        # Shape (n_chains,)

        dH = (proposal_log_probs + proposal_K) - (current_log_probs + current_K)
        log_accept_prob = jnp.minimum(0.0, dH) # Shape (n_chains,)

        # Create mask for accepted proposals. True means accept the proposal, false means reject the proposal
        accept_mask = log_u < log_accept_prob # Shape (n_chains,)
        current_x = jnp.where(accept_mask[:, None], x, current_x)
        
        accepts += accept_mask.astype(int) # shape (n_chains,)
        
        return (current_x, accepts), current_x
    
    carry, samples = jax.lax.scan(
        main_loop, 
        init=(chains_init, accepts_init), 
        xs=(p_rngs[1:], acceptance_rngs) # Removed the first momentum as it's already in the initial state
    )
    accepts = carry[1]

    # Calculate acceptance rates for all chains
    acceptance_rates = accepts / total_iterations
    
    return samples, acceptance_rates


def nuts(log_prob,
         initial: jnp.ndarray,
         n_samples,
         grad_fn = None,
         epsilon=0.1,
         n_chains=1,
         n_thin=1,
         key=jax.random.PRNGKey(0)):
    '''
    Implementation of No-U-Turn Sampler (NUTS)
    '''
    pass


########################################################
# Affine Invariant method
########################################################

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

def hamiltonian_side_move(potential_func: Callable, 
                          initial: jnp.ndarray, 
                          n_samples: int,
                          grad_fn: Callable = None, 
                          n_chains_per_group: int = 5,
                          epsilon: float = 0.01, 
                          L: int = 10, 
                          beta: float = 1.0,
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
    
    if grad_fn is None:
        grad_fn_vmap = jax.jit(jax.vmap(jax.grad(potential_func))) # F: (n_chains, dim) -> (n_chains, dim)
    else:
        grad_fn_vmap = jax.jit(jax.vmap(grad_fn)) # F: (n_chains, dim) -> (n_chains, dim)

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

        current_K1 = 0.5 * p1**2

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
        proposed_K1 = 0.5 * p1_current**2 # Shape (n_chains_per_group,)
        
        # Metropolis acceptance in log scale for numerical stability
        dH1 = (proposed_U1 + proposed_K1) - (current_U1 + current_K1) # Shape (n_chains_per_group,)
        
        # Log acceptance probability: min(0, -dH)
        log_accept_prob1 = jnp.minimum(0.0, -dH1) # Shape (n_chains_per_group,)
        
        # Generate log-uniform random numbers
        log_u1 = jnp.log(jax.random.uniform(keys[2], shape=(n_chains_per_group,), minval=1e-10, maxval=1.0))
        
        # Accept if log_u < log_accept_prob (equivalent to u < exp(log_accept_prob))
        accepts1 = log_u1 < log_accept_prob1 # Shape (n_chains_per_group,)
        
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
        current_K2 = 0.5 * p2**2

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
        proposed_K2 = 0.5 * p2_current**2

        # Metropolis acceptance in log scale for numerical stability
        dH2 = (proposed_U2 + proposed_K2) - (current_U2 + current_K2)
        
        # Log acceptance probability: min(0, -dH)
        log_accept_prob2 = jnp.minimum(0.0, -dH2)
        
        # Generate log-uniform random numbers
        log_u2 = jnp.log(jax.random.uniform(keys[5], shape=(n_chains_per_group,), minval=1e-10, maxval=1.0))
        
        # Accept if log_u < log_accept_prob (equivalent to u < exp(log_accept_prob))
        accepts2 = log_u2 < log_accept_prob2

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

########################################################
# Hamiltonian Walk Move (HWM)
########################################################

def leapfrog_walk_move(q: jnp.ndarray, 
                       p: jnp.ndarray, 
                       grad_fn: Callable, 
                       beta_eps: float, 
                       L: int,
                       centered: jnp.ndarray):
    '''
    Args:
        q: Shape (n_chains_per_group, dim)
        p: Shape (n_chinas_per_group, n_chains_per_group)
        grad_fn: Gradient of log probabiltiy vectorized. Maps (batch_size, dim) -> (batch_size, dim)
        beta_eps: beta times step size (epsilon)
        L: Number of steps
        centered: Shape (n_chains_per_group, dim)
    '''
    grad = grad_fn(q) # Shape (n_chains_per_group, dim)
    grad = jnp.nan_to_num(grad, nan=0.0) 

    p -= 0.5 * beta_eps * jnp.dot(grad, centered.T) # Shape (n_chains_per_group, n_chains_per_group)
   

    for step in range(L):
        q += beta_eps * jnp.dot(p, centered) # Shape (n_chains_per_group, dim)

        if (step < L - 1):
            grad = grad_fn(q) # Shape (n_chains_per_group, dim)
            grad = jnp.nan_to_num(grad, nan=0.0)

            p -= beta_eps * jnp.dot(grad, centered.T)

    grad = grad_fn(q) # Shape (n_chains_per_group, dim)
    grad = jnp.nan_to_num(grad, nan=0.0)

    p -= 0.5 * beta_eps * jnp.dot(grad, centered.T)

    return q, p

# def leapfrog_walk_move(q: jnp.ndarray, 
#                        p: jnp.ndarray, 
#                        grad_fn: Callable, 
#                        beta_eps: float, 
#                        L: int,
#                        centered: jnp.ndarray):
#     '''
#     Args:
#         q: Shape (n_chains_per_group, dim)
#         p: Shape (n_chinas_per_group, n_chains_per_group)
#         grad_fn: Gradient of log probabiltiy vectorized. Maps (batch_size, dim) -> (batch_size, dim)
#         beta_eps: beta times step size (epsilon)
#         L: Number of steps
#         centered: Shape (n_chains_per_group, dim)
#     '''

#     grad = grad_fn(q)
#     grad = jnp.nan_to_num(grad, nan=0.0) 

#     p -= 0.5 * beta_eps * jnp.dot(grad, centered.T)

#     for step in range(L):
#         q += beta_eps * jnp.dot(p, centered)

#         if (step < L - 1):
#             grad = grad_fn(q)
#             grad = jnp.nan_to_num(grad, nan=0.0)

#             p -= beta_eps * jnp.dot(grad, centered.T)
    
#     grad = grad_fn(q)
#     grad = jnp.nan_to_num(grad, nan=0.0) 

#     p -= 0.5 * beta_eps * jnp.dot(grad, centered)

#     return q, p




    


def hamiltonian_walk_move(potential_func: Callable, 
                          initial: jnp.ndarray, 
                          n_samples: int, 
                          grad_fn: Callable = None,
                          n_chains_per_group: int = 5, 
                          epsilon: float = 0.01, 
                          L: int = 10, 
                          beta: float = 0.05,
                          n_thin=1,
                          key=jax.random.PRNGKey(0)):
    """
    Hamiltonian Walk Move (HWM) sampler implementation using JAX.
    """
    # JIT
    potential_func_vmap = jax.jit(jax.vmap(potential_func))           # F: (n_chains, dim) -> (n_chains,)
    if grad_fn is None:
        grad_fn_vmap = jax.jit(jax.vmap(jax.grad(potential_func))) # F: (n_chains, dim) -> (n_chains, dim)
    else:
        grad_fn_vmap = jax.jit(jax.vmap(grad_fn)) # F: (n_chains, dim) -> (n_chains, dim)

    # Sizes
    dim = len(initial)
    total_chains = 2 * n_chains_per_group
    total_iterations = n_samples * n_thin

    # Initalize States
    spread = 0.1
    states = jnp.tile(initial.flatten(), (total_chains, 1)) + spread * jax.random.normal(key, shape=(total_chains, dim)) # Shape (total_chains, dim)
    states_group1 = states[:n_chains_per_group] # Shape (n_chains_per_group, dim)
    states_group2 = states[n_chains_per_group:] # Shape (n_chains_per_group, dim)

    accepts_group1 = jnp.zeros(n_chains_per_group)
    accepts_group2 = jnp.zeros(n_chains_per_group)

    keys_per_iter = 4
    all_keys = jax.random.split(key, total_iterations * keys_per_iter).reshape(total_iterations, keys_per_iter, 2)

    beta_eps = beta * epsilon

    def main_loop(carry, keys):
        #---------------------------------------------
        # Unpack input
        #---------------------------------------------
        states_group1, states_group2, accepts_group1, accepts_group2 = carry # Array shapped (total_chains, dim), integer

        q1 = states_group1
        q2 = states_group2
        q1_current = q1.copy()
        q2_current = q2.copy()


        ########################################################
        # Group 1
        ########################################################

        centered2 = (q2 - jnp.mean(q2, axis=0)[None, :]) / jnp.sqrt(n_chains_per_group) # Shape (n_chains_per_group, dim)

        # Random Momentum
        p1_current = jax.random.normal(keys[0], shape=(n_chains_per_group, n_chains_per_group))

        # Current Energy
        current_U1 = potential_func_vmap(q1_current) # Shape (n_chains_per_group,)
        current_K1 = 0.5 * jnp.sum(p1_current**2, axis=1) # Shape (n_chains_per_group,)

        # Leapfrog Integration
        q1_proposed, p1_proposed = leapfrog_walk_move(
            q1_current, 
            p1_current, 
            grad_fn_vmap, 
            beta_eps, 
            L, 
            centered2
        )
        proposed_U1 = potential_func_vmap(q1_proposed)
        proposed_K1 = 0.5 * jnp.sum(p1_proposed**2, axis=1) # Shape (n_chains_per_group,)

        # Metropolis Step in log scale for numerical stability
        dH1 = (proposed_U1 + proposed_K1) - (current_U1 + current_K1)
        
        # Log acceptance probability: min(0, -dH)
        log_accept_prob1 = jnp.minimum(0.0, -dH1)
        
        # Generate log-uniform random numbers
        log_u1 = jnp.log(jax.random.uniform(keys[1], shape=(n_chains_per_group,), minval=1e-10, maxval=1.0))
        
        # Accept if log_u < log_accept_prob (equivalent to u < exp(log_accept_prob))
        accepts1 = log_u1 < log_accept_prob1

        # Log Changes
        accepts_group1 += accepts1.astype(int)
        states_group1 = jnp.where(accepts1[:, None], q1_proposed, states_group1)


        ########################################################
        # Group 2
        ########################################################

        centered1 = (q1 - jnp.mean(q1, axis=0)) / jnp.sqrt(n_chains_per_group) # Shape (n_chains_per_group, dim)

        # Random Momentum
        p2_current = jax.random.normal(keys[2], shape=(n_chains_per_group, n_chains_per_group))

        # Current Energy
        current_U2 = potential_func_vmap(q2_current)
        current_K2 = 0.5 * jnp.sum(p2_current**2, axis=1) # Shape (n_chains_per_group,)

        # Leapfrog Integration
        q2_proposed, p2_proposed = leapfrog_walk_move(
            q2_current, 
            p2_current, 
            grad_fn_vmap, 
            beta_eps, 
            L, 
            centered1
        )
        proposed_U2 = potential_func_vmap(q2_proposed)
        proposed_K2 = 0.5 * jnp.sum(p2_proposed**2, axis=1) # Shape (n_chains_per_group,)

        # Metropolis Step in log scale for numerical stability
        dH2 = (proposed_U2 + proposed_K2) - (current_U2 + current_K2)
        
        # Log acceptance probability: min(0, -dH)
        log_accept_prob2 = jnp.minimum(0.0, -dH2)
        
        # Generate log-uniform random numbers
        log_u2 = jnp.log(jax.random.uniform(keys[3], shape=(n_chains_per_group,), minval=1e-10, maxval=1.0))
        
        # Accept if log_u < log_accept_prob (equivalent to u < exp(log_accept_prob))
        accepts2 = log_u2 < log_accept_prob2

        # Log Changes
        accepts_group2 += accepts2.astype(int)
        states_group2 = jnp.where(accepts2[:, None], q2_proposed, states_group2)

        ########################################################
        # Return
        ########################################################

        final_states = jnp.concatenate([states_group1, states_group2]) # Shape (total_chains, dim)

        return (states_group1, states_group2, accepts_group1, accepts_group2), final_states
    
    carry, previous_states = jax.lax.scan(main_loop, init=(states_group1, states_group2, accepts_group1, accepts_group2), xs=all_keys)
    states_group1, states_group2, accepts_group1, accepts_group2 = carry

    # Compute acceptance rates for all chains
    acceptance_rates_group1 = accepts_group1 / (total_iterations) # Shape (n_chains_per_group,)
    acceptance_rates_group2 = accepts_group2 / (total_iterations) # Shape (n_chains_per_group,)

    acceptance_rates = jnp.concatenate([acceptance_rates_group1, acceptance_rates_group2]) # Shape (total_chains,)

    return previous_states, acceptance_rates