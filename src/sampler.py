import numpy as np

'''
Copied from https://github.com/yifanc96/AffineInvariantSamplers
'''

def hmc(log_prob, 
        initial, 
        n_samples, 
        grad_fn, 
        epsilon=0.1, 
        L=10, 
        n_chains=1, 
        n_thin = 1):
    """
    Vectorized Hamiltonian Monte Carlo (HMC) sampler implementation.
    
    This function implements the HMC algorithm with support for multiple chains running in parallel.
    It uses the leapfrog integrator for Hamiltonian dynamics and includes a Metropolis acceptance step.
    
    Parameters
    ----------
    log_prob : callable
        Function that computes the log probability of the target distribution.
        Should accept a numpy array of shape (n_chains, dim) and return an array of shape (n_chains,).
    initial : numpy.ndarray
        Initial state for the chains. Shape should be (dim,).
    n_samples : int
        Number of samples to generate for each chain.
    grad_fn : callable
        Function that computes the gradient of the log probability.
        Should accept a numpy array of shape (n_chains, dim) and return an array of same shape.
    epsilon : float, optional
        Step size for the leapfrog integrator. Default is 0.1.
    L : int, optional
        Number of leapfrog steps per iteration. Default is 10.
    n_chains : int, optional
        Number of parallel chains to run. Default is 1.
    n_thin : int, optional
        Thinning factor - store every n_thin sample. Default is 1 (no thinning).
    
    Returns
    -------
    samples : numpy.ndarray
        Array of shape (n_chains, n_samples, dim) containing the samples.
    acceptance_rates : numpy.ndarray
        Array of shape (n_chains,) containing the acceptance rate for each chain.
    
    Notes
    -----
    - The implementation is fully vectorized for efficiency.
    - NaN values in gradients are handled by replacing them with zeros.
    - The acceptance probability is computed in log space for numerical stability.
    """
    
    dim = len(initial)
    
    # Initialize multiple chains with small random perturbations around initial
    chains = np.tile(initial, (n_chains, 1)) + 0.1 * np.random.randn(n_chains, dim)
    
    # Vectorized evaluation of initial log probabilities
    chain_log_probs = log_prob(chains)
    
    # Calculate total iterations needed based on thinning factor
    total_iterations = 1 + (n_samples - 1) * n_thin  # +1 for initial state
    
    # Storage for samples and tracking acceptance
    samples = np.zeros((n_chains, n_samples, dim))
    accepts = np.zeros(n_chains)
    
    # Store initial state
    samples[:, 0, :] = chains
    
    # Sample index to track where to store thinned samples (start at 1 since we stored initial state)
    sample_idx = 1
    
    # Main sampling loop
    for i in range(1, total_iterations):
        # Generate momentum variables for all chains at once
        p = np.random.normal(size=(n_chains, dim))
        current_p = p.copy()
        
        # Leapfrog integration (vectorized over all chains)
        x = chains.copy()
        
        # Get gradients for all chains
        x_grad = grad_fn(x)
        # Handle NaN values safely
        x_grad = np.nan_to_num(x_grad, nan=0.0)
        
        # Half step for momentum
        p -= 0.5 * epsilon * x_grad
        
        # Full steps for position and momentum
        # L_steps = np.random.randint(L//2, 2*L)
        L_steps = L
        for j in range(L_steps):
            # Full step for position
            x += epsilon * p
            
            if j < L_steps - 1:
                # Get gradients for all chains
                x_grad = grad_fn(x)
                # Handle NaN values safely
                x_grad = np.nan_to_num(x_grad, nan=0.0)
                
                # Full step for momentum
                p -= epsilon * x_grad
        
        # Get final gradients for all chains
        x_grad = grad_fn(x)
        # Handle NaN values safely
        x_grad = np.nan_to_num(x_grad, nan=0.0)
        
        # Half step for momentum
        p -= 0.5 * epsilon * x_grad
        
        # Flip momentum for reversibility
        p = -p
        
        # Metropolis acceptance (vectorized)
        proposal_log_probs = log_prob(x)
        
        # Compute log acceptance ratio directly in log space - avoiding exp
        # current_H = -chain_log_probs + 0.5 * sum(current_p²)
        # proposal_H = -proposal_log_probs + 0.5 * sum(p²)
        # log_accept_prob = min(0, current_H - proposal_H)
        current_K = 0.5 * np.sum(current_p**2, axis=1)
        proposal_K = 0.5 * np.sum(p**2, axis=1)
        
        # Calculate log acceptance probability directly to avoid overflow
        log_accept_prob = np.minimum(0, proposal_log_probs - chain_log_probs - proposal_K + current_K)
        
        # Generate uniform random numbers for acceptance decision
        log_u = np.log(np.random.uniform(size=n_chains))
        
        # Create mask for accepted proposals
        accept_mask = log_u < log_accept_prob
        
        # Update chains and log probabilities where accepted
        chains[accept_mask] = x[accept_mask]
        chain_log_probs[accept_mask] = proposal_log_probs[accept_mask]
        
        # Track acceptances
        accepts += accept_mask
        
        # Store current state for all chains (only every n_thin iterations after the first)
        if (i - 1) % n_thin == 0 and sample_idx < n_samples:
            samples[:, sample_idx, :] = chains
            sample_idx += 1
    
    # Calculate acceptance rates for all chains
    acceptance_rates = accepts / (total_iterations - 1)
    
    return samples, acceptance_rates

def hamiltonian_side_move(gradient_func, 
                          potential_func, 
                          initial, 
                          n_samples, 
                          n_chains_per_group=5, 
                          epsilon=0.01, 
                          n_leapfrog=10, 
                          beta=1.0,
                          n_thin = 1):
    """
    Vectorized Ensemble Hamiltonian Side Move sampler.
    
    This function implements an ensemble MCMC method where particles are split into two groups.
    Each particle randomly selects two particles from the complementary group for preconditioning
    the Hamiltonian dynamics. This helps in exploring the target distribution more efficiently.
    
    Parameters
    ----------
    gradient_func : callable
        Function that computes the gradient of the potential energy.
        Should accept a numpy array of shape (n_chains, *orig_dim) and return an array of same shape.
    potential_func : callable
        Function that computes the potential energy (negative log probability).
        Should accept a numpy array of shape (n_chains, *orig_dim) and return an array of shape (n_chains,).
    initial : numpy.ndarray
        Initial state for the chains. Shape should be (*orig_dim,).
    n_samples : int
        Number of samples to generate for each chain.
    n_chains_per_group : int, optional
        Number of chains in each group. Total chains will be 2 * n_chains_per_group. Default is 5.
    epsilon : float, optional
        Base step size for the leapfrog integrator. Default is 0.01.
    n_leapfrog : int, optional
        Number of leapfrog steps per iteration. Default is 10.
    beta : float, optional
        Scaling factor for the step size. Default is 1.0.
    n_thin : int, optional
        Thinning factor - store every n_thin sample. Default is 1 (no thinning).
    
    Returns
    -------
    samples : numpy.ndarray
        Array of shape (total_chains, n_samples, *orig_dim) containing the samples.
    acceptance_rates : numpy.ndarray
        Array of shape (total_chains,) containing the acceptance rate for each chain.
    
    Notes
    -----
    - The implementation is fully vectorized for efficiency.
    - NaN values in gradients are handled by replacing them with zeros.
    - The acceptance probability is computed with numerical safeguards for stability.
    - The total number of chains is 2 * n_chains_per_group.
    """
    
    # Initialize
    orig_dim = initial.shape
    flat_dim = np.prod(orig_dim)
    total_chains = 2 * n_chains_per_group
    
    # Create initial states with small random perturbations
    states = np.tile(initial.flatten(), (total_chains, 1)) + 0.1 * np.random.randn(total_chains, flat_dim)
    
    # Split into two groups
    group1_indices = np.arange(n_chains_per_group)
    group2_indices = np.arange(n_chains_per_group, total_chains)
    
    # Calculate total iterations needed based on thinning factor
    total_iterations = n_samples * n_thin
    
    # Storage for samples and acceptance tracking
    samples = np.zeros((total_chains, n_samples, flat_dim))
    accepts = np.zeros(total_chains)
    
    # Sample index to track where to store thinned samples
    sample_idx = 0
    
    # Precompute some constants for efficiency
    beta_eps = beta * epsilon
    beta_eps_half = beta_eps / 2
    
    # Main sampling loop
    for i in range(total_iterations):
        # Store current state from all chains (only every n_thin iterations)
        if i % n_thin == 0 and sample_idx < n_samples:
            samples[:, sample_idx] = states
            sample_idx += 1
        
        #---------------------------------------------
        # First group update - VECTORIZED
        #---------------------------------------------
        
        # For each particle in group 1, randomly select TWO particles from group 2
        random_indices1_from_group2 = np.random.choice(group2_indices, size=n_chains_per_group)
        random_indices2_from_group2 = np.random.choice(group2_indices, size=n_chains_per_group)
        
        # Ensure the two selected particles are different
        for j in range(n_chains_per_group):
            while random_indices1_from_group2[j] == random_indices2_from_group2[j]:
                random_indices2_from_group2[j] = np.random.choice(group2_indices)
        
        # Get the two sets of selected particles from group 2 (shape: n_chains_per_group x flat_dim)
        selected_particles1_group2 = states[random_indices1_from_group2]
        selected_particles2_group2 = states[random_indices2_from_group2]
        
        # Use the difference between the two particles (shape: n_chains_per_group x flat_dim)
        diff_particles_group2 = (selected_particles1_group2 - selected_particles2_group2) / np.sqrt(2*n_chains_per_group)
        
        # Generate momentum - one scalar per chain (shape: n_chains_per_group)
        p1 = np.random.randn(n_chains_per_group)
        
        # Store current state and energy
        current_q1 = states[group1_indices].copy()
        current_q1_reshaped = current_q1.reshape(n_chains_per_group, *orig_dim)
        current_U1 = potential_func(current_q1_reshaped)

        current_K1 = np.clip(0.5 * p1**2, 0, 1000)
        
        # Leapfrog integration with preconditioning
        q1 = current_q1.copy()
        p1_current = p1.copy()
        
        # Initial half-step for momentum - VECTORIZED
        grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad1 = np.nan_to_num(grad1, nan=0.0)
        
        # Compute dot products between gradients and difference particles - VECTORIZED
        # This gives one scalar per chain (shape: n_chains_per_group)
        gradient_projections = np.sum(grad1 * diff_particles_group2, axis=1)
        p1_current -= beta_eps_half * gradient_projections
        
        # Full leapfrog steps
        # n_steps = np.random.randint(n_leapfrog//2, 2*n_leapfrog)
        n_steps = n_leapfrog
        for step in range(n_steps):
            # Full position step - VECTORIZED with broadcasting
            # For each chain j, we're doing: q1[j] += beta_eps * p1_current[j] * diff_particles_group2[j]
            q1 += beta_eps * (p1_current[:, np.newaxis] * diff_particles_group2)
            
            if step < n_steps - 1:
                # Full momentum step (except at last iteration) - VECTORIZED
                grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
                # Handle NaNs safely
                grad1 = np.nan_to_num(grad1, nan=0.0)
                
                gradient_projections = np.sum(grad1 * diff_particles_group2, axis=1)
                p1_current -= beta_eps * gradient_projections
        
        # Final half-step for momentum - VECTORIZED
        grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad1 = np.nan_to_num(grad1, nan=0.0)
        
        gradient_projections = np.sum(grad1 * diff_particles_group2, axis=1)
        p1_current -= beta_eps_half * gradient_projections
        
        # Compute proposed energy
        proposed_U1 = potential_func(q1.reshape(n_chains_per_group, *orig_dim))
        proposed_K1 = np.clip(0.5 *p1_current**2, 0, 1000)
        

        # Metropolis acceptance - IMPROVED FOR NUMERICAL STABILITY
        dH1 = (proposed_U1 + proposed_K1) - (current_U1 + current_K1)
        
        # Calculate acceptance probabilities with numerical safeguards for exp()
        # Instead of: accept_probs1 = np.minimum(1.0, np.exp(-dH1))
        accept_probs1 = np.ones_like(dH1)  # Default to accept
        # Only calculate exp for positive dH (negative cases auto-accept with prob 1.0)
        exp_needed = dH1 > 0
        if np.any(exp_needed):
            # Clip extremely high values before exponentiating
            safe_dH = np.clip(dH1[exp_needed], None, 100)  # No lower bound, upper bound of 100
            accept_probs1[exp_needed] = np.exp(-safe_dH)
        
        accepts1 = np.random.random(n_chains_per_group) < accept_probs1
        
        # Update states - VECTORIZED
        states[group1_indices[accepts1]] = q1[accepts1]
        # Track acceptance for first chains
        accepts[group1_indices] += accepts1
        
        #---------------------------------------------
        # Second group update - VECTORIZED similarly
        #---------------------------------------------
        
        # For each particle in group 2, randomly select TWO particles from group 1
        random_indices1_from_group1 = np.random.choice(group1_indices, size=n_chains_per_group)
        random_indices2_from_group1 = np.random.choice(group1_indices, size=n_chains_per_group)
        
        # Ensure the two selected particles are different
        for j in range(n_chains_per_group):
            while random_indices1_from_group1[j] == random_indices2_from_group1[j]:
                random_indices2_from_group1[j] = np.random.choice(group1_indices)
        
        # Get the two sets of selected particles from group 1 (shape: n_chains_per_group x flat_dim)
        selected_particles1_group1 = states[random_indices1_from_group1]
        selected_particles2_group1 = states[random_indices2_from_group1]
        
        # Use the difference between the two particles (shape: n_chains_per_group x flat_dim)
        diff_particles_group1 = (selected_particles1_group1 - selected_particles2_group1) / np.sqrt(2*n_chains_per_group)
        
        # Generate momentum - one scalar per chain (shape: n_chains_per_group)
        p2 = np.random.randn(n_chains_per_group)
        
        # Store current state and energy
        current_q2 = states[group2_indices].copy()
        current_q2_reshaped = current_q2.reshape(n_chains_per_group, *orig_dim)
        current_U2 = potential_func(current_q2_reshaped)
        current_K2 = np.clip(0.5 *p2**2, 0, 1000)
        
        # Leapfrog integration with preconditioning
        q2 = current_q2.copy()
        p2_current = p2.copy()
        
        # Initial half-step for momentum - VECTORIZED
        grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad2 = np.nan_to_num(grad2, nan=0.0)
        
        gradient_projections = np.sum(grad2 * diff_particles_group1, axis=1)
        p2_current -= beta_eps_half * gradient_projections
        
        # Full leapfrog steps
        n_steps = n_leapfrog
        for step in range(n_steps):
            # Full position step - VECTORIZED with broadcasting
            q2 += beta_eps * (p2_current[:, np.newaxis] * diff_particles_group1)
            
            if step < n_steps - 1:
                # Full momentum step (except at last iteration) - VECTORIZED
                grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
                # Handle NaNs safely
                grad2 = np.nan_to_num(grad2, nan=0.0)
                
                gradient_projections = np.sum(grad2 * diff_particles_group1, axis=1)
                p2_current -= beta_eps * gradient_projections
        
        # Final half-step for momentum - VECTORIZED
        grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad2 = np.nan_to_num(grad2, nan=0.0)
        
        gradient_projections = np.sum(grad2 * diff_particles_group1, axis=1)
        p2_current -= beta_eps_half * gradient_projections
        
        # Compute proposed energy
        proposed_U2 = potential_func(q2.reshape(n_chains_per_group, *orig_dim))
        proposed_K2 = np.clip(0.5 *p2_current**2, 0, 1000)
        
        # Metropolis acceptance - IMPROVED FOR NUMERICAL STABILITY
        dH2 = (proposed_U2 + proposed_K2) - (current_U2 + current_K2)
        
        # Calculate acceptance probabilities with numerical safeguards for exp()
        accept_probs2 = np.ones_like(dH2)  # Default to accept
        # Only calculate exp for positive dH (negative cases auto-accept with prob 1.0)
        exp_needed = dH2 > 0
        if np.any(exp_needed):
            # Clip extremely high values before exponentiating
            safe_dH = np.clip(dH2[exp_needed], None, 100)  # No lower bound, upper bound of 100
            accept_probs2[exp_needed] = np.exp(-safe_dH)
        
        accepts2 = np.random.random(n_chains_per_group) < accept_probs2
        
        # Update states - VECTORIZED
        states[group2_indices[accepts2]] = q2[accepts2]
        
        # Track acceptance for second chains
        accepts[group2_indices] += accepts2
    
    # Reshape final samples to original dimensions
    samples = samples.reshape((total_chains, n_samples) + orig_dim)
    
    # Compute acceptance rates for all chains
    acceptance_rates = accepts / total_iterations
    
    return samples, acceptance_rates

def hamiltonian_walk_move(gradient_func, potential_func, initial, n_samples, n_chains_per_group=5, 
                       epsilon=0.01, n_leapfrog=10, beta=0.05, n_thin = 1):
    """
    Vectorized Hamiltonian Walk Move sampler.
    
    This function implements an ensemble MCMC method where particles are split into two groups.
    Each group uses the centered ensemble of the other group for preconditioning the Hamiltonian
    dynamics. This helps in exploring the target distribution more efficiently.
    
    Parameters
    ----------
    gradient_func : callable
        Function that computes the gradient of the potential energy.
        Should accept a numpy array of shape (n_chains, *orig_dim) and return an array of same shape.
    potential_func : callable
        Function that computes the potential energy (negative log probability).
        Should accept a numpy array of shape (n_chains, *orig_dim) and return an array of shape (n_chains,).
    initial : numpy.ndarray
        Initial state for the chains. Shape should be (*orig_dim,).
    n_samples : int
        Number of samples to generate for each chain.
    n_chains_per_group : int, optional
        Number of chains in each group. Total chains will be 2 * n_chains_per_group. Default is 5.
    epsilon : float, optional
        Base step size for the leapfrog integrator. Default is 0.01.
    n_leapfrog : int, optional
        Number of leapfrog steps per iteration. Default is 10.
    beta : float, optional
        Scaling factor for the step size. Default is 0.05.
    n_thin : int, optional
        Thinning factor - store every n_thin sample. Default is 1 (no thinning).
    
    Returns
    -------
    samples : numpy.ndarray
        Array of shape (total_chains, n_samples, *orig_dim) containing the samples.
    acceptance_rates : numpy.ndarray
        Array of shape (total_chains,) containing the acceptance rate for each chain.
    
    Notes
    -----
    - The implementation is fully vectorized for efficiency.
    - NaN values in gradients are handled by replacing them with zeros.
    - The acceptance probability is computed with numerical safeguards for stability.
    - The total number of chains is 2 * n_chains_per_group.
    - This method uses centered ensembles for preconditioning, which can help with exploring
      multimodal distributions more effectively.
    """
    
    # Initialize
    orig_dim = initial.shape
    flat_dim = np.prod(orig_dim)
    total_chains = 2 * n_chains_per_group
    
    # Create initial states with small random perturbations
    states = np.tile(initial.flatten(), (total_chains, 1)) + 0.1 * np.random.randn(total_chains, flat_dim)
    
    # Split into two groups
    group1 = slice(0, n_chains_per_group)
    group2 = slice(n_chains_per_group, total_chains)
    
    # Calculate total iterations needed based on thinning factor
    total_iterations = n_samples * n_thin
    
    # Storage for samples and acceptance tracking
    samples = np.zeros((total_chains, n_samples, flat_dim))
    accepts = np.zeros(total_chains)
    
    # Sample index to track where to store thinned samples
    sample_idx = 0
    
    # Precompute some constants for efficiency
    beta_eps = beta * epsilon
    beta_eps_half = beta_eps / 2
    
    # Main sampling loop
    for i in range(total_iterations):
        # Store current state from all chains (only every n_thin iterations)
        if i % n_thin == 0 and sample_idx < n_samples:
            samples[:, sample_idx] = states
            sample_idx += 1
        
        # Compute centered ensembles for preconditioning
        centered2 = (states[group2] - np.mean(states[group2], axis=0)) / np.sqrt(n_chains_per_group)
        
        # First group update
        # Generate momentum - fully vectorized with correct dimensions
        p1 = np.random.randn(n_chains_per_group, n_chains_per_group)
        
        # Store current state and energy
        current_q1 = states[group1].copy()
        current_q1_reshaped = current_q1.reshape(n_chains_per_group, *orig_dim)
        current_U1 = potential_func(current_q1_reshaped)
        current_K1 =  np.clip(0.5 *np.sum(p1**2, axis=1), 0, 1000)
        
        # Leapfrog integration with preconditioning
        q1 = current_q1.copy()
        p1_current = p1.copy()
        
        # Initial half-step for momentum - vectorized
        grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad1 = np.nan_to_num(grad1, nan=0.0)
        
        # Matrix multiplication for projection - fully vectorized
        p1_current -= beta_eps_half * np.dot(grad1, centered2.T)
        
        # Full leapfrog steps
        # n_steps = np.random.randint(n_leapfrog//2, 2*n_leapfrog)
        n_steps = n_leapfrog
        for step in range(n_steps):
            # Full position step - vectorized matrix multiplication
            q1 += beta_eps * np.dot(p1_current, centered2)
            
            if step < n_steps - 1:
                # Full momentum step (except at last iteration) - vectorized
                grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
                # Handle NaNs safely
                grad1 = np.nan_to_num(grad1, nan=0.0)
                
                p1_current -= beta_eps * np.dot(grad1, centered2.T)
        
        # Final half-step for momentum - vectorized
        grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad1 = np.nan_to_num(grad1, nan=0.0)
        
        p1_current -= beta_eps_half * np.dot(grad1, centered2.T)
        
        # Compute proposed energy
        proposed_U1 = potential_func(q1.reshape(n_chains_per_group, *orig_dim))
        proposed_K1 = np.clip(0.5 *np.sum(p1_current**2, axis=1), 0, 1000)
        
        # Metropolis acceptance - IMPROVED FOR NUMERICAL STABILITY
        dH1 = (proposed_U1 + proposed_K1) - (current_U1 + current_K1)
        
        # Calculate acceptance probabilities with numerical safeguards for exp()
        accept_probs1 = np.ones_like(dH1)  # Default to accept
        # Only calculate exp for positive dH (negative cases auto-accept with prob 1.0)
        exp_needed = dH1 > 0
        if np.any(exp_needed):
            # Clip extremely high values before exponentiating
            safe_dH = np.clip(dH1[exp_needed], None, 100)  # No lower bound, upper bound of 100
            accept_probs1[exp_needed] = np.exp(-safe_dH)
        
        accepts1 = np.random.random(n_chains_per_group) < accept_probs1
        
        # Update states - vectorized
        states[group1][accepts1] = q1[accepts1]
        accepts[group1] += accepts1

        # Second group update - vectorized the same way
        centered1 = (states[group1] - np.mean(states[group1], axis=0)) / np.sqrt(n_chains_per_group)
        
        p2 = np.random.randn(n_chains_per_group, n_chains_per_group)
        
        current_q2 = states[group2].copy()
        current_q2_reshaped = current_q2.reshape(n_chains_per_group, *orig_dim)
        current_U2 = potential_func(current_q2_reshaped)
        current_K2 = np.clip(0.5 *np.sum(p2**2, axis=1), 0, 1000)
        
        q2 = current_q2.copy()
        p2_current = p2.copy()
        
        # Initial half-step for momentum - vectorized
        grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad2 = np.nan_to_num(grad2, nan=0.0)
        
        p2_current -= beta_eps_half * np.dot(grad2, centered1.T)
        
        # Full leapfrog steps
        n_steps = n_leapfrog
        for step in range(n_steps):
            # Full position step - vectorized
            q2 += beta_eps * np.dot(p2_current, centered1)
            
            if step < n_steps - 1:
                # Full momentum step (except at last iteration) - vectorized
                grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
                # Handle NaNs safely
                grad2 = np.nan_to_num(grad2, nan=0.0)
                
                p2_current -= beta_eps * np.dot(grad2, centered1.T)
        
        # Final half-step for momentum - vectorized
        grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad2 = np.nan_to_num(grad2, nan=0.0)
        
        p2_current -= beta_eps_half * np.dot(grad2, centered1.T)
        
        # Compute proposed energy
        proposed_U2 = potential_func(q2.reshape(n_chains_per_group, *orig_dim))
        proposed_K2 = np.clip(0.5 *np.sum(p2_current**2, axis=1), 0, 1000)
        
        # Metropolis acceptance - IMPROVED FOR NUMERICAL STABILITY
        dH2 = (proposed_U2 + proposed_K2) - (current_U2 + current_K2)
        
        # Calculate acceptance probabilities with numerical safeguards for exp()
        accept_probs2 = np.ones_like(dH2)  # Default to accept
        # Only calculate exp for positive dH (negative cases auto-accept with prob 1.0)
        exp_needed = dH2 > 0
        if np.any(exp_needed):
            # Clip extremely high values before exponentiating
            safe_dH = np.clip(dH2[exp_needed], None, 100)  # No lower bound, upper bound of 100
            accept_probs2[exp_needed] = np.exp(-safe_dH)
        
        accepts2 = np.random.random(n_chains_per_group) < accept_probs2
        
        # Update states
        states[group2][accepts2] = q2[accepts2]
        
        # Track acceptance for second chains
        accepts[group2] += accepts2
    
    # Reshape final samples to original dimensions
    samples = samples.reshape((total_chains, n_samples) + orig_dim)
    
    # Compute acceptance rates for all chains
    acceptance_rates = accepts / total_iterations
    
    return samples, acceptance_rates