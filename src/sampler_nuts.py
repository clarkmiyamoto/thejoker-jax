import numpy as np
from tqdm import tqdm

from typing import Callable

# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------

# Kinetic Energy
def kinetic_energy(p):
    '''
    Args:
        p: (batch_size, dim)

    Returns:
        kinetic energy in shape (batch_size,)
    '''
    return 0.5 * np.sum(p * p, axis=-1)          # ½‖p‖²

def kinetic_energy_side_move(p):
    '''
    Args:
        p: (batch_size)

    Returns:
        kinetic energy in shape (batch_size,)
    '''
    return 0.5 * p**2 # Shape (batch_size,)

# No U Turn stop criterion
def stop_criterion(q_left, q_right, p_left, p_right):      # all (B,D)
    dq        = q_right - q_left
    proj_left = np.sum(dq * p_left,  axis=-1)              # (B,)
    proj_rght = np.sum(dq * p_right, axis=-1)              # (B,)
    return (proj_left >= 0) & (proj_rght >= 0)             # (B,) bool

def stop_criterion_side_move(q_left, q_right, p_left, p_right, diff_particles_group2):      # all (B,D)
    '''
    What Yifan thinks is the stop criterion
    '''
    dq        = q_right - q_left # Shape (B,D)
    proj_left = np.einsum('bd, bd -> b', dq, diff_particles_group2) * p_left
    proj_right = np.einsum('bd, bd -> b', dq, diff_particles_group2) * p_right
    return (proj_left >= 0) & (proj_right >= 0)             # (B,) bool



# ------------------------------------------------------------------
#  Leapfrog
# ------------------------------------------------------------------
def leapfrog(q, p, grad_logp, step):
    p_half = p + 0.5 * step * grad_logp(q)    # half-kick
    q_new  = q + step * p_half                # drift
    p_new  = p_half + 0.5 * step * grad_logp(q_new)   # second half-kick
    return q_new, p_new

def leapfrog_side_move(q1,
                       p1,
                       grad_logp,
                       step,
                       diff_particles_group2):
    '''Vectorized version of Eqn 4.11 in Yifan's paper
    
    Args:
        q1: (batch_size, dim)
        p1: (batch_size,)
        grad_logp: (batch_size, dim) -> (batch_size,)
        step: float
        diff_particles_group2: (dim)
    '''  
    p_half = p1 - 0.5 * step * np.einsum('bd,bd->b', diff_particles_group2, grad_logp(q1)) # (batch_size,)
    q_new = q1 + step * np.einsum('bd,b->bd', diff_particles_group2, p_half) # (batch_size, dim)
    p_new = p_half - 0.5 * step * np.einsum('bd,bd->b', diff_particles_group2, grad_logp(q_new)) # (batch_size,)

    return q_new, p_new

def leapfrog_walk_move(q1,
                          p1,
                          grad_logp,
                          step,
                          centerEnsemble):
    '''Eqn 4.8 in Yifan's paper'''
    p_half = p1 - 0.5 * step * np.einsum('db,d->b', centerEnsemble, grad_logp(q1)) # Shape (batch_size,)
    q_new = q1 + step * centerEnsemble @ p_half # Shape (dim,)
    p_new = p_half - 0.5 * step * np.einsum('db,d->b', centerEnsemble, grad_logp(q_new)) # Shape (batch_size,)

    return q_new, p_new

# ------------------------------------------------------------------
#  Recursive tree builder (Hoffman & Gelman Alg. 3)
# ------------------------------------------------------------------
def build_tree(q, p, grad_logp, log_prob,
               u,  v,  depth, step, rng):
    """
    All inputs are batched (B,*) except:
        v     : +1 or -1  (scalar) - same for every chain in this sub-tree
        depth : int       - scalar
        step  : float     - scalar
    Returns per-chain arrays (shape (B,…) except booleans):
        q_min, p_min, q_max, p_max, q_prop,
        n_valid, cont_mask, acc_sum, n_alpha
    """
    if depth == 0:
        q1, p1  = leapfrog(q, p, grad_logp, v * step)           # (B,D)
        joint1  = log_prob(q1) - kinetic_energy(p1)             # (B,)
        joint0  = log_prob(q)  - kinetic_energy(p)              # (B,)

        valid   = (u < np.exp(joint1)).astype(int)              # (B,) int
        accept  = np.minimum(1.0, np.exp(joint1 - joint0))      # (B,)

        return (q1, p1, q1, p1, q1,
                valid, np.ones_like(valid, bool), accept, np.ones_like(accept))
    
    # ─────────────────────────────────────────── recursive case
    (q_minus, p_minus, q_plus, p_plus, q_prop,
     n_valid, cont_mask, acc_sum, n_alpha) = build_tree(
        q, p, grad_logp, log_prob, u, v, depth - 1, step, rng)

    if cont_mask.any(): # at least one chain still expanding
        # Which direction to integrate in?
        if v == -1:
            (q_minus2, p_minus2, _, _, q_prop2,
             n2, cont2, acc2, n_a2) = build_tree(
                q_minus, p_minus, grad_logp, log_prob,
                u, v, depth - 1, step, rng)
            q_minus, p_minus = q_minus2, p_minus2
        else:
            (_, _, q_plus2, p_plus2, q_prop2,
             n2, cont2, acc2, n_a2) = build_tree(
                q_plus, p_plus, grad_logp, log_prob,
                u, v, depth - 1, step, rng)
            q_plus, p_plus = q_plus2, p_plus2

        # choose which proposal survives - each chain independently
        choose = rng.uniform(size=q.shape[0]) < n2 / np.maximum(n_valid + n2, 1)
        q_prop[choose] = q_prop2[choose]

        n_valid   += n2
        cont_mask  = cont2 & stop_criterion(q_minus, q_plus, p_minus, p_plus)
        acc_sum   += acc2
        n_alpha   += n_a2

    return (q_minus, p_minus, q_plus, p_plus, q_prop,
            n_valid, cont_mask, acc_sum, n_alpha)
    
# Affien Invariant Tree Builder

def build_tree_side_move(q, p, grad_logp, log_prob,
                         u,  v,  depth, step, rng,
                         diff_particles_group2):
    """
    All inputs are batched (B,*) except:
        v     : +1 or -1  (scalar) - same for every chain in this sub-tree
        depth : int       - scalar
        step  : float     - scalar
    Returns per-chain arrays (shape (B,…) except booleans):
        q_min, p_min, q_max, p_max, q_prop,
        n_valid, cont_mask, acc_sum, n_alpha
    """
    if depth == 0:
        q1, p1  = leapfrog_side_move(q, p, grad_logp, v * step, diff_particles_group2) 
        joint1  = log_prob(q1) - kinetic_energy_side_move(p1)             # (B,)
        joint0  = log_prob(q)  - kinetic_energy_side_move(p)              # (B,)

        valid   = (u < np.exp(joint1)).astype(int)              # (B,) int
        accept  = np.minimum(1.0, np.exp(joint1 - joint0))      # (B,)

        return (q1, p1, q1, p1, q1,
                valid, np.ones_like(valid, bool), accept, np.ones_like(accept))
    
    # ─────────────────────────────────────────── recursive case
    (q_minus, p_minus, q_plus, p_plus, q_prop,
     n_valid, cont_mask, acc_sum, n_alpha) = build_tree_side_move(
        q, p, grad_logp, log_prob, u, v, depth - 1, step, rng, diff_particles_group2)

    if cont_mask.any(): # at least one chain still expanding
        # Which direction to integrate in?
        if v == -1:
            (q_minus2, p_minus2, _, _, q_prop2,
             n2, cont2, acc2, n_a2) = build_tree_side_move(
                q_minus, p_minus, grad_logp, log_prob,
                u, v, depth - 1, step, rng, diff_particles_group2)
            q_minus, p_minus = q_minus2, p_minus2
        else:
            (_, _, q_plus2, p_plus2, q_prop2,
             n2, cont2, acc2, n_a2) = build_tree_side_move(
                q_plus, p_plus, grad_logp, log_prob,
                u, v, depth - 1, step, rng, diff_particles_group2)
            q_plus, p_plus = q_plus2, p_plus2

        # choose which proposal survives - each chain independently
        choose = rng.uniform(size=q.shape[0]) < n2 / np.maximum(n_valid + n2, 1)
        q_prop[choose] = q_prop2[choose]

        n_valid   += n2
        cont_mask  = cont2 & stop_criterion_side_move(q_minus, q_plus, p_minus, p_plus, diff_particles_group2)
        acc_sum   += acc2
        n_alpha   += n_a2

    return (q_minus, p_minus, q_plus, p_plus, q_prop,
            n_valid, cont_mask, acc_sum, n_alpha)

# ------------------------------------------------------------------
#  NUTs Sampler
# ------------------------------------------------------------------
def nuts_sampler(
        log_prob,           # accepts (B,D) → (B,)
        grad_log_prob,      # accepts (B,D) → (B,D)
        num_samples,        # integer
        initial_positions,  # (B,D) array
        step_size=0.25,     # scalar
        max_depth=10,
        rng=None
    ):
    """
    Returns:
        samples : (B, num_samples, D)
        stats   : dict with 'accept_rate_mean' (B,), 'step_size'
    """
    rng = np.random.default_rng() if rng is None else rng
    B, D  = initial_positions.shape
    jitter = 0.1
    q_cur = initial_positions.copy() + jitter * np.random.randn(B,D)        # (B,D)

    samples = np.empty((B, num_samples, D))
    accepts = np.empty((B, num_samples))

    

    for m in range(num_samples):
        p0      = rng.normal(size=(B, D))
        joint0  = log_prob(q_cur) - kinetic_energy(p0)           # (B,)
        u       = rng.uniform(size=B) * np.exp(joint0)           # (B,)

        q_minus = q_plus = q_cur.copy()
        p_minus = p_plus = p0.copy()
        q_prop  = q_cur.copy()

        n_valid  = np.ones(B, dtype=int)
        cont_mask = np.ones(B, bool)
        acc_sum  = np.zeros(B)
        n_alpha  = np.zeros(B)
        depth    = 0

        while cont_mask.any() and depth < max_depth:
            v = rng.choice([-1, 1])        # one direction shared by all chains this layer
            (q_minus, p_minus, q_plus, p_plus, q_candidate,
             n_new, cont_new, acc_new, n_a_new) = build_tree(
                (q_minus if v == -1 else q_plus),
                (p_minus if v == -1 else p_plus),
                grad_log_prob, log_prob,
                u, v, depth, step_size, rng)

            # Metropolis choice of which proposal to keep (independently per chain)
            take = rng.uniform(size=B) < n_new / np.maximum(n_valid + n_new, 1)
            q_prop[take] = q_candidate[take]

            # Update bookkeeping
            n_valid   += n_new
            cont_mask  = cont_new & stop_criterion(q_minus, q_plus, p_minus, p_plus)
            acc_sum   += acc_new
            n_alpha   += n_a_new
            depth     += 1

        # Record draw
        q_cur           = q_prop
        samples[:, m]   = q_cur
        accepts[:, m]   = acc_sum / np.maximum(n_alpha, 1)

    return samples, {
        "accept_rate_mean": accepts.mean(axis=1),   # per chain
        "step_size"       : step_size
    }

# Affine Invariant NUTs Sampler

def nuts_sampler_side_move(
    log_prob: Callable,
    grad_log_prob: Callable,
    num_samples: int,
    initial_positions: np.ndarray,
    n_chains_per_group: int = 5, 
    step_size: float =0.25,
    max_depth: int = 10,
    rng: np.random.Generator = None
    ) -> tuple[np.ndarray, dict]:
    """
    Affine invariant No-U-Turn sampler using side move.

    Args:
        - log_prob (Callable): Log probability function.
            Maps arrays of shape (batch_size, dim) -> (batch_size,)
        - grad_log_prob (Callable): Gradient of log probability. 
            Maps arrays of shape (batch_size, dim) -> (batch_size, dim)
        - num_samples (int): Number of samples to draw.
        - initial_positions (np.ndarray): Initial positions. Array of shape (2 * n_chains_per_group, dim)
        - n_chains_per_group (int): Number of chains per group.
        - step_size (float): Step size of leapfrog integrator.
        - max_depth (int): Maximum depth of recursion for finding the No-U-Turn stop criterion.
        - rng (np.random.Generator): Fix random number generator.

    Returns:
        - samples (np.ndarray): Samples.
            Shape (2 * n_chains_per_group, num_samples, dim)
        - stats (dict): Dictionary with 'accept_rate_mean' (2 * n_chains_per_group,), 'step_size'
            Shape (2 * n_chains_per_group,), 'step_size'
    """
    rng = np.random.default_rng() if rng is None else rng
    B, D  = initial_positions.shape
    jitter = 0.1
    q_current = initial_positions.copy() + jitter * np.random.randn(B,D)             # (B,D)

    samples = np.empty((B, num_samples, D))
    accepts = np.empty((B, num_samples))


    # Split into two groups
    total_chains = 2 * n_chains_per_group
    group1_indices = np.arange(n_chains_per_group)
    group2_indices = np.arange(n_chains_per_group, total_chains)

    for m in tqdm(range(num_samples)):
        #---------------------------------------------
        # First group update
        #---------------------------------------------

        ##### Extra setup for Side Move
        
        # For each particle in group 1, randomly select TWO particles from group 2
        random_indices1_from_group2 = np.random.choice(group2_indices, size=n_chains_per_group)
        random_indices2_from_group2 = np.random.choice(group2_indices, size=n_chains_per_group)
        # Ensure the two selected particles are different
        for j in range(n_chains_per_group):
            while random_indices1_from_group2[j] == random_indices2_from_group2[j]:
                random_indices2_from_group2[j] = np.random.choice(group2_indices)

        # Get the two sets of selected particles from group 2 (shape: n_chains_per_group x flat_dim)
        selected_particles1_group2 = q_current[random_indices1_from_group2] # Shape (n_chains_per_group, D)
        selected_particles2_group2 = q_current[random_indices2_from_group2] # Shape (n_chains_per_group, D)

        diff_particles_group2 = (selected_particles1_group2 - selected_particles2_group2) / np.sqrt(2*n_chains_per_group) # Shape (n_chains_per_group, D)
        
        ##### Main loop for NUTs algorithm, modified for side move     
        p0      = rng.normal(size=(n_chains_per_group)) # Shape (n_chains_per_group, D)
        joint0  = log_prob(q_current[group1_indices]) - kinetic_energy_side_move(p0) # Shape (n_chains_per_group,)
        u       = rng.uniform(size=n_chains_per_group) * np.exp(joint0) # Shape (n_chains_per_group,)

        q_minus = q_plus = q_current[group1_indices].copy() # Shape (n_chains_per_group, D)
        p_minus = p_plus = p0.copy() # Shape (n_chains_per_group,)
        q_prop  = q_current[group1_indices].copy() # Shape (n_chains_per_group, D)

        n_valid  = np.ones(n_chains_per_group, dtype=int)
        cont_mask = np.ones(n_chains_per_group, bool)
        acc_sum  = np.zeros(n_chains_per_group)
        n_alpha  = np.zeros(n_chains_per_group)
        depth    = 0

        while cont_mask.any() and depth < max_depth:
            v = rng.choice([-1, 1])        # one direction shared by all chains this layer
            (q_minus, p_minus, q_plus, p_plus, q_candidate,
             n_new, cont_new, acc_new, n_a_new) = build_tree_side_move(
                (q_minus if v == -1 else q_plus),
                (p_minus if v == -1 else p_plus),
                grad_log_prob, log_prob,
                u, v, depth, step_size, rng,
                diff_particles_group2)

            # Metropolis choice of which proposal to keep (independently per chain)
            take = rng.uniform(size=n_chains_per_group) < n_new / np.maximum(n_valid + n_new, 1)
            q_prop[take] = q_candidate[take] # Shape (n_chains_per_group, D)

            # Update bookkeeping
            n_valid   += n_new
            cont_mask  = cont_new & stop_criterion_side_move(q_minus, q_plus, p_minus, p_plus, diff_particles_group2)
            acc_sum   += acc_new
            n_alpha   += n_a_new
            depth     += 1

        # Record draw
        q_current[group1_indices] = q_prop
        samples[group1_indices, m, :] = q_current[group1_indices]
        accepts[group1_indices, m] = acc_sum / np.maximum(n_alpha, 1)

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
        selected_particles1_group1 = q_current[random_indices1_from_group1]
        selected_particles2_group1 = q_current[random_indices2_from_group1]

        diff_particles_group1 = (selected_particles1_group1 - selected_particles2_group1) / np.sqrt(2*n_chains_per_group)

        ##### Main loop for NUTs algorithm, modified for side move     
        p0      = rng.normal(size=(n_chains_per_group)) # Shape (n_chains_per_group, D)
        joint0  = log_prob(q_current[group2_indices]) - kinetic_energy_side_move(p0) # Shape (n_chains_per_group,)
        u       = rng.uniform(size=n_chains_per_group) * np.exp(joint0) # Shape (n_chains_per_group,)

        q_minus = q_plus = q_current[group2_indices].copy() # Shape (n_chains_per_group, D)
        p_minus = p_plus = p0.copy() # Shape (n_chains_per_group,)
        q_prop  = q_current[group2_indices].copy() # Shape (n_chains_per_group, D)

        n_valid  = np.ones(n_chains_per_group, dtype=int)
        cont_mask = np.ones(n_chains_per_group, bool)
        acc_sum  = np.zeros(n_chains_per_group)
        n_alpha  = np.zeros(n_chains_per_group)
        depth    = 0

        while cont_mask.any() and depth < max_depth:
            v = rng.choice([-1, 1])        # one direction shared by all chains this layer
            (q_minus, p_minus, q_plus, p_plus, q_candidate,
             n_new, cont_new, acc_new, n_a_new) = build_tree_side_move(
                (q_minus if v == -1 else q_plus),
                (p_minus if v == -1 else p_plus),
                grad_log_prob, log_prob,
                u, v, depth, step_size, rng,
                diff_particles_group1)

            # Metropolis choice of which proposal to keep (independently per chain)
            take = rng.uniform(size=n_chains_per_group) < n_new / np.maximum(n_valid + n_new, 1)
            q_prop[take] = q_candidate[take] # Shape (n_chains_per_group, D)

            # Update bookkeeping
            n_valid   += n_new
            cont_mask  = cont_new & stop_criterion_side_move(q_minus, q_plus, p_minus, p_plus, diff_particles_group1)
            acc_sum   += acc_new
            n_alpha   += n_a_new
            depth     += 1

        # Record draw
        q_current[group2_indices] = q_prop
        samples[group2_indices, m, :] = q_current[group2_indices]
        accepts[group2_indices, m] = acc_sum / np.maximum(n_alpha, 1)

    return samples, {
        "accept_rate_mean": accepts.mean(axis=1),   # per chain
        "step_size"       : step_size
    }