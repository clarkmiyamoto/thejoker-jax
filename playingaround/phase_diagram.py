import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
import ipywidgets as widgets
from IPython.display import display
import sys
import os
import argparse
from scipy.stats import beta
import random

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from src.velocity import velocity

# Enable double precision for JAX
jax.config.update("jax_enable_x64", True)


def sample_kipping_eccentricity(seed=None):
    """
    Sample eccentricity from Kipping's Beta distribution prior.
    Kipping's prior: Beta(0.867, 3.03) for eccentricity distribution.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Kipping's Beta distribution parameters for eccentricity
    a, b = 0.867, 3.03
    e = beta.rvs(a, b)
    
    # Ensure eccentricity is in valid range [0, 1)
    return min(e, 0.99)


def generate_poisson_observations(N, T, seed=None):
    """
    Generate observation times with Poisson-like distribution.
    First observation at t=0, last at t=T, remaining N-2 Poisson distributed.
    
    Parameters:
    - N: Number of observations
    - T: Total time span
    - seed: Random seed
    
    Returns:
    - time_obs: Array of observation times
    """
    if seed is not None:
        np.random.seed(seed)
    
    if N <= 2:
        return np.array([0.0, T])
    
    # Generate N-1 intervals from exponential distribution
    # Mean interval should be T/(N-1) to fill the total time
    mean_interval = T / (N - 1)
    intervals = np.random.exponential(mean_interval, N - 1)
    
    # Normalize intervals so their sum equals T
    intervals = intervals * T / np.sum(intervals)
    
    # Create observation times as cumulative sum
    time_obs = np.cumsum(np.concatenate([[0], intervals]))
    
    # Ensure last observation is exactly at T
    time_obs[-1] = T
    
    return time_obs


def generate_clustered_observations(total_time, n_clusters, obs_per_cluster, cluster_width, seed=42):
    """
    Generate observation times in clusters (similar to real observing runs).
    
    Parameters:
    - total_time: Total time span for observations (days)
    - n_clusters: Number of observation clusters
    - obs_per_cluster: Number of observations per cluster
    - cluster_width: Width of each cluster (days)
    - seed: Random seed for reproducibility
    
    Returns:
    - time_obs: Array of observation times
    """
    np.random.seed(seed)
    
    # Generate cluster centers
    cluster_centers = np.sort(np.random.uniform(0, total_time, n_clusters))
    
    time_obs = []
    for center in cluster_centers:
        # Generate observations within each cluster
        cluster_times = np.random.uniform(
            center - cluster_width/2, 
            center + cluster_width/2, 
            obs_per_cluster
        )
        time_obs.extend(cluster_times)
    
    return np.array(time_obs)


def generate_rv_data(time_obs, period, eccentricity, omega, phi0, K, v0, sigma, seed=42):
    """
    Generate radial velocity data with noise.
    
    Parameters:
    - time_obs: Array of observation times
    - period: Orbital period (days)
    - eccentricity: Orbital eccentricity
    - omega: Argument of periastron (radians)
    - phi0: Phase offset (radians)
    - K: Semi-amplitude of velocity (m/s)
    - v0: Systemic velocity offset (m/s)
    - sigma: RV uncertainty (m/s)
    - seed: Random seed for reproducibility
    
    Returns:
    - time_obs: Observation times
    - rv_obs: Observed radial velocities with noise
    - rv_err: RV uncertainties
    - rv_true: True radial velocities (no noise)
    """
    np.random.seed(seed)
    
    # Generate true RV curve
    rv_true = velocity(time_obs, period, eccentricity, omega, phi0, K, v0)
    
    # Add Gaussian noise
    rv_obs = rv_true + np.random.normal(0, sigma, len(time_obs))
    rv_err = np.full_like(rv_obs, sigma)
    
    return time_obs, rv_obs, rv_err, rv_true


def plot_rv_data(time_obs, rv_obs, rv_err, time_true, rv_true, period, eccentricity, K, sigma):
    """
    Plot radial velocity data with current parameters in title.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot true RV curve in blue
    plt.plot(time_true, rv_true, 'b-', linewidth=2, label='True RV Curve')
    
    # Plot observed velocities as black scatter plot with error bars
    plt.errorbar(time_obs, rv_obs, yerr=rv_err, fmt='ko', 
                capsize=3, capthick=1, markersize=4, label='Observed Data')
    
    plt.xlabel('Time (days)')
    plt.ylabel('Radial Velocity (m/s)')
    plt.title(f'RV Data: P={period:.1f}d, e={eccentricity:.2f}, '
             f'K={K:.1f}m/s, σ={sigma:.1f}m/s')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def interactive_rv_plot(time_period_ratio, K_sigma_ratio, period, eccentricity, omega, phi0, v0, 
                       n_clusters, obs_per_cluster, cluster_width, seed):
    """
    Interactive function for ipywidgets.interact.
    
    Parameters:
    - time_period_ratio: Ratio of total observation time to period
    - K_sigma_ratio: Ratio of amplitude to uncertainty
    - period: Orbital period (days)
    - eccentricity: Orbital eccentricity
    - omega: Argument of periastron (radians)
    - phi0: Phase offset (radians)
    - v0: Systemic velocity offset (m/s)
    - n_clusters: Number of observation clusters
    - obs_per_cluster: Number of observations per cluster
    - cluster_width: Width of each cluster (days)
    - seed: Random seed
    """
    # Calculate derived parameters
    total_time = period * time_period_ratio
    sigma = 5.0  # Fixed uncertainty
    K = sigma * K_sigma_ratio
    
    # Generate clustered observation times
    time_obs = generate_clustered_observations(
        total_time, n_clusters, obs_per_cluster, cluster_width, seed
    )
    
    # Generate RV data
    time_obs, rv_obs, rv_err, rv_true = generate_rv_data(
        time_obs, period, eccentricity, omega, phi0, K, v0, sigma, seed
    )
    
    # Generate smooth curve for plotting
    time_true = np.linspace(0, total_time, 500)
    rv_true_smooth = velocity(time_true, period, eccentricity, omega, phi0, K, v0)
    
    # Plot the data
    plot_rv_data(time_obs, rv_obs, rv_err, time_true, rv_true_smooth, 
                period, eccentricity, K, sigma)
    
    # Print summary
    print(f"Generated {len(time_obs)} observations over {total_time:.1f} days")
    print(f"Orbital parameters: P={period:.1f}d, e={eccentricity:.2f}, ω={omega:.2f}rad, φ₀={phi0:.2f}rad")
    print(f"RV parameters: K={K:.1f}m/s, v₀={v0:.1f}m/s, σ={sigma:.1f}m/s")
    print(f"Observation setup: {n_clusters} clusters, {obs_per_cluster} obs/cluster, {cluster_width:.1f}d width")


def create_interactive_widgets():
    """
    Create and display the interactive widgets for RV data visualization.
    """
    # Create widgets
    time_period_ratio_widget = widgets.FloatSlider(
        value=3.33,
        min=0.5,
        max=10.0,
        step=0.1,
        description='Time/Period:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    K_sigma_ratio_widget = widgets.FloatSlider(
        value=4.0,
        min=0.5,
        max=20.0,
        step=0.1,
        description='K/σ:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    period_widget = widgets.FloatText(
        value=3.0,
        description='Period (d):',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    eccentricity_widget = widgets.FloatSlider(
        value=0.5,
        min=0.0,
        max=0.99,
        step=0.01,
        description='Eccentricity:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    omega_widget = widgets.FloatSlider(
        value=1.0,
        min=0.0,
        max=2*np.pi,
        step=0.1,
        description='Omega (rad):',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    phi0_widget = widgets.FloatSlider(
        value=1.0,
        min=0.0,
        max=2*np.pi,
        step=0.1,
        description='Phi0 (rad):',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    v0_widget = widgets.FloatText(
        value=0.0,
        description='v0 (m/s):',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    n_clusters_widget = widgets.IntSlider(
        value=3,
        min=1,
        max=10,
        step=1,
        description='Clusters:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    obs_per_cluster_widget = widgets.IntSlider(
        value=5,
        min=1,
        max=20,
        step=1,
        description='Obs/cluster:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    cluster_width_widget = widgets.FloatSlider(
        value=0.5,
        min=0.1,
        max=2.0,
        step=0.1,
        description='Cluster width (d):',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    seed_widget = widgets.IntText(
        value=42,
        description='Seed:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    # Create output widget for the plot
    output_widget = widgets.Output()
    
    # Create a function that updates the plot
    def update_plot(*args, **kwargs):
        with output_widget:
            output_widget.clear_output(wait=True)
            # Get current values from all widgets
            interactive_rv_plot(
                time_period_ratio=time_period_ratio_widget.value,
                K_sigma_ratio=K_sigma_ratio_widget.value,
                period=period_widget.value,
                eccentricity=eccentricity_widget.value,
                omega=omega_widget.value,
                phi0=phi0_widget.value,
                v0=v0_widget.value,
                n_clusters=n_clusters_widget.value,
                obs_per_cluster=obs_per_cluster_widget.value,
                cluster_width=cluster_width_widget.value,
                seed=seed_widget.value
            )
    
    # Connect all widgets to the update function
    for widget in [time_period_ratio_widget, K_sigma_ratio_widget, period_widget, 
                   eccentricity_widget, omega_widget, phi0_widget, v0_widget,
                   n_clusters_widget, obs_per_cluster_widget, cluster_width_widget, seed_widget]:
        widget.observe(lambda change: update_plot(), names='value')
    
    # Create a layout for the widgets
    controls_box = widgets.VBox([
        widgets.HTML("<h3>Main Controls</h3>"),
        time_period_ratio_widget,
        K_sigma_ratio_widget,
        widgets.HTML("<h3>Orbital Parameters</h3>"),
        period_widget,
        eccentricity_widget,
        omega_widget,
        phi0_widget,
        v0_widget,
        widgets.HTML("<h3>Observation Parameters</h3>"),
        n_clusters_widget,
        obs_per_cluster_widget,
        cluster_width_widget,
        seed_widget
    ])
    
    # Create a horizontal layout with plot on left, controls on right
    main_layout = widgets.HBox([
        output_widget,
        controls_box
    ], layout=widgets.Layout(width='100%', height='600px'))
    
    # Display the combined layout
    display(main_layout)
    
    # Initial plot
    update_plot()
    
    return controls_box


def create_matplotlib_interactive():
    """
    Create interactive matplotlib version with sliders positioned to the right of the plot.
    """
    # Set up the figure with a larger width to accommodate sliders on the right
    fig = plt.figure(figsize=(20, 10))
    
    # Main plot area - positioned on the left side
    ax_plot = plt.axes([0.1, 0.1, 0.6, 0.8])  # [left, bottom, width, height]
    
    # Slider areas - positioned on the right side
    # First column of sliders
    ax_time_period = plt.axes([0.75, 0.85, 0.2, 0.03])
    ax_k_sigma = plt.axes([0.75, 0.80, 0.2, 0.03])
    ax_period = plt.axes([0.75, 0.75, 0.2, 0.03])
    ax_eccentricity = plt.axes([0.75, 0.70, 0.2, 0.03])
    ax_omega = plt.axes([0.75, 0.65, 0.2, 0.03])
    ax_phi0 = plt.axes([0.75, 0.60, 0.2, 0.03])
    ax_v0 = plt.axes([0.75, 0.55, 0.2, 0.03])
    ax_reset = plt.axes([0.75, 0.50, 0.2, 0.03])
    
    # Second column of sliders
    ax_n_obs = plt.axes([0.75, 0.40, 0.2, 0.03])
    ax_checkbox = plt.axes([0.75, 0.35, 0.2, 0.03])
    ax_seed = plt.axes([0.75, 0.25, 0.2, 0.03])
    
    # Initial parameters
    params = {
        'time_period_ratio': 3.33,
        'K_sigma_ratio': 4.0,
        'period': 3.0,
        'eccentricity': 0.5,
        'omega': 1.0,
        'phi0': 1.0,
        'v0': 0.0,
        'n_obs': 15,
        'seed': 42
    }
    
    # Create sliders
    slider_time_period = Slider(ax_time_period, 'Time/Period', 0.5, 10.0, valinit=params['time_period_ratio'], valstep=0.1)
    slider_k_sigma = Slider(ax_k_sigma, 'K/σ', 0.5, 20.0, valinit=params['K_sigma_ratio'], valstep=0.1)
    slider_period = Slider(ax_period, 'Period (d)', 0.1, 20.0, valinit=params['period'], valstep=0.1)
    slider_eccentricity = Slider(ax_eccentricity, 'Eccentricity', 0.0, 0.99, valinit=params['eccentricity'], valstep=0.01)
    slider_omega = Slider(ax_omega, 'Omega (rad)', 0.0, 2*np.pi, valinit=params['omega'], valstep=0.1)
    slider_phi0 = Slider(ax_phi0, 'Phi0 (rad)', 0.0, 2*np.pi, valinit=params['phi0'], valstep=0.1)
    slider_v0 = Slider(ax_v0, 'v0 (m/s)', -50.0, 50.0, valinit=params['v0'], valstep=1.0)
    slider_n_obs = Slider(ax_n_obs, 'N obs', 3, 50, valinit=params['n_obs'], valstep=1)
    checkbox_show_true = CheckButtons(ax_checkbox, ['Show True Curve'], [True])
    slider_seed = Slider(ax_seed, 'Seed', 1, 1000, valinit=params['seed'], valstep=1)
    
    # Reset button
    button_reset = Button(ax_reset, 'Reset')
    
    # Initial plot
    def update_plot(*args):
        # Get current slider values
        time_period_ratio = slider_time_period.val
        K_sigma_ratio = slider_k_sigma.val
        period = slider_period.val
        eccentricity = slider_eccentricity.val
        omega = slider_omega.val
        phi0 = slider_phi0.val
        v0 = slider_v0.val
        n_obs = int(slider_n_obs.val)
        seed = int(slider_seed.val)
        
        # Calculate derived parameters
        total_time = period * time_period_ratio
        sigma = 5.0  # Fixed uncertainty
        K = sigma * K_sigma_ratio
        
        # Generate Poisson-distributed observation times
        time_obs = generate_poisson_observations(n_obs, total_time, seed)
        
        # Generate RV data
        time_obs, rv_obs, rv_err, rv_true = generate_rv_data(
            time_obs, period, eccentricity, omega, phi0, K, v0, sigma, seed
        )
        
        # Generate smooth curve for plotting
        time_true = np.linspace(0, total_time, 500)
        rv_true_smooth = velocity(time_true, period, eccentricity, omega, phi0, K, v0)
        
        # Clear and replot
        ax_plot.clear()
        
        # Check if true curve should be shown
        show_true_curve = checkbox_show_true.get_status()[0]
        
        # Plot true RV curve in blue (if enabled)
        if show_true_curve:
            ax_plot.plot(time_true, rv_true_smooth, 'b-', linewidth=2, label='True RV Curve')
        
        # Plot observed velocities as black scatter plot with error bars
        ax_plot.errorbar(time_obs, rv_obs, yerr=rv_err, fmt='ko', 
                        capsize=3, capthick=1, markersize=4, label='Observed Data')
        
        ax_plot.set_xlabel('Time (days)')
        ax_plot.set_ylabel('Radial Velocity (m/s)')
        ax_plot.set_title(f'RV Data: P={period:.1f}d, e={eccentricity:.2f}, '
                         f'K={K:.1f}m/s, σ={sigma:.1f}m/s')
        ax_plot.legend()
        ax_plot.grid(True, alpha=0.3)
        
        plt.draw()
    
    def reset_sliders(event):
        slider_time_period.reset()
        slider_k_sigma.reset()
        slider_period.reset()
        slider_eccentricity.reset()
        slider_omega.reset()
        slider_phi0.reset()
        slider_v0.reset()
        slider_n_obs.reset()
        checkbox_show_true.set_active(0)  # Reset checkbox to checked (True)
        slider_seed.reset()
        update_plot()
    
    # Connect sliders to update function
    slider_time_period.on_changed(update_plot)
    slider_k_sigma.on_changed(update_plot)
    slider_period.on_changed(update_plot)
    slider_eccentricity.on_changed(update_plot)
    slider_omega.on_changed(update_plot)
    slider_phi0.on_changed(update_plot)
    slider_v0.on_changed(update_plot)
    slider_n_obs.on_changed(update_plot)
    checkbox_show_true.on_clicked(update_plot)
    slider_seed.on_changed(update_plot)
    
    # Connect reset button
    button_reset.on_clicked(reset_sliders)
    
    # Initial plot
    update_plot()
    
    plt.show()
    
    return fig


def main():
    """
    Main function to launch the interactive RV data visualization.
    """
    print("Interactive Radial Velocity Data Generator")
    print("=" * 50)
    print("Use the sliders and text boxes below to adjust parameters:")
    print("- Time/Period: Controls how many orbits are observed")
    print("- K/σ: Controls signal-to-noise ratio")
    print("- Other parameters: Direct input for orbital and observation settings")
    print()
    
    # Create and display the interactive widgets
    interactive_widget = create_interactive_widgets()
    
    return interactive_widget






def generate_gallery(seed=42, show_true_curve=True):
    """
    Generate a gallery of 27 RV plots varying three dimensionless numbers:
    N (observations), K/σ (signal-to-noise), T/P (time/period).
    Each parameter varies over [3, 10, 30].
    
    Parameters:
    - seed: Random seed for reproducibility
    - show_true_curve: Whether to plot the true RV curve
    """
    # Dimensionless parameter values
    N_values = [3, 10, 30]  # Number of observations
    K_sigma_values = [3, 10, 30]  # K/σ ratio
    T_P_values = [3, 10, 30]  # T/P ratio
    
    # Fixed parameters
    period = 1.0  # days (arbitrary, since we use T/P ratio)
    sigma = 1.0   # m/s (arbitrary, since we use K/σ ratio)
    v0 = 0.0      # m/s
    omega = 0.0   # radians (simplified)
    
    # Create figure with 3x3x3 = 27 subplots
    fig, axes = plt.subplots(9, 3, figsize=(10, 20))
    # plt.subplots_adjust(wspace=0.2, hspace=3.0) 

    axes = axes.flatten()
    
    plot_idx = 0
    
    for N in N_values:
        for K_sigma in K_sigma_values:
            for T_P in T_P_values:
                # Calculate derived parameters
                K = K_sigma * sigma
                T = T_P * period
                
                # Random parameters for this plot
                plot_seed = seed + plot_idx  # Different seed for each plot
                eccentricity = sample_kipping_eccentricity(plot_seed)
                np.random.seed(plot_seed)
                phi0 = np.random.uniform(0, 2*np.pi)  # Random phase
                
                # Generate Poisson-distributed observation times
                time_obs = generate_poisson_observations(N, T, plot_seed)
                
                # Generate RV data
                time_obs, rv_obs, rv_err, rv_true = generate_rv_data(
                    time_obs, period, eccentricity, omega, phi0, K, v0, sigma, plot_seed
                )
                
                # Generate smooth curve for plotting
                time_true = np.linspace(0, T, 500)
                rv_true_smooth = velocity(time_true, period, eccentricity, omega, phi0, K, v0)
                
                # Plot
                ax = axes[plot_idx]
                if show_true_curve:
                    ax.plot(time_true, rv_true_smooth, 'b-', linewidth=1.5, alpha=0.7, label='True RV')
                ax.errorbar(time_obs, rv_obs, yerr=rv_err, fmt='ko', 
                           capsize=2, capthick=1, markersize=3, alpha=0.8, label='Observed')
                ax.set_title(f'N={N}, K/σ={K_sigma}, T/P={T_P}\ne={eccentricity:.2f}, φ₀={phi0:.2f}', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
    
    plt.tight_layout()
    # plt.subplots_adjust()
    plt.savefig(f'phase_diagram_seed{seed}.png')
    plt.show()
    
    print(f"Generated gallery with {plot_idx} plots")
    print("Parameters varied:")
    print(f"- N (observations): {N_values}")
    print(f"- K/σ (signal-to-noise): {K_sigma_values}")
    print(f"- T/P (time/period): {T_P_values}")


def main_static():
    """
    Static version for generating and plotting RV data with fixed parameters.
    """
    # =============================================================================
    # TUNABLE PARAMETERS - Modify these to change the data generation
    # =============================================================================
    
    # Orbital parameters
    period = 3.0              # Orbital period (days)
    eccentricity = 0.5        # Orbital eccentricity (0 <= ecc < 1)
    omega = 1.0               # Argument of periastron (radians)
    phi0 = 1.0               # Phase offset (radians)
    K = 20.0                  # Semi-amplitude of velocity (m/s)
    v0 = 0.0                  # Systemic velocity offset (m/s)
    
    # Observation parameters
    total_time = 10.0         # Total observation time span (days)
    n_clusters = 3            # Number of observation clusters
    obs_per_cluster = 5       # Number of observations per cluster
    cluster_width = 0.5       # Width of each cluster (days)
    sigma = 5.0              # RV uncertainty (m/s)
    seed = 42                 # Random seed
    
    # =============================================================================
    # DATA GENERATION
    # =============================================================================
    
    # Generate clustered observation times
    time_obs = generate_clustered_observations(
        total_time, n_clusters, obs_per_cluster, cluster_width, seed
    )
    
    # Generate RV data
    time_obs, rv_obs, rv_err, rv_true = generate_rv_data(
        time_obs, period, eccentricity, omega, phi0, K, v0, sigma, seed
    )
    
    # Generate fine time array for smooth true curve
    time_true = np.linspace(0, total_time, 500)
    rv_true_smooth = velocity(time_true, period, eccentricity, omega, phi0, K, v0)
    
    # =============================================================================
    # PLOTTING
    # =============================================================================
    
    plot_rv_data(time_obs, rv_obs, rv_err, time_true, rv_true_smooth, 
                period, eccentricity, K, sigma)
    
    # Print summary
    print(f"Generated {len(time_obs)} observations over {total_time} days")
    print(f"Orbital parameters: P={period} d, e={eccentricity}, ω={omega} rad, φ₀={phi0} rad")
    print(f"RV parameters: K={K} m/s, v₀={v0} m/s, σ={sigma} m/s")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate RV data gallery with varying dimensionless parameters')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive matplotlib version with sliders (works outside Jupyter)')
    parser.add_argument('--widgets', action='store_true',
                       help='Run interactive widget version (requires Jupyter notebook)')
    parser.add_argument('--static', action='store_true',
                       help='Run static single plot version instead of gallery')
    parser.add_argument('--no-true-curve', action='store_true',
                       help='Hide the true RV curve in the plots (show only observed data)')
    
    args = parser.parse_args()
    
    if args.interactive:
        # Run interactive matplotlib version with sliders
        create_matplotlib_interactive()
    elif args.widgets:
        # Run interactive widget version (requires Jupyter notebook)
        main()
    elif args.static:
        # Run static single plot version
        main_static()
    else:
        # Generate the gallery of 27 plots as requested by advisor
        show_true = not args.no_true_curve
        print(f"Generating gallery with seed={args.seed}, show_true_curve={show_true}")
        generate_gallery(seed=args.seed, show_true_curve=show_true)