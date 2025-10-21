import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from velocity import velocity

# Enable double precision for JAX
jax.config.update("jax_enable_x64", True)


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

    # Print summary accounting for all arguments
    print(
        f"Generated {len(time_obs)} observations over {total_time:.1f} days\n"
        f"Arguments:\n"
        f"  time_period_ratio = {time_period_ratio}\n"
        f"  K_sigma_ratio = {K_sigma_ratio}\n"
        f"  period = {period:.1f} d\n"
        f"  eccentricity = {eccentricity:.2f}\n"
        f"  omega = {omega:.2f} rad\n"
        f"  phi0 = {phi0:.2f} rad\n"
        f"  v0 = {v0:.1f} m/s\n"
        f"  n_clusters = {n_clusters}\n"
        f"  obs_per_cluster = {obs_per_cluster}\n"
        f"  cluster_width = {cluster_width:.1f} d\n"
        f"  seed = {seed}\n"
        f"\nDerived parameters:\n"
        f"  total_time = {total_time:.1f} d\n"
        f"  K = {K:.1f} m/s (K = sigma * K_sigma_ratio)\n"
        f"  sigma = {sigma:.1f} m/s"
    )


def create_interactive_widgets():
    """
    Create and display the interactive widgets for RV data visualization.
    """
    # Create widgets
    time_period_ratio_widget = widgets.FloatSlider(
        value=1.0,
        min=0.01,
        max=5.0,
        step=0.05,
        description='Time/Period:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    K_sigma_ratio_widget = widgets.FloatSlider(
        value=4.0,
        min=0.01,
        max=8.0,
        step=0.1,
        description='K/σ:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    period_widget = widgets.FloatText(
        value=100.0,
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
        value=10.0,
        min=0.1,
        max=50.0,
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
    main()