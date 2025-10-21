'''
Visualize simulated radial velocity data with a given set of parameters.
'''
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
import matplotlib.gridspec as gridspec

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from velocity import velocity

# Enable double precision for JAX
jax.config.update("jax_enable_x64", True)


def generate_clustered_observations(total_time, n_clusters, obs_per_cluster, cluster_width):
    """
    Generate observation times in clusters (similar to real observing runs).
    
    Parameters:
    - total_time: Total time span for observations (days)
    - n_clusters: Number of observation clusters
    - obs_per_cluster: Number of observations per cluster
    - cluster_width: Width of each cluster (days)
    
    Returns:
    - time_obs: Array of observation times
    """
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


def generate_rv_data(time_obs, period, eccentricity, omega, phi0, K, v0, sigma):
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
    
    Returns:
    - time_obs: Observation times
    - rv_obs: Observed radial velocities with noise
    - rv_err: RV uncertainties
    - rv_true: True radial velocities (no noise)
    """
    # Generate true RV curve
    rv_true = velocity(time_obs, period, eccentricity, omega, phi0, K, v0)
    
    # Add Gaussian noise
    rv_obs = rv_true + np.random.normal(0, sigma, len(time_obs))
    rv_err = np.full_like(rv_obs, sigma)
    
    return time_obs, rv_obs, rv_err, rv_true


class InteractiveRVPlot:
    """
    Interactive GUI class for radial velocity data visualization.
    """
    
    def __init__(self):
        # Default parameter values
        self.period = 100.0
        self.eccentricity = 0.5
        self.omega = 1.0
        self.phi0 = 1.0
        self.v0 = 0.0
        self.n_clusters = 5
        self.obs_per_cluster = 5
        self.cluster_width = 10.0
        self.seed = 42
        
        # Ratio parameters (controlled by sliders)
        self.time_period_ratio = 1.0  # total_time / period
        self.K_sigma_ratio = 4.0       # K / sigma
        
        # Derived parameters
        self.total_time = self.period * self.time_period_ratio
        self.sigma = 5.0
        self.K = self.sigma * self.K_sigma_ratio
        
        # Set random seed
        np.random.seed(self.seed)
        
        # Create figure and layout
        self.fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 4, height_ratios=[3, 1, 1], width_ratios=[3, 1, 1, 1])
        
        # Main plot area
        self.ax = self.fig.add_subplot(gs[0, :2])
        
        # Sliders area
        self.ax_sliders = self.fig.add_subplot(gs[1, :2])
        self.ax_sliders.axis('off')
        
        # Text boxes area
        self.ax_text = self.fig.add_subplot(gs[0, 2:])
        self.ax_text.axis('off')
        
        # Initialize plot
        self.update_plot()
        self.setup_widgets()
        
    def setup_widgets(self):
        """Set up all the interactive widgets."""
        
        # Sliders
        slider_height = 0.02
        slider_width = 0.3
        
        # Time/Period ratio slider
        ax_time_ratio = plt.axes([0.1, 0.25, slider_width, slider_height])
        self.slider_time_ratio = Slider(ax_time_ratio, 'Time/Period', 0.01, 5.0, 
                                       valinit=self.time_period_ratio, valfmt='%.2f')
        self.slider_time_ratio.on_changed(self.update_time_ratio)
        
        # K/Sigma ratio slider
        ax_K_ratio = plt.axes([0.1, 0.20, slider_width, slider_height])
        self.slider_K_ratio = Slider(ax_K_ratio, 'K/σ', 0.5, 20.0, 
                                    valinit=self.K_sigma_ratio, valfmt='%.2f')
        self.slider_K_ratio.on_changed(self.update_K_ratio)
        
        # Text boxes for other parameters
        text_y_start = 0.85
        text_x_start = 0.7  # Moved slightly to the right
        text_height = 0.05
        text_width = 0.15
        
        # Period
        ax_period = plt.axes([text_x_start, text_y_start, text_width, text_height])
        self.text_period = TextBox(ax_period, 'Period (d): ', initial=str(self.period))
        self.text_period.on_submit(self.update_period)
        
        # Eccentricity
        ax_ecc = plt.axes([text_x_start, text_y_start - 0.08, text_width, text_height])
        self.text_ecc = TextBox(ax_ecc, 'Eccentricity: ', initial=str(self.eccentricity))
        self.text_ecc.on_submit(self.update_eccentricity)
        
        # Omega
        ax_omega = plt.axes([text_x_start, text_y_start - 0.16, text_width, text_height])
        self.text_omega = TextBox(ax_omega, 'Omega (rad): ', initial=str(self.omega))
        self.text_omega.on_submit(self.update_omega)
        
        # Phi0
        ax_phi0 = plt.axes([text_x_start, text_y_start - 0.24, text_width, text_height])
        self.text_phi0 = TextBox(ax_phi0, 'Phi0 (rad): ', initial=str(self.phi0))
        self.text_phi0.on_submit(self.update_phi0)
        
        # v0
        ax_v0 = plt.axes([text_x_start, text_y_start - 0.32, text_width, text_height])
        self.text_v0 = TextBox(ax_v0, 'v0 (m/s): ', initial=str(self.v0))
        self.text_v0.on_submit(self.update_v0)
        
        # Number of clusters
        ax_clusters = plt.axes([text_x_start, text_y_start - 0.40, text_width, text_height])
        self.text_clusters = TextBox(ax_clusters, 'Clusters: ', initial=str(self.n_clusters))
        self.text_clusters.on_submit(self.update_clusters)
        
        # Observations per cluster
        ax_obs_per = plt.axes([text_x_start, text_y_start - 0.48, text_width, text_height])
        self.text_obs_per = TextBox(ax_obs_per, 'Obs/cluster: ', initial=str(self.obs_per_cluster))
        self.text_obs_per.on_submit(self.update_obs_per_cluster)
        
        # Cluster width
        ax_cluster_width = plt.axes([text_x_start, text_y_start - 0.56, text_width, text_height])
        self.text_cluster_width = TextBox(ax_cluster_width, 'Cluster width (d): ', initial=str(self.cluster_width))
        self.text_cluster_width.on_submit(self.update_cluster_width)
        
        # Random seed
        ax_seed = plt.axes([text_x_start, text_y_start - 0.64, text_width, text_height])
        self.text_seed = TextBox(ax_seed, 'Seed: ', initial=str(self.seed))
        self.text_seed.on_submit(self.update_seed)
        
        # Update button
        ax_update = plt.axes([text_x_start, text_y_start - 0.72, text_width, text_height])
        self.button_update = Button(ax_update, 'Update')
        self.button_update.on_clicked(self.update_plot)
        
    def update_time_ratio(self, val):
        """Update time/period ratio and regenerate data."""
        self.time_period_ratio = val
        self.total_time = self.period * self.time_period_ratio
        self.update_plot()
        
    def update_K_ratio(self, val):
        """Update K/sigma ratio and regenerate data."""
        self.K_sigma_ratio = val
        self.K = self.sigma * self.K_sigma_ratio
        self.update_plot()
        
    def update_period(self, text):
        """Update period and regenerate data."""
        try:
            self.period = float(text)
            self.total_time = self.period * self.time_period_ratio
            self.update_plot()
        except ValueError:
            pass
            
    def update_eccentricity(self, text):
        """Update eccentricity and regenerate data."""
        try:
            val = float(text)
            if 0 <= val < 1:
                self.eccentricity = val
                self.update_plot()
        except ValueError:
            pass
            
    def update_omega(self, text):
        """Update omega and regenerate data."""
        try:
            self.omega = float(text)
            self.update_plot()
        except ValueError:
            pass
            
    def update_phi0(self, text):
        """Update phi0 and regenerate data."""
        try:
            self.phi0 = float(text)
            self.update_plot()
        except ValueError:
            pass
            
    def update_v0(self, text):
        """Update v0 and regenerate data."""
        try:
            self.v0 = float(text)
            self.update_plot()
        except ValueError:
            pass
            
    def update_clusters(self, text):
        """Update number of clusters and regenerate data."""
        try:
            val = int(text)
            if val > 0:
                self.n_clusters = val
                self.update_plot()
        except ValueError:
            pass
            
    def update_obs_per_cluster(self, text):
        """Update observations per cluster and regenerate data."""
        try:
            val = int(text)
            if val > 0:
                self.obs_per_cluster = val
                self.update_plot()
        except ValueError:
            pass
            
    def update_cluster_width(self, text):
        """Update cluster width and regenerate data."""
        try:
            val = float(text)
            if val > 0:
                self.cluster_width = val
                self.update_plot()
        except ValueError:
            pass
            
    def update_seed(self, text):
        """Update random seed and regenerate data."""
        try:
            self.seed = int(text)
            np.random.seed(self.seed)
            self.update_plot()
        except ValueError:
            pass
        
    def update_plot(self, event=None):
        """Update the plot with current parameters."""
        # Clear the plot
        self.ax.clear()
        
        # Generate new data
        time_obs = generate_clustered_observations(
            self.total_time, self.n_clusters, self.obs_per_cluster, self.cluster_width
        )
        
        time_obs, rv_obs, rv_err, rv_true = generate_rv_data(
            time_obs, self.period, self.eccentricity, self.omega, 
            self.phi0, self.K, self.v0, self.sigma
        )
        
        # Generate smooth curve for plotting
        time_true = np.linspace(0, self.total_time, 500)
        rv_true_smooth = velocity(time_true, self.period, self.eccentricity, 
                                self.omega, self.phi0, self.K, self.v0)
        
        # Plot true RV curve in blue
        self.ax.plot(time_true, rv_true_smooth, 'b-', linewidth=2, label='True RV Curve')
        
        # Plot observed velocities as black scatter plot with error bars
        self.ax.errorbar(time_obs, rv_obs, yerr=rv_err, fmt='ko', 
                        capsize=3, capthick=1, markersize=4, label='Observed Data')
        
        # Formatting
        self.ax.set_xlabel('Time (days)')
        self.ax.set_ylabel('Radial Velocity (m/s)')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        # Update the display
        self.fig.canvas.draw()
        
    def show(self):
        """Show the interactive plot."""
        plt.show()


def plot_rv_data(time_obs, rv_obs, rv_err, time_true, rv_true):
    """
    Plot radial velocity data (non-interactive version).
    
    Parameters:
    - time_obs: Observation times
    - rv_obs: Observed radial velocities
    - rv_err: RV uncertainties
    - time_true: Time array for true curve
    - rv_true: True radial velocity curve
    """
    plt.figure(figsize=(10, 6))
    
    # Plot true RV curve in blue
    plt.plot(time_true, rv_true, 'b-', linewidth=2, label='True RV Curve')
    
    # Plot observed velocities as black scatter plot with error bars
    plt.errorbar(time_obs, rv_obs, yerr=rv_err, fmt='ko', 
                capsize=3, capthick=1, markersize=4, label='Observed Data')
    
    plt.xlabel('Time (days)')
    plt.ylabel('Radial Velocity (m/s)')
    plt.title('Radial Velocity Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to launch the interactive RV data visualization GUI.
    """
    # Create and show the interactive plot
    interactive_plot = InteractiveRVPlot()
    interactive_plot.show()

if __name__ == "__main__":
    main()
