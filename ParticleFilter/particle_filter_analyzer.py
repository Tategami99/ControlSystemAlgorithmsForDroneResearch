import numpy as np
import matplotlib.pyplot as plt
import os

class ParticleFilterAnalyzer:
    """Collects and analyzes particle filter data during simulation"""
    def __init__(self, n_drones, n1, n_particles, steps, dt=0.1):
        self.n_drones = n_drones
        self.n1 = n1
        self.n_particles = n_particles
        self.steps = steps
        self.dt = dt
        
        # Storage for analysis
        self.time = np.arange(steps) * dt
        
        # True states
        self.true_positions = np.zeros((steps, n_drones, 2))
        self.true_velocities = np.zeros((steps, n_drones, 2))
        
        # Estimated states
        self.est_positions = np.zeros((steps, n_drones, 2))
        self.est_velocities = np.zeros((steps, n_drones, 2))
        
        # Particle cloud statistics
        self.particle_means = np.zeros((steps, n_drones, 2))
        self.particle_stds = np.zeros((steps, n_drones, 2))
        self.particle_weights = []  # Store weight distributions
        self.particle_positions = []  # Store all particles for each step
        self.effective_particles = np.zeros(steps)
        
        # Errors
        self.position_errors = np.zeros((steps, n_drones))
        self.velocity_errors = np.zeros((steps, n_drones))
        self.rmse_position = np.zeros(steps)
        self.rmse_velocity = np.zeros(steps)
        
        # Measurement info
        self.measurements = []
        self.measurement_predictions = []  # Store predicted measurements
        
        # Diversity metrics
        self.particle_diversity = np.zeros(steps)
        
        # Create results directory
        self.results_dir = './ParticleFilter/results/'
        os.makedirs(self.results_dir, exist_ok=True)
        
    def update(self, step, x_true, x_est, particles, weights, measurement, measurement_pred=None):
        """Update all analysis data for current step"""
        n1 = self.n1
        
        # x_true and x_est are column vectors (n, 1), so we need to flatten them or index properly
        x_true = x_true.flatten()  # Convert to 1D array for easier indexing
        x_est = x_est.flatten()    # Convert to 1D array for easier indexing
        
        # Store true and estimated positions/velocities for each drone
        for i in range(self.n_drones):
            px_idx = n1 + i*2
            py_idx = n1 + i*2 + 1
            
            # Make sure indices are within bounds
            if px_idx < len(x_true) and py_idx < len(x_true):
                self.true_positions[step, i, 0] = x_true[px_idx]
                self.true_positions[step, i, 1] = x_true[py_idx]
                
                self.est_positions[step, i, 0] = x_est[px_idx]
                self.est_positions[step, i, 1] = x_est[py_idx]
            
            # If we have velocity states (assuming positions at even indices, velocities at odd)
            if i*2 < n1:  # Check if we have velocity states in earth portion
                self.true_velocities[step, i, 0] = x_true[i*2] if i*2 < n1 else 0
                self.true_velocities[step, i, 1] = x_true[i*2 + 1] if i*2 + 1 < n1 else 0
                
                self.est_velocities[step, i, 0] = x_est[i*2] if i*2 < n1 else 0
                self.est_velocities[step, i, 1] = x_est[i*2 + 1] if i*2 + 1 < n1 else 0
        
        # Particle statistics
        for i in range(self.n_drones):
            px_idx = n1 + i*2
            py_idx = n1 + i*2 + 1
            
            if px_idx < particles.shape[1] and py_idx < particles.shape[1]:
                self.particle_means[step, i, 0] = np.average(particles[:, px_idx], weights=weights)
                self.particle_means[step, i, 1] = np.average(particles[:, py_idx], weights=weights)
                
                # Weighted standard deviation
                self.particle_stds[step, i, 0] = np.sqrt(np.average((particles[:, px_idx] - self.particle_means[step, i, 0])**2, weights=weights))
                self.particle_stds[step, i, 1] = np.sqrt(np.average((particles[:, py_idx] - self.particle_means[step, i, 1])**2, weights=weights))
        
        # Store particles and weights for later analysis
        self.particle_positions.append(particles.copy())
        self.particle_weights.append(weights.copy())
        
        # Effective sample size (measure of particle diversity)
        self.effective_particles[step] = 1.0 / np.sum(weights**2)
        
        # Calculate errors
        pos_error = np.linalg.norm(self.true_positions[step] - self.est_positions[step], axis=1)
        
        self.position_errors[step] = pos_error
        self.rmse_position[step] = np.sqrt(np.mean(pos_error**2))
        
        # Velocity errors if we have velocity data
        if np.any(self.true_velocities[step]):
            vel_error = np.linalg.norm(self.true_velocities[step] - self.est_velocities[step], axis=1)
            self.velocity_errors[step] = vel_error
            self.rmse_velocity[step] = np.sqrt(np.mean(vel_error**2))
        
        # Store measurement
        self.measurements.append(measurement.flatten())
        if measurement_pred is not None:
            self.measurement_predictions.append(measurement_pred.flatten())
        
        # Particle diversity (average pairwise distance - sample for efficiency)
        if self.n_particles <= 500:
            sample_size = min(50, self.n_particles)
            indices = np.random.choice(self.n_particles, sample_size, replace=False)
            sampled_particles = particles[indices, n1:]
            
            if len(sampled_particles) > 1:
                distances = []
                for ii in range(sample_size):
                    for jj in range(ii+1, sample_size):
                        dist = np.linalg.norm(sampled_particles[ii] - sampled_particles[jj])
                        distances.append(dist)
                self.particle_diversity[step] = np.mean(distances) if distances else 0
    
    def plot_trajectories(self):
        """Plot 1: True vs Estimated trajectories"""
        fig, ax = plt.subplots(figsize=(10, 8))
        for i in range(self.n_drones):
            ax.plot(self.true_positions[:, i, 0], self.true_positions[:, i, 1], 
                    'r-', linewidth=2, alpha=0.7, label=f'Drone {i+1} True' if i == 0 else '')
            ax.plot(self.est_positions[:, i, 0], self.est_positions[:, i, 1], 
                    'b--', linewidth=2, alpha=0.7, label=f'Drone {i+1} Est' if i == 0 else '')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('True vs Estimated Trajectories')
        ax.legend(['True', 'Estimated'])
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        plt.savefig(f'{self.results_dir}/01_trajectories.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  - Saved trajectories plot")
    
    def plot_rmse(self):
        """Plot 2: RMSE over time"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.time, self.rmse_position, 'r-', linewidth=2, label='Position RMSE')
        if np.any(self.rmse_velocity):
            ax.plot(self.time, self.rmse_velocity, 'b-', linewidth=2, label='Velocity RMSE')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('RMSE')
        ax.set_title('Root Mean Square Error Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(f'{self.results_dir}/02_rmse.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  - Saved RMSE plot")
    
    def plot_effective_particles(self):
        """Plot 3: Effective sample size"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.time, self.effective_particles, 'g-', linewidth=2)
        ax.axhline(y=self.n_particles, color='k', linestyle='--', alpha=0.5, label='Total particles')
        ax.axhline(y=self.n_particles/2, color='r', linestyle='--', alpha=0.5, label='Resample threshold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Effective particles')
        ax.set_title('Particle Diversity (Effective Sample Size)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(f'{self.results_dir}/03_effective_particles.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  - Saved effective particles plot")
    
    def plot_per_drone_errors(self):
        """Plot 4: Position errors for each drone"""
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(self.n_drones):
            ax.plot(self.time, self.position_errors[:, i], linewidth=1.5, label=f'Drone {i+1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position Error')
        ax.set_title('Position Error per Drone Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(f'{self.results_dir}/04_per_drone_errors.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  - Saved per-drone errors plot")
    
    def plot_particle_uncertainty(self):
        """Plot 5: Particle cloud spread (standard deviation)"""
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(self.n_drones):
            ax.plot(self.time, self.particle_stds[:, i, 0], '--', linewidth=1.5, label=f'Drone {i+1} X')
            ax.plot(self.time, self.particle_stds[:, i, 1], '-', linewidth=1.5, label=f'Drone {i+1} Y')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Std Deviation')
        ax.set_title('Particle Cloud Uncertainty (Standard Deviation)')
        ax.legend(loc='upper right', ncol=2, fontsize='small')
        ax.grid(True, alpha=0.3)
        plt.savefig(f'{self.results_dir}/05_particle_uncertainty.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  - Saved particle uncertainty plot")
    
    def plot_weight_entropy(self):
        """Plot 6: Weight distribution entropy"""
        fig, ax = plt.subplots(figsize=(10, 6))
        weight_array = np.array(self.particle_weights)
        weight_entropy = -np.sum(weight_array * np.log(weight_array + 1e-10), axis=1)
        ax.plot(self.time[:len(weight_entropy)], weight_entropy, 'purple', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Weight Entropy')
        ax.set_title('Particle Weight Diversity (Entropy)')
        ax.grid(True, alpha=0.3)
        plt.savefig(f'{self.results_dir}/06_weight_entropy.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  - Saved weight entropy plot")
    
    def plot_error_histogram(self):
        """Plot 7: Error distribution histogram"""
        fig, ax = plt.subplots(figsize=(10, 6))
        all_errors = self.position_errors.flatten()
        ax.hist(all_errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(np.mean(all_errors), color='r', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(all_errors):.3f}')
        ax.axvline(np.median(all_errors), color='g', linestyle='--', linewidth=2, 
                  label=f'Median: {np.median(all_errors):.3f}')
        ax.set_xlabel('Position Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Overall Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(f'{self.results_dir}/07_error_histogram.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  - Saved error histogram plot")
    
    def plot_cumulative_error(self):
        """Plot 8: Cumulative error over time"""
        fig, ax = plt.subplots(figsize=(10, 6))
        cumulative_error = np.cumsum(self.rmse_position) * self.dt
        ax.plot(self.time, cumulative_error, 'brown', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Cumulative Error')
        ax.set_title('Cumulative Position Error Over Time')
        ax.grid(True, alpha=0.3)
        plt.savefig(f'{self.results_dir}/08_cumulative_error.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  - Saved cumulative error plot")
    
    def plot_particle_diversity(self):
        """Plot 9: Particle cloud diversity"""
        fig, ax = plt.subplots(figsize=(10, 6))
        if np.any(self.particle_diversity):
            ax.plot(self.time, self.particle_diversity, 'orange', linewidth=2)
            ax.set_ylabel('Avg Pairwise Distance')
        else:
            ax.text(0.5, 0.5, 'No diversity data available', 
                   horizontalalignment='center', transform=ax.transAxes)
        ax.set_xlabel('Time (s)')
        ax.set_title('Particle Cloud Diversity Over Time')
        ax.grid(True, alpha=0.3)
        plt.savefig(f'{self.results_dir}/09_particle_diversity.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  - Saved particle diversity plot")
    
    def plot_error_boxplot(self):
        """Plot 10: Box plot of errors per drone"""
        fig, ax = plt.subplots(figsize=(12, 6))
        bp = ax.boxplot([self.position_errors[:, i] for i in range(self.n_drones)], 
                        patch_artist=True, labels=[f'Drone {i+1}' for i in range(self.n_drones)])
        
        # Color the boxes
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_drones))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Position Error')
        ax.set_title('Error Distribution per Drone (All Time Steps)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.savefig(f'{self.results_dir}/10_error_boxplot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  - Saved error boxplot")
    
    def plot_convergence(self):
        """Plot 11: Convergence of estimate to true value for first drone"""
        if self.n_drones > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # X coordinate convergence
            ax1.plot(self.time, self.true_positions[:, 0, 0], 'r-', linewidth=2, label='True')
            ax1.plot(self.time, self.est_positions[:, 0, 0], 'b--', linewidth=2, label='Estimated')
            ax1.fill_between(self.time, 
                             self.est_positions[:, 0, 0] - 2*self.particle_stds[:, 0, 0],
                             self.est_positions[:, 0, 0] + 2*self.particle_stds[:, 0, 0],
                             alpha=0.2, color='blue', label='2σ uncertainty')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('X Position')
            ax1.set_title('Drone 1 X Coordinate Convergence')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Y coordinate convergence
            ax2.plot(self.time, self.true_positions[:, 0, 1], 'r-', linewidth=2, label='True')
            ax2.plot(self.time, self.est_positions[:, 0, 1], 'b--', linewidth=2, label='Estimated')
            ax2.fill_between(self.time, 
                             self.est_positions[:, 0, 1] - 2*self.particle_stds[:, 0, 1],
                             self.est_positions[:, 0, 1] + 2*self.particle_stds[:, 0, 1],
                             alpha=0.2, color='blue', label='2σ uncertainty')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Y Position')
            ax2.set_title('Drone 1 Y Coordinate Convergence')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/11_convergence_drone1.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  - Saved convergence plot")
    
    def plot_measurement_innovation(self):
        """Plot 12: Measurement innovation if predictions were stored"""
        if len(self.measurement_predictions) == 0:
            print("  - Skipping measurement innovation plot (no predictions stored)")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        measurements = np.array(self.measurements)
        predictions = np.array(self.measurement_predictions)
        
        # Plot first few measurement dimensions
        n_plot = min(3, measurements.shape[1])
        for i in range(n_plot):
            innovation = measurements[:len(predictions), i] - predictions[:, i]
            ax.plot(self.time[:len(innovation)], innovation, linewidth=1.5, label=f'Dim {i+1}')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Innovation')
        ax.set_title('Measurement Innovation (Measurement - Prediction)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(f'{self.results_dir}/12_measurement_innovation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  - Saved measurement innovation plot")
    
    def generate_all_plots(self):
        """Generate all analysis plots"""
        print("\nGenerating analysis plots...")
        self.plot_trajectories()
        self.plot_rmse()
        self.plot_effective_particles()
        self.plot_per_drone_errors()
        self.plot_particle_uncertainty()
        self.plot_weight_entropy()
        self.plot_error_histogram()
        self.plot_cumulative_error()
        self.plot_particle_diversity()
        self.plot_error_boxplot()
        self.plot_convergence()
        self.plot_measurement_innovation()
        print(f"\nAll plots saved to {self.results_dir}")
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("PARTICLE FILTER PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Final RMSE Position: {self.rmse_position[-1]:.4f}")
        print(f"Average RMSE Position: {np.mean(self.rmse_position):.4f}")
        print(f"Max RMSE Position: {np.max(self.rmse_position):.4f}")
        print(f"Min RMSE Position: {np.min(self.rmse_position):.4f}")
        print(f"\nAverage Effective Particles: {np.mean(self.effective_particles):.1f}/{self.n_particles}")
        print(f"Min Effective Particles: {np.min(self.effective_particles):.1f}")
        print(f"\nPer-Drone Average Error:")
        for i in range(self.n_drones):
            print(f"  Drone {i+1}: {np.mean(self.position_errors[:, i]):.4f}")
        
        # Final position error
        print(f"\nFinal Position Error per Drone:")
        for i in range(self.n_drones):
            print(f"  Drone {i+1}: {self.position_errors[-1, i]:.4f}")
        print("="*60)