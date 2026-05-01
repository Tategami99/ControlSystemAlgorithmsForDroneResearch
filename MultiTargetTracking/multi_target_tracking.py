"""
multi_target_tracking.py
========================
Multi-target tracking with MOVING LANDMARKS.
Filters track the closest moving landmark to their current position.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import pickle as pkl
import os
from tqdm import tqdm
from scipy.linalg import solve_discrete_are

from state_dynamics import StateDynamics


class MovingLandmark:
    """A moving landmark that follows a trajectory."""
    
    def __init__(self, landmark_id, trajectory_type='circle', center=(0, 0), 
                 radius=5.0, speed=0.5, dt=0.05):
        self.id = landmark_id
        self.trajectory_type = trajectory_type
        self.center = np.array(center)
        self.radius = radius
        self.speed = speed
        self.dt = dt
        self.time = 0.0
        self.position = np.array(center)
        self.velocity = np.array([speed, speed]) * np.random.randn(2) * 0.5
        
    def update(self):
        self.time += self.dt
        
        if self.trajectory_type == 'circle':
            angle = self.speed * self.time
            self.position = self.center + self.radius * np.array([np.cos(angle), np.sin(angle)])
        elif self.trajectory_type == 'figure8':
            x = self.center[0] + self.radius * np.sin(self.speed * self.time)
            y = self.center[1] + self.radius * 0.5 * np.sin(2 * self.speed * self.time)
            self.position = np.array([x, y])
        elif self.trajectory_type == 'linear':
            x = self.center[0] + self.radius * np.sin(self.speed * self.time)
            y = self.center[1]
            self.position = np.array([x, y])
        elif self.trajectory_type == 'random_walk':
            self.velocity += np.random.randn(2) * 0.1
            self.velocity = np.clip(self.velocity, -self.speed, self.speed)
            self.position += self.velocity * self.dt
            self.position = np.clip(self.position, -12, 12)
        
        return self.position
    
    def get_position(self):
        return self.position


class RangeBearingSensorMoving:
    """Range-bearing sensor for measuring to landmarks."""
    
    def __init__(self, noise_scale=0.1):
        self.m = 2
        self.noise_scale = noise_scale
        range_std = noise_scale
        bearing_std = 0.05 * noise_scale
        self.V = np.diag([range_std**2, bearing_std**2])
    
    def measure(self, filter_position, landmark_position):
        dx = landmark_position[0] - filter_position[0]
        dy = landmark_position[1] - filter_position[1]
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        D = np.linalg.cholesky(self.V)
        noise = D @ np.random.randn(2, 1)
        return np.array([[r + noise[0, 0]], [theta + noise[1, 0]]])
    
    def measure_pred(self, filter_position, landmark_position):
        dx = landmark_position[0] - filter_position[0]
        dy = landmark_position[1] - filter_position[1]
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        return np.array([[r], [theta]])
    
    def g(self, filter_position, landmark_position):
        dx = landmark_position[0] - filter_position[0]
        dy = landmark_position[1] - filter_position[1]
        r = np.sqrt(dx**2 + dy**2)
        r_safe = max(r, 1e-6)
        dr_dx = -dx / r_safe
        dr_dy = -dy / r_safe
        dtheta_dx = dy / (r_safe**2)
        dtheta_dy = -dx / (r_safe**2)
        H = np.array([[dr_dx, 0, dr_dy, 0], [dtheta_dx, 0, dtheta_dy, 0]])
        return H


class LQGController:
    """LQG controller for moving toward a target."""
    
    def __init__(self, dt=0.05):
        self.dt = dt
        self.n = 4
        self.p = 2
        A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        B = np.array([[0, 0], [dt, 0], [0, 0], [0, dt]])
        Q = np.diag([10.0, 1.0, 10.0, 1.0])
        R = np.diag([1.0, 1.0]) * 0.1
        
        try:
            P = solve_discrete_are(A, B, Q, R)
            self.K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        except:
            self.K = np.array([[1.0, 0.5, 0, 0], [0, 0, 1.0, 0.5]])
    
    def compute_control(self, x_hat, goal_state):
        error = goal_state - x_hat
        u = self.K @ error
        return np.clip(u, -5.0, 5.0).reshape(-1, 1)


class ControlledAgent:
    """Agent that tracks the closest moving landmark."""
    
    def __init__(self, agent_id, dt=0.05, process_noise_scale=0.1, 
                 measurement_noise_scale=0.1, filter_type='ekf', n_particles=500):
        self.agent_id = agent_id
        self.filter_type = filter_type
        self.n_particles = n_particles
        self.dt = dt
        
        n1, n2, p = 0, 4, 2
        A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        B = np.array([[0, 0], [dt, 0], [0, 0], [0, dt]])
        W = np.eye(4) * process_noise_scale**2
        A_E = np.zeros((0, 0))
        
        self.F = StateDynamics(n1, n2, p, W, A_E, A, B)
        self.F.B = B
        self.sensor = RangeBearingSensorMoving(measurement_noise_scale)
        self.controller = LQGController(dt)
        
        self.reset_filter()
        self.target_landmark_id = None
        self.target_landmark_position = None
        self.true_positions = []
        self.estimated_positions = []
        self.controls = []
        self.tracked_landmark_history = []
    
    def reset_filter(self):
        n = 4
        x0 = self.F.get_x()
        if self.filter_type == 'pf':
            self.particles = x0.flatten() + np.random.randn(self.n_particles, 4) * 0.2
            self.weights = np.ones(self.n_particles) / self.n_particles
            self.x_hat = x0.copy()
            self.P_est = np.eye(4) * 0.1
        else:
            self.x_hat = x0.copy()
            self.P_est = np.eye(4) * 0.1
    
    def set_initial_state(self, x0):
        self.F.set_x(x0)
        self.reset_filter()
    
    def update_target_landmark(self, landmarks):
        if len(landmarks) == 0:
            return None
        current_pos = np.array([self.x_hat[0, 0], self.x_hat[2, 0]])
        landmark_positions = np.array([lm.get_position() for lm in landmarks])
        distances = np.linalg.norm(landmark_positions - current_pos, axis=1)
        closest_idx = np.argmin(distances)
        self.target_landmark_id = closest_idx
        self.target_landmark_position = landmark_positions[closest_idx]
        return self.target_landmark_position
    
    def step(self, landmarks):
        target_pos = self.update_target_landmark(landmarks)
        
        if target_pos is None:
            u = np.zeros((2, 1))
            self.F.set_u(u)
            self.F.forward()
            return
        
        x_true = self.F.get_x()
        true_pos = np.array([x_true[0, 0], x_true[2, 0]])
        y = self.sensor.measure(true_pos, target_pos)
        
        # Filter update
        if self.filter_type == 'ekf':
            self._update_ekf(y, target_pos)
        elif self.filter_type == 'ukf':
            self._update_ukf(y, target_pos)
        elif self.filter_type == 'qkf_numeric':
            self._update_qkf_numeric(y, target_pos)
        elif self.filter_type == 'pf':
            self._update_pf(y, target_pos)
        
        # Compute control
        goal_state = np.array([[target_pos[0]], [0], [target_pos[1]], [0]])
        u = self.controller.compute_control(self.x_hat, goal_state)
        self.F.set_u(u)
        
        # Store data
        self.true_positions.append(true_pos)
        self.estimated_positions.append(np.array([self.x_hat[0, 0], self.x_hat[2, 0]]))
        self.controls.append(u.flatten())
        self.tracked_landmark_history.append(self.target_landmark_id)
        
        self.F.forward()
    
    def _update_ekf(self, y, landmark_pos):
        x_pred = self.F.A @ self.x_hat + self.F.B @ self.F.get_u()
        P_pred = self.F.A @ self.P_est @ self.F.A.T + self.F.W
        H = self.sensor.g(np.array([self.x_hat[0, 0], self.x_hat[2, 0]]), landmark_pos)
        y_pred = self.sensor.measure_pred(np.array([x_pred[0, 0], x_pred[2, 0]]), landmark_pos)
        innov = y - y_pred
        innov[1, 0] = np.arctan2(np.sin(innov[1, 0]), np.cos(innov[1, 0]))
        S = H @ P_pred @ H.T + self.sensor.V
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x_hat = x_pred + K @ innov
        self.P_est = P_pred - K @ S @ K.T
    
    def _update_ukf(self, y, landmark_pos):
        n = 4
        alpha, beta, kappa = 0.5, 2, 0
        lambda_ = alpha**2 * (n + kappa) - n
        
        sigma_points = np.zeros((2*n + 1, n))
        sigma_points[0] = self.x_hat.flatten()
        
        try:
            sqrt_P = np.linalg.cholesky((n + lambda_) * self.P_est)
        except:
            eigenvals, eigenvecs = np.linalg.eigh(self.P_est)
            eigenvals = np.maximum(eigenvals, 1e-8)
            sqrt_P = eigenvecs @ np.diag(np.sqrt(eigenvals))
            sqrt_P = np.sqrt(n + lambda_) * sqrt_P
        
        for i in range(n):
            sigma_points[i+1] = self.x_hat.flatten() + sqrt_P[i]
            sigma_points[n+i+1] = self.x_hat.flatten() - sqrt_P[i]
        
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(2*n + 1):
            sigma_points_pred[i] = (self.F.A @ sigma_points[i].reshape(-1,1) + 
                                    self.F.B @ self.F.get_u()).flatten()
        
        weights_mean = np.full(2*n + 1, 1/(2*(n+lambda_)))
        weights_mean[0] = lambda_/(n+lambda_)
        x_pred = np.sum(weights_mean[:, np.newaxis] * sigma_points_pred, axis=0).reshape(-1,1)
        
        weights_cov = np.full(2*n + 1, 1/(2*(n+lambda_)))
        weights_cov[0] = lambda_/(n+lambda_) + (1 - alpha**2 + beta)
        P_pred = self.F.W.copy()
        for i in range(2*n + 1):
            diff = sigma_points_pred[i] - x_pred.flatten()
            P_pred += weights_cov[i] * np.outer(diff, diff)
        
        sigma_points_meas = np.zeros((2*n + 1, 2))
        for i in range(2*n + 1):
            pos = np.array([sigma_points_pred[i, 0], sigma_points_pred[i, 2]])
            sigma_points_meas[i] = self.sensor.measure_pred(pos, landmark_pos).flatten()
        
        y_pred = np.sum(weights_mean[:, np.newaxis] * sigma_points_meas, axis=0).reshape(-1,1)
        
        S = self.sensor.V.copy()
        for i in range(2*n + 1):
            diff = sigma_points_meas[i] - y_pred.flatten()
            S += weights_cov[i] * np.outer(diff, diff)
        
        C_tilde = np.zeros((n, 2))
        for i in range(2*n + 1):
            diff_state = sigma_points_pred[i] - x_pred.flatten()
            diff_meas = sigma_points_meas[i] - y_pred.flatten()
            C_tilde += weights_cov[i] * np.outer(diff_state, diff_meas)
        
        K = C_tilde @ np.linalg.pinv(S)
        innov = y - y_pred
        innov[1, 0] = np.arctan2(np.sin(innov[1, 0]), np.cos(innov[1, 0]))
        self.x_hat = x_pred + K @ innov
        self.P_est = P_pred - K @ S @ K.T
    
    def _update_qkf_numeric(self, y, landmark_pos, max_iter=5):
        x_pred = self.F.A @ self.x_hat + self.F.B @ self.F.get_u()
        P_pred = self.F.A @ self.P_est @ self.F.A.T + self.F.W
        x_current = x_pred.copy()
        
        for _ in range(max_iter):
            H = self.sensor.g(np.array([x_current[0,0], x_current[2,0]]), landmark_pos)
            y_pred = self.sensor.measure_pred(np.array([x_current[0,0], x_current[2,0]]), landmark_pos)
            innov = y - y_pred
            innov[1, 0] = np.arctan2(np.sin(innov[1, 0]), np.cos(innov[1, 0]))
            S = H @ P_pred @ H.T + self.sensor.V + np.eye(2) * 1e-6
            K = P_pred @ H.T @ np.linalg.inv(S)
            x_new = x_pred + K @ innov
            if np.linalg.norm(x_new - x_current) < 1e-4:
                break
            x_current = x_new
        
        H = self.sensor.g(np.array([x_current[0,0], x_current[2,0]]), landmark_pos)
        S = H @ P_pred @ H.T + self.sensor.V
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x_hat = x_current
        self.P_est = P_pred - K @ S @ K.T
    
    def _update_pf(self, y, landmark_pos):
        n = 4
        n_particles = self.n_particles
        particles_pred = (self.F.A @ self.particles.T).T + (self.F.B @ self.F.get_u()).flatten()
        chol_W = np.linalg.cholesky(self.F.W + np.eye(n) * 1e-10)
        particles_pred += (chol_W @ np.random.randn(n, n_particles)).T
        
        log_like = np.zeros(n_particles)
        for i in range(n_particles):
            pos = np.array([particles_pred[i, 0], particles_pred[i, 2]])
            y_pred = self.sensor.measure_pred(pos, landmark_pos)
            diff = y.flatten() - y_pred.flatten()
            diff[1] = np.arctan2(np.sin(diff[1]), np.cos(diff[1]))
            log_like[i] = -0.5 * diff @ np.linalg.solve(self.sensor.V, diff)
        
        log_w = np.log(self.weights + 1e-300) + log_like
        log_w -= log_w.max()
        weights_new = np.exp(log_w)
        weights_new /= weights_new.sum()
        
        n_eff = 1.0 / np.sum(weights_new**2)
        if n_eff < n_particles / 2:
            indices = np.random.choice(n_particles, n_particles, p=weights_new)
            particles_new = particles_pred[indices]
            weights_new = np.ones(n_particles) / n_particles
        else:
            particles_new = particles_pred
        
        self.particles = particles_new
        self.weights = weights_new
        self.x_hat = np.mean(particles_new, axis=0).reshape(-1, 1)
        diff = particles_new - self.x_hat.flatten()
        self.P_est = (diff.T @ diff) / n_particles + np.eye(n) * 1e-6
    
    def get_trajectories(self):
        return np.array(self.true_positions), np.array(self.estimated_positions)


class MultiTargetTrackerMovingLandmarks:
    """Multi-agent tracking with moving landmarks."""
    
    def __init__(self, n_agents=3, n_landmarks=4, H=200, dt=0.05,
                 process_noise_scale=0.05, measurement_noise_scale=0.1,
                 filters_to_use=['ekf', 'ukf', 'qkf_numeric', 'pf'],
                 n_particles=500):
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.H = H
        self.dt = dt
        self.filters_to_use = filters_to_use
        self.n_particles = n_particles
        self.landmarks = self._create_landmarks()
    
    def _create_landmarks(self):
        landmarks = []
        landmarks.append(MovingLandmark(0, 'circle', center=(5, 5), radius=4, speed=0.8, dt=self.dt))
        landmarks.append(MovingLandmark(1, 'circle', center=(-5, -5), radius=4, speed=0.8, dt=self.dt))
        landmarks.append(MovingLandmark(2, 'figure8', center=(0, 0), radius=6, speed=0.6, dt=self.dt))
        landmarks.append(MovingLandmark(3, 'linear', center=(0, 5), radius=8, speed=0.7, dt=self.dt))
        return landmarks[:self.n_landmarks]
    
    def run_filter(self, filter_name):
        agent_positions = np.zeros((self.H, self.n_agents, 2))
        agent_estimates = np.zeros((self.H, self.n_agents, 2))
        landmark_positions = np.zeros((self.H, self.n_landmarks, 2))
        tracked_landmarks = np.zeros((self.H, self.n_agents), dtype=int)
        
        agents = []
        np.random.seed(42)
        
        for agent_idx in range(self.n_agents):
            x0 = np.random.randn(4, 1) * 2
            x0[0, 0] = np.random.uniform(-8, 8)
            x0[2, 0] = np.random.uniform(-8, 8)
            agent = ControlledAgent(agent_idx, dt=self.dt, process_noise_scale=0.05,
                                    measurement_noise_scale=0.1, filter_type=filter_name,
                                    n_particles=self.n_particles)
            agent.set_initial_state(x0)
            agents.append(agent)
        
        for t in tqdm(range(self.H), desc=f"{filter_name.upper()}"):
            for lm in self.landmarks:
                lm.update()
                landmark_positions[t, lm.id, :] = lm.get_position()
            
            for agent_idx, agent in enumerate(agents):
                agent.step(self.landmarks)
                true_pos, est_pos = agent.get_trajectories()
                agent_positions[t, agent_idx, :] = true_pos[-1]
                agent_estimates[t, agent_idx, :] = est_pos[-1]
                tracked_landmarks[t, agent_idx] = agent.target_landmark_id if agent.target_landmark_id is not None else -1
        
        return agent_positions, agent_estimates, landmark_positions, tracked_landmarks
    
    def run_all_filters(self):
        results = {}
        for filter_name in self.filters_to_use:
            print(f"\nRunning {filter_name.upper()}...")
            self.landmarks = self._create_landmarks()
            agent_positions, agent_estimates, landmark_positions, tracked_landmarks = self.run_filter(filter_name)
            
            tracking_errors = np.zeros((self.H, self.n_agents))
            for t in range(self.H):
                for agent_idx in range(self.n_agents):
                    landmark_id = tracked_landmarks[t, agent_idx]
                    if landmark_id >= 0:
                        agent_pos = agent_positions[t, agent_idx]
                        lm_pos = landmark_positions[t, landmark_id]
                        tracking_errors[t, agent_idx] = np.linalg.norm(agent_pos - lm_pos)
                    else:
                        tracking_errors[t, agent_idx] = np.nan
            
            mean_tracking_error = np.nanmean(tracking_errors)
            print(f"  {filter_name.upper()}: Mean distance = {mean_tracking_error:.3f} m")
            
            results[filter_name] = {
                'agent_positions': agent_positions,
                'agent_estimates': agent_estimates,
                'landmark_positions': landmark_positions,
                'tracked_landmarks': tracked_landmarks,
                'tracking_errors': tracking_errors,
                'metrics': {
                    'mean_tracking_error': float(mean_tracking_error),
                    'final_tracking_error': float(np.nanmean(tracking_errors[-20:, :]))
                }
            }
        return results


def print_text_results(results):
    print("\n" + "="*80)
    print("LQG MULTI-AGENT TRACKING WITH MOVING LANDMARKS")
    print("="*80)
    
    first_filter = list(results.keys())[0]
    H = results[first_filter]['agent_positions'].shape[0]
    n_agents = results[first_filter]['agent_positions'].shape[1]
    n_landmarks = results[first_filter]['landmark_positions'].shape[1]
    
    print(f"\nSIMULATION PARAMETERS:")
    print(f"  Horizon: {H} timesteps")
    print(f"  Number of agents: {n_agents}")
    print(f"  Number of landmarks: {n_landmarks}")
    print(f"  Filters tested: {', '.join(results.keys())}")
    
    print("\n" + "-"*80)
    print("PERFORMANCE METRICS SUMMARY")
    print("-"*80)
    print(f"\n{'Filter':<15} {'Mean Distance to Landmark':<30} {'Final Distance':<20}")
    print(f"{'-'*15} {'-'*30} {'-'*20}")
    
    for filter_name, data in results.items():
        metrics = data['metrics']
        print(f"{filter_name.upper():<15} {metrics['mean_tracking_error']:>25.4f} m  {metrics['final_tracking_error']:>15.4f} m")
    
    print("\n" + "-"*80)
    print("RANKING (Best to Worst by Mean Distance)")
    print("-"*80)
    
    rankings = [(name, data['metrics']['mean_tracking_error']) for name, data in results.items()]
    rankings.sort(key=lambda x: x[1])
    
    for i, (name, error) in enumerate(rankings, 1):
        print(f"  {i}. {name.upper()}: {error:.4f} m")
    
    print("\n" + "="*80)


def plot_comparison(results, output_dir='MultiTargetTracking/results'):
    os.makedirs(output_dir, exist_ok=True)
    filter_names = list(results.keys())
    n_agents = results[filter_names[0]]['agent_positions'].shape[1]
    colors = {'ekf': '#1f77b4', 'ukf': '#2ca02c', 'qkf_numeric': '#ff7f0e', 'pf': '#9467bd'}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    for filter_name in filter_names:
        tracking_errors = results[filter_name]['tracking_errors']
        mean_error = np.nanmean(tracking_errors, axis=1)
        ax1.plot(mean_error, label=filter_name.upper(), color=colors.get(filter_name, 'black'), linewidth=2)
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Distance to Landmark (m)')
    ax1.set_title('Tracking Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    data = results[filter_names[0]]
    landmark_positions = data['landmark_positions'][-1]
    ax2.plot(landmark_positions[:, 0], landmark_positions[:, 1], 'k^', markersize=15,
            markerfacecolor='yellow', markeredgewidth=2, label='Landmarks')
    for filter_name in filter_names:
        positions = results[filter_name]['agent_positions'][-1]
        ax2.plot(positions[:, 0], positions[:, 1], 'o', markersize=10,
                color=colors.get(filter_name, 'black'), label=f'{filter_name.upper()}')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Final Positions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    ax3 = axes[1, 0]
    for filter_name in filter_names:
        tracked = results[filter_name]['tracked_landmarks']
        switches = np.sum(np.diff(tracked, axis=0) != 0, axis=0)
        ax3.bar(np.arange(n_agents) - 0.2, np.mean(switches, axis=0), 0.4,
               label=filter_name.upper(), color=colors.get(filter_name, 'black'))
    ax3.set_xlabel('Agent')
    ax3.set_ylabel('Number of Switches')
    ax3.set_title('Landmark Switching Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    x = np.arange(n_agents)
    width = 0.2
    for i, filter_name in enumerate(filter_names):
        errors = results[filter_name]['tracking_errors']
        mean_per_agent = np.nanmean(errors, axis=0)
        ax4.bar(x + i*width, mean_per_agent, width, 
               label=filter_name.upper(), color=colors.get(filter_name, 'black'))
    ax4.set_xlabel('Agent')
    ax4.set_ylabel('Mean Error (m)')
    ax4.set_title('Performance by Agent')
    ax4.set_xticks(x + width * (len(filter_names)-1)/2)
    ax4.set_xticklabels([f'Agent {i+1}' for i in range(n_agents)])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/moving_landmarks_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {output_dir}/")


def create_animation(results, filename='moving_landmarks.gif', fps=10, max_frames=100):
    print("\nCreating animation...")
    filter_names = list(results.keys())
    data = results[filter_names[0]]
    H = data['agent_positions'].shape[0]
    n_agents = data['agent_positions'].shape[1]
    n_landmarks = data['landmark_positions'].shape[1]
    step = max(1, H // max_frames)
    frames = list(range(0, H, step))
    colors = {'ekf': '#1f77b4', 'ukf': '#2ca02c', 'qkf_numeric': '#ff7f0e', 'pf': '#9467bd'}
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def animate(frame_idx):
        t = frames[frame_idx]
        ax.clear()
        
        # Landmarks
        lm_positions = data['landmark_positions'][t]
        ax.plot(lm_positions[:, 0], lm_positions[:, 1], 'ks', markersize=12,
                markerfacecolor='yellow', markeredgewidth=2, label='Landmarks')
        
        # Landmark trajectories
        for lm_id in range(n_landmarks):
            lm_traj = data['landmark_positions'][:t+1, lm_id, :]
            ax.plot(lm_traj[:, 0], lm_traj[:, 1], 'k--', alpha=0.3, linewidth=1)
        
        # Agents
        for filter_name in filter_names:
            positions = results[filter_name]['agent_positions'][t]
            ax.plot(positions[:, 0], positions[:, 1], 'o', markersize=10,
                   color=colors.get(filter_name, 'black'), label=filter_name.upper() if t == frames[0] else '')
            
            if t > 0:
                traj = results[filter_name]['agent_positions'][:t+1]
                ax.plot(traj[:, 0], traj[:, 1], '-', color=colors.get(filter_name, 'black'), alpha=0.3, linewidth=1)
        
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_title(f'Moving Landmark Tracking (t = {t}/{H})')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=1000/fps, repeat=False)
    output_dir = 'MultiTargetTracking/results'
    os.makedirs(output_dir, exist_ok=True)
    full_filename = os.path.join(output_dir, filename)
    
    try:
        anim.save(full_filename, writer='pillow', fps=fps)
        print(f"Animation saved to {full_filename}")
    except Exception as e:
        print(f"Could not save GIF: {e}")
    
    plt.close()


def main():
    print("LQG Multi-Agent Tracking with Moving Landmarks")
    print("=" * 60)
    
    n_agents = 3
    n_landmarks = 4
    H = 200
    dt = 0.05
    filters_to_use = ['ekf', 'ukf', 'qkf_numeric', 'pf']
    n_particles = 500
    
    print(f"  Agents: {n_agents}")
    print(f"  Landmarks: {n_landmarks}")
    print(f"  Horizon: {H} steps")
    print(f"  dt: {dt}s")
    print(f"  Filters: {filters_to_use}")
    print()
    
    tracker = MultiTargetTrackerMovingLandmarks(
        n_agents=n_agents, n_landmarks=n_landmarks, H=H, dt=dt,
        process_noise_scale=0.05, measurement_noise_scale=0.1,
        filters_to_use=filters_to_use, n_particles=n_particles
    )
    
    results = tracker.run_all_filters()
    
    output_dir = 'MultiTargetTracking/results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/moving_landmarks_results_{timestamp}.pkl"
    
    with open(filename, 'wb') as f:
        pkl.dump(results, f)
    print(f"\nResults saved to {filename}")
    
    print_text_results(results)
    plot_comparison(results, output_dir)
    create_animation(results, 'moving_landmarks.gif', fps=10)
    
    print("\nDone!")


if __name__ == '__main__':
    main()