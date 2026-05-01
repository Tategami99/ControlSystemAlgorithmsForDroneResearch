"""
showcase_qkf_advantages.py
===========================
Enhanced simulation specifically designed to showcase QKF advantages.

Key enhancements to highlight QKF benefits:
1. Stronger measurement nonlinearity (closer landmarks = more curvature)
2. More aggressive target maneuvers
3. Higher measurement noise (where QKF's second-order statistics help)
4. Multiple measurement types (range, bearing, range-rate)
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle as pkl
import os
from tqdm import tqdm
from scipy.linalg import solve_discrete_are

from state_dynamics import StateDynamics, Vec
from MultiTargetTracking.filters import RangeBearingSensor, update_lqe_ekf, update_lqe_ukf, update_lqe_qkf_numeric, update_lqe_pf
from MultiTargetTracking.multi_target_tracking import (
    EvasiveTarget, LQGController, TrackerAgent, 
    print_results, plot_results, create_animation
)


class EnhancedRangeBearingSensor:
    """
    Enhanced sensor with range-rate measurements for stronger nonlinearity.
    """
    
    def __init__(self, landmarks, noise_scale=0.15):
        self.landmarks = landmarks
        self.n_landmarks = landmarks.shape[0]
        self.m = 3 * self.n_landmarks  # range, bearing, range-rate per landmark
        self.n = 4
        
        # Higher noise to challenge filters
        range_std = noise_scale
        bearing_std = 0.15 * noise_scale  # Increased from 0.1
        rangerate_std = 0.1 * noise_scale
        
        noise_diag = np.tile([range_std**2, bearing_std**2, rangerate_std**2], 
                             self.n_landmarks)
        self.V = np.diag(noise_diag)
    
    def measure(self, x):
        """Measurement with noise: [range, bearing, range-rate] per landmark."""
        x = np.asarray(x).reshape(-1)
        px, vx, py, vy = x[0], x[1], x[2], x[3]
        measurements = []
        
        for lx, ly in self.landmarks:
            dx = px - lx
            dy = py - ly
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            # Range-rate: dr/dt = (dx*vx + dy*vy) / r
            r_safe = max(r, 1e-6)
            r_dot = (dx * vx + dy * vy) / r_safe
            
            measurements.extend([r, theta, r_dot])
        
        y = np.array(measurements).reshape(-1, 1)
        
        # Add noise
        D = np.linalg.cholesky(self.V + np.eye(self.m) * 1e-10)
        rng_noise = np.random.default_rng()
        noise = D @ rng_noise.standard_normal((self.m, 1))
        return y + noise
    
    def measure_pred(self, x):
        """Predicted measurement (no noise)."""
        x = np.asarray(x).reshape(-1)
        px, vx, py, vy = x[0], x[1], x[2], x[3]
        measurements = []
        
        for lx, ly in self.landmarks:
            dx = px - lx
            dy = py - ly
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            r_safe = max(r, 1e-6)
            r_dot = (dx * vx + dy * vy) / r_safe
            measurements.extend([r, theta, r_dot])
        
        return np.array(measurements).reshape(-1, 1)
    
    def g(self, x):
        """Jacobian H = dh/dx."""
        x = np.asarray(x).reshape(-1)
        px, vx, py, vy = x[0], x[1], x[2], x[3]
        H = np.zeros((self.m, 4))
        
        for i, (lx, ly) in enumerate(self.landmarks):
            dx = px - lx
            dy = py - ly
            r = np.sqrt(dx**2 + dy**2)
            r_safe = max(r, 1e-6)
            r2 = r_safe**2
            r3 = r_safe**3
            
            # Range derivatives
            dr_dpx = dx / r_safe
            dr_dpy = dy / r_safe
            
            # Bearing derivatives
            dtheta_dpx = -dy / r2
            dtheta_dpy = dx / r2
            
            # Range-rate derivatives (more complex!)
            dr_dot_dpx = (vx * r_safe - dx * (dx*vx + dy*vy) / r_safe) / r2
            dr_dot_dvx = dx / r_safe
            dr_dot_dpy = (vy * r_safe - dy * (dx*vx + dy*vy) / r_safe) / r2
            dr_dot_dvy = dy / r_safe
            
            # Fill Jacobian
            H[3*i, :] = [dr_dpx, 0, dr_dpy, 0]
            H[3*i+1, :] = [dtheta_dpx, 0, dtheta_dpy, 0]
            H[3*i+2, :] = [dr_dot_dpx, dr_dot_dvx, dr_dot_dpy, dr_dot_dvy]
        
        return H


class AggressiveTarget(EvasiveTarget):
    """
    More aggressive evasive target with sudden direction changes.
    Creates stronger nonlinearity to challenge filters.
    """
    
    def update(self):
        """Update with more aggressive maneuvers."""
        self.time += self.dt
        t = self.time
        
        if self.trajectory_type == 'sinusoidal':
            # Higher frequency, larger amplitude changes
            x = self.center[0] + self.amplitude * np.sin(2*self.omega * t + self.phase)
            y = self.center[1] + self.amplitude * np.cos(3*self.omega * t)
            vx = self.amplitude * 2*self.omega * np.cos(2*self.omega * t + self.phase)
            vy = -self.amplitude * 3*self.omega * np.sin(3*self.omega * t)
            
        elif self.trajectory_type == 'circular':
            # Variable radius creates radial acceleration
            radius = self.amplitude * (1 + 0.5 * np.sin(self.omega * t * 0.5))
            angle = 2 * self.omega * t + self.phase
            x = self.center[0] + radius * np.cos(angle)
            y = self.center[1] + radius * np.sin(angle)
            
            dr_dt = self.amplitude * 0.5 * self.omega * 0.5 * np.cos(self.omega * t * 0.5)
            vx = dr_dt * np.cos(angle) - radius * 2*self.omega * np.sin(angle)
            vy = dr_dt * np.sin(angle) + radius * 2*self.omega * np.cos(angle)
            
        elif self.trajectory_type == 'figure8':
            # Tighter figure-8
            x = self.center[0] + self.amplitude * np.sin(2.5*self.omega * t)
            y = self.center[1] + self.amplitude * np.sin(5*self.omega * t + self.phase)
            vx = self.amplitude * 2.5*self.omega * np.cos(2.5*self.omega * t)
            vy = self.amplitude * 5*self.omega * np.cos(5*self.omega * t + self.phase)
            
        elif self.trajectory_type == 'spiral':
            # Rapid spiral with oscillations
            radius = self.amplitude * (1 + 0.6 * np.sin(self.omega * t))
            angle = 3 * self.omega * t + self.phase
            x = self.center[0] + radius * np.cos(angle)
            y = self.center[1] + radius * np.sin(angle)
            
            dr_dt = self.amplitude * 0.6 * self.omega * np.cos(self.omega * t)
            vx = dr_dt * np.cos(angle) - radius * 3*self.omega * np.sin(angle)
            vy = dr_dt * np.sin(angle) + radius * 3*self.omega * np.cos(angle)
        
        self.state = np.array([[x], [vx], [y], [vy]])
        return self.state


class EnhancedTrackerAgent(TrackerAgent):
    """
    Tracker with enhanced sensor model.
    """
    
    def step(self, targets, landmarks):
        """Execute one step with enhanced sensor."""
        target_state = self.select_target(targets)
        
        if target_state is None:
            u = np.zeros((2, 1))
            self.F.set_u(u)
            self.F.forward()
            return
        
        # Enhanced sensor
        sensor = EnhancedRangeBearingSensor(landmarks, noise_scale=0.15)
        
        # Filter update
        if self.filter_type == 'ekf':
            self.x_hat, self.P_est, _ = update_lqe_ekf(
                self.F, sensor, self.x_hat, self.P_est
            )
        elif self.filter_type == 'ukf':
            self.x_hat, self.P_est, _ = update_lqe_ukf(
                self.F, sensor, self.x_hat, self.P_est
            )
        elif self.filter_type == 'qkf_numeric':
            self.x_hat, self.P_est, _ = update_lqe_qkf_numeric(
                self.F, sensor, self.x_hat, self.P_est, max_iter=15
            )
        elif self.filter_type == 'pf':
            self.particles, self.weights, self.x_hat, self.P_est = update_lqe_pf(
                self.F, sensor, self.particles, self.weights, self.n_particles
            )
        
        # Compute control
        goal_state = target_state.copy()
        goal_state[1, 0] = 0
        goal_state[3, 0] = 0
        
        u = self.controller.compute_control(self.x_hat, goal_state)
        self.F.set_u(u)
        
        # Store data
        x_true = self.F.get_x()
        self.true_states.append(x_true.copy())
        self.estimated_states.append(self.x_hat.copy())
        self.controls.append(u.copy())
        
        agent_pos = np.array([x_true[0, 0], x_true[2, 0]])
        target_pos = np.array([target_state[0, 0], target_state[2, 0]])
        tracking_error = np.linalg.norm(agent_pos - target_pos)
        self.tracking_errors.append(tracking_error)
        
        est_error = np.linalg.norm(x_true - self.x_hat)
        self.innovations.append(est_error)
        
        self.F.forward()


class QKFShowcaseSimulation:
    """
    Simulation specifically designed to showcase QKF advantages.
    """
    
    def __init__(self, n_agents=3, n_targets=4, n_landmarks=8, H=400, dt=0.05,
                 filters_to_use=None, n_particles=1000):
        
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.n_landmarks = n_landmarks
        self.H = H
        self.dt = dt
        self.n_particles = n_particles
        
        if filters_to_use is None:
            self.filters_to_use = ['ekf', 'ukf', 'qkf_numeric', 'pf']
        else:
            self.filters_to_use = filters_to_use
        
        # Closer landmarks = stronger measurement nonlinearity
        self.landmarks = self._initialize_landmarks()
        self.targets = self._initialize_aggressive_targets()
        self.results = {}
    
    def _initialize_landmarks(self):
        """Create landmarks closer to action for stronger nonlinearity."""
        landmarks = []
        
        # Inner ring (closer = more nonlinearity)
        for i in range(self.n_landmarks // 2):
            angle = 2 * np.pi * i / (self.n_landmarks // 2)
            radius = 6.0  # Closer than before
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            landmarks.append([x, y])
        
        # Outer ring
        for i in range(self.n_landmarks // 2):
            angle = 2 * np.pi * i / (self.n_landmarks // 2) + np.pi / (self.n_landmarks // 2)
            radius = 12.0
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            landmarks.append([x, y])
        
        return np.array(landmarks)
    
    def _initialize_aggressive_targets(self):
        """Create aggressive targets."""
        trajectory_types = ['sinusoidal', 'circular', 'figure8', 'spiral']
        targets = []
        
        for i in range(self.n_targets):
            angle = 2 * np.pi * i / self.n_targets
            center_radius = 3.0
            center = (center_radius * np.cos(angle), center_radius * np.sin(angle))
            traj_type = trajectory_types[i % len(trajectory_types)]
            
            target = AggressiveTarget(
                target_id=i,
                trajectory_type=traj_type,
                center=center,
                speed=1.2,  # Faster
                dt=self.dt
            )
            targets.append(target)
        
        return targets
    
    def _initialize_agents(self, filter_type):
        """Initialize enhanced agents."""
        agents = []
        
        for i in range(self.n_agents):
            angle = 2 * np.pi * i / self.n_agents
            radius = 1.5
            x0 = np.array([
                [radius * np.cos(angle)],
                [0],
                [radius * np.sin(angle)],
                [0]
            ])
            
            agent = EnhancedTrackerAgent(
                agent_id=i,
                dt=self.dt,
                process_noise_scale=0.15,  # Higher noise
                measurement_noise_scale=0.15,
                filter_type=filter_type,
                n_particles=self.n_particles,
                Q_scale=15.0,  # More aggressive control
                R_scale=0.05
            )
            agent.set_initial_state(x0)
            agents.append(agent)
        
        return agents
    
    def run_filter(self, filter_type):
        """Run simulation for one filter."""
        print(f"\nRunning {filter_type.upper()}...")
        
        agents = self._initialize_agents(filter_type)
        
        for target in self.targets:
            target.time = 0.0
            target.update()
        
        agent_positions = np.zeros((self.H, self.n_agents, 2))
        agent_states = np.zeros((self.H, self.n_agents, 4))
        target_positions = np.zeros((self.H, self.n_targets, 2))
        target_states = np.zeros((self.H, self.n_targets, 4))
        
        for t in tqdm(range(self.H), desc=f"{filter_type.upper()} steps", leave=False):
            for target in self.targets:
                target.update()
                target_states[t, target.id, :] = target.get_state().flatten()
                target_positions[t, target.id, :] = target.get_position()
            
            for agent in agents:
                agent.step(self.targets, self.landmarks)
                x = agent.F.get_x()
                agent_states[t, agent.agent_id, :] = x.flatten()
                agent_positions[t, agent.agent_id, :] = [x[0, 0], x[2, 0]]
        
        tracking_errors = np.array([agent.tracking_errors for agent in agents]).T
        estimation_errors = np.array([agent.innovations for agent in agents]).T
        controls = [agent.controls for agent in agents]
        
        mean_tracking_error = np.mean(tracking_errors)
        final_tracking_error = np.mean(tracking_errors[-1, :])
        mean_estimation_error = np.mean(estimation_errors)
        control_effort = np.mean([np.mean([np.linalg.norm(u) for u in agent.controls]) 
                                  for agent in agents])
        
        results = {
            'filter_type': filter_type,
            'agent_positions': agent_positions,
            'agent_states': agent_states,
            'target_positions': target_positions,
            'target_states': target_states,
            'tracking_errors': tracking_errors,
            'estimation_errors': estimation_errors,
            'controls': controls,
            'landmarks': self.landmarks,
            'metrics': {
                'mean_tracking_error': mean_tracking_error,
                'final_tracking_error': final_tracking_error,
                'mean_estimation_error': mean_estimation_error,
                'control_effort': control_effort
            }
        }
        
        return results
    
    def run_all_filters(self):
        """Run all filters."""
        print("="*80)
        print("QKF ADVANTAGE SHOWCASE - Enhanced Nonlinearity")
        print("="*80)
        print(f"Agents: {self.n_agents}")
        print(f"Aggressive Targets: {self.n_targets}")
        print(f"Landmarks: {self.n_landmarks}")
        print(f"Horizon: {self.H} steps")
        print(f"Enhanced measurements: range + bearing + range-rate")
        print(f"Filters: {self.filters_to_use}")
        print("="*80)
        
        for filter_type in self.filters_to_use:
            self.results[filter_type] = self.run_filter(filter_type)
        
        return self.results


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("QKF ADVANTAGE SHOWCASE")
    print("Enhanced nonlinearity to demonstrate QKF benefits")
    print("="*80 + "\n")
    
    # Enhanced parameters
    sim = QKFShowcaseSimulation(
        n_agents=3,
        n_targets=4,
        n_landmarks=8,
        H=400,
        dt=0.05,
        filters_to_use=['ekf', 'ukf', 'qkf_numeric', 'pf'],
        n_particles=1000
    )
    
    results = sim.run_all_filters()
    
    # Save results
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/qkf_showcase_{timestamp}.pkl"
    
    with open(filename, 'wb') as f:
        pkl.dump(results, f)
    print(f"\nResults saved to {filename}")
    
    # Print results
    print_results(results)
    
    # Create plots
    plot_results(results, output_dir)
    
    # Create animation
    anim_filename = f"{output_dir}/qkf_showcase.gif"
    create_animation(results, anim_filename, fps=10, max_frames=200)
    
    print(f"\n{'='*80}")
    print("QKF Showcase complete!")
    print(f"Check {output_dir}/ for outputs")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()