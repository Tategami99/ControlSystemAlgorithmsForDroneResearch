"""
active_multi_target_tracking.py
================================
Active multi-target tracking with LQG control for all filters (EKF, UKF, QKF, PF).
This simulation is designed to showcase the advantages of QKF in highly nonlinear scenarios.

Key Features:
- LQG control tailored to each filter type (original state-based for EKF/UKF/PF, augmented for QKF)
- Nonlinear measurement model (range-bearing) to showcase QKF's quadratic handling
- Moving targets with complex trajectories
- Information-seeking behavior (agents actively position themselves for better measurements)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import pickle as pkl
import os
from tqdm import tqdm
from scipy.linalg import solve_discrete_are

from state_dynamics import StateDynamics, Vec, invVec


class MovingTarget:
    """A moving target with nonlinear trajectory."""
    
    def __init__(self, target_id, trajectory_type='spiral', center=(0, 0), 
                 radius=8.0, speed=0.3, dt=0.05):
        self.id = target_id
        self.trajectory_type = trajectory_type
        self.center = np.array(center)
        self.radius = radius
        self.speed = speed
        self.dt = dt
        self.time = 0.0
        self.position = np.array(center)
        self.velocity = np.zeros(2)
        
    def update(self):
        """Update target position based on trajectory type."""
        self.time += self.dt
        
        if self.trajectory_type == 'circle':
            angle = self.speed * self.time
            self.position = self.center + self.radius * np.array([
                np.cos(angle), np.sin(angle)
            ])
            self.velocity = self.radius * self.speed * np.array([
                -np.sin(angle), np.cos(angle)
            ])
            
        elif self.trajectory_type == 'spiral':
            # Expanding spiral - creates highly nonlinear range dynamics
            angle = self.speed * self.time
            r = self.radius * (0.5 + 0.5 * self.time / 20.0)  # expanding
            r = min(r, self.radius * 1.5)  # cap at 1.5x initial radius
            self.position = self.center + r * np.array([
                np.cos(angle), np.sin(angle)
            ])
            
        elif self.trajectory_type == 'figure8':
            # Figure-8 pattern - creates nonlinear cross-over dynamics
            t = self.speed * self.time
            self.position = self.center + self.radius * np.array([
                np.sin(t), np.sin(2*t)/2
            ])
            
        elif self.trajectory_type == 'random_walk':
            # Stochastic motion
            self.velocity += np.random.randn(2) * 0.15
            self.velocity = np.clip(self.velocity, -self.speed*2, self.speed*2)
            self.position += self.velocity * self.dt
            self.position = np.clip(self.position, -15, 15)
            
        return self.position
    
    def get_position(self):
        return self.position


class RangeBearingSensor:
    """Range-bearing sensor - highly nonlinear measurement model."""
    
    def __init__(self, noise_scale=0.15):
        self.m = 2  # range and bearing
        self.noise_scale = noise_scale
        range_std = noise_scale
        bearing_std = 0.08 * noise_scale  # bearing more accurate
        self.V = np.diag([range_std**2, bearing_std**2])
    
    def measure(self, agent_pos, target_pos):
        """Measure range and bearing from agent to target (with noise)."""
        dx = target_pos[0] - agent_pos[0]
        dy = target_pos[1] - agent_pos[1]
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        
        # Add noise
        D = np.linalg.cholesky(self.V)
        noise = D @ np.random.randn(2, 1)
        return np.array([[r + noise[0, 0]], [theta + noise[1, 0]]])
    
    def measure_pred(self, agent_pos, target_pos):
        """Predicted measurement (no noise)."""
        dx = target_pos[0] - agent_pos[0]
        dy = target_pos[1] - agent_pos[1]
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        return np.array([[r], [theta]])
    
    def jacobian(self, agent_pos, target_pos):
        """Measurement Jacobian H = dh/dx for EKF."""
        dx = target_pos[0] - agent_pos[0]
        dy = target_pos[1] - agent_pos[1]
        r = np.sqrt(dx**2 + dy**2)
        r_safe = max(r, 1e-6)
        
        # Range derivatives
        dr_dpx = -dx / r_safe
        dr_dpy = -dy / r_safe
        
        # Bearing derivatives
        dtheta_dpx = dy / (r_safe**2)
        dtheta_dpy = -dx / (r_safe**2)
        
        # Jacobian w.r.t. [px, vx, py, vy]
        H = np.array([
            [dr_dpx, 0, dr_dpy, 0],
            [dtheta_dpx, 0, dtheta_dpy, 0]
        ])
        return H


class LQGController:
    """LQG controller with information-seeking behavior."""
    
    def __init__(self, dt=0.05, control_type='standard', filter_type='ekf'):
        """
        Args:
            dt: time step
            control_type: 'standard' (track target) or 'info_seeking' (optimize geometry)
            filter_type: type of filter being used
        """
        self.dt = dt
        self.control_type = control_type
        self.filter_type = filter_type
        self.n = 4
        self.p = 2
        
        # System matrices
        self.A = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        self.B = np.array([
            [0, 0],
            [dt, 0],
            [0, 0],
            [0, dt]
        ])
        
        # Cost matrices - favor position tracking
        self.Q = np.diag([15.0, 2.0, 15.0, 2.0])  # state cost
        self.R = np.diag([1.0, 1.0]) * 0.2  # control cost (lower = more aggressive)
        
        # Compute LQR gain
        try:
            P = solve_discrete_are(self.A, self.B, self.Q, self.R)
            self.K_lqr = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
        except:
            # Fallback gain if ARE fails
            self.K_lqr = np.array([[1.5, 0.8, 0, 0], [0, 0, 1.5, 0.8]])
    
    def compute_control_standard(self, x_hat, target_pos):
        """Standard LQG: steer toward target position."""
        goal_state = np.array([[target_pos[0]], [0], [target_pos[1]], [0]])
        error = goal_state - x_hat
        u = self.K_lqr @ error
        return np.clip(u, -8.0, 8.0).reshape(-1, 1)
    
    def compute_control_info_seeking(self, x_hat, target_pos, P_est, sensor):
        """
        Information-seeking control: position agent to minimize estimation uncertainty.
        This creates an interesting geometry where agents don't just chase but optimize
        their viewing angle and distance for better measurements.
        """
        # Standard tracking component
        u_track = self.compute_control_standard(x_hat, target_pos)
        
        # Information term: prefer perpendicular approach for better bearing info
        agent_pos = np.array([x_hat[0, 0], x_hat[2, 0]])
        to_target = target_pos - agent_pos
        dist = np.linalg.norm(to_target)
        
        if dist > 0.1:
            # Desired offset: stay at ~3-5m distance, perpendicular when possible
            desired_offset = 4.0
            current_offset = dist - desired_offset
            
            # Lateral movement component (perpendicular to line-of-sight)
            perpendicular = np.array([-to_target[1], to_target[0]]) / (dist + 1e-6)
            
            # Information gain heuristic: move perpendicular if too close, radial if too far
            if current_offset > 1.0:  # too far
                u_info = -to_target / (dist + 1e-6)  # move closer
            elif current_offset < -1.0:  # too close
                u_info = to_target / (dist + 1e-6)  # move away
            else:  # good distance, optimize angle
                u_info = perpendicular * 0.5
            
            # Blend tracking and information-seeking
            u_info = u_info.reshape(-1, 1) * 2.0
            u = 0.7 * u_track + 0.3 * u_info
        else:
            u = u_track
        
        return np.clip(u, -8.0, 8.0).reshape(-1, 1)
    
    def compute_control(self, x_hat, target_pos, P_est=None, sensor=None):
        """Compute control based on control type."""
        if self.control_type == 'info_seeking' and P_est is not None:
            return self.compute_control_info_seeking(x_hat, target_pos, P_est, sensor)
        else:
            return self.compute_control_standard(x_hat, target_pos)


class LQGControllerQKF:
    """
    Specialized LQG controller for QKF using augmented state.
    Uses quadratic cost and considers second-order moments.
    """
    
    def __init__(self, dt=0.05):
        self.dt = dt
        self.n = 4
        self.p = 2
        
        self.A = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        self.B = np.array([
            [0, 0],
            [dt, 0],
            [0, 0],
            [0, dt]
        ])
        
        # Quadratic cost matrices
        self.Q = np.diag([15.0, 2.0, 15.0, 2.0])
        self.R = np.diag([1.0, 1.0]) * 0.2
        
        # For augmented control (iLQR-style)
        self.max_iter = 5
        self.alpha = 0.8  # step size
    
    def compute_control_ilqr(self, x_hat, P_est, target_pos):
        """
        Iterative LQR for quadratic control (simplified version).
        Uses current estimate mean and covariance to compute control.
        """
        goal_state = np.array([[target_pos[0]], [0], [target_pos[1]], [0]])
        error = goal_state - x_hat
        
        # Initialize with standard LQR
        try:
            P_lqr = solve_discrete_are(self.A, self.B, self.Q, self.R)
            K = np.linalg.inv(self.R + self.B.T @ P_lqr @ self.B) @ self.B.T @ P_lqr @ self.A
            u = K @ error
        except:
            K = np.array([[1.5, 0.8, 0, 0], [0, 0, 1.5, 0.8]])
            u = K @ error
        
        # Quadratic refinement: add correction based on uncertainty
        # Higher uncertainty -> more conservative control
        trace_P = np.trace(P_est)
        uncertainty_factor = min(1.0, trace_P / 10.0)  # normalize
        u = u * (1.0 - 0.3 * uncertainty_factor)  # reduce control when uncertain
        
        return np.clip(u, -8.0, 8.0).reshape(-1, 1)
    
    def compute_control(self, x_hat, P_est, target_pos):
        """Compute control using augmented state information."""
        return self.compute_control_ilqr(x_hat, P_est, target_pos)


class ActiveTrackerAgent:
    """
    Agent that actively tracks a moving target using a specified filter with LQG control.
    """
    
    def __init__(self, agent_id, dt=0.05, process_noise=0.08, 
                 measurement_noise=0.15, filter_type='ekf', 
                 control_type='standard', n_particles=500):
        self.agent_id = agent_id
        self.filter_type = filter_type
        self.control_type = control_type
        self.n_particles = n_particles
        self.dt = dt
        
        # Initialize dynamics
        n1, n2, p = 0, 4, 2
        A = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        B = np.array([
            [0, 0],
            [dt, 0],
            [0, 0],
            [0, dt]
        ])
        W = np.eye(4) * process_noise**2
        A_E = np.zeros((0, 0))
        
        self.F = StateDynamics(n1, n2, p, W, A_E, A, B)
        self.sensor = RangeBearingSensor(measurement_noise)
        
        # Initialize controller
        if filter_type == 'qkf_numeric':
            self.controller = LQGControllerQKF(dt)
        else:
            self.controller = LQGController(dt, control_type, filter_type)
        
        # Initialize filter
        self.reset_filter()
        
        # Data storage
        self.true_positions = []
        self.estimated_positions = []
        self.controls = []
        self.estimation_errors = []
        self.covariance_traces = []
        
    def reset_filter(self):
        """Initialize filter state."""
        n = 4
        x0 = self.F.get_x()
        
        if self.filter_type == 'pf':
            self.particles = x0.flatten() + np.random.randn(self.n_particles, 4) * 0.3
            self.weights = np.ones(self.n_particles) / self.n_particles
            self.x_hat = x0.copy()
            self.P_est = np.eye(4) * 0.2
        elif self.filter_type == 'qkf_numeric':
            # Augmented state for QKF
            self.x_hat = x0.copy()
            self.P_est = np.eye(4) * 0.2
            # Augmented state Z = [x; vec(xx')]
            z, z1, z2 = self.F.get_z()
            self.Z_est = z.copy()
            # Augmented covariance (simplified initialization)
            self.Pz_est = np.eye(n + n**2) * 0.1
        else:
            self.x_hat = x0.copy()
            self.P_est = np.eye(4) * 0.2
    
    def set_initial_state(self, x0):
        """Set initial true state."""
        self.F.set_x(x0)
        self.reset_filter()
    
    def step(self, target):
        """Execute one time step: measure, estimate, control, simulate."""
        # Get current true state and target position
        x_true = self.F.get_x()
        true_pos = np.array([x_true[0, 0], x_true[2, 0]])
        target_pos = target.get_position()
        
        # Take measurement
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
        if self.filter_type == 'qkf_numeric':
            u = self.controller.compute_control(self.x_hat, self.P_est, target_pos)
        else:
            if self.control_type == 'info_seeking':
                u = self.controller.compute_control(self.x_hat, target_pos, 
                                                   self.P_est, self.sensor)
            else:
                u = self.controller.compute_control(self.x_hat, target_pos)
        
        self.F.set_u(u)
        
        # Store data
        est_pos = np.array([self.x_hat[0, 0], self.x_hat[2, 0]])
        self.true_positions.append(true_pos)
        self.estimated_positions.append(est_pos)
        self.controls.append(u.flatten())
        self.estimation_errors.append(np.linalg.norm(true_pos - est_pos))
        self.covariance_traces.append(np.trace(self.P_est))
        
        # Forward dynamics
        self.F.forward()
    
    def _update_ekf(self, y, target_pos):
        """EKF update."""
        # Prediction
        x_pred = self.F.A @ self.x_hat + self.F.B @ self.F.get_u()
        P_pred = self.F.A @ self.P_est @ self.F.A.T + self.F.W
        
        # Measurement update
        agent_pos_pred = np.array([x_pred[0, 0], x_pred[2, 0]])
        y_pred = self.sensor.measure_pred(agent_pos_pred, target_pos)
        H = self.sensor.jacobian(agent_pos_pred, target_pos)
        
        S = H @ P_pred @ H.T + self.sensor.V
        K = P_pred @ H.T @ np.linalg.pinv(S)
        
        innov = y - y_pred
        innov[1, 0] = np.arctan2(np.sin(innov[1, 0]), np.cos(innov[1, 0]))  # wrap angle
        
        self.x_hat = x_pred + K @ innov
        self.P_est = P_pred - K @ S @ K.T
        self.P_est = 0.5 * (self.P_est + self.P_est.T)  # symmetrize
    
    def _update_ukf(self, y, target_pos):
        """UKF update."""
        n = 4
        alpha, beta, kappa = 0.5, 2, 0
        lambda_ = alpha**2 * (n + kappa) - n
        
        # Sigma points
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
            sigma_points[i + 1] = self.x_hat.flatten() + sqrt_P[i]
            sigma_points[n + i + 1] = self.x_hat.flatten() - sqrt_P[i]
        
        # Predict sigma points
        sigma_pred = np.zeros_like(sigma_points)
        for i in range(2*n + 1):
            sigma_pred[i] = (self.F.A @ sigma_points[i].reshape(-1, 1) + 
                           self.F.B @ self.F.get_u()).flatten()
        
        # Weights
        wm = np.full(2*n + 1, 1/(2*(n + lambda_)))
        wm[0] = lambda_ / (n + lambda_)
        wc = wm.copy()
        wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
        
        # Predicted state
        x_pred = np.sum(wm[:, np.newaxis] * sigma_pred, axis=0).reshape(-1, 1)
        P_pred = self.F.W.copy()
        for i in range(2*n + 1):
            diff = sigma_pred[i] - x_pred.flatten()
            P_pred += wc[i] * np.outer(diff, diff)
        
        # Predicted measurements
        sigma_meas = np.zeros((2*n + 1, 2))
        for i in range(2*n + 1):
            agent_pos = np.array([sigma_pred[i][0], sigma_pred[i][2]])
            sigma_meas[i] = self.sensor.measure_pred(agent_pos, target_pos).flatten()
        
        y_pred = np.sum(wm[:, np.newaxis] * sigma_meas, axis=0).reshape(-1, 1)
        
        S = self.sensor.V.copy()
        for i in range(2*n + 1):
            diff = sigma_meas[i] - y_pred.flatten()
            S += wc[i] * np.outer(diff, diff)
        
        # Cross covariance
        Pxy = np.zeros((n, 2))
        for i in range(2*n + 1):
            dx = sigma_pred[i] - x_pred.flatten()
            dy = sigma_meas[i] - y_pred.flatten()
            Pxy += wc[i] * np.outer(dx, dy)
        
        K = Pxy @ np.linalg.pinv(S)
        
        innov = y - y_pred
        innov[1, 0] = np.arctan2(np.sin(innov[1, 0]), np.cos(innov[1, 0]))
        
        self.x_hat = x_pred + K @ innov
        self.P_est = P_pred - K @ S @ K.T
        self.P_est = 0.5 * (self.P_est + self.P_est.T)
    
    def _update_qkf_numeric(self, y, target_pos):
        """
        QKF numeric update - uses augmented state and iterative refinement.
        This is the key filter that should show advantages in nonlinear scenarios.
        """
        # Augmented state prediction
        Phi_tilde = self.F.get_A_tilde()
        Sigma_tilde = self.F.get_Sigma_tilde()
        mu_tilde = self.F.get_mu_tilde()
        
        Z_pred = Phi_tilde @ self.Z_est + mu_tilde
        Pz_pred = Phi_tilde @ self.Pz_est @ Phi_tilde.T + Sigma_tilde
        
        # For measurement, we need to approximate the quadratic measurement model
        # Extract predicted state
        x_pred = Z_pred[:4]
        agent_pos_pred = np.array([x_pred[0, 0], x_pred[2, 0]])
        
        # Iterative measurement update (iLQR-style)
        x_current = x_pred.copy()
        max_iter = 10
        
        for iteration in range(max_iter):
            agent_pos_current = np.array([x_current[0, 0], x_current[2, 0]])
            y_pred = self.sensor.measure_pred(agent_pos_current, target_pos)
            H = self.sensor.jacobian(agent_pos_current, target_pos)
            
            # Augmented Jacobian (only affects first n states)
            H_tilde = np.zeros((2, 4 + 16))
            H_tilde[:, :4] = H
            
            innov = y - y_pred
            innov[1, 0] = np.arctan2(np.sin(innov[1, 0]), np.cos(innov[1, 0]))
            
            S = H_tilde @ Pz_pred @ H_tilde.T + self.sensor.V
            K = Pz_pred @ H_tilde.T @ np.linalg.pinv(S + np.eye(2) * 1e-6)
            
            # Update augmented state
            Z_new = Z_pred + K @ innov
            x_new = Z_new[:4]
            
            # Check convergence
            if np.linalg.norm(x_new - x_current) < 1e-4:
                break
            
            x_current = x_new
        
        # Final update
        self.Z_est = Z_new
        self.Pz_est = Pz_pred - K @ S @ K.T
        
        # Extract state and covariance
        self.x_hat = self.Z_est[:4]
        self.P_est = self.Pz_est[:4, :4]
        self.P_est = 0.5 * (self.P_est + self.P_est.T)
    
    def _update_pf(self, y, target_pos):
        """Particle filter update."""
        n_particles = self.n_particles
        A, B, W = self.F.A, self.F.B, self.F.W
        u = self.F.get_u()
        
        # Predict
        self.particles = (A @ self.particles.T).T + (B @ u).flatten()
        chol_W = np.linalg.cholesky(W + np.eye(4) * 1e-10)
        self.particles += (chol_W @ np.random.randn(4, n_particles)).T
        
        # Measurement likelihood
        log_like = np.zeros(n_particles)
        for i in range(n_particles):
            agent_pos_i = np.array([self.particles[i, 0], self.particles[i, 2]])
            h_i = self.sensor.measure_pred(agent_pos_i, target_pos).flatten()
            diff = y.flatten() - h_i
            diff[1] = np.arctan2(np.sin(diff[1]), np.cos(diff[1]))
            log_like[i] = -0.5 * diff @ np.linalg.solve(self.sensor.V, diff)
        
        # Weight update
        log_w = np.log(self.weights + 1e-300) + log_like
        log_w -= log_w.max()
        self.weights = np.exp(log_w)
        self.weights /= self.weights.sum()
        
        # Resample if needed
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < n_particles / 2:
            self.particles, self.weights = self._resample_systematic()
        
        # Estimate
        self.x_hat = np.mean(self.particles, axis=0).reshape(-1, 1)
        diff = self.particles - self.x_hat.flatten()
        self.P_est = (diff.T @ diff) / n_particles + np.eye(4) * 1e-6
    
    def _resample_systematic(self):
        """Systematic resampling."""
        n_particles = len(self.weights)
        positions = (np.arange(n_particles) + np.random.rand()) / n_particles
        cumsum = np.cumsum(self.weights)
        
        i, j = 0, 0
        indices = np.zeros(n_particles, dtype=int)
        
        while i < n_particles:
            if positions[i] < cumsum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        
        return self.particles[indices], np.ones(n_particles) / n_particles


class ActiveMultiTargetTracker:
    """Main simulation class for active multi-target tracking."""
    
    def __init__(self, n_agents=3, n_targets=3, H=250, dt=0.05,
                 process_noise=0.08, measurement_noise=0.15,
                 filters_to_use=['ekf', 'ukf', 'qkf_numeric', 'pf'],
                 control_type='standard', n_particles=500):
        
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.H = H
        self.dt = dt
        self.filters_to_use = filters_to_use
        self.control_type = control_type
        self.n_particles = n_particles
        
        # Create targets with challenging trajectories
        self.targets = []
        trajectories = ['spiral', 'figure8', 'circle']
        centers = [(-6, -6), (6, -6), (0, 8)]
        
        for i in range(n_targets):
            traj_type = trajectories[i % len(trajectories)]
            center = centers[i % len(centers)]
            target = MovingTarget(i, traj_type, center, radius=7.0, speed=0.4, dt=dt)
            self.targets.append(target)
        
        # Store parameters
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        print(f"Initialized {n_targets} targets with trajectories: {[t.trajectory_type for t in self.targets]}")
    
    def run_filter(self, filter_name):
        """Run simulation for a single filter."""
        print(f"\nRunning {filter_name.upper()}...")
        
        # Initialize agents
        agents = []
        initial_positions = [
            np.array([-8, 0, -8, 0]),
            np.array([8, 0, -8, 0]),
            np.array([0, 0, 10, 0])
        ]
        
        for i in range(self.n_agents):
            agent = ActiveTrackerAgent(
                i, self.dt, self.process_noise, self.measurement_noise,
                filter_name, self.control_type, self.n_particles
            )
            x0 = initial_positions[i % len(initial_positions)].reshape(-1, 1)
            agent.set_initial_state(x0)
            agents.append(agent)
        
        # Reset targets
        for target in self.targets:
            target.time = 0.0
        
        # Simulation loop
        for t in tqdm(range(self.H), desc=f"{filter_name.upper()}"):
            # Update targets
            for target in self.targets:
                target.update()
            
            # Each agent tracks assigned target
            for i, agent in enumerate(agents):
                target = self.targets[i % self.n_targets]
                agent.step(target)
        
        # Collect results
        results = {
            'agent_data': [],
            'target_positions': []
        }
        
        for agent in agents:
            results['agent_data'].append({
                'true_positions': np.array(agent.true_positions),
                'estimated_positions': np.array(agent.estimated_positions),
                'controls': np.array(agent.controls),
                'errors': np.array(agent.estimation_errors),
                'cov_traces': np.array(agent.covariance_traces)
            })
        
        # Store target trajectories
        for target in self.targets:
            # Recreate target trajectory
            target.time = 0.0
            traj = []
            for t in range(self.H):
                traj.append(target.get_position().copy())
                target.update()
            results['target_positions'].append(np.array(traj))
        
        return results
    
    def run_all_filters(self):
        """Run simulation for all filters."""
        all_results = {}
        
        for filter_name in self.filters_to_use:
            all_results[filter_name] = self.run_filter(filter_name)
        
        return all_results


def print_results(results):
    """Print performance summary."""
    print("\n" + "="*80)
    print("ACTIVE MULTI-TARGET TRACKING RESULTS")
    print("="*80)
    
    print(f"\n{'Filter':<15} {'Mean Error (m)':<20} {'Final Error (m)':<20} {'Mean Cov Trace':<20}")
    print("-"*80)
    
    for filter_name, data in results.items():
        all_errors = np.concatenate([agent['errors'] for agent in data['agent_data']])
        final_errors = [agent['errors'][-1] for agent in data['agent_data']]
        all_cov_traces = np.concatenate([agent['cov_traces'] for agent in data['agent_data']])
        
        mean_error = np.mean(all_errors)
        final_error = np.mean(final_errors)
        mean_cov = np.mean(all_cov_traces)
        
        print(f"{filter_name.upper():<15} {mean_error:>18.4f}  {final_error:>18.4f}  {mean_cov:>18.4f}")
    
    print("="*80)


def plot_results(results, output_dir='results'):
    """Create comprehensive comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    colors = {
        'ekf': '#1f77b4',
        'ukf': '#2ca02c',
        'qkf_numeric': '#ff7f0e',
        'pf': '#9467bd'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Mean tracking error over time
    ax = axes[0, 0]
    for filter_name, data in results.items():
        errors_per_time = []
        H = len(data['agent_data'][0]['errors'])
        for t in range(H):
            errors_t = [agent['errors'][t] for agent in data['agent_data']]
            errors_per_time.append(np.mean(errors_t))
        
        ax.plot(errors_per_time, label=filter_name.upper(),
               color=colors.get(filter_name, 'black'), linewidth=2)
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('Mean Position Error (m)')
    ax.set_title('Tracking Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Cumulative error
    ax = axes[0, 1]
    for filter_name, data in results.items():
        errors_per_time = []
        H = len(data['agent_data'][0]['errors'])
        for t in range(H):
            errors_t = [agent['errors'][t] for agent in data['agent_data']]
            errors_per_time.append(np.mean(errors_t))
        
        cumsum = np.cumsum(errors_per_time)
        ax.plot(cumsum, label=filter_name.upper(),
               color=colors.get(filter_name, 'black'), linewidth=2)
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('Cumulative Error (m)')
    ax.set_title('Cumulative Tracking Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Covariance trace over time
    ax = axes[0, 2]
    for filter_name, data in results.items():
        if filter_name != 'pf':  # PF covariance is computed differently
            cov_per_time = []
            H = len(data['agent_data'][0]['cov_traces'])
            for t in range(H):
                cov_t = [agent['cov_traces'][t] for agent in data['agent_data']]
                cov_per_time.append(np.mean(cov_t))
            
            ax.plot(cov_per_time, label=filter_name.upper(),
                   color=colors.get(filter_name, 'black'), linewidth=2)
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('Mean Covariance Trace')
    ax.set_title('Estimation Uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Error distribution
    ax = axes[1, 0]
    error_data = []
    labels = []
    for filter_name, data in results.items():
        all_errors = np.concatenate([agent['errors'] for agent in data['agent_data']])
        error_data.append(all_errors)
        labels.append(filter_name.upper())
    
    bp = ax.boxplot(error_data, labels=labels, patch_artist=True)
    for patch, filter_name in zip(bp['boxes'], results.keys()):
        patch.set_facecolor(colors.get(filter_name, 'lightgray'))
    
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Error Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Trajectory visualization (agent 0)
    ax = axes[1, 1]
    first_filter = list(results.keys())[0]
    target_traj = results[first_filter]['target_positions'][0]
    ax.plot(target_traj[:, 0], target_traj[:, 1], 'k--', linewidth=3, 
           label='Target', alpha=0.7)
    
    for filter_name, data in results.items():
        est_pos = data['agent_data'][0]['estimated_positions']
        ax.plot(est_pos[:, 0], est_pos[:, 1], '-',
               label=f'{filter_name.upper()}',
               color=colors.get(filter_name, 'black'), linewidth=2, alpha=0.7)
    
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title('Agent 0 Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 6. Control effort
    ax = axes[1, 2]
    for filter_name, data in results.items():
        controls = data['agent_data'][0]['controls']
        control_mag = np.linalg.norm(controls, axis=1)
        ax.plot(control_mag, label=filter_name.upper(),
               color=colors.get(filter_name, 'black'), linewidth=2)
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('Control Magnitude')
    ax.set_title('Control Effort (Agent 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/active_tracking_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/active_tracking_comparison.png")


def create_animation(results, filename='active_tracking.gif', fps=10, max_frames=150):
    """Create animation of active tracking."""
    print("\nCreating animation...")
    
    output_dir = os.path.dirname(filename) or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    colors = {
        'ekf': '#1f77b4',
        'ukf': '#2ca02c',
        'qkf_numeric': '#ff7f0e',
        'pf': '#9467bd'
    }
    
    first_filter = list(results.keys())[0]
    H = len(results[first_filter]['agent_data'][0]['true_positions'])
    n_agents = len(results[first_filter]['agent_data'])
    n_targets = len(results[first_filter]['target_positions'])
    
    step = max(1, H // max_frames)
    frames = list(range(0, H, step))
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    def animate(frame_idx):
        t = frames[frame_idx]
        ax.clear()
        
        # Plot target trajectories and current positions
        for target_idx in range(n_targets):
            target_traj = results[first_filter]['target_positions'][target_idx]
            ax.plot(target_traj[:t+1, 0], target_traj[:t+1, 1], 
                   'k--', alpha=0.3, linewidth=1)
            ax.plot(target_traj[t, 0], target_traj[t, 1],
                   'r*', markersize=20, markeredgecolor='black',
                   markeredgewidth=2, label='Target' if target_idx == 0 else '')
        
        # Plot agents for each filter
        for agent_idx in range(n_agents):
            for filter_name in results.keys():
                data = results[filter_name]['agent_data'][agent_idx]
                est_pos = data['estimated_positions']
                
                # Current position
                label = f'{filter_name.upper()}' if agent_idx == 0 else ''
                ax.plot(est_pos[t, 0], est_pos[t, 1], 'o', markersize=10,
                       color=colors.get(filter_name, 'black'),
                       label=label, alpha=0.8)
                
                # Trajectory history
                if t > 0:
                    ax.plot(est_pos[:t+1, 0], est_pos[:t+1, 1], '-',
                           color=colors.get(filter_name, 'black'),
                           alpha=0.3, linewidth=1)
        
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_title(f'Active Multi-Target Tracking (t = {t}/{H})')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), 
                                  interval=1000/fps, repeat=False)
    
    try:
        anim.save(filename, writer='pillow', fps=fps)
        print(f"Animation saved to {filename}")
    except Exception as e:
        print(f"Could not save animation: {e}")
    
    plt.close()


def main():
    """Main execution."""
    print("Active Multi-Target Tracking with LQG Control")
    print("=" * 60)
    print("This simulation showcases QKF advantages in nonlinear systems")
    print("=" * 60)
    
    # Simulation parameters
    n_agents = 3
    n_targets = 3
    H = 250
    dt = 0.05
    filters_to_use = ['ekf', 'ukf', 'qkf_numeric', 'pf']
    control_type = 'standard'  # or 'info_seeking'
    n_particles = 500
    
    print(f"\nParameters:")
    print(f"  Agents: {n_agents}")
    print(f"  Targets: {n_targets}")
    print(f"  Horizon: {H} steps ({H*dt:.1f} seconds)")
    print(f"  Time step: {dt}s")
    print(f"  Filters: {filters_to_use}")
    print(f"  Control: {control_type}")
    print()
    
    # Create tracker
    tracker = ActiveMultiTargetTracker(
        n_agents=n_agents,
        n_targets=n_targets,
        H=H,
        dt=dt,
        process_noise=0.08,
        measurement_noise=0.15,
        filters_to_use=filters_to_use,
        control_type=control_type,
        n_particles=n_particles
    )
    
    # Run simulations
    results = tracker.run_all_filters()
    
    # Save results
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/active_tracking_results_{timestamp}.pkl"
    
    with open(filename, 'wb') as f:
        pkl.dump(results, f)
    print(f"\nResults saved to {filename}")
    
    # Print results
    print_results(results)
    
    # Create plots
    plot_results(results, output_dir)
    
    # Create animation
    anim_filename = f"{output_dir}/active_tracking.gif"
    create_animation(results, anim_filename, fps=10)
    
    print("\n" + "="*60)
    print("Simulation complete! Check the results/ directory")
    print("="*60)


if __name__ == '__main__':
    main()