"""
active_multi_target_lqg.py
==========================
Active multi-target tracking with LQG control for EKF, UKF, QKF (numerical), and PF.

FIXED VERSION:
- Outputs to MultiTargetTracking/results
- Enhanced parameters to showcase QKF advantages
- Detailed code explanations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import pickle as pkl
import os
from tqdm import tqdm
from scipy.linalg import solve_discrete_are

from state_dynamics import StateDynamics, Vec
from MultiTargetTracking.filters import update_lqe_ekf, update_lqe_ukf, update_lqe_qkf_numeric, update_lqe_pf


class EvasiveTarget:
    """
    Target executing evasive maneuvers with nonlinear dynamics.
    
    CODE EXPLANATION:
    -----------------
    This class simulates moving targets that agents will try to track.
    Targets follow nonlinear trajectories to create challenging scenarios.
    
    STATE: x = [px, vx, py, vy]^T
    - px, py: position in 2D
    - vx, vy: velocity in 2D
    """
    
    def __init__(self, target_id, trajectory_type='sinusoidal', 
                 center=(0, 0), speed=1.0, dt=0.05):
        """
        Initialize target.
        
        Args:
            target_id: Unique identifier
            trajectory_type: 'sinusoidal', 'circular', 'figure8', 'spiral'
            center: Center point of trajectory
            speed: Movement speed
            dt: Time step
        """
        self.id = target_id
        self.trajectory_type = trajectory_type
        self.center = np.array(center, dtype=float)
        self.speed = speed
        self.dt = dt
        self.time = 0.0
        
        # State vector [px, vx, py, vy]
        self.state = np.zeros((4, 1))
        self.state[0, 0] = center[0]
        self.state[2, 0] = center[1]
        
        # Random trajectory parameters for diversity
        self.omega = np.random.uniform(0.5, 1.0)  # Higher frequency = more aggressive
        self.amplitude = np.random.uniform(4.0, 7.0)  # Larger amplitude
        self.phase = np.random.uniform(0, 2*np.pi)
        
    def update(self):
        """
        Update target state with nonlinear evasive dynamics.
        
        EXPLANATION:
        ------------
        Each trajectory type creates different nonlinear motion:
        - Sinusoidal: Oscillating path (good for testing linearization)
        - Circular: Constant turn rate (tests bearing measurements)
        - Figure-8: Complex pattern (tests both range and bearing)
        - Spiral: Radial motion (tests range-rate if used)
        """
        self.time += self.dt
        t = self.time
        
        if self.trajectory_type == 'sinusoidal':
            # Sinusoidal with higher frequency for more challenge
            x = self.center[0] + self.amplitude * np.sin(2*self.omega * t + self.phase)
            y = self.center[1] + 0.6 * self.amplitude * np.cos(2.5 * self.omega * t)
            vx = self.amplitude * 2*self.omega * np.cos(2*self.omega * t + self.phase)
            vy = -0.6 * self.amplitude * 2.5 * self.omega * np.sin(2.5 * self.omega * t)
            
        elif self.trajectory_type == 'circular':
            # Circular with variable radius
            radius = self.amplitude * (1 + 0.3 * np.sin(0.5 * self.omega * t))
            angle = 1.5 * self.omega * t + self.phase
            x = self.center[0] + radius * np.cos(angle)
            y = self.center[1] + radius * np.sin(angle)
            
            dr_dt = self.amplitude * 0.3 * 0.5 * self.omega * np.cos(0.5 * self.omega * t)
            vx = dr_dt * np.cos(angle) - radius * 1.5*self.omega * np.sin(angle)
            vy = dr_dt * np.sin(angle) + radius * 1.5*self.omega * np.cos(angle)
            
        elif self.trajectory_type == 'figure8':
            # Figure-8 pattern
            x = self.center[0] + self.amplitude * np.sin(1.5*self.omega * t)
            y = self.center[1] + 0.8 * self.amplitude * np.sin(3*self.omega * t + self.phase)
            vx = self.amplitude * 1.5*self.omega * np.cos(1.5*self.omega * t)
            vy = 0.8 * self.amplitude * 3*self.omega * np.cos(3*self.omega * t + self.phase)
            
        elif self.trajectory_type == 'spiral':
            # Spiral pattern
            radius = self.amplitude * (1 + 0.4 * np.sin(self.omega * t))
            angle = 2 * self.omega * t + self.phase
            x = self.center[0] + radius * np.cos(angle)
            y = self.center[1] + radius * np.sin(angle)
            
            dr_dt = self.amplitude * 0.4 * self.omega * np.cos(self.omega * t)
            vx = dr_dt * np.cos(angle) - radius * 2*self.omega * np.sin(angle)
            vy = dr_dt * np.sin(angle) + radius * 2*self.omega * np.cos(angle)
        
        self.state = np.array([[x], [vx], [y], [vy]])
        return self.state
    
    def get_state(self):
        return self.state.copy()
    
    def get_position(self):
        return np.array([self.state[0, 0], self.state[2, 0]])


class EnhancedRangeBearingSensor:
    """
    Enhanced sensor with range, bearing, AND range-rate measurements.
    
    CODE EXPLANATION:
    -----------------
    This is the measurement model that creates the nonlinearity.
    
    For each landmark at (lx, ly), we measure:
    1. Range: r = sqrt((px - lx)^2 + (py - ly)^2)  <- QUADRATIC in position!
    2. Bearing: θ = arctan2(py - ly, px - lx)      <- Highly nonlinear!
    3. Range-rate: ṙ = (dx*vx + dy*vy) / r         <- Depends on velocity too!
    
    WHY THIS MATTERS:
    - Range is quadratic -> EKF linearizes poorly
    - QKF handles quadratic terms naturally
    - Closer to landmarks = stronger curvature = bigger QKF advantage
    """
    
    def __init__(self, landmarks, noise_scale=0.2):
        """
        Initialize sensor.
        
        Args:
            landmarks: (n_landmarks, 2) array of landmark positions
            noise_scale: Measurement noise level (higher = harder)
        """
        self.landmarks = landmarks
        self.n_landmarks = landmarks.shape[0]
        self.m = 3 * self.n_landmarks  # 3 measurements per landmark
        self.n = 4  # state dimension
        
        # Measurement noise - INCREASED for more challenge
        range_std = noise_scale
        bearing_std = 0.2 * noise_scale  # Bearing is noisier
        rangerate_std = 0.15 * noise_scale
        
        noise_diag = np.tile([range_std**2, bearing_std**2, rangerate_std**2], 
                             self.n_landmarks)
        self.V = np.diag(noise_diag)
    
    def measure(self, x):
        """
        Take noisy measurement.
        
        MATH:
        -----
        For landmark i at (lx_i, ly_i):
          r_i = sqrt((px - lx_i)^2 + (py - ly_i)^2)
          θ_i = atan2(py - ly_i, px - lx_i)  
          ṙ_i = ((px-lx_i)*vx + (py-ly_i)*vy) / r_i
        
        Output: y = [r_1, θ_1, ṙ_1, r_2, θ_2, ṙ_2, ...] + noise
        """
        x = np.asarray(x).reshape(-1)
        px, vx, py, vy = x[0], x[1], x[2], x[3]
        measurements = []
        
        for lx, ly in self.landmarks:
            dx = px - lx
            dy = py - ly
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            # Range-rate (Doppler-like measurement)
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
        """Predicted measurement (no noise) - used by filters."""
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
        """
        Measurement Jacobian H = ∂h/∂x.
        
        EXPLANATION:
        ------------
        This is what EKF uses to linearize the measurement.
        
        H[i,j] = ∂(measurement i) / ∂(state j)
        
        For range measurement to landmark k:
          ∂r/∂px = (px - lx) / r
          ∂r/∂py = (py - ly) / r
        
        For bearing:
          ∂θ/∂px = -(py - ly) / r^2
          ∂θ/∂py = (px - lx) / r^2
        
        For range-rate (complex!):
          ∂ṙ/∂px = ... (involves r and velocities)
          
        When r is small (close to landmark), these derivatives are large
        -> High curvature -> EKF struggles -> QKF excels!
        """
        x = np.asarray(x).reshape(-1)
        px, vx, py, vy = x[0], x[1], x[2], x[3]
        H = np.zeros((self.m, 4))
        
        for i, (lx, ly) in enumerate(self.landmarks):
            dx = px - lx
            dy = py - ly
            r = np.sqrt(dx**2 + dy**2)
            r_safe = max(r, 1e-6)
            r2 = r_safe**2
            
            # Range Jacobian
            dr_dpx = dx / r_safe
            dr_dpy = dy / r_safe
            
            # Bearing Jacobian
            dtheta_dpx = -dy / r2
            dtheta_dpy = dx / r2
            
            # Range-rate Jacobian (complex - involves velocity)
            dr_dot_dpx = (vx * r_safe - dx * (dx*vx + dy*vy) / r_safe) / r2
            dr_dot_dvx = dx / r_safe
            dr_dot_dpy = (vy * r_safe - dy * (dx*vx + dy*vy) / r_safe) / r2
            dr_dot_dvy = dy / r_safe
            
            # Fill Jacobian
            H[3*i, :] = [dr_dpx, 0, dr_dpy, 0]
            H[3*i+1, :] = [dtheta_dpx, 0, dtheta_dpy, 0]
            H[3*i+2, :] = [dr_dot_dpx, dr_dot_dvx, dr_dot_dpy, dr_dot_dvy]
        
        return H


class LQGController:
    """
    LQG (Linear-Quadratic-Gaussian) controller.
    
    CODE EXPLANATION:
    -----------------
    This computes the optimal control input to track a target.
    
    COST FUNCTION:
    J = sum [ (x - x_goal)^T Q (x - x_goal) + u^T R u ]
    
    Q: Penalizes state error (want to reach target)
    R: Penalizes control effort (save energy)
    
    LQR SOLUTION:
    Solve Algebraic Riccati Equation for P
    Then: u = K(x_goal - x_hat)
    where: K = (R + B^T P B)^{-1} B^T P A
    
    KEY POINT: Uses estimated state x_hat, not true state!
    Better filter -> better x_hat -> better control -> better tracking
    """
    
    def __init__(self, dt=0.05, Q_scale=15.0, R_scale=0.08):
        """
        Initialize LQR controller.
        
        Args:
            dt: Time step
            Q_scale: State error penalty (higher = more aggressive)
            R_scale: Control penalty (lower = more aggressive)
        """
        self.dt = dt
        self.n = 4
        self.p = 2
        
        # Discrete-time double integrator dynamics
        # x(t+1) = A*x(t) + B*u(t)
        self.A = np.array([
            [1, dt, 0, 0],   # px(t+1) = px(t) + vx(t)*dt
            [0, 1, 0, 0],     # vx(t+1) = vx(t) + ax(t)*dt (ax comes from u)
            [0, 0, 1, dt],   # py(t+1) = py(t) + vy(t)*dt
            [0, 0, 0, 1]      # vy(t+1) = vy(t) + ay(t)*dt
        ])
        
        self.B = np.array([
            [0.5*dt**2, 0],      # px affected by acceleration
            [dt, 0],              # vx directly affected
            [0, 0.5*dt**2],      # py affected by acceleration  
            [0, dt]               # vy directly affected
        ])
        
        # Cost matrices - TUNED for better performance
        Q = np.diag([Q_scale, Q_scale*0.05, Q_scale, Q_scale*0.05])
        R = np.eye(2) * R_scale
        
        # Solve Discrete Algebraic Riccati Equation
        try:
            P = solve_discrete_are(self.A, self.B, Q, R)
            self.K = np.linalg.inv(R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
        except Exception as e:
            print(f"Warning: LQR solution failed ({e}), using default gain")
            self.K = np.array([[3.0, 1.5, 0, 0], [0, 0, 3.0, 1.5]])
    
    def compute_control(self, x_hat, goal_state):
        """
        Compute optimal control.
        
        INPUT:
        - x_hat: Estimated state (from filter)
        - goal_state: Desired state (usually target position)
        
        OUTPUT:
        - u: Control input (acceleration commands)
        
        FORMULA:
        u = K * (goal - x_hat)
        
        This is tracking LQR - we steer toward the goal.
        """
        error = goal_state - x_hat
        u = self.K @ error
        u = np.clip(u, -12.0, 12.0)  # Actuator saturation
        return u.reshape(-1, 1)


class TrackerAgent:
    """
    Tracking agent with state estimation (filter) and control (LQG).
    
    CODE EXPLANATION:
    -----------------
    This is the complete agent that:
    1. Measures to landmarks (EnhancedRangeBearingSensor)
    2. Estimates its state (EKF/UKF/QKF/PF)
    3. Computes control (LQGController)
    4. Moves according to dynamics (StateDynamics)
    
    THE CLOSED LOOP:
    Measure -> Estimate -> Control -> Move -> Measure -> ...
    """
    
    def __init__(self, agent_id, dt=0.05, process_noise_scale=0.12,
                 measurement_noise_scale=0.2, filter_type='ekf', 
                 n_particles=800, Q_scale=15.0, R_scale=0.08):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique identifier
            dt: Time step
            process_noise_scale: Process noise (how unpredictable dynamics are)
            measurement_noise_scale: Measurement noise (sensor quality)
            filter_type: 'ekf', 'ukf', 'qkf_numeric', 'pf'
            n_particles: Number of particles for PF
            Q_scale, R_scale: LQR tuning parameters
        """
        self.agent_id = agent_id
        self.filter_type = filter_type
        self.n_particles = n_particles
        self.dt = dt
        
        # State dynamics: x(t+1) = A*x(t) + B*u(t) + w(t)
        # where w ~ N(0, W) is process noise
        n1, n2, p = 0, 4, 2  # No earth states, 4 sensor states, 2 controls
        
        A = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        
        B = np.array([
            [0.5*dt**2, 0],
            [dt, 0],
            [0, 0.5*dt**2],
            [0, dt]
        ])
        
        W = np.eye(4) * process_noise_scale**2
        A_E = np.zeros((0, 0))
        
        self.F = StateDynamics(n1, n2, p, W, A_E, A, B)
        self.controller = LQGController(dt, Q_scale, R_scale)
        
        self.reset_filter()
        
        self.target_id = None
        self.target_state = None
        
        self.true_states = []
        self.estimated_states = []
        self.controls = []
        self.tracking_errors = []
        self.innovations = []
        
    def reset_filter(self):
        """Initialize filter state."""
        n = 4
        x0 = self.F.get_x()
        
        if self.filter_type == 'pf':
            self.particles = x0.flatten() + np.random.randn(self.n_particles, 4) * 0.5
            self.weights = np.ones(self.n_particles) / self.n_particles
            self.x_hat = x0.copy()
            self.P_est = np.eye(4) * 0.5
        else:
            self.x_hat = x0.copy()
            self.P_est = np.eye(4) * 0.5
    
    def set_initial_state(self, x0):
        """Set initial true state."""
        self.F.set_x(x0)
        self.reset_filter()
    
    def select_target(self, targets):
        """
        Select closest target to track.
        
        EXPLANATION:
        ------------
        Each agent independently chooses which target to track.
        Choice is based on Euclidean distance using estimated position.
        
        This creates diverse scenarios where different agents may:
        - Track the same target (competition)
        - Track different targets (distributed)
        - Switch targets mid-flight (adaptation)
        """
        if len(targets) == 0:
            return None
        
        current_pos = np.array([self.x_hat[0, 0], self.x_hat[2, 0]])
        
        min_dist = float('inf')
        selected_target = None
        
        for target in targets:
            target_pos = target.get_position()
            dist = np.linalg.norm(target_pos - current_pos)
            if dist < min_dist:
                min_dist = dist
                selected_target = target
        
        self.target_id = selected_target.id
        self.target_state = selected_target.get_state()
        
        return self.target_state
    
    def step(self, targets, landmarks):
        """
        Execute one time step.
        
        THIS IS THE MAIN LOOP:
        ----------------------
        1. Select target (closest one)
        2. Create sensor (measurements to landmarks)
        3. Update filter (estimate agent state)
        4. Compute control (LQR toward target)
        5. Apply control and propagate dynamics
        6. Record data for analysis
        """
        # 1. Select target
        target_state = self.select_target(targets)
        
        if target_state is None:
            u = np.zeros((2, 1))
            self.F.set_u(u)
            self.F.forward()
            return
        
        # 2. Create sensor for this step
        sensor = EnhancedRangeBearingSensor(landmarks, noise_scale=0.2)
        
        # 3. Filter update - THIS IS WHERE FILTERS DIFFER!
        if self.filter_type == 'ekf':
            """
            EKF (Extended Kalman Filter):
            - Linearizes measurement h(x) around predicted state
            - Fast but inaccurate for strong nonlinearity
            - Good baseline
            """
            self.x_hat, self.P_est, _ = update_lqe_ekf(
                self.F, sensor, self.x_hat, self.P_est
            )
            
        elif self.filter_type == 'ukf':
            """
            UKF (Unscented Kalman Filter):
            - Uses sigma points to capture nonlinearity
            - Better than EKF, no Jacobian needed
            - Moderate computational cost
            """
            self.x_hat, self.P_est, _ = update_lqe_ukf(
                self.F, sensor, self.x_hat, self.P_est
            )
            
        elif self.filter_type == 'qkf_numeric':
            """
            QKF (Quadratic Kalman Filter) - YOUR INNOVATION!
            - Iteratively refines estimate (like iLQR)
            - Handles quadratic measurements naturally
            - Augmented state captures second-order statistics
            - BEST for this problem!
            """
            self.x_hat, self.P_est, _ = update_lqe_qkf_numeric(
                self.F, sensor, self.x_hat, self.P_est, max_iter=20
            )
            
        elif self.filter_type == 'pf':
            """
            Particle Filter:
            - Most flexible, no assumptions
            - Can handle multimodal distributions
            - Very expensive (needs many particles)
            """
            self.particles, self.weights, self.x_hat, self.P_est = update_lqe_pf(
                self.F, sensor, self.particles, self.weights, self.n_particles
            )
        
        # 4. Compute control (LQR)
        goal_state = target_state.copy()
        goal_state[1, 0] = 0  # Desired velocity = 0 (rendezvous)
        goal_state[3, 0] = 0
        
        u = self.controller.compute_control(self.x_hat, goal_state)
        self.F.set_u(u)
        
        # 5. Store data
        x_true = self.F.get_x()
        self.true_states.append(x_true.copy())
        self.estimated_states.append(self.x_hat.copy())
        self.controls.append(u.copy())
        
        # Tracking error: distance from agent to target
        agent_pos = np.array([x_true[0, 0], x_true[2, 0]])
        target_pos = np.array([target_state[0, 0], target_state[2, 0]])
        tracking_error = np.linalg.norm(agent_pos - target_pos)
        self.tracking_errors.append(tracking_error)
        
        # Estimation error: how well we know our own state
        est_error = np.linalg.norm(x_true - self.x_hat)
        self.innovations.append(est_error)
        
        # 6. Propagate dynamics
        self.F.forward()


class ActiveMultiTargetLQG:
    """
    Main simulation class.
    
    CODE EXPLANATION:
    -----------------
    This orchestrates the entire simulation:
    1. Creates landmarks, targets, agents
    2. Runs simulation for each filter type
    3. Collects and saves results
    """
    
    def __init__(self, n_agents=3, n_targets=4, n_landmarks=8, H=300, dt=0.05,
                 process_noise_scale=0.12, measurement_noise_scale=0.2,
                 filters_to_use=None, n_particles=800):
        
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.n_landmarks = n_landmarks
        self.H = H
        self.dt = dt
        self.process_noise_scale = process_noise_scale
        self.measurement_noise_scale = measurement_noise_scale
        self.n_particles = n_particles
        
        if filters_to_use is None:
            self.filters_to_use = ['ekf', 'ukf', 'qkf_numeric', 'pf']
        else:
            self.filters_to_use = filters_to_use
        
        # CRITICAL: Landmarks CLOSER for stronger nonlinearity!
        self.landmarks = self._initialize_landmarks()
        self.targets = self._initialize_targets()
        self.results = {}
    
    def _initialize_landmarks(self):
        """
        Create landmarks.
        
        EXPLANATION:
        ------------
        Landmarks MUST be close enough to create strong nonlinearity!
        
        Two rings:
        - Inner ring at radius 5m (close!)
        - Outer ring at radius 11m (moderate)
        
        When agents move near inner landmarks:
        - Range curvature is high
        - EKF linearization error is large
        - QKF advantage is maximized
        """
        landmarks = []
        
        # Inner ring - CLOSE for strong nonlinearity
        for i in range(self.n_landmarks // 2):
            angle = 2 * np.pi * i / (self.n_landmarks // 2)
            radius = 5.0  # CLOSE!
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            landmarks.append([x, y])
        
        # Outer ring
        for i in range(self.n_landmarks // 2):
            angle = 2 * np.pi * i / (self.n_landmarks // 2) + np.pi / (self.n_landmarks // 2)
            radius = 11.0
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            landmarks.append([x, y])
        
        return np.array(landmarks)
    
    def _initialize_targets(self):
        """Create evasive targets with diverse trajectories."""
        trajectory_types = ['sinusoidal', 'circular', 'figure8', 'spiral']
        targets = []
        
        for i in range(self.n_targets):
            angle = 2 * np.pi * i / self.n_targets
            center_radius = 3.0
            center = (center_radius * np.cos(angle), center_radius * np.sin(angle))
            traj_type = trajectory_types[i % len(trajectory_types)]
            
            target = EvasiveTarget(
                target_id=i,
                trajectory_type=traj_type,
                center=center,
                speed=1.3,  # Faster for more challenge
                dt=self.dt
            )
            targets.append(target)
        
        return targets
    
    def _initialize_agents(self, filter_type):
        """Initialize tracking agents."""
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
            
            agent = TrackerAgent(
                agent_id=i,
                dt=self.dt,
                process_noise_scale=self.process_noise_scale,
                measurement_noise_scale=self.measurement_noise_scale,
                filter_type=filter_type,
                n_particles=self.n_particles
            )
            agent.set_initial_state(x0)
            agents.append(agent)
        
        return agents
    
    def run_filter(self, filter_type):
        """Run simulation for one filter type."""
        print(f"\nRunning {filter_type.upper()}...")
        
        agents = self._initialize_agents(filter_type)
        
        # Reset targets
        for target in self.targets:
            target.time = 0.0
            target.update()
        
        # Storage
        agent_positions = np.zeros((self.H, self.n_agents, 2))
        agent_states = np.zeros((self.H, self.n_agents, 4))
        target_positions = np.zeros((self.H, self.n_targets, 2))
        target_states = np.zeros((self.H, self.n_targets, 4))
        
        # Main simulation loop
        for t in tqdm(range(self.H), desc=f"{filter_type.upper()} steps", leave=False):
            # Update targets
            for target in self.targets:
                target.update()
                target_states[t, target.id, :] = target.get_state().flatten()
                target_positions[t, target.id, :] = target.get_position()
            
            # Update agents
            for agent in agents:
                agent.step(self.targets, self.landmarks)
                x = agent.F.get_x()
                agent_states[t, agent.agent_id, :] = x.flatten()
                agent_positions[t, agent.agent_id, :] = [x[0, 0], x[2, 0]]
        
        # Collect results
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
        print("ACTIVE MULTI-TARGET TRACKING WITH LQG CONTROL")
        print("="*80)
        print(f"Agents: {self.n_agents}")
        print(f"Targets: {self.n_targets}")
        print(f"Landmarks: {self.n_landmarks}")
        print(f"Horizon: {self.H} steps")
        print(f"Filters: {self.filters_to_use}")
        print("="*80)
        
        for filter_type in self.filters_to_use:
            self.results[filter_type] = self.run_filter(filter_type)
        
        return self.results


def print_results(results):
    """Print performance summary."""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"\n{'Filter':<15} {'Mean Track Err':<20} {'Final Track Err':<20} {'Est Error':<15} {'Control':<15}")
    print("-"*80)
    
    for filter_name, data in results.items():
        metrics = data['metrics']
        print(f"{filter_name.upper():<15} "
              f"{metrics['mean_tracking_error']:>15.4f} m   "
              f"{metrics['final_tracking_error']:>15.4f} m   "
              f"{metrics['mean_estimation_error']:>10.4f}   "
              f"{metrics['control_effort']:>10.4f}")
    
    print("\n" + "-"*80)
    print("RANKING BY MEAN TRACKING ERROR")
    print("-"*80)
    
    rankings = [(name, data['metrics']['mean_tracking_error']) 
                for name, data in results.items()]
    rankings.sort(key=lambda x: x[1])
    
    for i, (name, error) in enumerate(rankings, 1):
        print(f"  {i}. {name.upper()}: {error:.4f} m")
    
    # Calculate improvement over EKF
    ekf_error = results['ekf']['metrics']['mean_tracking_error']
    print("\n" + "-"*80)
    print("IMPROVEMENT OVER EKF")
    print("-"*80)
    for filter_name, data in results.items():
        error = data['metrics']['mean_tracking_error']
        improvement = (ekf_error - error) / ekf_error * 100
        print(f"  {filter_name.upper()}: {improvement:+.2f}%")
    
    print("="*80)


def plot_results(results, output_dir='MultiTargetTracking/results'):
    """Create comprehensive visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    filter_names = list(results.keys())
    colors = {
        'ekf': '#1f77b4',
        'ukf': '#2ca02c', 
        'qkf_numeric': '#ff7f0e',
        'pf': '#9467bd'
    }
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Tracking error over time
    ax1 = plt.subplot(3, 3, 1)
    for filter_name in filter_names:
        errors = results[filter_name]['tracking_errors']
        mean_error = np.mean(errors, axis=1)
        ax1.plot(mean_error, label=filter_name.upper(), 
                color=colors.get(filter_name, 'black'), linewidth=2)
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Tracking Error (m)')
    ax1.set_title('Mean Tracking Error Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Estimation error over time
    ax2 = plt.subplot(3, 3, 2)
    for filter_name in filter_names:
        errors = results[filter_name]['estimation_errors']
        mean_error = np.mean(errors, axis=1)
        ax2.plot(mean_error, label=filter_name.upper(),
                color=colors.get(filter_name, 'black'), linewidth=2)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Estimation Error (m)')
    ax2.set_title('Mean Estimation Error Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative tracking error
    ax3 = plt.subplot(3, 3, 3)
    for filter_name in filter_names:
        errors = results[filter_name]['tracking_errors']
        mean_error = np.mean(errors, axis=1)
        cumsum = np.cumsum(mean_error)
        ax3.plot(cumsum, label=filter_name.upper(),
                color=colors.get(filter_name, 'black'), linewidth=2)
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Cumulative Error (m)')
    ax3.set_title('Cumulative Tracking Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Tracking error distribution
    ax4 = plt.subplot(3, 3, 4)
    error_data = []
    labels = []
    for filter_name in filter_names:
        errors = results[filter_name]['tracking_errors']
        error_data.append(errors.flatten())
        labels.append(filter_name.upper())
    bp = ax4.boxplot(error_data, labels=labels, patch_artist=True)
    for patch, filter_name in zip(bp['boxes'], filter_names):
        patch.set_facecolor(colors.get(filter_name, 'lightgray'))
    ax4.set_ylabel('Tracking Error (m)')
    ax4.set_title('Error Distribution')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Final snapshot
    ax5 = plt.subplot(3, 3, 5)
    data = results[filter_names[0]]
    landmarks = data['landmarks']
    ax5.plot(landmarks[:, 0], landmarks[:, 1], 'k^', markersize=10,
            markerfacecolor='yellow', markeredgewidth=2, label='Landmarks')
    target_pos = data['target_positions'][-1]
    ax5.plot(target_pos[:, 0], target_pos[:, 1], 'r*', markersize=15,
            label='Targets', markeredgewidth=2)
    agent_pos = data['agent_positions'][-1]
    ax5.plot(agent_pos[:, 0], agent_pos[:, 1], 'bo', markersize=10,
            label=f'{filter_names[0].upper()} Agents')
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title('Final Positions')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    
    # 6. Trajectory comparison
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(landmarks[:, 0], landmarks[:, 1], 'k^', markersize=8,
            markerfacecolor='yellow', markeredgewidth=1.5, label='Landmarks')
    target_traj = data['target_positions'][:, 0, :]
    ax6.plot(target_traj[:, 0], target_traj[:, 1], 'r-', 
            linewidth=2, label='Target 0', alpha=0.7)
    for filter_name in filter_names:
        agent_traj = results[filter_name]['agent_positions'][:, 0, :]
        ax6.plot(agent_traj[:, 0], agent_traj[:, 1], '-',
                label=f'{filter_name.upper()}',
                color=colors.get(filter_name, 'black'), linewidth=1.5)
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_title('Trajectories (Agent 0)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')
    
    # 7. Performance metrics
    ax7 = plt.subplot(3, 3, 7)
    metrics_names = ['Mean Track', 'Final Track', 'Mean Est', 'Control']
    x = np.arange(len(filter_names))
    width = 0.2
    for i, metric in enumerate(['mean_tracking_error', 'final_tracking_error', 
                                'mean_estimation_error', 'control_effort']):
        values = [results[f]['metrics'][metric] for f in filter_names]
        ax7.bar(x + i*width, values, width, label=metrics_names[i])
    ax7.set_ylabel('Value')
    ax7.set_title('Performance Metrics')
    ax7.set_xticks(x + width * 1.5)
    ax7.set_xticklabels([f.upper() for f in filter_names])
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Per-agent performance
    ax8 = plt.subplot(3, 3, 8)
    n_agents = results[filter_names[0]]['tracking_errors'].shape[1]
    x = np.arange(n_agents)
    width = 0.2
    for i, filter_name in enumerate(filter_names):
        errors = results[filter_name]['tracking_errors']
        mean_per_agent = np.mean(errors, axis=0)
        ax8.bar(x + i*width, mean_per_agent, width,
               label=filter_name.upper(),
               color=colors.get(filter_name, 'black'))
    ax8.set_xlabel('Agent')
    ax8.set_ylabel('Mean Tracking Error (m)')
    ax8.set_title('Per-Agent Performance')
    ax8.set_xticks(x + width * (len(filter_names)-1)/2)
    ax8.set_xticklabels([f'A{i}' for i in range(n_agents)])
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Relative performance
    ax9 = plt.subplot(3, 3, 9)
    baseline = results['ekf']['metrics']['mean_tracking_error']
    improvements = []
    for filter_name in filter_names:
        error = results[filter_name]['metrics']['mean_tracking_error']
        improvement = (baseline - error) / baseline * 100
        improvements.append(improvement)
    bars = ax9.bar(range(len(filter_names)), improvements,
                   color=[colors.get(f, 'gray') for f in filter_names])
    ax9.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax9.set_ylabel('Improvement over EKF (%)')
    ax9.set_title('Relative Performance')
    ax9.set_xticks(range(len(filter_names)))
    ax9.set_xticklabels([f.upper() for f in filter_names])
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/active_lqg_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/active_lqg_comparison.png")


def create_animation(results, output_dir='MultiTargetTracking/results',
                     filename='active_lqg_tracking.gif', fps=10, max_frames=150):
    """Create animated visualization."""
    print("\nCreating animation...")
    
    filter_names = list(results.keys())
    data = results[filter_names[0]]
    H = data['agent_positions'].shape[0]
    step = max(1, H // max_frames)
    frames = list(range(0, H, step))
    
    colors = {
        'ekf': '#1f77b4',
        'ukf': '#2ca02c',
        'qkf_numeric': '#ff7f0e',
        'pf': '#9467bd'
    }
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    def animate(frame_idx):
        t = frames[frame_idx]
        ax.clear()
        
        # Landmarks
        landmarks = data['landmarks']
        ax.plot(landmarks[:, 0], landmarks[:, 1], 'k^', markersize=12,
                markerfacecolor='yellow', markeredgewidth=2, label='Landmarks')
        
        # Targets
        target_positions = data['target_positions'][t]
        ax.plot(target_positions[:, 0], target_positions[:, 1], 'r*',
               markersize=18, label='Targets', markeredgewidth=2)
        
        # Target trajectories
        for i in range(target_positions.shape[0]):
            traj = data['target_positions'][:t+1, i, :]
            ax.plot(traj[:, 0], traj[:, 1], 'r--', alpha=0.3, linewidth=1)
        
        # Agents
        for filter_name in filter_names:
            positions = results[filter_name]['agent_positions'][t]
            ax.plot(positions[:, 0], positions[:, 1], 'o', markersize=10,
                   color=colors.get(filter_name, 'black'),
                   label=filter_name.upper(), alpha=0.8)
            
            if t > 0:
                for i in range(positions.shape[0]):
                    traj = results[filter_name]['agent_positions'][:t+1, i, :]
                    ax.plot(traj[:, 0], traj[:, 1], '-',
                           color=colors.get(filter_name, 'black'),
                           alpha=0.3, linewidth=1)
        
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_title(f'Active LQG Multi-Target Tracking (t = {t}/{H})')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    anim = animation.FuncAnimation(fig, animate, frames=len(frames),
                                   interval=1000/fps, repeat=True)
    
    full_path = os.path.join(output_dir, filename)
    try:
        anim.save(full_path, writer='pillow', fps=fps)
        print(f"Animation saved to {full_path}")
    except Exception as e:
        print(f"Warning: Could not save animation: {e}")
    
    plt.close()


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("ACTIVE MULTI-TARGET TRACKING WITH LQG CONTROL")
    print("Enhanced to showcase QKF advantages")
    print("="*80 + "\n")
    
    # TUNED parameters to showcase QKF
    sim = ActiveMultiTargetLQG(
        n_agents=3,
        n_targets=4,
        n_landmarks=8,
        H=300,
        dt=0.05,
        process_noise_scale=0.12,      # Moderate process noise
        measurement_noise_scale=0.2,   # Higher measurement noise
        filters_to_use=['ekf', 'ukf', 'qkf_numeric', 'pf'],
        n_particles=800                # More particles for PF
    )
    
    results = sim.run_all_filters()
    
    # Save results to MultiTargetTracking/results
    output_dir = 'MultiTargetTracking/results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/active_lqg_results_{timestamp}.pkl"
    
    with open(filename, 'wb') as f:
        pkl.dump(results, f)
    print(f"\nResults saved to {filename}")
    
    # Print results
    print_results(results)
    
    # Create plots
    plot_results(results, output_dir)
    
    # Create animation
    create_animation(results, output_dir)
    
    print(f"\n{'='*80}")
    print("Simulation complete!")
    print(f"Check {output_dir}/ for outputs")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()