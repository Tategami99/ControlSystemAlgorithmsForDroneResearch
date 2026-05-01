"""
filters.py
==========
Filter implementations extracted from LQG_QKF.py and adapted for multi-target tracking
with range-bearing sensors.

Contains: EKF, UKF, QKF (numerical via iLQR-style iteration), PF
Based on the LQG_QKF.py implementation by the user.
"""

import numpy as np
from state_dynamics import StateDynamics, sensor, Vec, invVec


class RangeBearingSensor:
    """
    Range-bearing sensor compatible with your sensor class structure.
    Measures range and bearing to fixed landmarks.
    
    This creates a 'fake' sensor object that works like your quadratic sensor
    but handles range-bearing measurements via measure_pred() and g() methods.
    """
    
    def __init__(self, landmarks: np.ndarray, noise_scale: float):
        """
        Args:
            landmarks: (n_landmarks, 2) array of [x, y] positions
            noise_scale: scale for measurement noise
        """
        self.landmarks = landmarks  # (n_landmarks, 2)
        self.n_landmarks = landmarks.shape[0]
        self.m = 2 * self.n_landmarks  # range + bearing per landmark
        self.n = 4  # state dimension [px, vx, py, vy]
        
        # Measurement noise covariance
        range_std = noise_scale
        bearing_std = 0.1 * noise_scale
        noise_diag = np.tile([range_std**2, bearing_std**2], self.n_landmarks)
        self.V = np.diag(noise_diag)
    
    def measure(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement with noise: y = h(x) + v
        Compatible with your sensor.measure(x) interface.
        """
        x = np.asarray(x).reshape(-1)
        px, py = x[0], x[2]
        measurements = []
        
        for lx, ly in self.landmarks:
            dx = px - lx
            dy = py - ly
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            measurements.extend([r, theta])
        
        y = np.array(measurements).reshape(-1, 1)
        
        # Add noise
        D = np.linalg.cholesky(self.V)
        rng_noise = np.random.default_rng()
        noise = D @ rng_noise.standard_normal((self.m, 1))
        return y + noise
    
    def measure_pred(self, x: np.ndarray) -> np.ndarray:
        """
        Predicted measurement (no noise): y = h(x)
        Compatible with your sensor.measure_pred(x) interface.
        """
        x = np.asarray(x).reshape(-1)
        px, py = x[0], x[2]
        measurements = []
        
        for lx, ly in self.landmarks:
            dx = px - lx
            dy = py - ly
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            measurements.extend([r, theta])
        
        return np.array(measurements).reshape(-1, 1)
    
    def g(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian H = dh/dx for EKF.
        Compatible with your sensor.g(x) interface which returns the measurement Jacobian.
        """
        x = np.asarray(x).reshape(-1)
        px, py = x[0], x[2]
        H = np.zeros((self.m, 4))
        
        for i, (lx, ly) in enumerate(self.landmarks):
            dx = px - lx
            dy = py - ly
            r = np.sqrt(dx**2 + dy**2)
            r_safe = max(r, 1e-6)  # prevent division by zero
            
            # Range derivatives: dr/dpx, dr/dpy
            dr_dpx = dx / r_safe
            dr_dpy = dy / r_safe
            
            # Bearing derivatives: dtheta/dpx, dtheta/dpy
            dtheta_dpx = -dy / (r_safe**2)
            dtheta_dpy = dx / (r_safe**2)
            
            # Fill Jacobian: [dr/dpx, dr/dvx, dr/dpy, dr/dvy]
            #                 [dtheta/dpx, dtheta/dvx, dtheta/dpy, dtheta/dvy]
            H[2*i, :] = [dr_dpx, 0, dr_dpy, 0]
            H[2*i+1, :] = [dtheta_dpx, 0, dtheta_dpy, 0]
        
        return H


def update_lqe_ekf(F: StateDynamics, sensor: RangeBearingSensor, x_hat, P_est):
    """
    EKF update - extracted from LQG_QKF.py update_lqe_ekf()
    
    Args:
        F: StateDynamics object
        sensor: RangeBearingSensor object
        x_hat: current state estimate (n, 1)
        P_est: current covariance estimate (n, n)
    
    Returns:
        x_hat_new: updated state estimate (n, 1)
        P_est_new: updated covariance (n, n)
        K: Kalman gain
    """
    mu = F.B @ F.u
    Phi = F.A
    Sigma = F.W
    
    # State prediction
    X_pred = mu + Phi @ x_hat
    P_pred = Phi @ P_est @ Phi.T + Sigma
    
    # Measurement prediction
    Y_pred = sensor.measure_pred(X_pred)
    g = sensor.g(X_pred)
    M = g @ P_pred @ g.T + sensor.V
    
    # Kalman gain
    K = P_pred @ g.T @ np.linalg.inv(M)
    
    # Measurement
    Y_meas = sensor.measure(F.get_x())
    innov = Y_meas - Y_pred
    
    # Wrap angles in innovation (every other measurement starting from index 1)
    for i in range(1, len(innov), 2):
        innov[i] = np.arctan2(np.sin(innov[i]), np.cos(innov[i]))
    
    # State update
    x_hat_new = X_pred + K @ innov
    P_est_new = P_pred - K @ M @ K.T
    
    return x_hat_new, P_est_new, K


def update_lqe_ukf(F: StateDynamics, sensor: RangeBearingSensor, x_hat, P_est, 
                   alpha=0.5, beta=2, kappa=0):
    """
    UKF update - extracted from LQG_QKF.py update_lqe_ukf()
    
    Args:
        F: StateDynamics object
        sensor: RangeBearingSensor object
        x_hat: current state estimate (n, 1)
        P_est: current covariance estimate (n, n)
        alpha, beta, kappa: UKF parameters
    
    Returns:
        x_hat_new: updated state estimate (n, 1)
        P_est_new: updated covariance (n, n)
        K: Kalman gain
    """
    n = x_hat.shape[0]
    lambda_ = alpha**2 * (n + kappa) - n
    
    # Compute sigma points
    sigma_points = np.zeros((2 * n + 1, n))
    sigma_points[0] = x_hat.flatten()
    
    # Cholesky decomposition for numerical stability
    try:
        sqrt_P = np.linalg.cholesky((n + lambda_) * P_est)
    except np.linalg.LinAlgError:
        # If not positive definite, use eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(P_est)
        eigenvals = np.maximum(eigenvals, 1e-8)
        sqrt_P = eigenvecs @ np.diag(np.sqrt(eigenvals))
        sqrt_P = np.sqrt(n + lambda_) * sqrt_P
    
    for i in range(n):
        sigma_points[i + 1] = x_hat.flatten() + sqrt_P[i]
        sigma_points[n + i + 1] = x_hat.flatten() - sqrt_P[i]
    
    # Predict sigma points through state dynamics
    sigma_points_pred = np.zeros_like(sigma_points)
    for i in range(2 * n + 1):
        x_pred = F.A @ sigma_points[i].reshape(-1, 1) + F.B @ F.u
        sigma_points_pred[i] = x_pred.flatten()
    
    # Compute state mean
    weights_mean = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
    weights_mean[0] = lambda_ / (n + lambda_)
    x_predicted = np.sum(weights_mean[:, np.newaxis] * sigma_points_pred, axis=0).reshape(-1, 1)
    
    # Compute state covariance
    weights_cov = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
    weights_cov[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
    sigma_0 = F.W.copy()
    for i in range(2 * n + 1):
        diff = sigma_points_pred[i] - x_predicted.flatten()
        sigma_0 += weights_cov[i] * np.outer(diff, diff)
    
    # Predict measurements using sigma points
    sigma_points_meas = np.zeros((2 * n + 1, sensor.m))
    for i in range(2 * n + 1):
        sigma_points_meas[i] = sensor.measure_pred(sigma_points_pred[i].reshape(-1, 1)).flatten()
    
    # Predict measurement mean
    y_predicted = np.sum(weights_mean[:, np.newaxis] * sigma_points_meas, axis=0).reshape(-1, 1)
    
    # Predict measurement covariance
    S = sensor.V.copy()
    for i in range(2 * n + 1):
        diff = sigma_points_meas[i] - y_predicted.flatten()
        S += weights_cov[i] * np.outer(diff, diff)
    
    # Cross covariance
    C_tilde = np.zeros((n, sensor.m))
    for i in range(2 * n + 1):
        diff_state = sigma_points_pred[i] - x_predicted.flatten()
        diff_meas = sigma_points_meas[i] - y_predicted.flatten()
        C_tilde += weights_cov[i] * np.outer(diff_state, diff_meas)
    
    # Kalman gain
    K = C_tilde @ np.linalg.pinv(S)
    
    # Measurement
    y = sensor.measure(F.get_x())
    delta_y = y - y_predicted
    
    # Wrap angles
    for i in range(1, len(delta_y), 2):
        delta_y[i] = np.arctan2(np.sin(delta_y[i]), np.cos(delta_y[i]))
    
    # Update the state estimate
    x_hat_new = x_predicted + K @ delta_y
    
    # Update the covariance estimate
    P_est_new = sigma_0 - K @ S @ K.T
    
    return x_hat_new, P_est_new, K


def update_lqe_qkf_numeric(F: StateDynamics, sensor: RangeBearingSensor, x_hat, P_est, 
                            max_iter=10):
    """
    QKF (numerical) - iterative refinement similar to iLQR approach in LQG_QKF.py
    
    This uses a numerical optimization approach for the measurement update,
    iteratively refining the state estimate to minimize the innovation.
    
    Args:
        F: StateDynamics object
        sensor: RangeBearingSensor object
        x_hat: current state estimate (n, 1)
        P_est: current covariance estimate (n, n)
        max_iter: maximum iterations for refinement
    
    Returns:
        x_hat_new: updated state estimate (n, 1)
        P_est_new: updated covariance (n, n)
        K: Kalman gain
    """
    n = x_hat.shape[0]
    
    # Prediction step (same as EKF)
    mu = F.B @ F.u
    Phi = F.A
    Sigma = F.W
    X_pred = mu + Phi @ x_hat
    P_pred = Phi @ P_est @ Phi.T + Sigma
    
    # Measurement
    Y_meas = sensor.measure(F.get_x())
    
    # Iterative refinement of the measurement update
    x_current = X_pred.copy()
    
    for iteration in range(max_iter):
        # Predicted measurement at current estimate
        Y_pred = sensor.measure_pred(x_current)
        H = sensor.g(x_current)
        
        # Innovation with angle wrapping
        innov = Y_meas - Y_pred
        for i in range(1, len(innov), 2):
            innov[i] = np.arctan2(np.sin(innov[i]), np.cos(innov[i]))
        
        # Kalman gain
        S = H @ P_pred @ H.T + sensor.V + np.eye(sensor.m) * 1e-6
        K = P_pred @ H.T @ np.linalg.pinv(S)
        
        # Update
        x_new = X_pred + K @ innov
        
        # Check convergence
        if np.linalg.norm(x_new - x_current) < 1e-4:
            break
        
        x_current = x_new
    
    # Final covariance update
    Y_pred = sensor.measure_pred(x_current)
    H = sensor.g(x_current)
    S = H @ P_pred @ H.T + sensor.V
    K = P_pred @ H.T @ np.linalg.pinv(S)
    P_est_new = P_pred - K @ S @ K.T
    
    return x_current, P_est_new, K


def update_lqe_pf(F: StateDynamics, sensor: RangeBearingSensor, particles, weights, 
                  n_particles=500):
    """
    Particle Filter update - extracted from LQG_QKF.py update_lqe_pf()
    
    Args:
        F: StateDynamics object
        sensor: RangeBearingSensor object
        particles: (n_particles, n) current particles
        weights: (n_particles,) current weights
        n_particles: number of particles
    
    Returns:
        particles_new: updated particles (n_particles, n)
        weights_new: updated weights (n_particles,)
        x_hat: mean estimate (n, 1)
        P_est: covariance estimate (n, n)
    """
    n = F.n
    A, B, W = F.A, F.B, F.W
    u = F.get_u()
    
    # Predict: propagate particles through dynamics
    particles_pred = (A @ particles.T).T + (B @ u).flatten()
    chol_W = np.linalg.cholesky(W + np.eye(n) * 1e-10)
    particles_pred += (chol_W @ np.random.randn(n, n_particles)).T
    
    # Measurement
    y = sensor.measure(F.get_x())
    y = y.flatten()
    V = sensor.V
    
    # Log-likelihood for each particle
    log_like = np.zeros(n_particles)
    for i in range(n_particles):
        h_i = sensor.measure_pred(particles_pred[i].reshape(-1, 1)).flatten()
        diff = y - h_i
        
        # Wrap angles
        for j in range(1, len(diff), 2):
            diff[j] = np.arctan2(np.sin(diff[j]), np.cos(diff[j]))
        
        log_like[i] = -0.5 * diff @ np.linalg.solve(V, diff)
    
    # Weight update (log-domain for stability)
    log_w = np.log(weights + 1e-300) + log_like
    log_w -= log_w.max()
    weights_new = np.exp(log_w)
    weights_new /= weights_new.sum()
    
    # Resample if effective sample size is low
    n_eff = 1.0 / np.sum(weights_new**2)
    if n_eff < n_particles / 2:
        particles_new, weights_new = _resample_systematic(particles_pred, weights_new)
    else:
        particles_new = particles_pred
    
    # Compute mean and covariance
    x_hat = np.mean(particles_new, axis=0).reshape(-1, 1)
    diff = particles_new - x_hat.flatten()
    P_est = (diff.T @ diff) / n_particles + np.eye(n) * 1e-6
    
    return particles_new, weights_new, x_hat, P_est


def _resample_systematic(particles, weights):
    """
    Systematic resampling - from LQG_QKF.py
    
    Args:
        particles: (n_particles, n) array
        weights: (n_particles,) array
    
    Returns:
        resampled_particles: (n_particles, n) array
        uniform_weights: (n_particles,) array of 1/n_particles
    """
    n_particles = len(weights)
    positions = (np.arange(n_particles) + np.random.rand()) / n_particles
    cumsum = np.cumsum(weights)
    
    i, j = 0, 0
    indices = np.zeros(n_particles, dtype=int)
    
    while i < n_particles:
        if positions[i] < cumsum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    
    return particles[indices], np.ones(n_particles) / n_particles