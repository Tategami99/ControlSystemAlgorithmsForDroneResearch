import numpy as np
from scipy.stats import multivariate_normal

class ParticleFilter:
    def __init__(self, n_particles, dynamics, sensor_obj):
        self.n_p = n_particles
        self.dyn = dynamics
        self.sensor = sensor_obj
        self.n = dynamics.get_state_size()
        self.n1 = dynamics.get_earth_state_size()
        
        # Spread particles evenly across the state space (0 to 10 for each dimension)
        self.particles = np.random.uniform(0, 10, (self.n_p, self.n))

        # Each particle starts with equal weight
        self.weights = np.ones(self.n_p) / self.n_p

    def predict(self, u):
        """
        Moves particles based on Ax + Bu + w.
        u: Control input (p, 1)
        """
        A = self.dyn.get_A()
        B = self.dyn.get_B()
        W = self.dyn.get_W() # Covariance matrix (how much uncertainty each state has)

        # Using Cholesky decomposition
        # Finds a lower trianglular matrix L such that L @ L.T = W
        # Basically L is to scale the noise by the uncertainty in W
        L = np.linalg.cholesky(W)
        
        # Vectorized prediction: x_next = x @ A.T + u.T @ B.T + noise
        # This is much faster than a for-loop for 200+ 
        
        # Generate random movement based on noise
        noise = (L @ np.random.standard_normal((self.n, self.n_p))).T

        # Move every particle forward using the same control input sent to the 
        # real drones
        self.particles = (self.particles @ A.T) + (u.T @ B.T) + noise

    def update(self, measurement):
        """
        Weights particles based on the Quadratic Sensor Model.
        measurement: The real sensor reading (m, 1)
        """
        # Measurement Covariance
        V = self.sensor.get_V()

        # Ensure measurement is flat for comparison
        z_real = measurement.flatten()
        
        for i in range(self.n_p):
            p_vec = self.particles[i].reshape(-1, 1)

            # If the drone was at the particle's state, what would the sensor read
            y_pred = self.sensor.measure_pred(p_vec).flatten()
            
            diff = z_real - y_pred
            try:
                # Multidimensional Normal Distribution with 0 distance from
                # the real measurement being the highest
                prob = multivariate_normal.pdf(diff, mean=np.zeros(len(diff)), cov=V)
                self.weights[i] = prob
            except:
                # If it's too far off, make sure weight isn't 0
                self.weights[i] = 1e-300

        self.weights += 1e-300 # Avoid numerical zero
        self.weights /= np.sum(self.weights)

    def resample(self):
        """
        Duplicate particles with higher weights; discard those with low weights
        """
        indices = np.random.choice(np.arange(self.n_p), size=self.n_p, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_p) / self.n_p

    def get_estimate(self):
        """Returns the weighted average state"""
        return np.average(self.particles, axis=0, weights=self.weights).reshape(-1, 1)