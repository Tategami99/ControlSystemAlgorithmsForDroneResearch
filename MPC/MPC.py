import numpy as np
from scipy.optimize import minimize

class MPC:
    def __init__(self, dynamics, horizon, n_drones, drone_radius):
        self.dyn = dynamics
        self.N = horizon
        self.n_drones = n_drones
        self.p = dynamics.get_input_size()
        self.coverage_radius = drone_radius

    def objective(self, u_flattened, x0, density_func):
        u_seq = u_flattened.reshape((self.N, self.p))
        total_cost = 0
        state = x0.copy()
        A, B = self.dyn.get_A(), self.dyn.get_B()

        for k in range(self.N):
            # 1. Predict state at step k
            state = A @ state + B @ u_seq[k].reshape(-1, 1) # future state prediction
            drones = state[self.dyn.n1:].flatten()
            positions = [drones[i:i+2] for i in range(0, len(drones), 2)]
            
            # 2. Calculate Predicted Reward
            stage_reward = 0
            for i, pos_i in enumerate(positions):
                val = density_func(pos_i[0], pos_i[1])
                
                # Non-Redundancy Penalty: 
                # If drones move too close to each other in the future, 
                # their perceived reward is slashed.
                penalty = 0
                for j, pos_j in enumerate(positions):
                    if i != j:
                        dist = np.linalg.norm(pos_i - pos_j)
                        if dist < (2 * self.coverage_radius):
                            penalty += val * (1.2 / (dist + 0.4))
                
                stage_reward += (val - penalty)

            # 3. Acceleration Cost (Control Effort)
            effort = 0.05 * np.sum(u_seq[k]**2)
            total_cost += (-25 * stage_reward + effort)
            
        return total_cost

    def solve(self, x_current, density_func):
        u_guess = np.zeros(self.N * self.p)
        bounds = [(-4.0, 4.0)] * (self.N * self.p)
        res = minimize(self.objective, u_guess, args=(x_current, density_func),
                       method='SLSQP', bounds=bounds, 
                       options={'ftol': 1e-3, 'maxiter': 20})
        return res.x[:self.p].reshape(-1, 1)