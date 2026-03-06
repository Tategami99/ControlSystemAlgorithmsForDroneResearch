import numpy as np
import matplotlib.pyplot as plt
from state_dynamics import StateDynamics
from MPC.MPC import MPC
from scipy.optimize import differential_evolution
from matplotlib.animation import FuncAnimation, PillowWriter
import imageio

def density_map(x, y):
    """3 Peaks: High (8,8), Medium (2,3), and Mid-High (7,2)"""
    p1 = 4.0 * np.exp(-((x-2)**2 + (y-3)**2) / 3.0)
    p2 = 5.0 * np.exp(-((x-8)**2 + (y-8)**2) / 4.0)
    p3 = 3.5 * np.exp(-((x-7)**2 + (y-2)**2) / 2.5)
    return p1 + p2 + p3

def calculate_current_coverage(drone_positions, gx, gy, gz, radius=1.0):
    """Calculates Union of density covered by multiple drones."""
    covered_mask = np.zeros(gz.shape, dtype=bool)
    for i in range(0, len(drone_positions), 2):
        dx, dy = drone_positions[i], drone_positions[i+1]
        dist_sq = (gx - dx)**2 + (gy - dy)**2
        covered_mask |= (dist_sq <= radius**2)
    return np.sum(gz[covered_mask])

def calculate_redundant_coverage(drone_positions, gx, gy, gz, radius=1.0):
    """Calculates overlap/redundant coverage (areas covered by multiple drones)."""
    coverage_count = np.zeros(gz.shape)
    for i in range(0, len(drone_positions), 2):
        dx, dy = drone_positions[i], drone_positions[i+1]
        dist_sq = (gx - dx)**2 + (gy - dy)**2
        coverage_count += (dist_sq <= radius**2).astype(int)
    
    # Areas covered by more than 1 drone
    redundant_mask = coverage_count > 1
    return np.sum(gz[redundant_mask])

def find_max_theoretical_coverage(n_drones, gx, gy, gz, radius=1.0):
    """Finds the absolute best 5 positions for stationary drones."""
    print(f"Calculating best possible 5-drone placement for this map...")
    def objective(p):
        return -calculate_current_coverage(p, gx, gy, gz, radius)
    bounds = [(0, 10), (0, 10)] * n_drones
    # Differential Evolution finds global max without needing a good initial guess
    res = differential_evolution(objective, bounds, popsize=5, tol=0.05)
    return -res.fun

def run_simulation(steps=50, n_drones=5, drone_radius=1.0, visualize=True, save_gif=True):
    # Dimensions
    n1, n2, p = 2, n_drones * 2, n_drones * 2
    dt = 0.1
    
    # Init Dynamics & MPC
    dyn = StateDynamics(n1, n2, p, np.eye(n1+n2)*0.005, np.eye(n1), np.eye(n2), np.eye(n2)*dt)
    controller = MPC(dyn, horizon=4, n_drones=n_drones, drone_radius=drone_radius)
    
    # Start drones in a cluster (forces them to coordinate and spread)
    initial_x = np.zeros((n1 + n2, 1))
    initial_x[n1:] = np.random.uniform(3, 7, (n2, 1))
    dyn.x = initial_x

    # Environment Prep
    res_grid = 100
    gx, gy = np.mgrid[0:10:complex(res_grid), 0:10:complex(res_grid)]
    gz = density_map(gx, gy)
    
    # Benchmark: Max Coverage Possible with 5 Drones
    max_n_drone_coverage = find_max_theoretical_coverage(n_drones, gx, gy, gz, radius=drone_radius)

    if visualize or save_gif:
        fig, ax = plt.subplots(figsize=(10, 8))
        if visualize:
            plt.ion()
        else:
            plt.ioff()

    #to store frames for GIF
    frames = []

    for t in range(steps):
        # 1. Solve MPC
        x_now = dyn.get_x()
        u_opt = controller.solve(x_now, density_map)
        
        # 2. Move Drones
        dyn.set_u(u_opt)
        dyn.forward()
        
        # 3. Calculate Real Coverage (Non-Redundant)
        current_drones = dyn.get_x()[n1:].flatten()
        covered_val = calculate_current_coverage(current_drones, gx, gy, gz, radius=1.0)
        coverage_percent = (covered_val / max_n_drone_coverage) * 100

        if visualize or save_gif:
            ax.clear()
            ax.contourf(gx, gy, gz, cmap='viridis', alpha=0.6)
            for i in range(0, len(current_drones), 2):
                dx, dy = current_drones[i], current_drones[i+1]
                ax.scatter(dx, dy, color='red', s=100, edgecolors='black', zorder=5)
                circle = plt.Circle((dx, dy), 1.0, color='red', fill=False, linestyle='--', alpha=0.3)
                ax.add_patch(circle)

            title_str = (f"Step: {t+1}/{steps} | Drones: {n_drones}\n"
                         f"Coverage: {covered_val:.1f} / {max_n_drone_coverage:.1f} Max ({coverage_percent:.1f}%)")
            ax.set_title(title_str)
            ax.set_xlim(0, 10); ax.set_ylim(0, 10)
            if visualize:
                plt.pause(0.01)

            if save_gif:
                fig.canvas.draw()
                image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(np.uint8)
                frames.append(image.copy())

    if save_gif and frames:
        print("Saving GIF...")
        imageio.mimsave('./MPC/results/drone_coverage.gif', frames, fps=10)
        print("GIF Saved.")

    print("Simulation Finished.")
    if visualize:
        plt.ioff()
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    run_simulation(steps=50, n_drones=5, drone_radius=1.0, visualize=True)