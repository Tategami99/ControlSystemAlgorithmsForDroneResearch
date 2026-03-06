import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from state_dynamics import StateDynamics, sensor
from ParticleFilter.particle_filter import ParticleFilter
from ParticleFilter.particle_filter_analyzer import ParticleFilterAnalyzer

def density_map(x, y):
    """3 Peaks: High (8,8), Medium (2,3), and Mid-High (7,2)"""
    p1 = 4.0 * np.exp(-((x-2)**2 + (y-3)**2) / 3.0)
    p2 = 5.0 * np.exp(-((x-8)**2 + (y-8)**2) / 4.0)
    p3 = 3.5 * np.exp(-((x-7)**2 + (y-2)**2) / 2.5)
    return p1 + p2 + p3

def run_estimation_sim(steps=50, n_drones=5, visualize=True, save_gif=True, analyze=True):
    # Dimensions: n1 earth states, n2 sensor/drone states
    n1, n2, p = 2, n_drones * 2, n_drones * 2
    dt = 0.1
    
    # Dynamics and Sensor setup
    W = np.eye(n1 + n2) * 0.005 # Process noise
    dyn = StateDynamics(n1, n2, p, W, np.eye(n1), np.eye(n2), np.eye(n2)*dt)
    
    # Quadratic Sensor Setup: Measure drone positions directly for this test
    C = np.zeros((n_drones*2, n1+n2))
    for i in range(n_drones*2):
        C[i, n1 + i] = 1.0 
    M = np.zeros((n_drones*2, n1+n2, n1+n2))
    V = np.eye(n_drones*2) * 0.05 # Measurement noise
    sns = sensor(C, M, V)

    # Initialize drones and Particle Filter
    dyn.x[n1:] = np.random.uniform(2, 8, (n2, 1))
    pf = ParticleFilter(n_particles=300, dynamics=dyn, sensor_obj=sns)

    if analyze:
        analyzer = ParticleFilterAnalyzer(n_drones, n1, pf.n_p, steps, dt)

    # Environment for background
    res_grid = 100
    gx, gy = np.mgrid[0:10:complex(res_grid), 0:10:complex(res_grid)]
    gz = density_map(gx, gy)

    # Visualization Setup (Standard from your MPC code)
    if visualize or save_gif:
        fig, ax = plt.subplots(figsize=(10, 8))
        if visualize: plt.ion()
        else: plt.ioff()

    frames = []

    for t in range(steps):
        # 1. Control Input
        # TODO: Use the mpc controller as the input
        # For now, move drones towards the top-right peak (8,8)
        u = (np.tile([8.0, 8.0], n_drones).reshape(-1, 1) - dyn.x[n1:]) * 0.1
        
        # 2. Move System & Measure
        dyn.set_u(u)
        dyn.forward()
        x_true = dyn.get_x()
        z = sns.measure(x_true)
        
        # 3. Particle Filter Step
        pf.predict(u)
        pf.update(z)
        pf.resample()
        x_est = pf.get_estimate()

        if analyze:
            analyzer.update(t, x_true, x_est, pf.particles, pf.weights, z)

        # 4. Visualization (Following your MPC style)
        if visualize or save_gif:
            ax.clear()
            ax.contourf(gx, gy, gz, cmap='viridis', alpha=0.6)
            
            # Plot Particles for Drone 1 (Black cloud)
            ax.scatter(pf.particles[:, n1], pf.particles[:, n1+1], 
                       color='black', s=2, alpha=0.1, label='Particles')
            
            # Plot All Drones
            for i in range(n_drones):
                ix, iy = n1 + i*2, n1 + i*2 + 1
                # True Position (Red)
                ax.scatter(x_true[ix], x_true[iy], color='red', s=100, edgecolors='black', zorder=5)
                # Estimate Position (Blue X)
                ax.scatter(x_est[ix], x_est[iy], color='blue', marker='x', s=150, zorder=6)
                
            ax.set_title(f"Step: {t+1}/{steps} | Particle Filter Tracking\nRed=True, BlueX=Estimate")
            ax.set_xlim(0, 10); ax.set_ylim(0, 10)
            
            if visualize:
                plt.pause(0.01)

            if save_gif:
                fig.canvas.draw()
                image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(np.uint8)
                frames.append(image.copy())

    if save_gif and frames:
        print("Saving GIF...")
        imageio.mimsave('./ParticleFilter/results/drone_estimation.gif', frames, fps=10)
        print("GIF Saved.")

    if visualize: plt.ioff(); plt.show()
    else: plt.close()

    if analyze:
        analyzer.generate_all_plots()
        analyzer.print_summary()

if __name__ == "__main__":
    run_estimation_sim(steps=50, n_drones=5, visualize=True)