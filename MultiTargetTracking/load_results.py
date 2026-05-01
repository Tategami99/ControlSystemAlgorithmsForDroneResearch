"""
load_results.py
===============
Load results from pickle file and regenerate visualizations.

Usage:
    python load_results.py results/multi_target_results_TIMESTAMP.pkl
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle as pkl
import sys
import os


def load_results(filename):
    """
    Load results from pickle file.
    
    Args:
        filename: path to .pkl file
    
    Returns:
        results: dict with filter results
    """
    with open(filename, 'rb') as f:
        results = pkl.load(f)
    
    print(f"Loaded results from: {filename}")
    print(f"Filters: {list(results.keys())}")
    
    # Print basic info
    first_filter = list(results.keys())[0]
    H, n_targets, n_states = results[first_filter]['true_states'].shape
    n_landmarks = results[first_filter]['landmarks'].shape[0]
    
    print(f"Horizon: {H} timesteps")
    print(f"Targets: {n_targets}")
    print(f"Landmarks: {n_landmarks}")
    print()
    
    return results


def print_performance_summary(results):
    """
    Print performance summary for all filters.
    
    Args:
        results: dict from load_results()
    """
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for filter_name in results.keys():
        data = results[filter_name]
        true_states = data['true_states']
        estimates = data['estimates']
        
        # Compute position errors
        pos_errors = np.sqrt(
            (true_states[:, :, 0] - estimates[:, :, 0])**2 +
            (true_states[:, :, 2] - estimates[:, :, 2])**2
        )
        
        mean_rmse = np.mean(pos_errors)
        final_rmse = np.mean(pos_errors[-1, :])
        max_rmse = np.max(pos_errors)
        
        print(f"{filter_name.upper():15s} | Mean RMSE: {mean_rmse:.4f} m | "
              f"Final RMSE: {final_rmse:.4f} m | Max RMSE: {max_rmse:.4f} m")
    
    print("=" * 60)
    print()


def plot_comparison(results, output_dir=None):
    """
    Create comparison plots for all filters.
    
    Args:
        results: dict from load_results()
        output_dir: directory to save plots (if None, shows plot)
    """
    filter_names = list(results.keys())
    
    # Color scheme
    colors = {
        'ekf': '#1f77b4',
        'ukf': '#2ca02c',
        'qkf_numeric': '#ff7f0e',
        'pf': '#9467bd'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # 1. Position error over time
    ax = axes[0]
    for filter_name in filter_names:
        data = results[filter_name]
        true_states = data['true_states']
        estimates = data['estimates']
        
        pos_errors = np.sqrt(
            (true_states[:, :, 0] - estimates[:, :, 0])**2 +
            (true_states[:, :, 2] - estimates[:, :, 2])**2
        )
        mean_error = np.mean(pos_errors, axis=1)
        
        ax.plot(mean_error, label=filter_name.upper(), 
                color=colors.get(filter_name, 'black'), linewidth=2)
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('Position RMSE (m)')
    ax.set_title('Position Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Cumulative error
    ax = axes[1]
    for filter_name in filter_names:
        data = results[filter_name]
        true_states = data['true_states']
        estimates = data['estimates']
        
        pos_errors = np.sqrt(
            (true_states[:, :, 0] - estimates[:, :, 0])**2 +
            (true_states[:, :, 2] - estimates[:, :, 2])**2
        )
        mean_error = np.mean(pos_errors, axis=1)
        cumsum = np.cumsum(mean_error)
        
        ax.plot(cumsum, label=filter_name.upper(),
                color=colors.get(filter_name, 'black'), linewidth=2)
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('Cumulative RMSE (m)')
    ax.set_title('Cumulative Position Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Error distribution
    ax = axes[2]
    error_data = []
    labels = []
    for filter_name in filter_names:
        data = results[filter_name]
        true_states = data['true_states']
        estimates = data['estimates']
        
        pos_errors = np.sqrt(
            (true_states[:, :, 0] - estimates[:, :, 0])**2 +
            (true_states[:, :, 2] - estimates[:, :, 2])**2
        )
        error_data.append(pos_errors.flatten())
        labels.append(filter_name.upper())
    
    bp = ax.boxplot(error_data, labels=labels, patch_artist=True)
    for patch, filter_name in zip(bp['boxes'], filter_names):
        patch.set_facecolor(colors.get(filter_name, 'lightgray'))
    
    ax.set_ylabel('Position RMSE (m)')
    ax.set_title('Error Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Trajectory visualization (first target only)
    ax = axes[3]
    filter_name = filter_names[0]
    data = results[filter_name]
    landmarks = data['landmarks']
    
    # Plot landmarks
    ax.plot(landmarks[:, 0], landmarks[:, 1], 'k^', markersize=12, 
            label='Landmarks', markerfacecolor='yellow', markeredgewidth=2)
    
    # Plot true and estimated trajectories
    for i, filter_name in enumerate(filter_names):
        data = results[filter_name]
        true_states = data['true_states']
        estimates = data['estimates']
        
        if i == 0:
            ax.plot(true_states[:, 0, 0], true_states[:, 0, 2], 
                   'k-', linewidth=3, label='True', alpha=0.7)
        
        ax.plot(estimates[:, 0, 0], estimates[:, 0, 2], '--',
               label=f'{filter_name.upper()} Est', 
               color=colors.get(filter_name, 'black'), linewidth=2)
    
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title('Trajectory (Target 1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/filter_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Plots saved to {output_dir}/filter_comparison.png")
    else:
        plt.show()
    
    plt.close()


def create_animation(results, filename='tracking.gif', fps=10):
    """
    Create animated visualization of tracking.
    
    Args:
        results: dict from load_results()
        filename: output filename
        fps: frames per second
    """
    print("Creating animation...")
    
    filter_names = list(results.keys())
    data = results[filter_names[0]]
    landmarks = data['landmarks']
    H = data['true_states'].shape[0]
    n_targets = data['true_states'].shape[1]
    
    # Colors
    colors = {
        'ekf': '#1f77b4',
        'ukf': '#2ca02c',
        'qkf_numeric': '#ff7f0e',
        'pf': '#9467bd'
    }
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def animate(t):
        ax.clear()
        
        # Plot landmarks
        ax.plot(landmarks[:, 0], landmarks[:, 1], 'k^', markersize=15,
                markerfacecolor='yellow', markeredgewidth=2, label='Landmarks')
        
        # Plot each target
        for target_idx in range(n_targets):
            # True position
            true_x = results[filter_names[0]]['true_states'][t, target_idx, 0]
            true_y = results[filter_names[0]]['true_states'][t, target_idx, 2]
            ax.plot(true_x, true_y, 'ko', markersize=12, 
                   markerfacecolor='red', label='True' if target_idx == 0 else '')
            
            # Filter estimates
            for filter_name in filter_names:
                data = results[filter_name]
                est_x = data['estimates'][t, target_idx, 0]
                est_y = data['estimates'][t, target_idx, 2]
                
                label = filter_name.upper() if target_idx == 0 else ''
                ax.plot(est_x, est_y, 'o', markersize=8,
                       color=colors.get(filter_name, 'black'),
                       label=label, alpha=0.7)
                
                # Trajectory history
                if t > 0:
                    hist_x = data['estimates'][:t+1, target_idx, 0]
                    hist_y = data['estimates'][:t+1, target_idx, 2]
                    ax.plot(hist_x, hist_y, '-', 
                           color=colors.get(filter_name, 'black'),
                           alpha=0.3, linewidth=1)
        
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_title(f'Multi-Target Tracking (t = {t})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    anim = animation.FuncAnimation(fig, animate, frames=H, interval=1000/fps)
    anim.save(filename, writer='pillow', fps=fps)
    print(f"Animation saved to {filename}")
    plt.close()


def main():
    """Main execution"""
    if len(sys.argv) < 2:
        print("Usage: python load_results.py <path_to_pkl_file>")
        print("\nExample:")
        print("  python load_results.py results/multi_target_results_20240501_123456.pkl")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        sys.exit(1)
    
    # Load results
    results = load_results(filename)
    
    # Print performance summary
    print_performance_summary(results)
    
    # Get output directory from input filename
    output_dir = os.path.dirname(filename)
    if output_dir == '':
        output_dir = '.'
    
    # Create plots
    plot_comparison(results, output_dir)
    
    # Create animation
    anim_filename = os.path.join(output_dir, 'tracking.gif')
    create_animation(results, anim_filename)
    
    print("\nDone!")
    print(f"Check {output_dir}/ for outputs")


if __name__ == '__main__':
    main()