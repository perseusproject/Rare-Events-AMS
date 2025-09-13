"""
Utility functions for visualization and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.colors as mcolors
try:
    from .potentials import create_potential_grid
except ImportError:
    # For standalone execution
    from potentials import create_potential_grid


def plot_potential_field(omega=1.0, x_range=(-2, 2), y_range=(-2, 2), n_points=100,
                         show_regions=True, r_A=0.2, r_B=0.2, ax=None):
    """
    Plot the double well potential field with regions A and B.

    Parameters:
    -----------
    omega : float, optional
        Stiffness parameter, default=1.0
    x_range, y_range : tuple, optional
        Coordinate ranges
    n_points : int, optional
        Number of grid points
    show_regions : bool, optional
        Whether to show regions A and B
    r_A, r_B : float, optional
        Radii for regions A and B
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new figure if None)

    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # Create potential grid
    X, Y, V = create_potential_grid(x_range, y_range, n_points, omega)

    # Plot potential
    contour = ax.contourf(X, Y, V, levels=50, cmap='viridis', alpha=0.8)
    ax.contour(X, Y, V, levels=20, colors='black', alpha=0.3, linewidths=0.5)

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Potential Energy V(x,y)')

    # Mark regions A and B
    if show_regions:
        # Region A (left well)
        circle_A = Circle((-1.0, 0.0), r_A, fill=False, edgecolor='red',
                          linewidth=2, linestyle='--', label='Region A')
        ax.add_patch(circle_A)

        # Region B (right well)
        circle_B = Circle((1.0, 0.0), r_B, fill=False, edgecolor='blue',
                          linewidth=2, linestyle='--', label='Region B')
        ax.add_patch(circle_B)

        ax.legend()

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Double Well Potential (ω={omega})')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_trajectory(trajectory, absorption_state=0, ax=None, **kwargs):
    """
    Plot a single trajectory.

    Parameters:
    -----------
    trajectory : ndarray
        Array of shape (n_steps, 2) containing positions
    absorption_state : int
        0: not absorbed, 1: absorbed in A, 2: absorbed in B
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    **kwargs : additional plotting arguments

    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # Default plotting style
    plot_kwargs = {
        'alpha': 0.7,
        'linewidth': 1.5,
        'marker': 'o',
        'markersize': 3,
        'markevery': 50
    }
    plot_kwargs.update(kwargs)

    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], **plot_kwargs)

    # Mark start and end points
    ax.plot(trajectory[0, 0], trajectory[0, 1],
            'go', markersize=8, label='Start')

    if absorption_state == 1:
        ax.plot(trajectory[-1, 0], trajectory[-1, 1],
                'ro', markersize=8, label='End (A)')
    elif absorption_state == 2:
        ax.plot(trajectory[-1, 0], trajectory[-1, 1],
                'bo', markersize=8, label='End (B)')
    else:
        ax.plot(trajectory[-1, 0], trajectory[-1, 1],
                'ko', markersize=8, label='End')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Particle Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    return fig, ax


def plot_absorption_time_histogram(results, ax=None):
    """
    Plot histogram of absorption times.

    Parameters:
    -----------
    results : dict
        Results from naive MC simulation
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Filter absorption times by region
    times_A = results['a_times']
    times_B = results['b_times']

    # Create histogram
    if len(times_A) > 0 or len(times_B) > 0:
        all_times = np.concatenate([times_A, times_B])
        bins = np.linspace(0, np.max(all_times) + 100, 50)

        if len(times_A) > 0:
            ax.hist(times_A, bins=bins, alpha=0.7, label=f'Absorbed in A (n={len(times_A)})',
                    color='red', density=True)

        if len(times_B) > 0:
            ax.hist(times_B, bins=bins, alpha=0.7, label=f'Absorbed in B (n={len(times_B)})',
                    color='blue', density=True)

        ax.set_xlabel('Absorption Time (steps)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Absorption Times')
        ax.legend()
        ax.grid(True, alpha=0.3)

    return fig, ax


def save_figure(fig, filename, dpi=300, format='png'):
    """
    Save figure to experiments/figures directory.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Output filename (without extension)
    dpi : int, optional
        Resolution, default=300
    format : str, optional
        File format, default='png'
    """
    import os
    os.makedirs('experiments/figures', exist_ok=True)
    filepath = f'experiments/figures/{filename}.{format}'
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', format=format)
    print(f"Figure saved to {filepath}")


def test_visualization():
    """
    Test visualization functions.
    """
    print("Testing visualization functions...")

    # Test potential field plot
    fig1, ax1 = plot_potential_field(omega=1.0)
    plt.show()

    # Generate a test trajectory
    try:
        from .detectors import simulate_until_absorption
    except ImportError:
        # For standalone execution
        from detectors import simulate_until_absorption

    trajectory, absorption_state, _ = simulate_until_absorption(
        -1.0, 0.0, 0.01, 0.5, 1000, omega=1.0
    )

    # Test trajectory plot
    fig2, ax2 = plot_trajectory(trajectory, absorption_state)
    plt.show()

    return fig1, fig2


if __name__ == "__main__":
    test_visualization()
