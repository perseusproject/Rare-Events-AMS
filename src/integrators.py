"""
Euler-Maruyama integrator for overdamped Langevin dynamics.
"""

import numpy as np
from numba import jit
try:
    from .potentials import double_well_gradient
except ImportError:
    # For standalone execution
    from potentials import double_well_gradient


def euler_maruyama_step(x, y, dt, beta_inv, omega=1.0):
    """
    Single Euler-Maruyama step for overdamped Langevin dynamics.

    X_{n+1} = X_n - ∇V(X_n)Δt + √(2β⁻¹Δt) G_n

    Parameters:
    -----------
    x, y : float
        Current position coordinates
    dt : float
        Time step
    beta_inv : float
        Inverse temperature (k_B T)
    omega : float, optional
        Stiffness parameter for potential, default=1.0

    Returns:
    --------
    x_new, y_new : float
        New position coordinates
    """
    # Compute gradient of potential
    dV_dx, dV_dy = double_well_gradient(x, y, omega)

    # Generate Gaussian noise
    noise_x = np.random.normal(0, 1)
    noise_y = np.random.normal(0, 1)

    # Compute diffusion coefficient
    diffusion_coeff = np.sqrt(2 * beta_inv * dt)

    # Update position
    x_new = x - dV_dx * dt + diffusion_coeff * noise_x
    y_new = y - dV_dy * dt + diffusion_coeff * noise_y

    return x_new, y_new


def euler_maruyama_trajectory(x0, y0, dt, beta_inv, max_steps, omega=1.0):
    """
    Generate a full trajectory using Euler-Maruyama integration.

    Parameters:
    -----------
    x0, y0 : float
        Initial position
    dt : float
        Time step
    beta_inv : float
        Inverse temperature
    max_steps : int
        Maximum number of steps
    omega : float, optional
        Stiffness parameter, default=1.0

    Returns:
    --------
    trajectory : ndarray
        Array of shape (max_steps+1, 2) containing positions
    """
    trajectory = np.zeros((max_steps + 1, 2))
    trajectory[0, 0] = x0
    trajectory[0, 1] = y0

    x, y = x0, y0
    for i in range(1, max_steps + 1):
        x, y = euler_maruyama_step(x, y, dt, beta_inv, omega)
        trajectory[i, 0] = x
        trajectory[i, 1] = y

    return trajectory


def test_integrator_stability():
    """
    Test the integrator stability for different time steps.
    """
    # Test parameters
    x0, y0 = -1.0, 0.0  # Start near left well
    beta_inv = 0.1      # Temperature parameter
    omega = 1.0
    max_steps = 10000

    dt_values = [0.001, 0.005, 0.01, 0.05]

    print("Testing Euler-Maruyama integrator stability:")
    print(f"Initial position: ({x0}, {y0})")
    print(f"beta_inv = {beta_inv}")
    print(f"omega = {omega}")
    print()

    for dt in dt_values:
        trajectory = euler_maruyama_trajectory(
            x0, y0, dt, beta_inv, max_steps, omega)

        # Check for numerical instability (NaN or extreme values)
        has_nan = np.any(np.isnan(trajectory))
        max_value = np.max(np.abs(trajectory))

        print(f"Δt = {dt:.4f}:")
        print(
            f"  Final position: ({trajectory[-1, 0]:.4f}, {trajectory[-1, 1]:.4f})")
        print(f"  Contains NaN: {has_nan}")
        print(f"  Max absolute value: {max_value:.4f}")

        if max_value > 1000:  # Arbitrary threshold for instability
            print("  WARNING: Potential numerical instability")
        print()


if __name__ == "__main__":
    test_integrator_stability()
