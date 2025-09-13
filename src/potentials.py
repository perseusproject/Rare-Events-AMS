"""
Double well potential and gradient functions for rare event simulation.
"""

import numpy as np


def double_well_potential(x, y, omega=1.0):
    """
    Double well potential: V(x,y) = ¼(x²-1)² + ω/2 y²

    Parameters:
    -----------
    x, y : float or array-like
        Coordinates
    omega : float, optional
        Stiffness parameter for y-direction, default=1.0

    Returns:
    --------
    V : float or array-like
        Potential energy
    """
    return 0.25 * (x**2 - 1)**2 + (omega / 2) * y**2


def double_well_gradient(x, y, omega=1.0):
    """
    Gradient of double well potential: ∇V(x,y) = [x(x²-1), ωy]

    Parameters:
    -----------
    x, y : float or array-like
        Coordinates
    omega : float, optional
        Stiffness parameter for y-direction, default=1.0

    Returns:
    --------
    gradV : tuple or array-like
        Gradient components (dV/dx, dV/dy)
    """
    dV_dx = x * (x**2 - 1)  # dV/dx = x(x²-1)
    dV_dy = omega * y        # dV/dy = ωy
    return dV_dx, dV_dy


def double_well_hessian(x, y, omega=1.0):
    """
    Hessian matrix of double well potential.

    Parameters:
    -----------
    x, y : float or array-like
        Coordinates
    omega : float, optional
        Stiffness parameter for y-direction, default=1.0

    Returns:
    --------
    H : array-like
        Hessian matrix [[d²V/dx², d²V/dxdy], [d²V/dydx, d²V/dy²]]
    """
    d2V_dx2 = 3*x**2 - 1    # d²V/dx² = 3x² - 1
    d2V_dxdy = 0.0          # d²V/dxdy = 0
    d2V_dydx = 0.0          # d²V/dydx = 0
    d2V_dy2 = omega         # d²V/dy² = ω

    return np.array([[d2V_dx2, d2V_dxdy],
                     [d2V_dydx, d2V_dy2]])


def create_potential_grid(x_range=(-2, 2), y_range=(-2, 2), n_points=100, omega=1.0):
    """
    Create a grid of potential values for visualization.

    Parameters:
    -----------
    x_range, y_range : tuple, optional
        Range for x and y coordinates
    n_points : int, optional
        Number of points in each dimension
    omega : float, optional
        Stiffness parameter

    Returns:
    --------
    X, Y, V : array-like
        Meshgrid coordinates and potential values
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    V = double_well_potential(X, Y, omega)
    return X, Y, V


if __name__ == "__main__":
    # Test the potential functions
    print("Testing double well potential functions:")
    print(f"V(0, 0) = {double_well_potential(0, 0):.4f}")
    print(f"V(1, 0) = {double_well_potential(1, 0):.4f}")
    print(f"V(-1, 0) = {double_well_potential(-1, 0):.4f}")

    grad = double_well_gradient(0.5, 0.3)
    print(f"∇V(0.5, 0.3) = ({grad[0]:.4f}, {grad[1]:.4f})")
