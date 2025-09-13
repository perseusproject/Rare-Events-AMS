"""
Region detection functions for absorption in sets A and B.
"""

import numpy as np


def in_region_A(x, y, r_A=0.2):
    """
    Check if point (x,y) is in region A.
    A = {x < -0.8 ∧ y² ≤ r_A²}

    Parameters:
    -----------
    x, y : float
        Coordinates
    r_A : float, optional
        Radius for y-direction, default=0.2

    Returns:
    --------
    bool : True if point is in region A
    """
    return (x < -0.8) and (y**2 <= r_A**2)


def in_region_B(x, y, r_B=0.2):
    """
    Check if point (x,y) is in region B.
    B = {x > 0.8 ∧ y² ≤ r_B²}

    Parameters:
    -----------
    x, y : float
        Coordinates
    r_B : float, optional
        Radius for y-direction, default=0.2

    Returns:
    --------
    bool : True if point is in region B
    """
    return (x > 0.8) and (y**2 <= r_B**2)


def check_absorption(x, y, r_A=0.2, r_B=0.2):
    """
    Check if point has been absorbed in region A or B.

    Parameters:
    -----------
    x, y : float
        Coordinates
    r_A, r_B : float, optional
        Radii for regions A and B, default=0.2

    Returns:
    --------
    absorption_state : int
        0: not absorbed
        1: absorbed in A
        2: absorbed in B
    """
    if in_region_A(x, y, r_A):
        return 1
    elif in_region_B(x, y, r_B):
        return 2
    else:
        return 0


def simulate_until_absorption(x0, y0, dt, beta_inv, max_steps, omega=1.0, r_A=0.2, r_B=0.2):
    """
    Simulate trajectory until absorption in A or B, or maximum steps reached.

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
    r_A, r_B : float, optional
        Radii for regions A and B, default=0.2

    Returns:
    --------
    trajectory : ndarray
        Array of positions until absorption
    absorption_state : int
        0: not absorbed (max steps reached)
        1: absorbed in A
        2: absorbed in B
    absorption_time : int
        Step number when absorption occurred (0 if not absorbed)
    """
    try:
        from .integrators import euler_maruyama_step
    except ImportError:
        # For standalone execution
        from integrators import euler_maruyama_step

    # Pre-allocate trajectory array
    trajectory = np.zeros((max_steps + 1, 2))
    trajectory[0, 0] = x0
    trajectory[0, 1] = y0

    x, y = x0, y0
    absorption_state = 0
    absorption_time = 0

    for step in range(1, max_steps + 1):
        x, y = euler_maruyama_step(x, y, dt, beta_inv, omega)
        trajectory[step, 0] = x
        trajectory[step, 1] = y

        # Check for absorption
        absorption_state = check_absorption(x, y, r_A, r_B)
        if absorption_state != 0:
            absorption_time = step
            break

    # Trim trajectory to actual length
    if absorption_time > 0:
        trajectory = trajectory[:absorption_time + 1]
    else:
        trajectory = trajectory[:max_steps + 1]

    return trajectory, absorption_state, absorption_time


def test_region_detection():
    """
    Test the region detection functions.
    """
    print("Testing region detection functions:")
    print()

    # Test points in different regions
    test_points = [
        (-1.0, 0.0, "Should be in A"),
        (-1.0, 0.1, "Should be in A"),
        (-1.0, 0.3, "Should NOT be in A (y too large)"),
        (1.0, 0.0, "Should be in B"),
        (1.0, 0.1, "Should be in B"),
        (1.0, 0.3, "Should NOT be in B (y too large)"),
        (0.0, 0.0, "Should NOT be absorbed"),
        (-0.5, 0.0, "Should NOT be absorbed"),
        (0.5, 0.0, "Should NOT be absorbed"),
    ]

    for x, y, description in test_points:
        in_a = in_region_A(x, y)
        in_b = in_region_B(x, y)
        absorption = check_absorption(x, y)

        print(f"Point ({x:.1f}, {y:.1f}): {description}")
        print(f"  In A: {in_a}, In B: {in_b}, Absorption state: {absorption}")
        print()


if __name__ == "__main__":
    test_region_detection()
