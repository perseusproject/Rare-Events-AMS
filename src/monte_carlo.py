"""
Naive Monte Carlo simulation for estimating transition probabilities.
"""

import numpy as np
import scipy.stats
from tqdm import tqdm
try:
    from .detectors import simulate_until_absorption
except ImportError:
    # For standalone execution
    from detectors import simulate_until_absorption


def run_single_simulation(x0, y0, dt, beta_inv, max_steps, omega=1.0, r_A=0.2, r_B=0.2):
    """
    Run a single simulation and return absorption results.

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
    absorption_state : int
        0: not absorbed, 1: absorbed in A, 2: absorbed in B
    absorption_time : int
        Step number when absorption occurred
    """
    _, absorption_state, absorption_time = simulate_until_absorption(
        x0, y0, dt, beta_inv, max_steps, omega, r_A, r_B
    )
    return absorption_state, absorption_time


def run_naive_mc(n_simulations, x0, y0, dt, beta_inv, max_steps,
                 omega=1.0, r_A=0.2, r_B=0.2, seed=None):
    """
    Run naive Monte Carlo simulation to estimate transition probability.

    Parameters:
    -----------
    n_simulations : int
        Number of simulations to run
    x0, y0 : float
        Initial position
    dt : float
        Time step
    beta_inv : float
        Inverse temperature
    max_steps : int
        Maximum number of steps per simulation
    omega : float, optional
        Stiffness parameter, default=1.0
    r_A, r_B : float, optional
        Radii for regions A and B, default=0.2
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    results : dict
        Dictionary containing simulation results and statistics
    """
    if seed is not None:
        np.random.seed(seed)

    absorption_states = np.zeros(n_simulations, dtype=int)
    absorption_times = np.zeros(n_simulations, dtype=int)

    # Run simulations with progress bar
    for i in tqdm(range(n_simulations), desc="Running MC simulations"):
        absorption_states[i], absorption_times[i] = run_single_simulation(
            x0, y0, dt, beta_inv, max_steps, omega, r_A, r_B
        )

    # Calculate statistics
    n_A = np.sum(absorption_states == 1)
    n_B = np.sum(absorption_states == 2)
    n_unabsorbed = np.sum(absorption_states == 0)

    p_estimate = n_B / n_simulations if n_simulations > 0 else 0.0

    # Clopper-Pearson confidence interval for binomial proportion
    if n_B > 0 and n_B < n_simulations:
        ci_low, ci_high = scipy.stats.beta.interval(
            0.95, n_B + 1, n_simulations - n_B + 1)
    elif n_B == 0:
        ci_low, ci_high = 0.0, scipy.stats.beta.ppf(0.95, 1, n_simulations + 1)
    elif n_B == n_simulations:
        ci_low, ci_high = scipy.stats.beta.ppf(0.05, n_simulations, 1), 1.0
    else:
        ci_low, ci_high = 0.0, 1.0

    # Filter absorption times for successful transitions
    b_times = absorption_times[absorption_states == 2]
    a_times = absorption_times[absorption_states == 1]

    results = {
        'n_simulations': n_simulations,
        'n_A': n_A,
        'n_B': n_B,
        'n_unabsorbed': n_unabsorbed,
        'p_estimate': p_estimate,
        'ci_95': (ci_low, ci_high),
        'absorption_times': absorption_times,
        'absorption_states': absorption_states,
        'b_times': b_times,
        'a_times': a_times,
        'mean_time_B': np.mean(b_times) if len(b_times) > 0 else np.nan,
        'mean_time_A': np.mean(a_times) if len(a_times) > 0 else np.nan,
        'std_time_B': np.std(b_times) if len(b_times) > 0 else np.nan,
        'std_time_A': np.std(a_times) if len(a_times) > 0 else np.nan,
    }

    return results


def print_results_summary(results):
    """
    Print a summary of Monte Carlo simulation results.

    Parameters:
    -----------
    results : dict
        Results dictionary from run_naive_mc
    """
    print("=" * 50)
    print("NAIVE MONTE CARLO SIMULATION RESULTS")
    print("=" * 50)
    print(f"Number of simulations: {results['n_simulations']}")
    print(
        f"Absorbed in A: {results['n_A']} ({results['n_A']/results['n_simulations']*100:.2f}%)")
    print(
        f"Absorbed in B: {results['n_B']} ({results['n_B']/results['n_simulations']*100:.2f}%)")
    print(
        f"Unabsorbed: {results['n_unabsorbed']} ({results['n_unabsorbed']/results['n_simulations']*100:.2f}%)")
    print()
    print(f"Estimated p = P(τ_B < τ_A) = {results['p_estimate']:.6f}")
    print(
        f"95% Confidence Interval: [{results['ci_95'][0]:.6f}, {results['ci_95'][1]:.6f}]")
    print()

    if len(results['b_times']) > 0:
        print(
            f"Mean absorption time for B: {results['mean_time_B']:.2f} steps")
        print(f"Std absorption time for B: {results['std_time_B']:.2f} steps")

    if len(results['a_times']) > 0:
        print(
            f"Mean absorption time for A: {results['mean_time_A']:.2f} steps")
        print(f"Std absorption time for A: {results['std_time_A']:.2f} steps")

    print("=" * 50)


def test_naive_mc():
    """
    Test the naive Monte Carlo simulation with reasonable parameters.
    """
    print("Testing naive Monte Carlo simulation...")

    # Test parameters - chosen to give measurable transition probability
    n_simulations = 1000
    x0, y0 = -1.0, 0.0  # Start in left well
    dt = 0.01
    beta_inv = 0.5      # Moderate temperature
    max_steps = 10000
    omega = 1.0
    seed = 42

    results = run_naive_mc(n_simulations, x0, y0, dt, beta_inv, max_steps,
                           omega=omega, seed=seed)

    print_results_summary(results)

    return results


if __name__ == "__main__":
    test_naive_mc()
