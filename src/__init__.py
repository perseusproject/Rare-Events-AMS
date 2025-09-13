"""
Rare Event Simulation Package - Phase 1: Naive Monte Carlo

This package implements tools for simulating rare events in double well potential systems
using Euler-Maruyama integration and naive Monte Carlo methods.
"""

from .potentials import (
    double_well_potential,
    double_well_gradient,
    double_well_hessian,
    create_potential_grid
)

from .integrators import (
    euler_maruyama_step,
    euler_maruyama_trajectory,
    test_integrator_stability
)

from .detectors import (
    in_region_A,
    in_region_B,
    check_absorption,
    simulate_until_absorption,
    test_region_detection
)

from .monte_carlo import (
    run_single_simulation,
    run_naive_mc,
    print_results_summary,
    test_naive_mc
)

from .utils import (
    plot_potential_field,
    plot_trajectory,
    plot_absorption_time_histogram,
    save_figure,
    test_visualization
)

__version__ = "0.1.0"
__author__ = "Thibaud Montagne"
__description__ = "Rare Event Simulation with Adaptive Multilevel Splitting"
