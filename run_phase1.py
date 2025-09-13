#!/usr/bin/env python3
"""
Run Phase 1: Naive Monte Carlo simulation and generate results.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.monte_carlo import run_naive_mc, print_results_summary
from src.detectors import simulate_until_absorption
from src.utils import plot_trajectory, plot_potential_field, plot_absorption_time_histogram, save_figure
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))


def main():
    print("Running Phase 1: Naive Monte Carlo Simulation")
    print("=" * 60)

    # Create figures directory
    os.makedirs('experiments/figures', exist_ok=True)
    os.makedirs('experiments/results', exist_ok=True)

    # 1. Plot potential field
    print("1. Generating potential field visualization...")
    fig1, ax1 = plot_potential_field(omega=1.0, show_regions=True)
    save_figure(fig1, 'double_well_potential')
    plt.close(fig1)

    # 2. Generate example trajectories
    print("2. Generating example trajectories...")

    # Example trajectory that stays in A
    trajectory_A, absorption_state_A, _ = simulate_until_absorption(
        -1.0, 0.0, 0.01, 0.1, 1000, omega=1.0
    )
    fig2, ax2 = plot_trajectory(trajectory_A, absorption_state_A)
    save_figure(fig2, 'trajectory_stays_in_A')
    plt.close(fig2)

    # Example trajectory that transitions to B (may need to run multiple times)
    found_transition = False
    attempts = 0
    while not found_transition and attempts < 100:
        trajectory_B, absorption_state_B, _ = simulate_until_absorption(
            -1.0, 0.0, 0.01, 0.5, 10000, omega=1.0
        )
        if absorption_state_B == 2:  # Successfully transitioned to B
            found_transition = True
            fig3, ax3 = plot_trajectory(trajectory_B, absorption_state_B)
            save_figure(fig3, 'trajectory_transition_to_B')
            plt.close(fig3)
        attempts += 1

    if not found_transition:
        print(
            "   Could not find a transition to B in 100 attempts (expected for rare events)")

    # 3. Run Monte Carlo simulation
    print("3. Running Monte Carlo simulation...")
    results = run_naive_mc(
        n_simulations=1000,
        x0=-1.0, y0=0.0,
        dt=0.01,
        beta_inv=0.5,
        max_steps=10000,
        omega=1.0,
        seed=42
    )

    # Print results
    print_results_summary(results)

    # 4. Plot absorption time histogram
    print("4. Generating absorption time histogram...")
    fig4, ax4 = plot_absorption_time_histogram(results)
    save_figure(fig4, 'absorption_time_histogram')
    plt.close(fig4)

    # 5. Save results
    print("5. Saving results...")
    import json
    with open('experiments/results/phase1_results.json', 'w') as f:
        json.dump({
            'parameters': {
                'n_simulations': 1000,
                'x0': -1.0, 'y0': 0.0,
                'dt': 0.01,
                'beta_inv': 0.5,
                'max_steps': 10000,
                'omega': 1.0,
                'seed': 42
            },
            'results': {
                'p_estimate': float(results['p_estimate']),
                'ci_95': [float(results['ci_95'][0]), float(results['ci_95'][1])],
                'n_A': int(results['n_A']),
                'n_B': int(results['n_B']),
                'n_unabsorbed': int(results['n_unabsorbed']),
                'mean_time_B': float(results['mean_time_B']),
                'mean_time_A': float(results['mean_time_A'])
            }
        }, f, indent=2)

    print("\nPhase 1 completed successfully!")
    print("Results saved to experiments/results/phase1_results.json")
    print("Figures saved to experiments/figures/")


if __name__ == "__main__":
    main()
