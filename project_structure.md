# Project Structure - Rare Event Simulation with AMS

## Directory Structure
```
EA_Recherche/
├── src/
│   ├── __init__.py
│   ├── potentials.py          # Potential functions (double well, etc.)
│   ├── integrators.py         # Euler-Maruyama and other integrators
│   ├── detectors.py           # Region detection (A, B absorption)
│   ├── monte_carlo.py         # Naive MC simulation
│   └── utils.py               # Utility functions
├── experiments/
│   ├── configs/               # YAML/JSON configuration files
│   ├── results/               # Simulation results
│   └── figures/               # Generated figures
├── notebooks/
│   ├── 01_naive_mc.ipynb      # Phase 1: Naive Monte Carlo
│   ├── 02_ams_experiments.ipynb # Phase 2: AMS experiments
│   └── 03_committor_study.ipynb # Phase 3: Committor analysis
├── requirements.txt
└── README.md
```

## Phase 1 Implementation Plan (Naive Monte Carlo)
### Core Components to Implement:
1. **Double Well Potential**: V(x,y) = ¼(x²-1)² + ω/2 y² with ω ∈ [0.5, 3]
2. **Euler-Maruyama Integrator**: X_{n+1} = X_n - ∇V(X_n)Δt + √(2β⁻¹Δt) G_n
3. **Region Detection**: 
   - A = {x < -0.8 ∧ y² ≤ r_A²}
   - B = {x > 0.8 ∧ y² ≤ r_B²}
4. **Simulation Runner**: Generate trajectories until absorption
5. **Statistics**: Estimate p = ℙ(τ_B < τ_A) with confidence intervals
6. **Visualization**: Trajectories, potential field, histograms

### Parameters to Study:
- Δt values: [0.001, 0.005, 0.01, 0.05] for stability analysis
- β values: Choose to make p measurable (10⁻² to 10⁻³ range)
- ω values: [0.5, 1.0, 2.0, 3.0]
- Region radii: r_A = r_B = 0.2 (default)

### Deliverables:
- Stable, reproducible code
- Figures: trajectories, potential maps, τ distributions
- Table: p estimates with 95% confidence intervals for different β, Δt
