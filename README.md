# Simulation of Metastable Problems through AMS

This project implements stochastic algorithms for rare event simulation in molecular dynamics and materials science, supervised by Tony Lelièvre (École Polytechnique, CERMICS).

## Project Structure

The project is divided into multiple phases:

### Phase 1 — Naïve Monte Carlo Simulation ✅ COMPLETED
- Implementation of overdamped Langevin dynamics
- Double well potential system
- Transition probability estimation
- Confidence interval calculation

### Phase 2 — Arrhenius Law Modeling ✅ IMPLEMENTED
- Temperature dependence of transition rates
- Arrhenius plot analysis
- Rareness quantification vs temperature
- Verification of exponential dependence

### Phase 3 — Dynamic Adaptive Multilevel Splitting (AMS) ✅ IMPLEMENTED
- Efficient rare event probability estimation
- Trajectory selection and branching
- Performance comparison with naive MC
- Parameter sensitivity analysis

### Phase 4 — Committor Function Approximation #TODO
- Committor function estimation
- Continuous vs discrete committor analysis
- Integration with AMS algorithm

### Phase 5 — Validation & Extensions #TODO
- Algorithm validation
- Performance optimization
- Extensions to more complex systems

## Notebooks

- `notebooks/01_naive_mc.ipynb` - Phase 1: Naive Monte Carlo simulation
- `notebooks/02_arrhenius_law.ipynb` - Phase 2: Arrhenius law modeling
- `notebooks/03_dynamic_ams.ipynb` - Phase 3: Dynamic AMS implementation

## Mathematical Framework

The project implements:
- **Langevin Equations**: Overdamped dynamics for molecular systems
- **Double Well Potential**: $V(x,y) = \frac{1}{4}(x^2-1)^2 + \frac{\omega}{2}y^2$
- **Arrhenius Law**: $k = A \exp\left(-\frac{\Delta E}{k_B T}\right)$
- **Adaptive Multilevel Splitting**: Efficient rare event simulation

## Key Features

- Modular Python implementation
- Comprehensive visualization and analysis
- Parameter studies and sensitivity analysis
- Performance comparison between methods
- Extensible architecture for future phases
