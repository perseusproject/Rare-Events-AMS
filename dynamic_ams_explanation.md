# Explanation of `dynamic_ams.ipynb`

This document provides a detailed explanation of the `dynamic_ams.ipynb` Jupyter notebook, which implements and analyzes the Dynamic Adaptive Multilevel Splitting (AMS) algorithm for efficient rare event simulation. The notebook is designed to estimate the probability of rare events in stochastic systems, a task that is often computationally prohibitive using traditional Monte Carlo methods.

---

## Phase 3: Dynamic Adaptive Multilevel Splitting (AMS)

This notebook focuses on the implementation and analysis of the AMS algorithm.

### Setup and Imports

This section prepares the Python environment by importing all necessary libraries and custom modules.

-   **Core Libraries**: Imports `numpy` for numerical operations, `matplotlib.pyplot` for plotting, `scipy.stats` for statistical functions (though not explicitly used in the provided snippets, it's a common companion), `tqdm` for progress bars, and standard `sys`, `os`, `copy` modules.
-   **Custom Modules**: Crucially, it adds the `src` directory to Python's path to import custom modules. These modules define:
    -   `potentials`: Functions for defining the potential energy landscape (e.g., a double-well potential).
    -   `integrators`: Functions for simulating particle trajectories (e.g., Euler-Maruyama integration).
    -   `detectors`: Functions to identify if a particle is in specific regions (e.g., start region A, target region B) or has been absorbed.
    -   `monte_carlo`: Functions for performing naive Monte Carlo simulations.
    -   `utils`: General utility functions, including plotting and saving figures.
    -   `ams`: The core AMS algorithm functions.
-   **Plotting Configuration**: Sets up `matplotlib` to use a default style, specifies figure size, font size, and enables grid lines for all plots, ensuring visual consistency and clarity.

### 1. AMS Algorithm Implementation

This section introduces the theoretical foundation of the Adaptive Multilevel Splitting (AMS) algorithm and provides its practical implementation within a Python class named `DynamicAMS`.

-   **`DynamicAMS` Class**: This class encapsulates the entire AMS algorithm.
    -   **Initialization (`__init__`)**:
        -   Takes `n_trajectories` (the number of trajectories in the ensemble), `selection_fraction` (the proportion of "worst" trajectories to replace in each step, defaulting to 0.1), and `max_iterations` (a safety limit for the simulation, defaulting to 100).
        -   Calculates `self.k`, the exact number of trajectories to replace per iteration.
        -   Initializes empty lists to store `self.trajectories`, `self.scores` (for each trajectory), and `self.history` (to record the state of the ensemble at each iteration for later analysis).
    -   **Trajectory Initialization (`initialize_trajectories`)**:
        -   Generates the initial ensemble of `n_trajectories` independent paths.
        -   For each path, it uses a helper function (`simulate_until_absorption`) to simulate the particle's movement from a starting point (`x0`, `y0`) until it either reaches the initial region A, the target region B, or a maximum number of steps (`max_steps`) is reached.
        -   A `score` is computed for each trajectory using the `compute_score` method, reflecting its progress towards the rare event.
        -   The initial state of the ensemble (all trajectories, their scores, and summary statistics like min/max/mean scores) is saved into `self.history`.
    -   **Score Computation (`compute_score`)**:
        -   This method assigns a numerical value to a trajectory based on its outcome and path.
        -   If a trajectory successfully reaches the target region B, it's given an infinitely high score (`float('inf')`), indicating the best possible outcome.
        -   If it returns to the starting region A, it receives an infinitely low score (`float('-inf')`), representing the worst outcome.
        -   For trajectories that are still evolving or reached the maximum simulation steps without absorption, the score is defined as the maximum x-coordinate achieved. This is a common choice when the rare event involves crossing a barrier along the x-axis, as a higher x-coordinate implies more progress towards the target.
    -   **Single AMS Iteration (`run_iteration`)**:
        -   This is the core of the AMS algorithm's adaptive nature.
        -   **Sorting**: All current trajectories are sorted based on their scores from lowest to highest.
        -   **Selection**: The `k` trajectories with the lowest scores are identified as those to be replaced.
        -   **Branching**: The remaining `n_trajectories - k` trajectories (those with better scores) form a "breeding pool."
        -   **Replacement**: For each trajectory to be replaced:
            -   A random trajectory is chosen from the breeding pool.
            -   A "branching point" is found within this chosen trajectory. This point is the last position where the trajectory's score (x-coordinate) is at least as good as the score of the trajectory being replaced. This ensures that new simulations always start from a point that has made sufficient progress.
            -   A new sub-trajectory is simulated starting from this branching point.
            -   This new sub-trajectory is appended to the initial part of the "parent" trajectory to form a complete new path.
            -   The replaced trajectory is updated with this new path and its corresponding score.
        -   **History Update**: The state of the ensemble after this iteration (including updated trajectories, scores, and statistics) is recorded in `self.history`.
    -   **Full AMS Run (`run_ams`)**:
        -   Orchestrates the entire AMS simulation.
        -   First, it calls `initialize_trajectories` to set up the initial ensemble.
        -   Then, it iteratively calls `run_iteration` for a maximum of `max_iterations` or until a specified `target_B_count` (number of trajectories reaching region B) is achieved.
        -   Progress is printed to the console at each iteration.
        -   Finally, it calls `get_results` to compile and return the overall simulation outcomes.
    -   **Result Retrieval (`get_results`)**:
        -   Calculates and returns a dictionary containing key results from the AMS simulation.
        -   Includes the total `n_trajectories`, the `n_reached_B` (total successful events), the `p_estimate` (estimated probability of the rare event), `survival_probs` (a list showing the proportion of trajectories that haven't returned to region A at each iteration), the full `history` of the simulation, and the `final_scores` of all trajectories.

### 2. AMS Algorithm Testing

This section demonstrates how to use the `DynamicAMS` class by setting up and running a simulation for a rare event.

-   **`test_ams_algorithm` Function**:
    -   Defines a set of parameters for a rare event scenario. These include:
        -   `n_trajectories = 100`: Ensemble size.
        -   `selection_fraction = 0.1`: 10% replacement rate.
        -   `max_iterations = 50`: Iteration limit.
        -   `x0, y0 = -1.0, 0.0`: Starting point in the left potential well.
        -   `dt = 0.01`: Time step.
        -   `beta_inv = 0.0`: A very low inverse temperature, making the event rare.
        -   `max_steps = 5000`: Maximum steps for a single trajectory.
        -   `omega = 1.0`: Potential stiffness.
        -   `target_B_count = 10`: The simulation aims to find 10 successful trajectories.
    -   An instance of `DynamicAMS` is created with these parameters.
    -   The `run_ams` method is called to start the simulation.
    -   The function returns the `DynamicAMS` object itself (which holds the full history) and the final `results` dictionary.
-   **Execution**: The `test_ams_algorithm` function is executed, and its outputs are stored in `ams` and `ams_results` variables for subsequent analysis. The console output provides a real-time view of the simulation's progress, showing how many trajectories have reached region B in each iteration.

### 3. AMS Results Analysis

This section provides a detailed visualization and analysis of the AMS simulation's performance and outcomes.

-   **`analyze_ams_results` Function**:
    -   Generates a 2x2 grid of plots to offer a comprehensive view of the simulation.
    -   **Plot 1: Survival Probability Evolution**: Displays a line plot showing the percentage of trajectories that have not returned to the starting region A over the course of the AMS iterations. This indicates the overall "survival" rate of the ensemble as it progresses towards the rare event.
    -   **Plot 2: Progress Toward Rare Event**: Presents a line plot tracking the cumulative number of trajectories that have successfully reached the target region B across iterations. This directly illustrates the algorithm's effectiveness in finding rare events.
    -   **Plot 3: Score Evolution**: Shows three line plots on the same axes: the minimum, maximum, and mean scores of the trajectories in the ensemble at each iteration. This helps to understand how the distribution of trajectory "progress" shifts and improves over time, with the minimum score generally increasing as the algorithm prunes less successful paths.
    -   **Plot 4: Final Score Distribution**: Creates a histogram of the final scores of all trajectories. It specifically filters out infinite scores (trajectories that reached A or B) to focus on the distribution of scores for trajectories that were still evolving when the simulation stopped. A vertical dashed line indicates the approximate threshold for region B, providing context for the distribution.
    -   The function ensures a tight layout for the plots and displays the figure.
    -   It returns the Matplotlib figure object.
-   **Execution**: The `analyze_ams_results` function is called with the `ams` object and `ams_results` obtained from the testing phase. The generated figure is then saved as `ams_algorithm_analysis.png` in the `experiments/figures` directory.

### 4. Comparison with Naive Monte Carlo

This section quantitatively compares the efficiency of the AMS algorithm against a traditional (naive) Monte Carlo simulation for the same rare event.

-   **`compare_ams_vs_naive_mc_grid` Function**:
    -   This function performs a systematic comparison across a grid of `dt` (time step) and `beta_inv` (inverse temperature) values.
    -   For each combination of `dt` and `beta_inv`:
        -   It runs a naive Monte Carlo simulation (`n_mc_simulations = 10000`) using the `run_naive_mc` function.
        -   It calculates the total number of individual trajectory simulations performed by AMS (which is `n_trajectories * (1 + ams_iterations)`).
        -   It calculates the "efficiency" for both AMS and naive MC as the number of successful events (reaching B) divided by the total simulations.
        -   A `speedup` factor is computed by dividing AMS efficiency by naive MC efficiency. If naive MC observes no events, the speedup is considered infinite.
        -   All these comparison metrics are stored in a list of dictionaries.
    -   The function prints a detailed summary for each grid point, showing the probability estimates, number of events, total simulations, and the calculated speedup factor for both methods.
-   **Execution**: The `compare_ams_vs_naive_mc_grid` function is called with the `ams_results` and predefined lists of `dt_values` and `beta_inv_values`. The console output clearly illustrates the significant (often infinite) speedup of AMS, especially when the rare event is too infrequent for naive Monte Carlo to observe within a reasonable number of simulations.

### 5. AMS Trajectory Visualization

This section provides a visual representation of selected trajectories from the AMS simulation, offering insights into how the algorithm explores the state space.

-   **`visualize_ams_trajectories` Function**:
    -   Retrieves the final set of `trajectories` and their `scores` from the `ams` object.
    -   **Trajectory Selection**: It intelligently selects a small number of trajectories (defaulting to 5) for visualization. The selection aims to include a mix of:
        -   "Successful" trajectories (those that reached region B, indicated by `float('inf')` score).
        -   "Progressing" trajectories (those that made some progress towards B but didn't necessarily reach it, having a finite score).
        -   "Failed" trajectories (those that returned to region A, indicated by `float('-inf')` score) are also considered.
    -   **Background Plot**: It first plots the `double_well_potential_field` as a background, clearly showing regions A and B.
    -   **Trajectory Overlay**: Each selected trajectory is then plotted on top of the potential field.
        -   Different colors and line widths are used to distinguish between successful (red, thick), failed (gray, thin), and progressing (various colors, medium thickness) trajectories.
        -   The starting and ending points of each trajectory are marked to show their evolution.
    -   The plot is titled, a legend is added, and the figure is displayed.
    -   The function returns the Matplotlib figure object.
-   **Execution**: The `visualize_ams_trajectories` function is called, and the resulting figure is saved as `ams_representative_trajectories.png` in the `experiments/figures` directory.

### 6. AMS Parameter Sensitivity Analysis

This section investigates how the AMS algorithm's performance is influenced by changes in its key parameters, specifically the `selection_fraction`.

-   **`study_ams_parameters` Function**:
    -   Establishes a `base_params` dictionary containing a standard set of simulation parameters.
    -   **Parameter Variation**: It defines a list of `selection_fractions` (e.g., 0.05, 0.1, 0.2, 0.3) to systematically test.
    -   For each `selection_fraction`:
        -   A new `DynamicAMS` instance is created, and the `run_ams` method is executed.
        -   The results, including the number of iterations taken to reach the target and the final count of trajectories in region B, are recorded.
    -   **Sensitivity Plots**: Two subplots are generated to visualize the impact of the `selection_fraction`:
        -   **Plot 1: Convergence Speed vs Selection Fraction**: Shows how the number of iterations required to achieve the `target_B_count` changes as the `selection_fraction` varies. This helps identify the optimal `selection_fraction` for faster convergence.
        -   **Plot 2: Final Performance vs Selection Fraction**: Illustrates how the `final_B_count` (number of successful events) is affected by the `selection_fraction`. This provides insight into the algorithm's robustness and efficiency at different pruning rates.
    -   The plots are displayed, and the function returns the collected `selection_results` and the figure object.
-   **Execution**: The `study_ams_parameters` function is executed, and the resulting figure is saved as `ams_parameter_sensitivity.png`. The console output details the testing for each selection fraction.

### 7. AMS for Very Rare Events

This section pushes the AMS algorithm to its limits by testing it on events that are so rare that even AMS might struggle, and naive Monte Carlo would be entirely infeasible.

-   **`test_ams_very_rare_events` Function**:
    -   Configures parameters for an *extremely* rare event, notably setting `beta_inv = 0.05` (a very low inverse temperature, implying extremely high energy barriers for transitions).
    -   It explicitly states that naive Monte Carlo would require millions of simulations for such an event.
    -   An `DynamicAMS` instance is initialized and `run_ams` is executed with these extreme parameters.
    -   **Performance Evaluation**: After the simulation, it checks if AMS detected any rare events.
        -   If `p_estimate > 0`, it calculates the estimated probability, AMS total simulations, and an estimated number of naive MC simulations needed for comparison, along with the speedup factor.
        -   If `p_estimate` remains 0, it concludes that the event is too rare even for the current AMS setup.
    -   The function returns the `ams_rare` object and its `results_rare`.
-   **Execution**: The `test_ams_very_rare_events` function is called. The console output shows the iterations, and in the provided example, AMS failed to detect any events, underscoring the extreme rarity of the chosen scenario.

### 8. Summary and Conclusions

This section provides a high-level summary of the AMS algorithm's capabilities and the findings from the notebook.

-   **Summary Printout**: This cell prints a structured summary that covers:
    -   **Algorithm Features**: Highlights adaptive selection, dynamic score-based progression, efficient probability estimation, and significant speedup.
    -   **Key Advantages**: Emphasizes exponential speedup for rare events, adaptability, ability to handle very rare events (p < 10^-6), and providing trajectory insights.
    -   **Implementation Details**: Briefly describes the chosen score function (max x-coordinate), branching mechanism, selection strategy, and stopping conditions.
    -   **Performance Highlights**: Summarizes the results from the initial AMS test, including the number of detected events and the observed speedup.

### 9. Save AMS Results

The final section of the notebook is dedicated to persisting the simulation results for future reference or further analysis.

-   **Result Serialization**:
    -   It imports the `json` module for data serialization.
    -   A Python dictionary named `ams_data` is constructed. This dictionary meticulously organizes all critical information from the simulation:
        -   `algorithm_parameters`: Stores the initial configuration of the AMS algorithm (e.g., `n_trajectories`, `selection_fraction`, `max_iterations`).
        -   `system_parameters`: Contains the physical parameters of the stochastic system that was simulated (e.g., initial position `x0, y0`, time step `dt`, inverse temperature `beta_inv`, `max_steps`, `omega`).
        -   `results`: Holds the key outcomes of the simulation, such as the `p_estimate` (estimated rare event probability), `n_reached_B` (number of trajectories that successfully reached region B), `total_iterations`, and `final_scores`. Special handling is applied to `final_scores` to convert `float('inf')` values to strings, as JSON does not natively support infinity.
    -   **File Output**: The `ams_data` dictionary is then written to a JSON file named `ams_results.json`. This file is placed in the `../experiments/results/` directory, ensuring that the results are stored in a designated location relative to the notebook. The `indent=2` argument ensures the JSON output is human-readable.
-   **Confirmation**: Print statements confirm that the AMS results have been successfully saved and that "Phase 3 implementation is complete!".

---

This detailed explanation covers the purpose, methodology, and outcomes of each significant part of the `dynamic_ams.ipynb` notebook, providing a thorough understanding of its contents without including the actual code.
