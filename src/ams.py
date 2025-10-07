import numpy as np
import copy
from detectors import simulate_until_absorption


class DynamicAMS:
    """
    Dynamic Adaptive Multilevel Splitting algorithm for rare event simulation.
    """

    def __init__(self, n_trajectories, selection_fraction=0.1, max_iterations=100):
        """
        Initialize AMS algorithm.

        Parameters:
        -----------
        n_trajectories : int
            Number of trajectories to maintain
        selection_fraction : float
            Fraction of trajectories to replace each iteration (0 < fraction < 1)
        max_iterations : int
            Maximum number of iterations
        """
        self.n_trajectories = n_trajectories
        self.k = int(selection_fraction * n_trajectories)  # Number to replace
        self.max_iterations = max_iterations
        self.trajectories = []
        self.scores = []
        self.history = []

    def initialize_trajectories(self, x0, y0, dt, beta_inv, max_steps, omega=1.0):
        """
        Initialize trajectories from starting point.

        Parameters:
        -----------
        x0, y0 : float
            Initial position
        dt : float
            Time step
        beta_inv : float
            Inverse temperature
        max_steps : int
            Maximum steps per trajectory
        omega : float
            Stiffness parameter
        """
        self.trajectories = []
        self.scores = []

        for i in range(self.n_trajectories):
            traj, absorption_state, absorption_time = simulate_until_absorption(
                x0, y0, dt, beta_inv, max_steps, omega
            )
            self.trajectories.append(traj)
            score = self.compute_score(traj, absorption_state)
            self.scores.append(score)

        self.history.append({
            'iteration': 0,
            'trajectories': copy.deepcopy(self.trajectories),
            'scores': self.scores.copy(),
            'min_score': np.min(self.scores),
            'max_score': np.max(self.scores),
            'mean_score': np.mean(self.scores)
        })

    def compute_score(self, trajectory, absorption_state):
        """
        Compute score for a trajectory.

        Current implementation: maximum x-coordinate reached.
        For rare events from A to B, higher x is better.

        Parameters:
        -----------
        trajectory : ndarray
            Array of positions
        absorption_state : int
            Absorption state (0: unabsorbed, 1: A, 2: B)

        Returns:
        --------
        score : float
            Trajectory score (higher is better progress toward B)
        """
        if absorption_state == 2:  # Reached B
            return float('inf')  # Best possible score
        elif absorption_state == 1:  # Returned to A
            return float('-inf')  # Worst possible score
        else:  # Still evolving
            return np.max(trajectory[:, 0])  # Maximum x-coordinate reached

    def run_iteration(self, dt, beta_inv, max_steps, omega=1.0):
        """
        Run one AMS iteration.

        Parameters:
        -----------
        dt : float
            Time step
        beta_inv : float
            Inverse temperature
        max_steps : int
            Maximum steps per trajectory
        omega : float
            Stiffness parameter
        """
        scores = np.array(self.scores, dtype=float)

        # 1. Calcul du seuil global z_q = k-ième ordre statistique
        kth = max(1, self.k) - 1   # index 0-based
        z_q = np.partition(scores, kth)[kth]

        # 2. Sélection des trajectoires à remplacer : toutes celles <= z_q
        replace_indices = np.where(scores <= z_q)[0]

        # 3. Sélection des trajectoires parentes : celles > z_q
        branch_indices = np.where(scores > z_q)[0]

        # (Sécurité : si aucune survivante, prendre les meilleures)
        if branch_indices.size == 0:
            branch_indices = np.argsort(
                scores)[-max(1, self.n_trajectories // 10):]

        # Replace worst trajectories by branching from better ones
        for replace_idx in replace_indices:
            # Randomly select a trajectory to branch from
            branch_idx = np.random.choice(branch_indices)
            branch_traj = self.trajectories[branch_idx]

            # Find branching point (last point where score equals current threshold)
            threshold_score = self.scores[replace_idx]
            branch_points = []

            for i, point in enumerate(branch_traj):
                x, y = point
                # Simple score at this point (x-coordinate)
                point_score = x
                if point_score >= threshold_score:
                    # s'assurer d'un tuple (x, y)
                    branch_points.append((i, (x, y)))

            if branch_points:
                # --- CORRECTIF ICI : on tire l'indice, pas l'objet ---
                # ou np.random.choice(len(branch_points))
                idx = np.random.randint(len(branch_points))
                branch_step, (branch_x, branch_y) = branch_points[idx]
                # ----------------------------------------------------

                # Generate new trajectory from branching point
                new_traj, absorption_state, absorption_time = simulate_until_absorption(
                    branch_x, branch_y, dt, beta_inv, max_steps - branch_step, omega
                )

                # Combine trajectories
                combined_traj = np.vstack(
                    [branch_traj[:branch_step], new_traj])

                # Update trajectory and score
                self.trajectories[replace_idx] = combined_traj
                self.scores[replace_idx] = self.compute_score(
                    combined_traj, absorption_state)
            else:
                # If no suitable branching point, keep original
                pass

        # Record iteration history
        self.history.append({
            'iteration': len(self.history),
            'trajectories': copy.deepcopy(self.trajectories),
            'scores': self.scores.copy(),
            'min_score': np.min(self.scores),
            'max_score': np.max(self.scores),
            'mean_score': np.mean(self.scores),
            'n_reached_B': np.sum(np.array(self.scores) == float('inf'))
        })

    def run_ams(self, x0, y0, dt, beta_inv, max_steps, omega=1.0, target_B_count=None):
        """
        Run full AMS algorithm.

        Parameters:
        -----------
        x0, y0 : float
            Initial position
        dt : float
            Time step
        beta_inv : float
            Inverse temperature
        max_steps : int
            Maximum steps per trajectory
        omega : float
            Stiffness parameter
        target_B_count : int, optional
            Stop when this many trajectories reach B

        Returns:
        --------
        results : dict
            AMS algorithm results
        """
        print("Initializing AMS trajectories...")
        self.initialize_trajectories(x0, y0, dt, beta_inv, max_steps, omega)

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            print(
                f"Iteration {iteration}: {np.sum(np.array(self.scores) == float('inf'))} trajectories reached B")

            self.run_iteration(dt, beta_inv, max_steps, omega)

            # Check stopping condition
            n_reached_B = np.sum(np.array(self.scores) == float('inf'))
            if target_B_count and n_reached_B >= target_B_count:
                print(
                    f"Stopping: {n_reached_B} trajectories reached B (target: {target_B_count})")
                break

        return self.get_results()

    def get_results(self):
        """
        Get AMS algorithm results.

        Returns:
        --------
        results : dict
            Dictionary containing AMS results
        """
        n_reached_B = np.sum(np.array(self.scores) == float('inf'))
        p_estimate = n_reached_B / self.n_trajectories

        # Calculate survival probabilities at each iteration
        survival_probs = []
        for hist in self.history:
            n_surviving = np.sum(np.array(hist['scores']) > float('-inf'))
            survival_probs.append(n_surviving / self.n_trajectories)

        results = {
            'n_trajectories': self.n_trajectories,
            'n_reached_B': n_reached_B,
            'p_estimate': p_estimate,
            'survival_probs': survival_probs,
            'history': self.history,
            'final_scores': self.scores
        }

        return results


def adaptive_multilevel_splitting(x0, y0, dt, beta_inv, max_steps, n_trajectories, selection_fraction=0.1, max_iterations=100, omega=1.0, target_B_count=None):
    """
    Wrapper function to run the Dynamic Adaptive Multilevel Splitting algorithm.
    """
    ams = DynamicAMS(n_trajectories, selection_fraction, max_iterations)
    results = ams.run_ams(x0, y0, dt, beta_inv,
                          max_steps, omega, target_B_count)
    return results


def print_ams_summary(results):
    """
    Prints a summary of the AMS results.
    """
    print("\n--- AMS Results Summary ---")
    print(f"Number of trajectories: {results['n_trajectories']}")
    print(f"Trajectories reached B: {results['n_reached_B']}")
    print(f"Probability estimate P_AB: {results['p_estimate']:.2e}")
    print(f"Total iterations: {len(results['history']) - 1}")
    print("---------------------------\n")


def plot_ams_trajectories(history, potential_func, x_range, y_range, n_trajectories_to_plot=5, filename=None):
    """
    Plots a selection of AMS trajectories from the history.
    """
    print("Plotting AMS trajectories (placeholder)...")
    # This is a placeholder. Actual plotting logic would go here.
    # It would typically involve iterating through the history and plotting
    # a subset of trajectories, potentially with the potential field.
    pass
