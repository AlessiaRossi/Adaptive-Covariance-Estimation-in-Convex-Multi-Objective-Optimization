# aco_active_learning.py
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from surr_model_toy import fit_gp_model, optimize_for_lambda, train_and_prepare_surrogate, evaluate_model, \
    load_lambda_covariance_data, build_dataset
import matplotlib.pyplot as plt
import os
import joblib
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, r2_score
import concurrent.futures
from filelock import FileLock
import time
from scipy.optimize import minimize
import tempfile
import matplotlib.pyplot as plt
import random

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

NUM_PERTURBATIONS = 20
PERTURBATION_STRENGTH = 0.025

a = 1.0
b = -2.0

# Problem definition
def f1(x):
    return 0.5 * (x - a) ** 2


def f2(x):
    return 0.5 * (x - b) ** 2


def scalarized_objective(x, lmbd):
    return lmbd * f1(x) + (1 - lmbd) * f2(x)



class ACOActiveLearner:
    def __init__(self, lambda_data, surrogate_model=None, rho=0.1, random_state=GLOBAL_SEED):
        ''' Initialize the ACOActiveLearner with lambda data, surrogate model, and parameters '''
        self.lambda_data = lambda_data.copy()
        self.lambdas = self.lambda_data['lambda'].values.reshape(-1, 1)  # shape (N, 1)

        self.tau = defaultdict(lambda: 1.0) # Pheromone levels
        self.eta = {} # Heuristic values
        self.surrogate = surrogate_model

        self.Selected = set() # Selected lambdas
        self.C_e = None # Empirical covariance matrix

        self.epsilon_threshold = 1e-2  # Convergence threshold
        self.rho = rho # Pheromone evaporation rate
        self.x_opt_cache = {}  # Cache for x_opt values

        self.random_state = np.random.RandomState(random_state)

    def get_x_opt(self, lam):
        ''' Get x_opt for a given lambda, using cache to avoid recomputation '''
        lam_tuple = tuple(lam)
        if lam_tuple not in self.x_opt_cache:
            x_opt, _ = optimize_for_lambda(lam_tuple)
            self.x_opt_cache[lam_tuple] = x_opt
        return self.x_opt_cache[lam_tuple]

    def compute_heuristic(self):
        ''' Calculate heuristic eta for each lambda, representing the norm of the covariance matrix using the surrogate model '''
        if self.surrogate is None:
            raise ValueError("Surrogate model is not provided.")
        print("Calcolating heuristic eta...")
        X = self.lambda_data[['lambda']].values  # shape (N, 1)
        y_pred, _ = self.surrogate['model'].predict(X, return_std=True)
        y_pred_mean = y_pred.mean()
        y_pred_std = y_pred.std() if y_pred.std() > 0 else 1.0
        for i, lam in enumerate(self.lambdas):
            lam_tuple = tuple(lam)
            # Penalty for diversity
            diversity_penalty = 1 / (1 + min([np.linalg.norm(lam - np.array(sel)) for sel in self.Selected] + [1.0]))
            self.eta[lam_tuple] = ((y_pred[i] - y_pred_mean) / y_pred_std) * diversity_penalty

    def select_diverse_lambdas(self, n_select=20, min_dist=0.085, score_dict=None, exclude_lambdas=None,
                               random_state=None):
        ''' Select diverse lambdas based on the heuristic eta '''
        print(f"Selecting diverse lambdas with min_dist={min_dist}...")
        if score_dict is None:
            score_dict = self.eta
        if exclude_lambdas is None:
            exclude_lambdas = set()
        # Initialize random state
        rng = np.random.RandomState(random_state if random_state is not None else GLOBAL_SEED)

        # Sort lambdas by score
        sorted_lambdas = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        rng.shuffle(sorted_lambdas)  # Shuffle to introduce randomness

        selected = []
        for lam, score in sorted_lambdas:
            lam_tuple = tuple(lam)
            if lam_tuple in selected or lam_tuple in exclude_lambdas:
                continue
            lam_arr = np.array(lam_tuple)
            if selected:
                dists = np.linalg.norm(np.array(selected) - lam_arr, axis=1)
                if np.all(dists >= min_dist):
                    selected.append(lam_tuple)
            else:
                selected.append(lam_tuple)
            if len(selected) >= n_select:
                break

        # If you haven't reached n_select yet, add the best remaining (even if close)
        if len(selected) < n_select:
            for lam, score in sorted_lambdas:
                lam_tuple = tuple(lam)
                if lam_tuple not in selected and lam_tuple not in exclude_lambdas:
                    selected.append(lam_tuple)
                if len(selected) >= n_select:
                    break

        return [tuple(lam) for lam in selected]

    def sample_candidates(self, n_ants=100, alpha=1.0, beta=1.0, random_state=None):
        ''' Sample candidates based on pheromone and heuristic values '''
        print("Sampling candidates")
        keys = list(self.eta.keys())
        scores = np.array([
            (self.tau[lam] ** alpha) * (self.eta[lam] ** beta) for lam in keys
        ])
        min_score = np.min(scores)
        if min_score < 0:
            scores = scores - min_score + 1e-8
        if np.sum(scores) == 0:
            probs = np.ones_like(scores) / len(scores)
        else:
            probs = scores / np.sum(scores)
        print(f" Samplin candidates with scores: {probs}")
        # Initialize random state
        if random_state is not None:
            rng = np.random.RandomState(random_state)
        else:
            rng = np.random.RandomState(GLOBAL_SEED)


        indices = rng.choice(len(keys), size=n_ants, replace=False, p=probs)
        sampled = [keys[i] for i in indices]
        print(f"Total candidates sampled: {len(sampled)}")
        return sampled

    def run_aco_active_learning(self, C_ref,alpha=1.0, beta=1.0,epsilon=None, budget=10,n_init_diverse=5,min_lambda=10, top_k=20,model_path="surrogate_toy_gp1.pkl", reload_surrogate=True,colony_id=None, selection_criterion=None):
        ''' Run the Ant Colony Optimization (ACO) active learning algorithm to iteratively select a diverse and informative set of lambdas that minimize the error between the empirical covariance matrix and the reference matrix C_ref.'''
        if epsilon is None:
            epsilon = 1e-6

        # Initialize the selected lambdas and compute the initial empirical covariance matrix.
        if len(self.Selected) == 0:
            self.Selected = set(self.select_diverse_lambdas(n_select=n_init_diverse, min_dist=0.085))
            self.C_e = self.compute_Ce_from_lambdas(self.Selected)

        val_count = len(self.Selected)
        print(f"Nummber of selected lambdas: {val_count}")

        # Initialize variables for tracking the best configuration and errors
        best_error = float('inf')
        best_config = None
        error_list = []
        no_improve_counter = 0
        selected_count_list = []
        selected_history = []
        max_iter = budget
        last_mtime = None

        for iter_idx in range(max_iter):
            print(f"\n--- Iteration {iter_idx + 1}/{max_iter} | Colony: {colony_id} ---", flush=True)
            selected_history.append(list(self.Selected))

            # Reload the surrogate model if required.
            if reload_surrogate and model_path is not None:
                self.surrogate, last_mtime = maybe_reload_surrogate(self.surrogate, model_path, last_mtime)
            print(f"\n--- Iteration {iter_idx + 1}/{max_iter} ---", flush=True)
            print(f" Selected lambdas: {len(self.Selected)}", flush=True)
            print(f" Current error: {np.linalg.norm(C_ref - self.C_e, ord='fro'):.8f}" if self.C_e is not None else "Current error: N/A",flush=True)

            # 1. Sample new candidates based on the selection criterion.
            if selection_criterion == "heuristic":
                # Use heuristic eta and pheromone tau
                candidates = self.sample_candidates(n_ants=top_k, alpha=alpha, beta=beta, random_state=GLOBAL_SEED)
            else:
                # Use only pheromone tau (beta=0.0)
                candidates = self.sample_candidates(n_ants=top_k, alpha=alpha, beta=0.0, random_state=GLOBAL_SEED)

            # Exclude already selected lambdas
            new_candidates = [lam for lam in candidates if lam not in self.Selected]
            print(f"Total candidates sampled: {len(candidates)}")
            print(f"New candidates : {len(new_candidates)}")

            # 2. Compute delta_k for each new candidate
            delta_k_list = []
            for lam in new_candidates:
                delta_k = self.compute_delta_k(C_ref, lam)
                delta_k_list.append((lam, delta_k))

            # Sort by delta_k
            delta_k_list.sort(key=lambda x: x[1])
            selected_candidates = [lam for lam, _ in delta_k_list[:max(1, len(delta_k_list) // 2)]]
            print(f"Selected samples for update: {len(selected_candidates)}")

            # 3: Update pheromones and add new candidates to the selected set.
            self.evaporate_pheromones()
            self.update_pheromones(selected_candidates, C_ref)

            n_added = 0
            for lam in selected_candidates:
                lam_tuple = tuple(lam)
                if lam_tuple not in self.Selected:
                    self.Selected.add(lam_tuple)
                    n_added += 1
            print(f"New candidates added to selected set: {n_added}")
            print(f"Total selected lambdas after addition: {len(self.Selected)}")

            # 4. Pruning : Prune the selected lambdas if their count exceeds the maximum allowed.
            max_lambdas = 175  # max number of lambdas to keep
            if len(self.Selected) > max_lambdas:
                before_pruning = set(self.Selected)
                self.Selected = set(
                    self.select_diverse_lambdas(n_select=max_lambdas, min_dist=0.085, score_dict=self.tau))
                n_removed = len(before_pruning) - len(self.Selected)
                print(f"Lambda removed during pruning: {n_removed}")
                print(f"Lambda selected after pruning: {len(self.Selected)}")
            else:
                print("No pruning needed")

            # Update the empirical covariance matrix and compute the current error.
            self.C_e = self.compute_Ce_from_lambdas(self.Selected)
            if self.C_e is not None:
                error = np.linalg.norm(C_ref - self.C_e, ord='fro')
            else:
                error = None
            error_list.append(error)

            # Update the best error and configuration if an improvement is found.
            selected_count_list.append(len(self.Selected))
            # # Early stopping
            if error is not None and error < best_error:
                best_error = error
                best_config = list(self.Selected)
                print(f"[Iteration {iter_idx + 1}] Improvement: error {error:.8f} < best error {best_error:.8f}")
                no_improve_counter = 0
            else:
                print(f"[Iteration {iter_idx + 1}] No improvement: error {error:.8f} >= best error {best_error:.8f}")
                no_improve_counter += 1

            print(f"Actual error: {error:.8f}" if error is not None else "Actual error: N/A")
            print(f"Best configuration: {best_config} with error: {best_error:.8f}")

            # Stopping criteria: convergence or lack of improvement.
            if (error is not None and error < epsilon) or (len(self.Selected) < min_lambda):
                print(f"STOP: error < {epsilon} or selected < {min_lambda}")
                break
            if no_improve_counter >= 10:  # <--- early stopping
                print ("STOP: no improvement for 10 consecutive iterations.")
                # Make sure to return the same structure as in the normal exit
                return (
                    self.Selected,
                    self.C_e,
                    error_list,
                    selected_count_list,
                    best_config,
                    best_error
                )
        else:
            print(f"STOP:maximum number of iterations reached ({max_iter})")
        return self.Selected, self.C_e, error_list, selected_count_list, best_config, best_error

        
    
    def compute_Ce_from_lambdas(self, lambdas_list):
        ''' Compute the empirical covariance matrix from selected lambdas '''
        if len(lambdas_list) <= 1:
            return None
        x_list = []
        for lam in lambdas_list:
            x_opt = self.get_x_opt(lam)
            x_list.append(x_opt)
        X = np.array(x_list).reshape(-1, 1)
        x_bar = np.mean(X, axis=0)
        deviations = X - x_bar
        C_e = (deviations.T @ deviations) / (len(lambdas_list) - 1)
        return C_e

    def compute_delta_k(self, C_ref, lam):
        ''' Compute the incremental change in error (delta_k) when adding a new lambda to the current set of selected lambdas.'''

        # If no lambdas are currently selected, compute the empirical covariance matrix for the single lambda.
        lam_tuple = tuple(lam)
        if not self.Selected:
            C_e_single = self.compute_Ce_from_lambdas([lam_tuple])
            delta_k = -np.sum(np.abs(C_ref - C_e_single))  # norm L1
        else:
            # Compute the empirical covariance matrix before and after adding the new lambda.
            C_e_old = self.compute_Ce_from_lambdas(list(self.Selected))
            C_e_new = self.compute_Ce_from_lambdas(list(self.Selected) + [lam_tuple])
            # Calculate the change in error (delta_k) as the difference in L1 norms.
            delta_k = np.sum(np.abs(C_ref - C_e_new)) - np.sum(np.abs(C_ref - C_e_old))
        return delta_k

    def evaporate_pheromones(self):
        ''' Evaporate pheromones for all lambdas '''
        for lam in self.tau:
            self.tau[lam] *= (1 - self.rho)

    def update_pheromones(self, selected_lambdas, C_ref):
        ''' Update pheromones for selected lambdas '''
        for lam in selected_lambdas:
            lam_tuple = tuple(lam)
            C_e_old = self.compute_Ce_from_lambdas(list(self.Selected - {lam_tuple}))
            C_e_new = self.compute_Ce_from_lambdas(list(self.Selected))
            if C_e_old is not None and C_e_new is not None:
                error_old = np.linalg.norm(C_ref - C_e_old, ord='fro')
                error_new = np.linalg.norm(C_ref - C_e_new, ord='fro')
                delta_error = error_old - error_new
                if delta_error > 0:
                    self.tau[lam_tuple] += delta_error


def optimize_for_lambda(lmbd, x_range=(-5, 5), seed=42):
    if isinstance(lmbd, (tuple, list, np.ndarray)):
        lmbd = float(lmbd[0])

    best_result = None
    lowest_loss = float('inf')
    rng = np.random.RandomState(seed)
    for i in range(5):
        start = rng.uniform(x_range[0], x_range[1], size=1)
        result = minimize(lambda x: scalarized_objective(x[0], lmbd), start, bounds=[x_range])
        if result.success and result.fun < lowest_loss:
            lowest_loss = result.fun
            best_result = result
    if best_result is None:
        raise RuntimeError("Optimization failed")
    return best_result.x[0], best_result.fun


def get_or_train_model(archive_file, model_path, n_training=100, random_state=42):
    ''' Load or train the surrogate model '''
    if os.path.exists(model_path):
        print(f"Loading surrogate model from {model_path}...")
        surrogate_model = joblib.load(model_path)
    else:
        print(f"Training surrogate model from {archive_file}...")
        data = load_lambda_covariance_data(archive_file)
        surrogate_model, *_ = train_and_prepare_surrogate(data, n_training, random_state)
        joblib.dump(surrogate_model, model_path)
    return surrogate_model
def sample_uniform_simplex(n, d=3, random_state=GLOBAL_SEED):
    """
    Sample n points (or as close as possible) uniformly from the (d-1)-simplex (sum=1, all >=0)
    using a regular grid (combinatorial approach). For d=1, returns [1.0] * n.
    """
    if d == 1:
        # In 1D, the only point on the simplex is [1.0]
        return [(1.0,)] * n

    def num_points(level, d):
        from math import comb
        return comb(level + d - 1, d - 1)

    level = 1
    while num_points(level, d) < n:
        level += 1
    points = []

    def gen_points(level, d):
        if d == 1:
            yield (level,)
        else:
            for i in range(level + 1):
                for rest in gen_points(level - i, d - 1):
                    yield (i,) + rest

    for idx in gen_points(level, d):
        pt = tuple(i / level for i in idx)
        points.append(pt)
    rng = np.random.RandomState(random_state if random_state is not None else GLOBAL_SEED)
    rng.shuffle(points)
    return points[:n]


def sample_random_simplex(n, d=3, random_state=GLOBAL_SEED):
    """Sample n points uniformly from the (d-1)-simplex (sum=1, all >=0) using np.random.uniform."""
    if d == 1:
        return [(1.0,)] * n
    rng = np.random.RandomState(random_state if random_state is not None else GLOBAL_SEED)
    samples = rng.uniform(0, 1, (n, d))
    samples = samples / samples.sum(axis=1, keepdims=True)
    return [tuple(row) for row in samples]


def run_single_colony(colony_id, archive_data, gp_model, C_ref, params, already_selected=None, random_state=None, selection_criterion="heuristic"):
    '''Execute a single colony in the ACO (Ant Colony Optimization) active learning process.'''
    np.random.seed(random_state)
    random.seed(random_state)
    # Initialize the ACOActiveLearner instance
    print(f"Colony {colony_id}: starting ")
    aco = ACOActiveLearner(archive_data, gp_model)
    # Compute heuristic values (eta) for the lambdas
    print(f"Colony {colony_id}: calculating heuristic eta")
    aco.compute_heuristic()
    # Select initial diverse lambdas based on the selection criterion
    print(f"Colony {colony_id}: selecting initial diverse lambdas")
    if already_selected is None:
        already_selected = set()
    n_init = params['n_init_diverse']

    # Initial selection of lambdas based on the specified criterion
    if selection_criterion == "heuristic":
        # Use heuristic values and enforce a minimum distance between lambdas
        min_dist = 0.085
        initial = aco.select_diverse_lambdas(
            n_select=n_init,
            min_dist=min_dist,
            exclude_lambdas=already_selected,
            random_state=GLOBAL_SEED
        )
    elif selection_criterion == "uncertainty":
        # Use uncertainty (standard deviation) from the surrogate model and enforce a minimum distance
        min_dist = 0.085
        X = archive_data[['lambda']].values
        X_scaled = gp_model['scaler_X'].transform(X)
        _, std_pred = gp_model['model'].predict(X_scaled, return_std=True)
        std_dict = {tuple(row): std for row, std in zip(aco.lambdas, std_pred)}
        initial = aco.select_diverse_lambdas(
            n_select=n_init,
            min_dist=min_dist,
            score_dict=std_dict,
            exclude_lambdas=already_selected,
            random_state=GLOBAL_SEED
        )
    elif selection_criterion == "uniform":
        # Uniform sampling from the simplex
        initial = sample_uniform_simplex(n_init, d=3)
        initial = [lam for lam in initial if tuple(lam) not in already_selected]
    elif selection_criterion == "random":
        # Random sampling from the simplex
        initial = sample_random_simplex(n_init, d=3, random_state=GLOBAL_SEED)
        initial = [lam for lam in initial if tuple(lam) not in already_selected]
    else:
        raise ValueError(f"Unknown selection_criterion: {selection_criterion}")

    # Remove duplicates from the initial selection
    initial = list({tuple(lam): lam for lam in initial}.values())
    aco.Selected = set(initial)
    already_selected.update(aco.Selected)
    print(f"Colony {colony_id}: selected {len(aco.Selected)} initial diverse lambdas")

    # Run the ACO active learning process
    Selected, C_e, error_list, selected_count_list, best_config, best_error = aco.run_aco_active_learning(
        C_ref=C_ref,
        alpha=params['alpha'],
        beta=params['beta'],
        epsilon=params['epsilon'],
        budget=params['budget'],
        n_init_diverse=params['n_init_diverse'],
        min_lambda=params['n_init_diverse'] // 2,
        top_k=params['top_k'],
        model_path=params.get('model_path', 'surrogate_toy_gp1.pkl'),
        reload_surrogate=True,
        colony_id=colony_id,
        selection_criterion=selection_criterion  # <--- Passa il criterio
    )

    return {
        "selected": list(Selected),
        "error_list": error_list,
        "selected_count_list": selected_count_list,
        "best_config": best_config,
        "best_error": best_error,
        "criterion": selection_criterion
    }


def append_new_selected(lam, results_dict, shared_file="shared_selected.csv"):
    df = pd.DataFrame([results_dict])
    lock = FileLock(shared_file + ".lock")
    with lock:
        if not os.path.exists(shared_file):
            df.to_csv(shared_file, index=False)
        else:
            df.to_csv(shared_file, mode='a', header=False, index=False)


def retrain_surrogate(self, random_state=None):
    retrain_lambdas = np.array([lam[0] for lam in self.Selected]).reshape(-1, 1)
    retrain_xopt = np.array([self.get_x_opt(lam) for lam in self.Selected]).reshape(-1, 1)
    retrain_df = pd.DataFrame({
        'lambda': retrain_lambdas.flatten(),
        'x_opt': retrain_xopt.flatten(),
        
    })
    surrogate_model, model, X_train, X_test, y_train, y_test, scaler_X, scaler_y = train_and_prepare_surrogate(
        retrain_df, n_training=len(retrain_df), random_state=random_state
    )
    self.surrogate = surrogate_model
    y_pred, y_std, metrics = evaluate_model(model, X_test, y_test)
    print("Surrogate model evaluation metrics after retrain:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


def retrain_surrogate_with_selected(self, random_state=None):
    # 1. Prepara i lambda selezionati come lista di tuple
    selected_lambdas = [tuple(lam) for lam in self.Selected]
    n_samples = len(selected_lambdas)
    # 2. Salva i lambda selezionati in un file temporaneo (o passa direttamente la lista)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmpfile:
        output_file = tmpfile.name
    # 3. Usa compute_and_save_covariance_samples per calcolare tutte le feature derivate
    # Modifica compute_and_save_covariance_samples per accettare una lista di lambda opzionale:
    # compute_and_save_covariance_samples(n_samples, output_file, lambda_list=None)
    df = build_dataset(n_samples, output_file, lambda_list=selected_lambdas)
    # 4. Riallena il modello surrogato con il nuovo dataset
    surrogate_model, model, X_train, X_test, y_train, y_test, scaler_X, scaler_y = train_and_prepare_surrogate(
        df, n_training=len(df), random_state=random_state
    )
    self.surrogate = surrogate_model
    print(f"Surrogate model retrained with {n_samples} selected lambdas.")


def maybe_reload_surrogate(local_model, model_path="surrogate_toy_gp1.pkl", last_mtime=None):
    if os.path.exists(model_path):
        mtime = os.path.getmtime(model_path)
        if last_mtime is None or mtime > last_mtime:
            surrogate = joblib.load(model_path)
            print("Surrogate model reloaded.")
            return surrogate, mtime
    return local_model, last_mtime

def plot_results(colony_results, criteri_colonie):
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan']

    # 1. Grafico dell'errore per iterazione
    plt.figure(figsize=(10, 6))
    for idx, criterion in enumerate(criteri_colonie):
        res = colony_results[criterion]
        errors = res.get("error_list", [])
        plt.plot(range(1, len(errors) + 1), errors, marker='o', label=criterion, color=colors[idx % len(colors)])
    plt.xlabel("Iteration")
    plt.ylabel("Error (Frobenius Norm)")
    plt.title("Error per Iteration for Each Criterion")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("error_per_iteration.png", dpi=300)
    plt.show()

    # 2. Grafico del numero di campioni selezionati per iterazione
    plt.figure(figsize=(10, 6))
    for idx, criterion in enumerate(criteri_colonie):
        res = colony_results[criterion]
        selected_counts = res.get("selected_count_list", [])
        plt.plot(range(1, len(selected_counts) + 1), selected_counts, marker='o', label=criterion,
                 color=colors[idx % len(colors)])
    plt.xlabel("Iteration")
    plt.ylabel("Number of Selected Samples")
    plt.title("Number of Selected Samples per Iteration for Each Criterion")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("selected_samples_per_iteration.png", dpi=300)
    plt.show()

    # 3. Grafico a barre del valore di errore più basso raggiunto
    plt.figure(figsize=(8, 6))
    lowest_errors = [min(colony_results[criterion].get("error_list", [float('inf')])) for criterion in criteri_colonie]
    plt.bar(criteri_colonie, lowest_errors, color=colors[:len(criteri_colonie)])
    plt.xlabel("Criterion")
    plt.ylabel("Lowest Error (Frobenius Norm)")
    plt.title("Lowest Error Achieved for Each Criterion")
    plt.tight_layout()
    plt.savefig("lowest_error_per_criterion.png", dpi=300)
    plt.show()


# main.py

def main():
    print("Synergy iterativa ACO ↔ Active Learning TOY")

    # Define file paths for input data and model
    archive_file = 'toy_quadratic_dataset.csv'
    ground_truth_file = 'results_covariance_toy.csv'
    model_path = 'surrogate_toy_gp1.pkl'

    archive_data = pd.read_csv(archive_file)
    print(f"Loaded archive data with shape: {archive_data.shape}", flush=True)

    gp_model = get_or_train_model(archive_file, model_path, n_training=500)

    results_df = pd.read_csv(ground_truth_file, header=None)
    C_ref = results_df.values
    print(f" C_ref shape: {C_ref.shape}", flush=True)

    # Define parameters for the ACO process
    params = dict(
        n_ants=550,
        top_k=60,
        alpha=1.0,
        beta=1.0,
        omega=1.0,
        epsilon=0.001,
        budget=15,
        retrain_every=5,
        n_init_diverse=110
    )
    print("\n--- Parameters used for ACO and Active Learning ---")
    for k, v in params.items():
        print(f"  {k}: {v}")
    # Define the number of colonies and selection criteria
    n_colonies = 4
    criteri_colonie = ["heuristic", "uncertainty", "uniform", "random"]  # scegli quelli che vuoi

    # Run colonies in parallel using a ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_colonies) as executor:
        futures = [
            executor.submit(
                run_single_colony,
                i,
                archive_data,
                gp_model,
                C_ref,
                params,
                random_state=GLOBAL_SEED + i,
                selection_criterion=criteri_colonie[i % len(criteri_colonie)]
            )
            for i in range(n_colonies)
        ]
        colony_results = {}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            colony_results[result["criterion"]] = result

    # Combine all selected lambdas from all colonies
    all_selected_unique = list({
                                   tuple(lam): lam
                                   for res in colony_results.values()
                                   if res["best_config"] is not None and isinstance(res["best_config"], list)
                                   for lam in res["best_config"]
                               }.values())
    print(f"Total unique lambdas selected: {len(all_selected_unique)}")



    # Compute the final empirical covariance matrix
    C_e_final = ACOActiveLearner(archive_data, gp_model).compute_Ce_from_lambdas(all_selected_unique)
    final_error = np.linalg.norm(C_ref - C_e_final, ord='fro') if C_e_final is not None else None
    
    #print("\n C_ref matrix (Ground Truth):")
    #print(np.array2string(C_ref, precision=4, suppress_small=True))

    #print("\nC_e_final matrix (Empirical):")
    #if C_e_final is not None:
    #    print(np.array2string(C_e_final, precision=4, suppress_small=True))
    #else:
    #    print("C_e_final not available")

    print("\n--- Final Results ---")
    print(f"Lambdas selected from all colonies: {len(all_selected_unique)}")
    print(f"Final error (unione): {final_error:.4f}")
    print(f"\n C_ref matrix:\n{C_ref}")
    print(f"\n C_e_union matrix:\n{C_e_final}")
    print("Norma Frobenius  C_e_union:", np.linalg.norm(C_ref - C_e_final, ord='fro'))
    print("Shape C_e_union:", C_e_final.shape)
    print(f"Mean of C_e_final elements: {np.mean(C_e_final):.6f}")
    print(f"Variance of C_e_final elements: {np.var(C_e_final):.6f}")
    print(f"Mean of C_ref elements: {np.mean(C_ref):.6f}")
    print(f"Variance of C_ref elements: {np.var(C_ref):.6f}")

    print("\n--- Colony Results ---")
    for i, criterion in enumerate(criteri_colonie):
        res = colony_results[criterion]
        print(f"Colony {i}:")
        print(f"  Selected lambdas: {len(res['selected'])}")
        print(f"  Best configuration: {res['best_config']}")
        print(f"  Best error: {res['best_error']:.4f}")
        print(f"  Selection criterion: {res['criterion']}")
            # Calcolo e stampa della matrice di covarianza, media e varianza per la best configuration della colonia
        best_lambdas = res['best_config']
        if best_lambdas is not None and len(best_lambdas) > 1:
            C_e_colony = ACOActiveLearner(archive_data, gp_model).compute_Ce_from_lambdas(best_lambdas)
            print(f"  [Colony {i}] Covariance matrix (C_e_colony):\n{np.array2string(C_e_colony, precision=4, suppress_small=True)}")
            print(f"  [Colony {i}] Mean of C_e_colony elements: {np.mean(C_e_colony):.6f}")
            print(f"  [Colony {i}] Variance of C_e_colony elements: {np.var(C_e_colony):.6f}")
        else:
            print(f"  [Colony {i}] Not enough lambdas to compute covariance matrix.")
            
      
        
    plot_results(colony_results, criteri_colonie)
    print("Plots saved as PNG files.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(" Execuion error:", e, flush=True)
        traceback.print_exc()