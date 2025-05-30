# aco_active_learning.py
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from surrogate_model import fit_gp_model, optimize_for_lambda, hessian_estimation_for_lambda, \
    estimate_local_covariances_from_lambdas, load_lambda_covariance_data, train_and_prepare_surrogate, evaluate_model
import matplotlib.pyplot as plt
import os
import joblib
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, r2_score
import concurrent.futures
from filelock import FileLock
import time

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

NUM_PERTURBATIONS = 20
PERTURBATION_STRENGTH = 0.025


class ACOActiveLearner:
    def __init__(self, lambda_data, surrogate_model=None, rho=0.1):
        self.lambda_data = lambda_data.copy()
        self.lambda_data['lambda3'] = 1 - self.lambda_data['lambda1'] - self.lambda_data['lambda2']
        self.lambdas = self.lambda_data[['lambda1', 'lambda2', 'lambda3']].values

        self.tau = defaultdict(lambda: 1.0)
        self.eta = {}
        self.surrogate = surrogate_model

        self.Selected = set()
        self.C_e = None

        self.epsilon_threshold = 1e-2
        self.rho = rho
        self.x_opt_cache = {}  # Cache for x_opt values

    def get_x_opt(self, lam):
        ''' Get x_opt for a given lambda, using cache to avoid recomputation '''
        lam_tuple = tuple(lam)
        if lam_tuple not in self.x_opt_cache:
            x_opt, _ = optimize_for_lambda(lam_tuple)
            self.x_opt_cache[lam_tuple] = x_opt
        return self.x_opt_cache[lam_tuple]

    def compute_heuristic(self):
        ''' Calculate heuristic eta for each lambda, rappresenting the norm of the covariance matrix using the surrogate model '''
        if self.surrogate is None:
            raise ValueError("Surrogate model is not provided.")
        print("Calcolating heuristic eta...")
        X = self.lambda_data[['lambda1', 'lambda2']].values
        X_scaled = self.surrogate['scaler_X'].transform(X)
        y_pred_scaled, _ = self.surrogate['model'].predict(X_scaled, return_std=True)
        y_pred = self.surrogate['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_pred_mean = y_pred.mean()
        y_pred_std = y_pred.std() if y_pred.std() > 0 else 1.0
        for i, lam in enumerate(self.lambdas):
            lam_tuple = tuple(lam)
            # Penality for diversity
            diversity_penalty = 1 / (1 + min([np.linalg.norm(lam - np.array(sel)) for sel in self.Selected] + [1.0]))
            self.eta[lam_tuple] = ((y_pred[i] - y_pred_mean) / y_pred_std) * diversity_penalty

    def select_diverse_lambdas(self, n_select=20, min_dist=0.08, score_dict=None, exclude_lambdas=None,
                               random_state=None):
        ''' Select diverse lambdas based on the heuristic eta '''
        print("Selecting diverse lambdas...")
        if score_dict is None:
            score_dict = self.eta
        if exclude_lambdas is None:
            exclude_lambdas = set()
        sorted_lambdas = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            rng.shuffle(sorted_lambdas)
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

    def select_initial_diverse_lambdas(self, n_init_diverse=20, random_state=None):
        ''' Select initial diverse lambdas for the first iteration '''
        print("Selecting initial diverse lambdas...")
        sorted_lambdas = sorted(self.eta.items(), key=lambda x: x[1], reverse=True)
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            rng.shuffle(sorted_lambdas)
        selected = []
        for lam, score in sorted_lambdas:
            lam_tuple = tuple(lam)
            if lam_tuple in selected:
                continue
            selected.append(lam_tuple)
            if len(selected) >= n_init_diverse:
                break
        print(f"Number of initial diverse lambdas selected: {len(selected)}")
        return selected

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
        if random_state is not None:
            rng = np.random.RandomState(random_state)
        else:
            rng = np.random.RandomState(GLOBAL_SEED)
        n_ants = min(n_ants, len(keys))
        indices = rng.choice(len(keys), size=n_ants, replace=False, p=probs)
        sampled = [keys[i] for i in indices]
        print(f"Total candidates sampled: {len(sampled)}")
        return sampled

    def run_aco_active_learning(self, C_ref, archive_data, n_ants=150,
                                alpha=1.0, beta=1.0,
                                epsilon=None, budget=10, retrain_every=6,
                                n_init_diverse=5, exclude_lambdas=None, random_state=None,
                                min_lambda=10, top_k=20,
                                model_path="surrogate_model.pkl", reload_surrogate=True):
        ''' Run the Ant Colony Optimization (ACO) active learning algorithm to iteratively select
    a diverse and informative set of lambdas that minimize the error between the empirical
    covariance matrix and the reference matrix C_ref.

    At each iteration:
      - Candidates are sampled based on pheromone and heuristic values.
      - The incremental error (delta_k) for each candidate is computed.
      - The best candidates (with lowest delta_k) are selected and added to the set.
      - Pheromones are updated based on the improvement in error.
      - If the number of selected lambdas exceeds a threshold, pruning is performed to keep only the most diverse ones.
      - The best configuration (with the lowest error so far) is tracked.

    The process stops if the error drops below epsilon or the number of selected lambdas falls below min_lambda.
    '''
        if epsilon is None:
            epsilon = 1e-6
        if len(self.Selected) == 0:
            self.Selected = set(self.select_diverse_lambdas(n_select=n_init_diverse, min_dist=0.08))
            self.C_e = self.compute_Ce_from_lambdas(self.Selected)
        val_count = len(self.Selected)
        print(f"Nummber of selected lambdas: {val_count}")
        best_error = float('inf')
        best_config = None
        error_list = []
        max_iter = budget
        last_mtime = None
        for iter_idx in range(max_iter):
            # --- AGGIUNTA: retrain modello surrogato ogni retrain_every ---
            '''if retrain_every and iter_idx > 0 and iter_idx % retrain_every == 0:
                print(f" Retraining surrogate model at iteration {iter_idx}...")
                # Riaddestra il modello con i dati aggiornati (puoi personalizzare la funzione di retrain)
                data = archive_data  # oppure aggiorna con nuovi dati se necessario
                surrogate_model, model, X_train, X_test, y_train, y_test, scaler_X, scaler_y = train_and_prepare_surrogate(
                    data, n_training=len(data), random_state=random_state
                )
                joblib.dump(surrogate_model, model_path)
                self.surrogate = surrogate_model
                print(f"Surrogate model retrained and updated at iteration {iter_idx}.")

                # Valuta il modello e stampa le metriche
                y_pred, y_std, metrics = evaluate_model(model, X_test, y_test, scaler_X, scaler_y)
                print("Surrogate model evaluation metrics after retrain:")
                for k, v in metrics.items():
                    print(f"  {k}: {v:.4f}")'''

            # --- reload modello surrogato se richiesto ---
            if reload_surrogate and model_path is not None:
                self.surrogate, last_mtime = maybe_reload_surrogate(self.surrogate, model_path, last_mtime)
            print(f"\n--- Iteration {iter_idx + 1}/{max_iter} ---", flush=True)
            print(f" Selected lambdas: {len(self.Selected)}", flush=True)
            print(
                f" Current error: {np.linalg.norm(C_ref - self.C_e, ord='fro'):.8f}" if self.C_e is not None else "Current error: N/A",
                flush=True)

            # 1. Sapling new candidates
            candidates = self.sample_candidates(n_ants=top_k, alpha=alpha, beta=beta, random_state=random_state)
            # Exclude already selected lambdas
            new_candidates = [lam for lam in candidates if lam not in self.Selected]
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

            self.evaporate_pheromones()
            self.update_pheromones(selected_candidates, C_ref)

            # 3. Add new candidates to the selected set
            n_added = 0
            for lam in selected_candidates:
                lam_tuple = tuple(lam)
                if lam_tuple not in self.Selected:
                    self.Selected.add(lam_tuple)
                    n_added += 1
            print(f"New candidates added to selected set: {n_added}")
            print(f"Total selected lambdas after addition: {len(self.Selected)}")

            # 4. Pruning
            max_lambdas = 90  # max number of lambdas to keep
            if len(self.Selected) > max_lambdas:
                before_pruning = set(self.Selected)
                self.Selected = set(
                    self.select_diverse_lambdas(n_select=max_lambdas, min_dist=0.08, score_dict=self.tau))
                n_removed = len(before_pruning) - len(self.Selected)
                print(f"Lambda removed during pruning: {n_removed}")
                print(f"Lambda selected after pruning: {len(self.Selected)}")
            else:
                print("No pruning needed")

            self.C_e = self.compute_Ce_from_lambdas(self.Selected)
            if self.C_e is not None:
                error = np.linalg.norm(C_ref - self.C_e, ord='fro')
            else:
                error = None
            error_list.append(error)
            # Save the best configuration
            if error is not None and error < best_error:
                best_error = error
                best_config = list(self.Selected)
                print(f"[Iteration {iter_idx + 1}] Improvement: error {error:.8f} < best error {best_error:.8f}")
            else:
                print(f"[Iteration {iter_idx + 1}] No improvement: error {error:.8f} >= best error {best_error:.8f}")
            print(f"Actual error: {error:.8f}" if error is not None else "Actual error: N/A")
            print(f"Best configuration: {best_config} with error: {best_error:.8f}")
            if (error is not None and error < epsilon) or (len(self.Selected) < min_lambda):
                print(f"STOP: error < {epsilon} or selected < {min_lambda}")
                break

        plt.figure(figsize=(8, 4))
        plt.plot(error_list, marker='o')
        plt.title("Frobenius Error for Each Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Frobenius Error")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("error_iteration.png")
        plt.show()
        return list(self.Selected), self.C_e, error_list[-1] if error_list else None, best_config, best_error

    def compute_Ce_from_lambdas(self, lambdas_list):
        ''' Compute the empirical covariance matrix from selected lambdas '''
        if len(lambdas_list) <= 1:
            return None
        x_list = []
        for lam in lambdas_list:
            x_opt = self.get_x_opt(lam)
            x_list.append(x_opt)
            # print(f"[DEBUG] Lambda: {lam}, x_opt: {x_opt}")
        X = np.array(x_list)
        # print("[DEBUG] X shape:", X.shape)
        # print("[DEBUG] X mean:", np.mean(X, axis=0))
        # print("[DEBUG] X std:", np.std(X, axis=0))
        x_bar = np.mean(X, axis=0)
        deviations = X - x_bar
        C_e = (deviations.T @ deviations) / (len(lambdas_list) - 1)
        return C_e

    def compute_delta_k(self, C_ref, lam):
        ''' Compute the incremental change in error (delta_k) when adding a new lambda to the current set of selected lambdas.

        If no lambdas have been selected yet, it computes the L1 norm (sum of absolute differences)
      between the reference matrix C_ref and the empirical matrix computed using only the new lambda.
      The result is negated so that a lower error means a higher (better) delta_k.'''

        lam_tuple = tuple(lam)
        if not self.Selected:
            C_e_single = self.compute_Ce_from_lambdas([lam_tuple])
            delta_k = -np.sum(np.abs(C_ref - C_e_single))  # norm L1
        else:
            C_e_old = self.compute_Ce_from_lambdas(list(self.Selected))
            C_e_new = self.compute_Ce_from_lambdas(list(self.Selected) + [lam_tuple])
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


def run_single_colony(colony_id, archive_data, gp_model, C_ref, params, already_selected=None, random_state=None):
    ''' Run a single colony of the Ant Colony Optimization (ACO) algorithm.

    This function initializes an ACOActiveLearner instance for a given colony, computes the heuristic values,
    selects an initial set of diverse lambdas, and then runs the ACO active learning loop to iteratively select
    the most informative lambdas that minimize the error with respect to the reference covariance matrix.
    '''
    print(f"Colony {colony_id}: starting ")
    aco = ACOActiveLearner(archive_data, gp_model)
    print(f"Colny {colony_id}: calculating heuristic eta")
    aco.compute_heuristic()
    print(f"Colony {colony_id}: selecting initial diverse lambdas")
    if already_selected is None:
        already_selected = set()
    aco.Selected = set(aco.select_diverse_lambdas(
        n_select=params['n_init_diverse'],
        min_dist=0.08,
        exclude_lambdas=already_selected,
        random_state=random_state
    ))
    print(f"Colony {colony_id}: selected {len(aco.Selected)} initial diverse lambdas")

    Selected, C_e, final_error, best_config, best_error = aco.run_aco_active_learning(
        C_ref=C_ref,
        archive_data=archive_data,
        n_ants=params['n_ants'],
        alpha=params['alpha'],
        beta=params['beta'],
        epsilon=params['epsilon'],
        budget=params['budget'],
        retrain_every=params['retrain_every'],
        n_init_diverse=params['n_init_diverse'],
        exclude_lambdas=already_selected,
        random_state=random_state,
        min_lambda=params['n_init_diverse'] // 2,
        top_k=params['top_k'],
        model_path=params.get('model_path', 'surrogate_model.pkl'),  # Passa il path
        reload_surrogate=True
    )
    return {
        "selected": list(Selected),
        "best_config": best_config,
        "best_error": best_error
    }


def append_new_selected(lam, results_dict, shared_file="shared_selected.csv"):
    df = pd.DataFrame([results_dict])
    lock = FileLock(shared_file + ".lock")
    with lock:
        if not os.path.exists(shared_file):
            df.to_csv(shared_file, index=False)
        else:
            df.to_csv(shared_file, mode='a', header=False, index=False)


def retrain_loop(shared_file="shared_selected.csv", model_path="surrogate_global.pkl", interval=600):
    while True:
        lock = FileLock(shared_file + ".lock")
        with lock:
            if os.path.exists(shared_file):
                df = pd.read_csv(shared_file)
                model, scaler_X, scaler_y = fit_gp_model(df)
                joblib.dump({'model': model, 'scaler_X': scaler_X, 'scaler_y': scaler_y}, model_path)
                print("Surrogate model retrained and saved.")
        time.sleep(interval)


def maybe_reload_surrogate(local_model, model_path="surrogate_global.pkl", last_mtime=None):
    if os.path.exists(model_path):
        mtime = os.path.getmtime(model_path)
        if last_mtime is None or mtime > last_mtime:
            surrogate = joblib.load(model_path)
            print("Surrogate model reloaded.")
            return surrogate, mtime
    return local_model, last_mtime


# main.py

def main():
    print("Synergy iterativa ACO ↔ Active Learning")
    archive_file = 'losses_cov.csv'
    ground_truth_file = 'results_covariance1.csv'
    model_path = 'surrogate_model.pkl'

    archive_data = pd.read_csv(archive_file)
    print(f"Loaded archive data with shape: {archive_data.shape}", flush=True)

    gp_model = get_or_train_model(archive_file, model_path, n_training=500)

    results_df = pd.read_csv(ground_truth_file, header=None)
    C_ref = results_df.values
    print(f" C_ref shape: {C_ref.shape}", flush=True)

    params = dict(
        n_ants=250,
        top_k=50,
        alpha=1.0,
        beta=1.0,
        omega=0.7,
        epsilon=0.1,
        budget=10,
        retrain_every=5,
        n_init_diverse=70
    )

    n_colonies = 3
    all_best_configs = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_colonies) as executor:
        futures = [
            executor.submit(
                run_single_colony,
                i,
                archive_data,
                gp_model,
                C_ref,
                params,
                random_state=GLOBAL_SEED + i
            )
            for i in range(n_colonies)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            all_best_configs.append(result["best_config"])

    # Combine all selected lambdas from all colonies
    all_selected_unique = list({tuple(lam): lam for config in all_best_configs for lam in config}.values())
    print(f"Total unique lambdas selected: {len(all_selected_unique)}")
    C_e_final = ACOActiveLearner(archive_data, gp_model).compute_Ce_from_lambdas(all_selected_unique)
    final_error = np.linalg.norm(C_ref - C_e_final, ord='fro') if C_e_final is not None else None
    print(f"Final error (union): {final_error:.4f}")

    print("\n C_ref matrix (Ground Truth):")
    print(np.array2string(C_ref, precision=4, suppress_small=True))

    print("\nC_e_final matrix (Empirical):")
    if C_e_final is not None:
        print(np.array2string(C_e_final, precision=4, suppress_small=True))
    else:
        print("C_e_final not available")

    colony_errors = []
    for i in range(n_colonies):
        colony_selected = futures[i].result()["selected"]
        C_e_colony = ACOActiveLearner(archive_data, gp_model).compute_Ce_from_lambdas(colony_selected)
        err = np.linalg.norm(C_ref - C_e_colony, ord='fro') if C_e_colony is not None else None
        colony_errors.append(err)

    plt.figure(figsize=(6, 4))
    plt.bar(range(n_colonies), colony_errors)
    plt.xlabel("Colony")
    plt.ylabel("Final Error (Frobenius Norm)")
    plt.title("Final Error per Colony")
    plt.tight_layout()
    plt.savefig("final_error_per_colony.png")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(C_ref, aspect='auto', cmap='viridis')
    plt.title("C_ref (Ground Truth)")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    if C_e_final is not None and len(C_e_final.shape) == 2:
        plt.imshow(C_e_final, aspect='auto', cmap='viridis')
        plt.title("C_e_final (Empirical)")
        plt.colorbar()
    else:
        plt.text(0.5, 0.5, "C_e_final not available", ha='center', va='center')
        plt.title("C_e_final (Empirica.)")
    plt.tight_layout()
    plt.savefig("Cref_vs_Cefinal.png")
    plt.show()
    print("\n--- Final Results ---")
    print(f"Lambdas selected from all colonies: {len(all_selected_unique)}")
    print(f"Final error (unione): {final_error:.4f}")
    print(f"\n C_ref matrix:\n{C_ref}")
    print(f"\n C_e_union matrix:\n{C_e_final}")
    print("Norma Frobenius  C_e_union:", np.linalg.norm(C_ref - C_e_final, ord='fro'))
    print("Shape C_e_union:", C_e_final.shape)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(" Execuion error:", e, flush=True)
        traceback.print_exc()