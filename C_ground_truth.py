import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Problem definition
def problem(x, lambdas):
    f1 = np.sum(x * np.cos(x) - x ** 3 + np.sin(x))
    f2 = np.sum(5 * np.sin(x) + 5 * x ** 3 - np.cos(x) * np.exp(x))
    f3 = np.sum(-4 * x ** 4 * np.cos(x) + np.sin(x) * np.exp(-x))

    if not np.isclose(np.sum(lambdas), 1.0) or np.any(np.array(lambdas) < 0):
        raise ValueError("lambdas must be non-negative and sum to 1")
    return lambdas[0] * f1 + lambdas[1] * f2 + lambdas[2] * f3

# Optimization function
def optimize_for_lambda(lambdas, x_range=(-2, 2)):
    objective = lambda x: problem(x, lambdas)

    best_result = None
    lowest_loss = float('inf')

    for _ in range(10):
        start = np.random.uniform(x_range[0], x_range[1], size=3)
        result = minimize(objective, start, bounds=[x_range]*3)
        if result.success and result.fun < lowest_loss:
            lowest_loss = result.fun
            best_result = result

    if best_result is None:
        raise RuntimeError("Optimization failed for all attempts.")

    return best_result.x

# Generate random lambda vectors
def generate_lambdas(n_samples, n_objectives=3):
    return np.random.dirichlet(np.ones(n_objectives), size=n_samples)


# Compute and save results in separate CSV files
def compute_and_save_separate_csv(N_ref=10000,
                                  data_filename='C:\\Users\\aless\\Desktop\\Tesi\\Articolo\\Articolo\\results_data.csv',
                                  covariance_filename='C:\\Users\\aless\\Desktop\\Tesi\\Articolo\\Articolo\\results_covariance.csv'):
    lambdas_samples = generate_lambdas(N_ref)
    solutions = []

    for lambdas in lambdas_samples:
        x_opt = optimize_for_lambda(lambdas)
        solutions.append({
            'lambda_1': lambdas[0],
            'lambda_2': lambdas[1],
            'lambda_3': lambdas[2],
            'x_opt_1': x_opt[0],
            'x_opt_2': x_opt[1],
            'x_opt_3': x_opt[2],
        })

    df = pd.DataFrame(solutions)
    X = df[['x_opt_1', 'x_opt_2', 'x_opt_3']].values
    x_mean = np.mean(X, axis=0)
    X_centered = X - x_mean
    C_ref = (X_centered.T @ X_centered) / (N_ref - 1)

    df.to_csv(data_filename, index=False)

    with open(data_filename, 'a') as f:
        f.write('\n# Media\n')
        f.write(','.join(map(str, x_mean)) + '\n')

    with open(covariance_filename, 'w') as f:
        for row in C_ref:
            f.write(','.join(map(str, row)) + '\n')

    print(f"Data saved in '{data_filename}' and covariance matrix in '{covariance_filename}'")
    return df, C_ref, x_mean




df_sol, cov_mat, media = compute_and_save_separate_csv(N_ref=10000)
print("Mean vector:\n", media)
print("Covariance matrix:\n", cov_mat)
