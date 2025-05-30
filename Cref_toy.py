import numpy as np
import pandas as pd

# Parametri del problema
a = 1.0
b = -2.0

def f1(x):
    return 0.5 * (x - a) ** 2

def f2(x):
    return 0.5 * (x - b) ** 2

def scalarized_objective(x, lmbd):
    """Funzione scalare per dato x e lambda."""
    return lmbd * f1(x) + (1 - lmbd) * f2(x)

def optimize_for_lambda(lmbd):
    """Restituisce x ottimo per dato lambda (soluzione analitica)."""
    # Per il problema quadratico la soluzione ottima è:
    x_opt = lmbd * a + (1 - lmbd) * b
    return x_opt

def generate_lambda_samples(n_samples):
    """Genera n_samples valori di lambda in [0,1]."""
    return np.linspace(0, 1, n_samples)

def compute_and_save_covariance(
    N_ref=10000,
    data_filename='results_data_toy.csv',
    covariance_filename='results_covariance_toy.csv'
):
    """
    Calcola le soluzioni ottime per molte combinazioni di lambda,
    salva i risultati e la matrice di covarianza.
    """
    lambda_samples = generate_lambda_samples(N_ref)
    solutions = []

    for lmbd in lambda_samples:
        x_opt = optimize_for_lambda(lmbd)
        solutions.append({
            'lambda': lmbd,
            'x_opt': x_opt
        })

    df = pd.DataFrame(solutions)
    X = df[['x_opt']].values  # shape (N_ref, 1)
    x_mean = np.mean(X, axis=0)
    X_centered = X - x_mean
    # Per una sola variabile, la "matrice" di covarianza è uno scalare (la varianza)
    C_ref = (X_centered.T @ X_centered) / (N_ref - 1)

    # Salva i dati principali nel file data_filename
    df.to_csv(data_filename, index=False)

    # Aggiungi la media al file data_filename
    with open(data_filename, 'a') as f:
        f.write('\n# Media\n')
        f.write(','.join(map(str, x_mean)) + '\n')

    # Salva la matrice di covarianza nel file covariance_filename
    with open(covariance_filename, 'w') as f:
        for row in C_ref:
            f.write(','.join(map(str, row)) + '\n')

    print(f"Dati salvati in '{data_filename}' e matrice di covarianza in '{covariance_filename}'")
    return df, C_ref, x_mean

# Esempio di esecuzione
if __name__ == "__main__":
    df_sol, cov_mat, media = compute_and_save_covariance(N_ref=10000)
    print("Media x̄:", media)
    print("Matrice di covarianza C_ref:\n", cov_mat)