import numpy as np
import pandas as pd

# Problem definition
a = 1.0
b = -2.0

def f1(x):
    return 0.5 * (x - a) ** 2

def f2(x):
    return 0.5 * (x - b) ** 2

def scalarized_objective(x, lmbd):
    """ Scalar objective function for a given x and lambda. Combines f1 and f2 using the weight lambda."""
    return lmbd * f1(x) + (1 - lmbd) * f2(x)

def optimize_for_lambda(lmbd):
    """ Returns the optimal x for a given lambda (analytical solution).For the quadratic problem, the optimal solution is a weighted average of 'a' and 'b'."""
    x_opt = lmbd * a + (1 - lmbd) * b
    return x_opt

def generate_lambda_samples(n_samples, random_state=None):
    """Generates 'n_samples' values of lambda in the range [0, 1]."""
    if random_state is not None:
        np.random.seed(random_state)
    return np.linspace(0, 1, n_samples)

def compute_and_save_covariance(N_ref=10000, data_filename='results_data_toy.csv', covariance_filename='results_covariance_toy.csv'):
    """ Computes the optimal solutions for many lambda values,saves the results and the covariance matrix to files."""
    # Generate lambda samples
    lambda_samples = generate_lambda_samples(N_ref, random_state=random_state)
    solutions = []

    # Compute the optimal solution for each lambda
    for lmbd in lambda_samples:
        x_opt = optimize_for_lambda(lmbd)
        solutions.append({
            'lambda': lmbd,
            'x_opt': x_opt
        })

    # Create a DataFrame with the results
    df = pd.DataFrame(solutions)
    X = df[['x_opt']].values  # shape (N_ref, 1)
    x_mean = np.mean(X, axis=0)
    X_centered = X - x_mean
    # Compute the covariance matrix (for a single variable, this is the variance)
    C_ref = (X_centered.T @ X_centered) / (N_ref - 1)

    # Save the main data to the file 'data_filename'
    df.to_csv(data_filename, index=False)

    # Append the mean to the file 'data_filename'
    with open(data_filename, 'a') as f:
        f.write('\n# Media\n')
        f.write(','.join(map(str, x_mean)) + '\n')

    # Save the covariance matrix to the file 'covariance_filename'
    with open(covariance_filename, 'w') as f:
        for row in C_ref:
            f.write(','.join(map(str, row)) + '\n')

    print(f"Data saved to '{data_filename}' and covariance matrix to '{covariance_filename}'")
    return df, C_ref, x_mean


if __name__ == "__main__":
    df_sol, cov_mat, mean = compute_and_save_covariance(N_ref=10000,random_state=42)
    print("Mean x̄:", mean)
    print("Covariance matrix  C_ref:\n", cov_mat)