import numpy as np
import pandas as pd
from scipy.optimize import minimize, approx_fprime

# Problem definition

def f1(x, Q=0.4, rho=1.4, c=1.7, Dx=0.6):
    """
    Convexified Eu based on simplified quadratic formulation.
    x = [a, b, Dx, x50]
    a	Inlet width (e.g., longer side of a rectangular inlet)
    b	Inlet height (e.g., shorter side of the rectangular inlet)
    Dx	Diameter of the vortex finder (outlet tube)
    x50 particle diameter that has a 50% collection efficiency
    
    - Q: volumetric flow rate (m^3/s)
    - rho: gas density (kg/m^3)
    - c: proportionality constant for v_theta_CS (default 1.5)
    """
    a, b = x[0], x[1]
    
    # Inlet area and velocity
    A_in = a * b
    V_in = Q / A_in
    
    # Vortex finder area and axial velocity
    A_vf = np.pi * (Dx**2) / 4
    vx = Q / A_vf
    
    # Tangential velocity at CS (approximation)
    v_theta_CS = c * V_in
    
    # Compute v_theta_CS / vx ratio
    v_ratio = v_theta_CS / vx
    
    # Compute Rcx using Eq. 4.3.15
    exponent = -0.7
    numerator = 0.03 * (v_ratio ** exponent) + 1
    denominator = 0.700 * (v_ratio ** exponent) + 1
    Rcx = numerator / denominator
    
    
    delta_p = 0.5 * rho * (
        (1 / (1 - Rcx**2)**2) * vx**2 +
        (1 / Rcx**2) * v_theta_CS**2
    )
    
    Eu = (
        1 / (1 - Rcx**2)**2 +
        (1 / Rcx**2) * (v_theta_CS / vx)**2 +
        1
    )
    return Eu



def f2(x, Q=0.1, rho_p=1000, mu=1.8e-3,  D=0.5):
    """
    Realistic Stk50 based on geometry and flow.
    x = [a, b, Dx, x50]
    """
    x50= x[2]
    a, b= x[0], x[1]
    A_in = a * b
    V_in = Q / A_in
    Stk50 = (rho_p * x50**2 * V_in) / (18 * mu * D)
    return Stk50

def combined_loss(x, lmbd):
    """ Scalar objective function for a given x and lambda. Combines f1 and f2 using the weight lambda."""
    return lmbd * f1(x) + (1 - lmbd) * f2(x)

def optimize_for_lambda(lam, x_bounds=None):
    if x_bounds is None:
        # Default bounds for the cyclone design parameters
        x_bounds = [
            (0.1, 2.0),   # a: inlet width [m]
            (0.05, 1.0),  # b: inlet height [m]
            #(0.05, 0.8),  # Dx: vortex finder diameter [m]
            #(0.2, 3.0),   # D: cyclone body diameter [m]
            (1e-6, 50e-6) # x50: particle cut-off diameter [m] (1 micron to 50 micron)
        ]
    objective = lambda x: combined_loss(x, lam)

    # Use multiple starting points to avoid local minima

    best_result = None
    lowest_loss = float('inf')

    for _ in range(30):
        start = np.array([np.random.uniform(b[0], b[1]) for b in x_bounds])
        result = minimize(objective, start,bounds=x_bounds, method='L-BFGS-B')
        if result.success and result.fun < lowest_loss:
            lowest_loss = result.fun
            best_result = result

    if best_result is None:
        raise RuntimeError("Optimization failed for all starting points.")

    return best_result.x, best_result.fun


def generate_lambda_samples(n_samples, random_state=None):
    """Generates 'n_samples' values of lambda in the range [0, 1]."""
    if random_state is not None:
        np.random.seed(random_state)
    return np.linspace(0, 1, n_samples)

def compute_and_save_covariance(N_ref=60000, data_filename='results_data_problem.csv', covariance_filename='results_covariance_problem.csv', random_state=None):
    """ Computes the optimal solutions for many lambda values,saves the results and the covariance matrix to files."""
    # Generate lambda samples
    lambda_samples = generate_lambda_samples(N_ref, random_state=random_state)
    solutions = []
    convex_flags = []

    # Compute the optimal solution for each lambda
    for lmbd in lambda_samples:
        x_opt, _ = optimize_for_lambda(lmbd)
        hess = numerical_hessian(lambda x: combined_loss(x, lmbd), x_opt)
        eigvals = np.linalg.eigvalsh(hess)
        is_convex = np.all(eigvals > 1e-8)  # Soglia numerica per evitare problemi di arrotondamento
        convex_flags.append(is_convex)
        solutions.append({
            'lambda': lmbd,
            'x_opt': x_opt
        })

    # Create a DataFrame with the results
    df = pd.DataFrame(solutions)
    x_opt_cols = pd.DataFrame(df['x_opt'].tolist(), columns=[f'x{i+1}' for i in range(5)])
    df = pd.concat([df.drop(columns=['x_opt']), x_opt_cols], axis=1)
    X = df[[f'x{i+1}' for i in range(5)]].values

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
    print(f"Lambda {lmbd}:")
    print("Hessiana nel punto ottimo:")
    print(hess)
    print("Autovalori:", eigvals)
    print("Convessa:" if is_convex else "Non convessa!", "\n")
    return df, C_ref, x_mean

def numerical_hessian(f, x, eps=1e-5):
    """Calcola l'Hessiana numerica della funzione f nel punto x."""
    x = np.asarray(x)
    n = x.size
    hess = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        x1 = np.array(x, copy=True)
        x1[i] += eps
        fxi = f(x1)
        for j in range(i, n):
            x2 = np.array(x, copy=True)
            x2[i] += eps
            x2[j] += eps
            fxij = f(x2)
            x3 = np.array(x, copy=True)
            x3[j] += eps
            fxj = f(x3)
            hess[i, j] = (fxij - fxi - fxj + fx) / (eps ** 2)
            if i != j:
                hess[j, i] = hess[i, j]
    return hess


if __name__ == "__main__":
    df_sol, cov_mat, mean = compute_and_save_covariance(N_ref=10000,random_state=42)
    print("Mean x̄:", mean)
    print("Covariance matrix  C_ref:\n", cov_mat)