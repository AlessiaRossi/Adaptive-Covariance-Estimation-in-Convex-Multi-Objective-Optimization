import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel, DotProduct, RBF, RationalQuadratic
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import random

# --- Covariance computation parameters ---
NUM_PERTURBATIONS = 20  # Number of perturbations for covariance estimation
PERTURBATION_STRENGTH = 0.025  # Strength of lambda perturbations
GLOBAL_SEED = 42  # Define a global seed for reproducibility
np.random.seed(GLOBAL_SEED)
# --- Data generation parameters ---
NUM_LAMBDA_SAMPLES = 4000  # Number of base lambda vectors for dataset


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


'''   
# Problem Definition
n_obj = 3
k = 20
n_vars = n_obj + k - 1 
problem = DTLZ2(n_var=n_vars, n_obj=n_obj)

def scalarized_objective(x, lambdas):

    f = problem.evaluate(np.array([x]))[0]  

    return np.dot(lambdas, f)  


def optimize_for_lambda(lambdas, x_range=(0.0, 1.0)):

    lambdas = np.array(lambdas)

    if not np.isclose(np.sum(lambdas), 1.0) or np.any(lambdas < 0):
        raise ValueError("lambdas must be non-negative and sum to 1")

    objective = lambda x: scalarized_objective(x, lambdas)

    # Use multiple starting points to avoid local minima
    starting_points = np.linspace(x_range[0], x_range[1], 10)
    best_result = None
    lowest_loss = float('inf')

    for start in starting_points:
        result = minimize(objective, 
                          x0=np.full(n_vars, start), 
                          bounds=[(x_range[0], x_range[1])] * n_vars, 
                          method='L-BFGS-B')
        if result.fun < lowest_loss:
            lowest_loss = result.fun
            best_result = result

    return best_result.x, best_result.fun

'''


def generate_perturbed_lambdas(lam, num_perturbations, strength, rng):
    """
    Generates perturbed lambda vectors around a central lambda_vec using a provided RNG.
    All generated vectors are normalized to sum to 1 and clipped to [0,1].
    rng: An instance of numpy.random.Generator for deterministic random number generation.
    """
    perturbed = []
    for _ in range(num_perturbations):
        l = lam + rng.normal(0, strength)
        l = np.clip(l, 0, 1)
        perturbed.append(l)
    return np.unique(perturbed)

# Wrapper for find_optimal_x to control its 'workers' and 'seed' parameter

def find_optimal_x_for_cov_wrapper(lambda_coeffs_tuple, workers_for_de, seed_for_optimizer):
    """
    Finds the optimal x for given lambdas using the differential evolution algorithm.
    lambda_coeffs_tuple: Tuple of lambda coefficients (must sum to 1 and be non-negative).
    workers_for_de: Number of workers for parallel execution in differential evolution.
    seed_for_optimizer: Seed for the optimizer to ensure reproducibility.
    """
    lambda_coeffs = np.array(lambda_coeffs_tuple)

    # Define the objective function using the problem formulation
    objective_func = lambda x_params: combined_loss(x_params[0], lambda_coeffs)

    # Perform optimization using differential evolution
    result = differential_evolution(
        objective_func,
        bounds=[(-2, 2)],  # Bounds for x
        maxiter=1000,  # Maximum number of iterations
        tol=1e-6,  # Tolerance for convergence
        workers=workers_for_de,
        seed=seed_for_optimizer  # Seed for reproducibility
    )

    return result.x[0], result.fun


def hessian_estimation_for_lambda(lam, delta=0.01):
    lam = float(lam)
    lam_p = np.clip(lam + delta, 0, 1)
    lam_m = np.clip(lam - delta, 0, 1)
    x0, _ = optimize_for_lambda(lam)
    x_p, _ = optimize_for_lambda(lam_p)
    x_m, _ = optimize_for_lambda(lam_m)
    f0 = combined_loss(x0, lam)
    fp = combined_loss(x_p, lam_p)
    fm = combined_loss(x_m, lam_m)
    hess = (fp - 2 * f0 + fm) / (delta ** 2)
    return hess, x0



def estimate_local_covariances_from_lambdas(lambda_vec, num_perturbations=10, delta=0.01):
    """
    Estimate the covariance of the optimal solution and objective function
    for a given lambda vector by perturbing it.
    """
    # Create a random number generator
    rng = np.random.default_rng(GLOBAL_SEED)

    # Generate perturbed lambda vectors
    lambda_perturbed_set = generate_perturbed_lambdas(lambda_vec, num_perturbations, delta, rng)

    x_list = []
    f_list = []

    for lam in lambda_perturbed_set:
        x_opt, _ = optimize_for_lambda(lam)
        f_val = combined_loss(x_opt, lam)
        x_list.append(x_opt)
        f_list.append(f_val)

    X = np.array(x_list)
    F = np.array(f_list)

    Sigma_x = np.cov(X.T)  # How the optimal x changes through perturbations of lambda
    Sigma_f = np.cov(F.T)  # How the optimal f changes through perturbations of lambda

    return Sigma_x, Sigma_f


def generate_lambda_samples(num_samples):
    """
    Generates diverse lambda samples that sum to 1.
    Uses Dirichlet distribution properties for more uniform sampling on the simplex.
    """
    rng = np.random.default_rng(GLOBAL_SEED)
    samples = rng.uniform(0, 1, num_samples)
    samples = np.concatenate([samples, [0.0, 0.5, 1.0]])
    return np.unique(samples)

    return list(set(samples))  # Remove duplicates


def compute_and_save_covariance_samples(n_samples, output_file):
    """
    Generates lambda samples, computes covariance matrices, solution covariance,
    objective covariance, and saves the results to a CSV file.
    """
    lambda_samples = generate_lambda_samples(n_samples)
    print(f"Generated {len(lambda_samples)} unique lambda samples.")
    records = []
    for i, lam in enumerate(lambda_samples):
        # Compute the covariance matrix and optimal solution for the given lambda
        cov_matrix, x_opt = hessian_estimation_for_lambda(
            lam,
            delta=0.01
        )
        # --- Fix: ensure cov_matrix is always 2D ---
        cov_matrix = np.atleast_2d(cov_matrix)
        print(f"Processing lambda {i + 1}/{len(lambda_samples)}: {lam:.4f} - Covariance shape: {cov_matrix.shape}")
        # -------------------------------------------
        # Compute the solution covariance and objective covariance
        Sigma_x, Sigma_f = estimate_local_covariances_from_lambdas(
            lambda_vec=lam,
            num_perturbations=NUM_PERTURBATIONS,
            delta=PERTURBATION_STRENGTH
        )
        # Calculate triangular matrix P from covariance matrix
        try:
            P = np.linalg.cholesky(cov_matrix).T
            P_flattened = P[np.triu_indices_from(P)]
        except np.linalg.LinAlgError:
            P = np.full_like(cov_matrix, np.nan)
            P_flattened = np.full((cov_matrix.shape[0] * (cov_matrix.shape[0] + 1)) // 2, np.nan)
        
        print(f"Lambda {lam:.4f} - P shape: {P.shape}")
        # Append the results to the records
        records.append({
            'lambda': lam,
            'x_opt': x_opt.tolist(),
            'cov_matrix': cov_matrix.tolist(),
            'solution_covariance': Sigma_x.tolist(),  # Add Sigma_x from estimate_local_covariances_from_lambdas
            'objective_covariance': Sigma_f.tolist(),  # Add Sigma_f from estimate_local_covariances_from_lambdas
            'sensitivity_norm': np.linalg.norm(cov_matrix, ord='fro'),
            'P_matrix': P.tolist(),
            'P_flattened': P_flattened.tolist()
        })
        print("Processed record for lambda")

    # Save the records to a CSV file
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    return df


def load_lambda_covariance_data(file_path='problem.csv'):
    """Load the lambda-covariance data from CSV file"""
    return pd.read_csv(file_path)


def fit_gp_model(data, n_training=100, random_state=42):
    """
    Fit a Gaussian Process model to predict sensitivity_norm from lambda values
    using a state-of-the-art kernel configuration with learnable parameters.

    Parameters:
    data: DataFrame with lambda and sensitivity data
    n_training: Number of samples to use for training
    random_state: Random seed for reproducibility

    Returns:
    model: Fitted GP model
    X_train, X_test: Training and test feature sets
    y_train, y_test: Training and test target values
    scaler_X, scaler_y: Data scalers
    """
    # Extract features (lambda values) and target (sensitivity norm)
    X = data[['lambda']].values  
    y = data['sensitivity_norm'].values.reshape(-1, 1)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=random_state
    )

    # Scale the data
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train_scaled = scaler_X.transform(X_train)
    y_train_scaled = scaler_y.transform(y_train)

    # Define a state-of-the-art kernel combination:

    # 1. Amplitude component - scales the overall variance
    amplitude = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.01, 10.0))

    # 2. RBF kernel with automatic relevance determination (ARD)
    # Individual length scale for each dimension to capture different importance
    rbf = RBF(length_scale=1.0, length_scale_bounds=(0.01, 10.0))

    # 3. Rational Quadratic kernel - handles multiple length scales
    # Better than RBF for modeling functions with varying smoothness
    rational_quad = RationalQuadratic(length_scale=1.0, alpha=0.5,
                                      length_scale_bounds=(0.01, 10.0),
                                      alpha_bounds=(0.1, 10.0))

    # 4. Matérn kernel - can model less smooth functions than RBF
    # nu=1.5 is less smooth than the standard nu=2.5
    matern = Matern(length_scale=1.0, nu=1.5,
                    length_scale_bounds=(0.01, 10.0))

    # 5. WhiteKernel - represents the noise in the data (fully learnable)
    noise = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1.0))

    # Combine the kernels
    # The sum of kernels allows for modeling different aspects of the data
    # The product with amplitude scales everything appropriately
    kernel = amplitude * (0.5 * rbf + 0.3 * rational_quad + 0.2 * matern) + noise
    # kernel = amplitude * (0.5 * rbf + 0.2 * matern) + noise
    # kernel = amplitude * (0.5 * rbf) + noise

    print("Initial kernel configuration:")
    print(kernel)

    # Create and fit the GP model
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1,  # Small alpha for numerical stability
        normalize_y=False,  # We already scaled the data
        n_restarts_optimizer=35,  # More restarts to find better hyperparameters
        random_state=random_state 
    )

    # Fit the model
    print("\nFitting Gaussian Process model with optimized kernel...")
    model.fit(X_train_scaled, y_train_scaled)

    print(f"\nOptimized kernel parameters:")
    print(model.kernel_)

    # Print the learned noise level
    if hasattr(model.kernel_, 'k2') and hasattr(model.kernel_.k2, 'noise_level'):
        print(f"\nLearned noise level: {model.kernel_.k2.noise_level:.6f}")
    else:
        # Navigate the kernel structure to find the WhiteKernel
        for param_name, param in model.kernel_.get_params().items():
            if isinstance(param, WhiteKernel):
                print(f"\nLearned noise level: {param.noise_level:.6f}")

    # Log marginal likelihood (higher is better)
    print(f"\nLog marginal likelihood: {model.log_marginal_likelihood(model.kernel_.theta):.4f}")

    return model, X_train, X_test, y_train, y_test, scaler_X, scaler_y


def evaluate_model(model, X_test, y_test, scaler_X, scaler_y):
    """
    Evaluate the GP model on test data

    Returns:
    y_pred: Predicted values
    metrics: Dictionary of evaluation metrics
    """
    # Scale the test data
    X_test_scaled = scaler_X.transform(X_test)

    # Make predictions
    y_pred_scaled, y_std_scaled = model.predict(X_test_scaled, return_std=True)

    # Unscale the predictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_std = y_std_scaled * scaler_y.scale_

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

    return y_pred, y_std, metrics


# Plot for problem visualization

def plot_predicted_vs_true(y_test, y_pred, y_std):
    """
    Create a scatter plot of predicted vs true sensitivity norms
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the diagonal perfect prediction line
    max_val = max(np.max(y_test), np.max(y_pred))
    min_val = min(np.min(y_test), np.min(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect prediction')

    # Plot the predictions with error bars (95% confidence intervals)
    ax.errorbar(y_test.ravel(), y_pred.ravel(), yerr=1.96 * y_std,
                fmt='o', markersize=8, alpha=0.6,
                ecolor='lightgray', capsize=5)

    # Add correlation coefficient
    correlation = np.corrcoef(y_test.ravel(), y_pred.ravel())[0, 1]
    ax.annotate(f'Correlation: {correlation:.4f}', xy=(0.05, 0.95),
                xycoords='axes fraction', fontsize=12)

    # Set labels and title
    ax.set_xlabel('True Sensitivity Norm')
    ax.set_ylabel('Predicted Sensitivity Norm')
    ax.set_title('Predicted vs. True Sensitivity Norm (losses)')

    # Add a grid for better readability
    ax.grid(True, alpha=0.3)

    # Make the plot square
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('predicted_vs_true_losses.png', dpi=300)

    return fig



def train_and_prepare_surrogate(data, n_training=100, random_state=42):
    """
    Allena un modello surrogato e restituisce un dizionario compatibile con ACOActiveLearner.

    Parameters:
    data: DataFrame con i dati di input (lambda1, lambda2) e target (sensitivity_norm)
    n_training: Numero di campioni da usare per il training
    random_state: Seed per la riproducibilità

    Returns:
    surrogate_model: Dizionario contenente il modello surrogato e gli scaler
    """
    print("Columns in data:", data.columns)
    # Allena il modello surrogato
    model, X_train, X_test, y_train, y_test, scaler_X, scaler_y = fit_gp_model(data, n_training, random_state)

    # Prepara il dizionario del modello surrogato
    surrogate_model = {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }

    return surrogate_model, model, X_train, X_test, y_train, y_test, scaler_X, scaler_y