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
from pymoo.problems.many import DTLZ2
from scipy.stats import norm

# --- Covariance computation parameters ---
NUM_PERTURBATIONS = 20  # Number of perturbations for covariance estimation
PERTURBATION_STRENGTH = 0.025  # Strength of lambda perturbations
GLOBAL_SEED = 42  # Define a global seed for reproducibility
np.random.seed(GLOBAL_SEED)
# --- Data generation parameters ---
NUM_LAMBDA_SAMPLES = 5000  # Number of base lambda vectors for dataset


# Problem Definition
def loss_1(x):
    # x vettore 3D, applichiamo la formula elemento per elemento e sommiamo
    return np.sum(x * np.cos(x) - x ** 3 + np.sin(x))


def loss_2(x):
    return np.sum(5 * np.sin(x) + 5 * x ** 3 - np.cos(x) * np.exp(x))


def loss_3(x):
    return np.sum(-4 * x ** 4 * np.cos(x) + np.sin(x) * np.exp(-x))


def combined_loss(x, lambdas):
    lambdas = np.array(lambdas)
    if not np.isclose(np.sum(lambdas), 1.0) or np.any(lambdas < 0):
        raise ValueError("lambdas must be non-negative and sum to 1")

    return lambdas[0] * loss_1(x) + lambdas[1] * loss_2(x) + lambdas[2] * loss_3(x)


def optimize_for_lambda(lambdas, x_range=(-2, 2)):
    objective = lambda x: combined_loss(x, lambdas)

    # Use multiple starting points to avoid local minima

    best_result = None
    lowest_loss = float('inf')

    for _ in range(10):
        start = np.random.uniform(x_range[0], x_range[1], 3)
        result = minimize(objective, start, bounds=[x_range] * 3, method='L-BFGS-B')
        if result.success and result.fun < lowest_loss:
            lowest_loss = result.fun
            best_result = result

    if best_result is None:
        raise RuntimeError("Ottimizzazione fallita")

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


def generate_perturbed_lambdas(lambda_vec, num_perturbations, strength, rng):
    """
    Generates perturbed lambda vectors around a central lambda_vec using a provided RNG.
    All generated vectors are normalized to sum to 1 and clipped to [0,1].
    rng: An instance of numpy.random.Generator for deterministic random number generation.
    """
    perturbed_lambdas_list = []
    base_lambda = np.array(lambda_vec)

    # Generate a fixed set of unique perturbations based on the rng state
    # To ensure num_perturbations are distinct if possible, we might need more attempts
    # or a different strategy if strength is very low or num_perturbations is high.
    # For simplicity, we generate and then take unique ones.

    generated_count = 0
    attempts = 0
    max_attempts = num_perturbations * 5  # Try more times to get unique perturbations

    temp_perturbed_lambdas = []

    while generated_count < num_perturbations and attempts < max_attempts:
        perturbation = rng.normal(0, strength, 3)
        # The original 'while np.all(perturbation == 0)' loop is unlikely with rng.normal
        # but good to be mindful of extremely small strengths.
        # If strength is 0, this will always be base_lambda.

        perturbed = base_lambda + perturbation
        perturbed = np.clip(perturbed, 0, 1)  # Ensure lambdas are in [0,1]

        sum_perturbed = np.sum(perturbed)
        if sum_perturbed == 0:
            # This case happens if all components are clipped to 0.
            # A deterministic outcome if base_lambda + perturbation leads to all <=0.
            perturbed = np.ones(3) / 3.0
        else:
            perturbed /= sum_perturbed

        temp_perturbed_lambdas.append(tuple(perturbed))
        generated_count += 1
        attempts += 1

    # Use set to remove duplicates, then convert back to list
    # Sort to ensure order is deterministic if content is the same across runs
    perturbed_lambdas_list = sorted(list(set(temp_perturbed_lambdas)))

    # If we still don't have enough unique perturbations, we might return fewer.
    # Or, one could add the base_lambda itself if not present.
    if not perturbed_lambdas_list and num_perturbations > 0:  # If list is empty but we wanted some
        perturbed_lambdas_list.append(tuple(base_lambda / np.sum(base_lambda)))  # Add normalized base as a fallback

    return perturbed_lambdas_list[:num_perturbations]  # Return up to num_perturbations


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


def hessian_estimation_for_lambda(lambdas, delta=0.01, base_seed=None):
    """
    Compute the covariance matrix for a given lambda vector by perturbing its components.

    Args:
        lambdas: Array-like, the lambda vector (must sum to 1).
        delta: Float, the perturbation step size.
        base_seed: Optional, seed for reproducibility.

    Returns:
        perturbed_losses: 3x3 covariance matrix.
        x_opt: Optimal x for the base lambda vector.

    #descrive quanto cambia il valore ottimale del loss scalarizzato rispetto ai pesi
    """
    # Converti lambdas in un array NumPy per supportare operazioni come .copy()
    lambdas = np.array(lambdas)

    # Calculate the optimal loss at the base lambda point
    x_opt, base_loss = optimize_for_lambda(lambdas)

    # Store perturbed losses
    perturbed_losses = np.zeros((3, 3))

    # For each pair of lambdas, calculate the perturbation effect
    for i in range(3):
        for j in range(3):
            if i == j:
                # Diagonal elements - single parameter perturbation
                if i < 2:  # Only perturb λ1 and λ2 directly
                    lambda_perturbed = lambdas.copy()
                    lambda_perturbed[i] += delta
                    lambda_perturbed[2] -= delta

                    if np.all(lambda_perturbed >= 0):
                        x_pert, pert_loss = optimize_for_lambda(lambda_perturbed)
                        perturbed_losses[i, i] = (pert_loss - base_loss) / delta
                    else:
                        lambda_perturbed = lambdas.copy()
                        lambda_perturbed[i] -= delta
                        lambda_perturbed[2] += delta

                        if np.all(lambda_perturbed >= 0):
                            x_pert, pert_loss = optimize_for_lambda(lambda_perturbed)
                            perturbed_losses[i, i] = (base_loss - pert_loss) / delta
                        else:
                            perturbed_losses[i, i] = 0
                else:
                    lambda_perturbed = lambdas.copy()
                    lambda_perturbed[0] -= delta / 2
                    lambda_perturbed[1] -= delta / 2
                    lambda_perturbed[2] += delta

                    if np.all(lambda_perturbed >= 0):
                        x_pert, pert_loss = optimize_for_lambda(lambda_perturbed)
                        perturbed_losses[2, 2] = (pert_loss - base_loss) / delta
                    else:
                        perturbed_losses[2, 2] = 0
            elif i < 2 and j < 2:
                lambda_perturbed = lambdas.copy()
                lambda_perturbed[i] += delta / 2
                lambda_perturbed[j] += delta / 2
                lambda_perturbed[2] -= delta

                if np.all(lambda_perturbed >= 0):
                    x_pert, pert_loss = optimize_for_lambda(lambda_perturbed)
                    single_i_change = perturbed_losses[i, i] * delta / 2
                    single_j_change = perturbed_losses[j, j] * delta / 2
                    total_change = pert_loss - base_loss
                    interaction = total_change - single_i_change - single_j_change
                    perturbed_losses[i, j] = interaction / ((delta / 2) * (delta / 2))
                    perturbed_losses[j, i] = perturbed_losses[i, j]
                else:
                    perturbed_losses[i, j] = 0
                    perturbed_losses[j, i] = 0
            elif i < 2 and j == 2:
                lambda_perturbed = lambdas.copy()
                lambda_perturbed[i] += delta / 2
                lambda_perturbed[j] += delta / 2
                lambda_perturbed[1 - i] -= delta

                if np.all(lambda_perturbed >= 0):
                    x_pert, pert_loss = optimize_for_lambda(lambda_perturbed)
                    single_i_change = perturbed_losses[i, i] * delta / 2
                    single_j_change = perturbed_losses[j, j] * delta / 2
                    total_change = pert_loss - base_loss
                    interaction = total_change - single_i_change - single_j_change
                    perturbed_losses[i, j] = interaction / ((delta / 2) * (delta / 2))
                    perturbed_losses[j, i] = perturbed_losses[i, j]
                else:
                    perturbed_losses[i, j] = 0
                    perturbed_losses[j, i] = 0
    # perturbed_losses : how changes the optimal loss (F) with respect to the perturbations of the lambdas
    return perturbed_losses, x_opt


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
    samples = []
    # Generate samples using random numbers and normalization
    # y_i = -log(u_i), where u_i ~ U(0,1)
    # lambda_i = y_i / sum(y_j)
    rng = np.random.default_rng(GLOBAL_SEED)
    raw_samples = -np.log(rng.random((num_samples, 3)))
    for sample in raw_samples:
        samples.append(tuple(sample / np.sum(sample)))

    # Add some specific boundary and central cases for better coverage
    samples.append((1.0, 0.0, 0.0))
    samples.append((0.0, 1.0, 0.0))
    samples.append((0.0, 0.0, 1.0))
    samples.append((0.5, 0.5, 0.0))
    samples.append((0.5, 0.0, 0.5))
    samples.append((0.0, 0.5, 0.5))
    samples.append((1 / 3, 1 / 3, 1 / 3))

    return list(set(samples))  # Remove duplicates


def compute_and_save_covariance_samples(n_samples, output_file):
    """
    Generates lambda samples, computes covariance matrices, solution covariance,
    objective covariance, and saves the results to a CSV file.
    """
    lambda_samples = generate_lambda_samples(n_samples)

    records = []
    for i, lambdas in enumerate(lambda_samples):
        # Compute the covariance matrix and optimal solution for the given lambda
        cov_matrix, x_opt = hessian_estimation_for_lambda(
            lambdas,
            delta=0.01,
            base_seed=GLOBAL_SEED + i
        )
        # Compute the solution covariance and objective covariance
        Sigma_x, Sigma_f = estimate_local_covariances_from_lambdas(
            lambda_vec=lambdas,
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

        # Append the results to the records
        records.append({
            'lambda1': lambdas[0],
            'lambda2': lambdas[1],
            'lambda3': lambdas[2],
            'x_opt': x_opt.tolist(),
            'cov_matrix': cov_matrix.tolist(),
            'solution_covariance': Sigma_x.tolist(),  # Add Sigma_x from estimate_local_covariances_from_lambdas
            'objective_covariance': Sigma_f.tolist(),  # Add Sigma_f from estimate_local_covariances_from_lambdas
            'sensitivity_norm': np.linalg.norm(cov_matrix, ord='fro'),
            'P_matrix': P.tolist(),
            'P_flattened': P_flattened.tolist()
        })

    # Save the records to a CSV file
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    return df


def load_lambda_covariance_data(file_path='losses_cov.csv'):
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
    X = data[['lambda1', 'lambda2']].values  # lambda3 is redundant (sum to 1)
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
    rbf = RBF(length_scale=[1.0, 1.0], length_scale_bounds=(0.01, 10.0))

    # 3. Rational Quadratic kernel - handles multiple length scales
    # Better than RBF for modeling functions with varying smoothness
    rational_quad = RationalQuadratic(length_scale=1.0, alpha=0.5,
                                      length_scale_bounds=(0.01, 10.0),
                                      alpha_bounds=(0.1, 10.0))

    # 4. Matérn kernel - can model less smooth functions than RBF
    # nu=1.5 is less smooth than the standard nu=2.5
    matern = Matern(length_scale=[1.0, 1.0], nu=1.5,
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
        alpha=1e-10,  # Small alpha for numerical stability
        normalize_y=False,  # We already scaled the data
        n_restarts_optimizer=15,  # More restarts to find better hyperparameters
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

def plot_optimal_values_surface():
    ''' Plots the optimal values of x for different lambda combinations. '''

    resolution = 20
    lambda1_vals = np.linspace(0, 1, resolution)
    lambda2_vals = np.linspace(0, 1, resolution)
    # OSS: lambda3 = 1 - lambda1 - lambda2

    valid_lambda1 = []
    valid_lambda2 = []
    valid_lambda3 = []
    optimal_losses = []
    optimal_x_values = []

    for l1 in lambda1_vals:
        for l2 in lambda2_vals:
            l3 = 1 - l1 - l2

            if l3 >= 0:
                lambdas = np.array([l1, l2, l3])
                x_min, loss_min = optimize_for_lambda(lambdas)

                valid_lambda1.append(l1)
                valid_lambda2.append(l2)
                valid_lambda3.append(l3)
                optimal_losses.append(loss_min)
                optimal_x_values.append(x_min)

    valid_lambda1 = np.array(valid_lambda1)
    valid_lambda2 = np.array(valid_lambda2)
    valid_lambda3 = np.array(valid_lambda3)
    optimal_losses = np.array(optimal_losses)
    optimal_x_values = np.array(optimal_x_values)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(valid_lambda1, valid_lambda2, valid_lambda3,
                         c=optimal_losses, cmap=cm.viridis,
                         s=50, alpha=0.8)

    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    colorbar.set_label('Optimal Loss Value', fontsize=12)

    ax.set_xlabel('Lambda 1')
    ax.set_ylabel('Lambda 2')
    ax.set_zlabel('Lambda 3')
    ax.set_title('Optimal loss for Different Lambda Combinations')

    plt.savefig('optimal_loss_surface_3d.png', dpi=300)
    return fig


def visualize_covariance_results(df):
    """ Create visualizations for the lambda-covariance samples """

    # 1. Plot the sensitivity norm on the simplex
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    scatter = ax1.scatter(df['lambda1'], df['lambda2'], c=df['sensitivity_norm'],
                          cmap='viridis', s=30, alpha=0.7)
    plt.colorbar(scatter, label='Sensitivity Norm (Covariance Matrix)')

    # Add the simplex boundary
    ax1.plot([0, 1, 0, 0], [0, 0, 1, 0], 'k-', linewidth=1.5)

    # Add labels
    ax1.text(0, 0, 'λ3 = 1', fontsize=12, ha='center', va='center')
    ax1.text(1, 0, 'λ1 = 1', fontsize=12, ha='center', va='center')
    ax1.text(0, 1, 'λ2 = 1', fontsize=12, ha='center', va='center')

    ax1.set_xlabel('Lambda 1', fontsize=12)
    ax1.set_ylabel('Lambda 2', fontsize=12)
    ax1.set_title('Sensitivity of Optimal Loss to Lambda Perturbations (losses)', fontsize=14)
    ax1.set_aspect('equal')

    plt.savefig('sensitivity_norm_simplex_losses.png', dpi=300)

    # 2. Histogram of sensitivity norms
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(df['sensitivity_norm'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Sensitivity Norm', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Sensitivity Norms (losses)', fontsize=14)

    plt.savefig('sensitivity_norm_histogram_losses.png', dpi=300)

    # 3. 3D scatter plot of lambdas and sensitivity
    fig3 = plt.figure(figsize=(12, 10))
    ax3 = fig3.add_subplot(111, projection='3d')

    scatter3 = ax3.scatter(df['lambda1'], df['lambda2'], df['lambda3'],
                           c=df['sensitivity_norm'], cmap='plasma', s=30, alpha=0.7)

    ax3.set_xlabel('Lambda 1', fontsize=12)
    ax3.set_ylabel('Lambda 2', fontsize=12)
    ax3.set_zlabel('Lambda 3', fontsize=12)
    ax3.set_title('Lambda Values, Sensitivity, and Optimal Loss (losses)', fontsize=14)

    fig3.colorbar(scatter3, ax=ax3, label='Sensitivity Norm')

    plt.savefig('lambda_sensitivity_3d_losses.png', dpi=300)

    return fig1, fig2, fig3


def plot_gp_surface_with_test_points(model, X_train, X_test, y_train, y_test, y_pred,
                                     y_std, scaler_X, scaler_y):
    """
    Plot the GP model surface and the test points
    """
    # Create a grid for the lambda simplex
    resolution = 50
    l1 = np.linspace(0, 1, resolution)
    l2 = np.linspace(0, 1, resolution)
    L1, L2 = np.meshgrid(l1, l2)

    # Filter out points outside the simplex (lambda1 + lambda2 <= 1)
    valid_indices = (L1 + L2 <= 1)

    # Create a mesh grid of lambda values
    X_mesh = np.column_stack((L1.ravel(), L2.ravel()))

    # Only keep points inside the simplex
    valid_points = X_mesh[np.ravel(valid_indices)]

    # Scale the points and predict
    X_mesh_scaled = scaler_X.transform(valid_points)
    y_mesh_scaled, y_mesh_std_scaled = model.predict(X_mesh_scaled, return_std=True)

    # Unscale the predictions
    y_mesh = scaler_y.inverse_transform(y_mesh_scaled.reshape(-1, 1)).ravel()
    y_mesh_std = y_mesh_std_scaled * scaler_y.scale_

    # Set up the 3D figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Reshape for plotting
    L1_valid = valid_points[:, 0]
    L2_valid = valid_points[:, 1]

    # Plot the GP surface using triangulation
    ax.plot_trisurf(L1_valid, L2_valid, y_mesh,
                    cmap=cm.viridis, alpha=0.7, linewidth=0.2, edgecolor='gray')

    # Plot training points in red
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train,
               color='red', s=5, label='Training points')

    # Plot test points in blue
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test,
               color='blue', s=8, label='Test points')

    # Set labels and title
    ax.set_xlabel('Lambda 1')
    ax.set_ylabel('Lambda 2')
    ax.set_zlabel('Sensitivity Norm')
    ax.set_title('GP Model of Sensitivity for losses')
    ax.legend()

    plt.savefig('gp_sensitivity_surface_losses.png', dpi=300, bbox_inches='tight')

    # Create a 2D plot of the model prediction on the simplex with uncertainty
    fig2, ax2 = plt.subplots(figsize=(10, 8))

    # Create a triangulation for plotting
    triang_plot = plt.matplotlib.tri.Triangulation(L1_valid, L2_valid)

    # Plot the mean prediction
    tcf = ax2.tricontourf(triang_plot, y_mesh, levels=20, cmap='viridis')
    plt.colorbar(tcf, ax=ax2, label='Predicted Sensitivity Norm')

    # Plot the training and test points
    ax2.scatter(X_train[:, 0], X_train[:, 1], c='red', s=30, alpha=0.7, label='Training')
    ax2.scatter(X_test[:, 0], X_test[:, 1], c='blue', s=30, alpha=0.7, label='Test')

    # Add the simplex boundary
    ax2.plot([0, 1, 0, 0], [0, 0, 1, 0], 'k-')

    # Labels
    ax2.text(0, 0, 'λ3 = 1', fontsize=12)
    ax2.text(1, 0, 'λ1 = 1', fontsize=12)
    ax2.text(0, 1, 'λ2 = 1', fontsize=12)

    ax2.set_xlabel('Lambda 1')
    ax2.set_ylabel('Lambda 2')
    ax2.set_title('GP Model Prediction on Lambda Simplex (losses)')
    ax2.set_aspect('equal')
    ax2.legend()

    plt.savefig('gp_sensitivity_2d_losses.png', dpi=300, bbox_inches='tight')

    return fig, fig2


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


def plot_gp_surface_with_test_points_enhanced(model, X_train_orig, y_train_orig, X_test_orig, y_test_orig,
                                              y_pred_test, test_point_errors, scaler_X_fitted, scaler_y_fitted,
                                              title_suffix="Covariance Norm"):
    """
    Plot the GP model surface, training points, and test points (colored by error).
    X_train_orig, y_train_orig, X_test_orig, y_test_orig are unscaled original values.
    y_pred_test are unscaled predictions for X_test_orig.
    test_point_errors are absolute errors for test points.
    scaler_X_fitted, scaler_y_fitted are the fitted scalers.
    """
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Create a grid for the lambda simplex surface
    resolution = 40
    l1_lin = np.linspace(0, 1, resolution)
    l2_lin = np.linspace(0, 1, resolution)
    L1_mesh, L2_mesh = np.meshgrid(l1_lin, l2_lin)

    # Valid points in the simplex (l1 + l2 <= 1)
    valid_simplex_indices = (L1_mesh + L2_mesh <= 1.001)  # Add small tolerance

    grid_l1 = L1_mesh[valid_simplex_indices]
    grid_l2 = L2_mesh[valid_simplex_indices]

    X_grid = np.column_stack((grid_l1.ravel(), grid_l2.ravel()))

    # Scale grid points and predict
    X_grid_scaled = scaler_X_fitted.transform(X_grid)
    y_grid_scaled, _ = model.predict(X_grid_scaled, return_std=True)  # std not used for surface here
    y_grid_pred = scaler_y_fitted.inverse_transform(y_grid_scaled.reshape(-1, 1)).ravel()

    # Plot the GP surface using triangulation
    # The surface is for the mean prediction
    surf = ax.plot_trisurf(grid_l1, grid_l2, y_grid_pred, cmap=cm.viridis, alpha=0.7,
                           linewidth=0.1, antialiased=True, edgecolor='none', shade=True)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label=f'Predicted {title_suffix}')

    # Plot training points (unscaled)
    ax.scatter(X_train_orig[:, 0], X_train_orig[:, 1], y_train_orig.ravel(),
               c='blue', marker='o', s=30, label='Training Points', alpha=0.7, depthshade=False, edgecolors='w',
               linewidth=0.5)

    # Plot test points (unscaled), colored by their absolute error
    # Ensure test_point_errors has a suitable range for colormap
    min_err, max_err = np.min(test_point_errors), np.max(test_point_errors)
    if min_err == max_err:  # Handle case with uniform error (e.g., single test point)
        normalized_errors = np.ones_like(test_point_errors) * 0.5
    else:
        normalized_errors = (test_point_errors - min_err) / (max_err - min_err)

    error_cmap = cm.get_cmap('Reds')
    test_point_colors = error_cmap(normalized_errors)

    sc_test = ax.scatter(X_test_orig[:, 0], X_test_orig[:, 1], y_test_orig.ravel(),
                         c=test_point_colors, marker='^', s=50, label='Test Points (Actual)',
                         alpha=0.9, depthshade=False, edgecolors='k', linewidth=0.5)

    # Create a colorbar for test point errors
    # This requires a bit of manual setup for a scatter plot with colors mapped to values
    # One way is to use a ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=error_cmap, norm=plt.Normalize(vmin=min_err, vmax=max_err))
    sm.set_array([])  # You need to set an array for the ScalarMappable, even an empty one
    cbar_errors = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.02,
                               label='Absolute Prediction Error on Test Points')

    # Aesthetics and Labels
    ax.set_xlabel('$\lambda_1$', fontsize=14)
    ax.set_ylabel('$\lambda_2$', fontsize=14)
    ax.set_zlabel(f'True/Predicted {title_suffix}', fontsize=14)
    ax.set_title(f'GP Model of {title_suffix} with Test Points', fontsize=16, pad=20)

    # Simplex boundary lines for clarity
    ax.plot([0, 1], [0, 0], [ax.get_zlim()[0], ax.get_zlim()[0]], 'k-', alpha=0.5, linewidth=1.5)  # l1 axis
    ax.plot([0, 0], [0, 1], [ax.get_zlim()[0], ax.get_zlim()[0]], 'k-', alpha=0.5, linewidth=1.5)  # l2 axis
    ax.plot([1, 0], [0, 1], [ax.get_zlim()[0], ax.get_zlim()[0]], 'k-', alpha=0.5, linewidth=1.5)  # l1+l2=1 line

    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
    ax.view_init(elev=25, azim=-40)  # Adjust view angle for better visualization
    plt.tight_layout()
    plt.show()
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
    # Allena il modello surrogato
    model, X_train, X_test, y_train, y_test, scaler_X, scaler_y = fit_gp_model(data, n_training, random_state)

    # Prepara il dizionario del modello surrogato
    surrogate_model = {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }

    return surrogate_model, model, X_train, X_test, y_train, y_test, scaler_X, scaler_y