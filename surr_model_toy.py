import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Parametri del problema
a = 1.0
b = -2.0


def f1(x):
    return 0.5 * (x - a) ** 2


def f2(x):
    return 0.5 * (x - b) ** 2


def scalarized_objective(x, lmbd):
    """Calcola la funzione scalare per dato x e lambda."""
    return lmbd * f1(x) + (1 - lmbd) * f2(x)


def optimize_for_lambda(lmbd):
    """Restituisce x ottimo e valore ottimo per dato lambda."""
    x_opt = lmbd * a + (1 - lmbd) * b
    f_opt = scalarized_objective(x_opt, lmbd)
    return x_opt, f_opt


def generate_lambda_samples(n_samples):
    """Genera n_samples valori di lambda in [0,1]."""
    return np.linspace(0, 1, n_samples)


def generate_perturbed_lambdas(lmbd, n_perturb=10, delta=0.05):
    """Genera n_perturb valori di lambda perturbati attorno a lmbd."""
    perturbed = lmbd + np.random.uniform(-delta, delta, n_perturb)
    perturbed = np.clip(perturbed, 0, 1)
    return perturbed


def hessian_estimation_for_lambda(lmbd, delta=1e-3):
    """Stima la derivata seconda numerica della funzione ottima rispetto a lambda."""
    lmbd = float(lmbd)
    x0, _ = optimize_for_lambda(lmbd)
    x_p, _ = optimize_for_lambda(lmbd + delta)
    x_m, _ = optimize_for_lambda(lmbd - delta)
    f0 = scalarized_objective(x0, lmbd)
    fp = scalarized_objective(x_p, lmbd + delta)
    fm = scalarized_objective(x_m, lmbd - delta)
    hess = (fp - 2 * f0 + fm) / (delta ** 2)
    return hess


def estimate_local_covariances_from_lambdas(lambda_samples, n_perturb=10, delta=0.05):
    """Stima la varianza locale della soluzione ottima rispetto a lambda."""
    covariances = []
    for lmbd in lambda_samples:
        perturbed = generate_perturbed_lambdas(lmbd, n_perturb, delta)
        xs = [optimize_for_lambda(l)[0] for l in perturbed]
        cov = np.var(xs)
        covariances.append(cov)
    return np.array(covariances)


def build_dataset(
        n_lambda=50,
        n_perturb=10,
        delta=0.05,
        lambda_list=None,
        n_samples=5000,
        previous_df=None
):
    """
    Crea un dataset con lambda, x_opt, f_opt, hessian, local_cov.
    Se previous_df è fornito, aggiunge solo i nuovi lambda non già presenti.
    """
    if lambda_list is not None:
        lambda_samples = list(lambda_list)
    else:
        lambda_samples = list(generate_lambda_samples(n_samples))

    # Se previous_df è fornito, evita duplicati
    if previous_df is not None and not previous_df.empty:
        existing_lambdas = set(previous_df["lambda"].round(8))
        lambda_samples = [l for l in lambda_samples if round(float(l), 8) not in existing_lambdas]
        data = previous_df.to_dict("records")
    else:
        data = []

    for lmbd in lambda_samples:
        lmbd_float = float(lmbd)
        x_opt, f_opt = optimize_for_lambda(lmbd_float)
        hess = hessian_estimation_for_lambda(lmbd_float)
        local_cov = estimate_local_covariances_from_lambdas([lmbd_float], n_perturb, delta)[0]
        data.append({
            "lambda": lmbd_float,
            "x_opt": x_opt,
            "f_opt": f_opt,
            "hessian": hess,
            "local_cov": local_cov
        })
    return pd.DataFrame(data)


def fit_gp_model(data, n_training=100, random_state=42):
    X = data[['lambda']].values
    y = data["x_opt"].values

    # Scaling
    scaler_X = StandardScaler().fit(X)
    X_scaled = scaler_X.transform(X)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled,
                                                        train_size=min(n_training, len(X_scaled) - 1),
                                                        random_state=random_state)

    kernel = C(1.0, (1e-3, 1e3)) * RBF(0.5, (1e-2, 1e2))
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25, alpha=1e-8, normalize_y=True)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test, scaler_X, scaler_y


def train_and_prepare_surrogate(data, n_training=100, random_state=42, model_path="surrogate_toy_gp.pkl"):
    """
    Allena un modello surrogato e restituisce un dizionario compatibile con ACOActiveLearner.
    Salva anche il modello su disco.
    """
    model, X_train, X_test, y_train, y_test, scaler_X, scaler_y = fit_gp_model(data, n_training, random_state)
    surrogate_model = {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    # Salva il modello surrogato su disco
    joblib.dump(surrogate_model, model_path)
    print(f"Surrogate model saved to {model_path}")
    return surrogate_model, model, X_train, X_test, y_train, y_test, scaler_X, scaler_y


def predict_gp(gp, lambda_grid):
    """Predice x_opt per una griglia di lambda usando il GP."""
    X_pred = lambda_grid.reshape(-1, 1)
    y_pred, y_std = gp.predict(X_pred, return_std=True)
    return y_pred, y_std


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the GP model on test data (no scaling needed for toy problem).

    Returns:
    y_pred: Predicted values
    y_std: Standard deviation of predictions
    metrics: Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred, y_std = model.predict(X_test, return_std=True)

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


def load_lambda_covariance_data(file_path='toy_quadratic_dataset.csv'):
    """Load the lambda-covariance data from CSV file"""
    return pd.read_csv(file_path)


if __name__ == "__main__":
    # Costruisci dataset
    df = build_dataset(n_lambda=5000, n_perturb=30, delta=0.05)
    df.to_csv("toy_quadratic_dataset.csv", index=False)
    print(df.head())

    # Allena e salva il modello surrogato
    surrogate_model, model, X_train, X_test, y_train, y_test, scaler_X, scaler_y = train_and_prepare_surrogate(df,
                                                                                                               n_training=100,
                                                                                                               random_state=42)

    # Predizione su griglia per il plot
    lambda_grid = np.linspace(0, 1, 100)
    y_pred_grid, y_std_grid = predict_gp(model, lambda_grid)

    # Valutazione su test set
    y_pred_test, y_std_test, metrics = evaluate_model(model, X_test, y_test)
    print(metrics)

    # Plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(df["lambda"], df["x_opt"], "o", label="x_opt true")
    plt.plot(lambda_grid, y_pred_grid, "-", label="GP mean")
    plt.fill_between(lambda_grid, y_pred_grid - 2 * y_std_grid, y_pred_grid + 2 * y_std_grid, color="gray", alpha=0.3,
                     label="GP 2 std")
    plt.xlabel("lambda")
    plt.ylabel("x_opt")
    plt.legend()
    plt.tight_layout()
    plt.show()