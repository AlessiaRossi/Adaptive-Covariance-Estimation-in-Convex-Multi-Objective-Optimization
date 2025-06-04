from surrogate_model import fit_gp_model, evaluate_model, plot_gp_surface_with_test_points, plot_predicted_vs_true, \
    plot_gp_surface_with_test_points_enhanced, load_lambda_covariance_data, compute_and_save_covariance_samples, \
    visualize_covariance_results, plot_optimal_values_surface, optimize_for_lambda, train_and_prepare_surrogate
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

# --- Optimization parameters ---
OPTIMIZER_BOUNDS = [(0.0, 2.0), (0.0, 2.0)]  # Domain for x = [x_a, x_b]
DE_MAX_ITER = 50  # Max iterations for differential_evolution
DE_TOL = 1e-3  # Tolerance for differential_evolution

# --- Covariance computation parameters ---
NUM_PERTURBATIONS = 20  # Number of perturbations for covariance estimation
PERTURBATION_STRENGTH = 0.025  # Strength of lambda perturbations
GLOBAL_SEED = 42  # Define a global seed for reproducibility
np.random.seed(GLOBAL_SEED)
# --- Data generation parameters ---
NUM_LAMBDA_SAMPLES = 5000  # Number of base lambda vectors for dataset


# Execute the code
def main():
    dataset_file = 'losses_cov1.csv'

    # Plot the optimal loss 3D surface
    plot_optimal_values_surface()

    # Sample a few specific lambda combinations and print their optimal values
    lambda_combinations = [
        [1 / 3, 1 / 3, 1 / 3],  # Equal weights
        [0.8, 0.1, 0.1],  # Mostly λ1
        [0.1, 0.8, 0.1],  # Mostly λ2
        [0.1, 0.1, 0.8]  # Mostly λ3
    ]
    print("\nOptimal solutions for specific lambda combinations:")
    print("-" * 50)
    print(f"{'Lambda':20s} | {'Optimal Loss':12s}")
    print("-" * 50)

    for lambdas in lambda_combinations:
        _, loss_min = optimize_for_lambda(lambdas)
        lambda_str = f"[{lambdas[0]:.2f}, {lambdas[1]:.2f}, {lambdas[2]:.2f}]"
        print(f"{lambda_str:20s} | {loss_min:12.6f}")

    plt.show()

    if os.path.exists(dataset_file):
        print(f"Dataset '{dataset_file}' found. Loading existing samples...")
        lambda_cov_samples = load_lambda_covariance_data(file_path=dataset_file)
    else:
        print(
            f"Dataset '{dataset_file}' not found. Generating covariance samples using compute_covariance_for_lambda()...")
        lambda_cov_samples = compute_and_save_covariance_samples(NUM_LAMBDA_SAMPLES, dataset_file)
        print(f"Saved '{dataset_file}' with {len(lambda_cov_samples)} samples.")

    # Visualize the results
    print("Creating visualizations...")
    visualize_covariance_results(lambda_cov_samples)

    # Display summary statistics
    print("\nSummary statistics of sensitivity norms:")
    print(lambda_cov_samples['sensitivity_norm'].describe())

    # Show the plots
    plt.show()

    """Main function to orchestrate the GP modeling and evaluation """
    # Load the data
    print("Loading lambda-covariance data ...")
    data = load_lambda_covariance_data()
    print(f"Loaded {len(data)} samples.")

    # Ensure lambda3 is calculated to respect the simplex constraint
    data['lambda3'] = 1 - data['lambda1'] - data['lambda2']

    # Fit the GP model
    surrogate_model, model, X_train, X_test, y_train, y_test, scaler_X, scaler_y = train_and_prepare_surrogate(data,
                                                                                                               n_training=500)
    surrogate_model_path = 'surrogate_model1.pkl'

    joblib.dump(surrogate_model, surrogate_model_path)
    print(f"Modello surrogate salvato in: {surrogate_model_path}")
    # Evaluate the model
    y_pred, y_std, metrics = evaluate_model(model, X_test, y_test, scaler_X, scaler_y)

    # Print metrics
    print("\nModel performance on test data (before Active Learning):")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plot the GP surface with test points
    print("\nCreating visualizations...")
    plot_gp_surface_with_test_points(model, X_train, X_test, y_train, y_test, y_pred,
                                     y_std, scaler_X, scaler_y)

    # Plot predicted vs true values
    plot_predicted_vs_true(y_test, y_pred, y_std)
    if len(X_test) > 0:  # Ensure there is test data to plot
        print("\nGenerating 3D GP Surface Plot...")
        test_errors = np.abs(y_test - y_pred)
        fig3d = plot_gp_surface_with_test_points_enhanced(
            model,
            X_train, y_train,  # Original unscaled training data
            X_test, y_test,  # Original unscaled test data
            y_pred,  # Unscaled predictions for test data
            test_errors,  # Absolute errors for test data
            scaler_X, scaler_y,  # Fitted scalers
            title_suffix="Covariance Norm"
        )
        # Save the figure
        fig3d.savefig('3d_gp_surface_plot.png', dpi=300)
    else:
        print("Skipping plots as there is no test data (or too few overall data points).")

    print("\nAll visualizations have been saved.")
    print("Done!")


if __name__ == "__main__":
    main()