print("Import 1 OK", flush=True)
from surr import fit_gp_model, evaluate_model, plot_predicted_vs_true, load_lambda_covariance_data, compute_and_save_covariance_samples, optimize_for_lambda,train_and_prepare_surrogate
print("Import 2 OK", flush=True)
import numpy as np
print("Import 3 OK", flush=True)
import matplotlib.pyplot as plt
print("Import 4 OK", flush=True)
import os
print("Import 5 OK", flush=True)
import joblib
print("Import 6 OK", flush=True)
import sys
print("Import 7 OK", flush=True)


# --- Covariance computation parameters ---
NUM_PERTURBATIONS = 20 # Number of perturbations for covariance estimation
PERTURBATION_STRENGTH = 0.025 # Strength of lambda perturbations
GLOBAL_SEED = 42 # Define a global seed for reproducibility
np.random.seed(GLOBAL_SEED)
# --- Data generation parameters ---
NUM_LAMBDA_SAMPLES = 2000 # Number of base lambda vectors for dataset
print("Import 8 OK", flush=True)
# Execute the code


def main():
    print("SCRIPT STARTED")
    print("Starting the GP modeling and evaluation process...")
    dataset_file = 'problem2.csv'

    print(f"Checking for existing dataset file: {dataset_file}")
    
    
    if os.path.exists(dataset_file):
        print(f"Dataset '{dataset_file}' found. Loading existing samples...")
        lambda_cov_samples = load_lambda_covariance_data(file_path=dataset_file)
    else:
        print(f"Dataset '{dataset_file}' not found. Generating covariance samples using compute_covariance_for_lambda()...")
        lambda_cov_samples = compute_and_save_covariance_samples(NUM_LAMBDA_SAMPLES, dataset_file)
        print(f"Saved '{dataset_file}' with {len(lambda_cov_samples)} samples.")

    
    # Visualize the results
    
    
    # Display summary statistics
    print("\nSummary statistics of sensitivity norms:")
    print(lambda_cov_samples['sensitivity_norm'].describe())
    
    
    """Main function to orchestrate the GP modeling and evaluation """
    # Load the data
    print("Loading lambda-covariance data ...")
    data = load_lambda_covariance_data()
    print(f"Loaded {len(data)} samples.")
    
    
    
    # Fit the GP model
    surrogate_model, model, X_train, X_test, y_train, y_test, scaler_X, scaler_y= train_and_prepare_surrogate(data, n_training=500)
    surrogate_model_path = 'surrogate_problem2.pkl'

    joblib.dump(surrogate_model, surrogate_model_path)
    print(f"Modello surrogate salvato in: {surrogate_model_path}")
    # Evaluate the model
    y_pred, y_std, metrics = evaluate_model(model, X_test, y_test, scaler_X, scaler_y)

    # Print metrics
    print("\nModel performance on test data (before Active Learning):")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    
    # Plots
    plot_predicted_vs_true(y_test, y_pred, y_std)
    plt.hist(lambda_cov_samples['sensitivity_norm'], bins=30)
    plt.xlabel('Sensitivity Norm')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sensitivity Norm')
    plt.savefig('hist_sensitivity_norm.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.scatter(lambda_cov_samples['lambda'], lambda_cov_samples['sensitivity_norm'], alpha=0.5)
    plt.xlabel('Lambda')
    plt.ylabel('Sensitivity Norm')
    plt.title('Sensitivity Norm vs Lambda')
    plt.savefig('scatter_lambda_vs_sensitivity_norm.png', dpi=300, bbox_inches='tight')
    plt.close()

    lambda_grid = X_test
    plt.plot(lambda_grid, y_std)
    plt.xlabel('Lambda')
    plt.ylabel('Predicted Std (Uncertainty)')
    plt.title('GP Predicted Uncertainty vs Lambda')
    plt.savefig('gp_predicted_uncertainty_vs_lambda.png', dpi=300, bbox_inches='tight')
    plt.close()
       
    print("\nAll visualizations have been saved.")
    print("Done!")

if __name__ == "__main__":
    print("Starting...", flush=True)
    main()
    