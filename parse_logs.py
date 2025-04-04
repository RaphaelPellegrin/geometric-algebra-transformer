# Parses the giant file

import pandas as pd
from log_parser import save_experiment_results, extract_metrics_from_log

# Path to your log file
log_file = "/Users/pellegrinraphael/Desktop/Repos_Equivariant/logs/gatr_nbody_7618646.log"

# Directory to save the results
output_dir = "/Users/pellegrinraphael/Desktop/Repos_Equivariant/logs/results"

# Process the logs and save results
json_path, csv_path = save_experiment_results(log_file, output_dir)
print(f"Results saved as JSON: {json_path}")
print(f"Results saved as CSV: {csv_path}")

# Load the results into a DataFrame
df = pd.read_csv(csv_path)

# Display the results
print("\nExperiment Results Summary:")
print(df)

# You can now analyze the data
if len(df) > 0:
    # Plot results
    import matplotlib.pyplot as plt

    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot training steps vs different metrics (using eval metrics instead of validation)
    axes[0, 0].plot(df["training_steps"], df["eval_rmse"], "o-", label="Eval RMSE")
    axes[0, 0].set_xlabel("Training Steps")
    axes[0, 0].set_ylabel("RMSE")
    axes[0, 0].set_title("Eval RMSE vs Training Steps")
    axes[0, 0].grid(True)

    # Plot training steps vs MSE
    axes[0, 1].plot(df["training_steps"], df["eval_mse"], "o-", label="Eval MSE")
    axes[0, 1].set_xlabel("Training Steps")
    axes[0, 1].set_ylabel("MSE")
    axes[0, 1].set_title("Eval MSE vs Training Steps")
    axes[0, 1].grid(True)

    # Plot training steps vs MAE
    axes[1, 0].plot(df["training_steps"], df["eval_mae"], "o-", label="Eval MAE")
    axes[1, 0].set_xlabel("Training Steps")
    axes[1, 0].set_ylabel("MAE")
    axes[1, 0].set_title("Eval MAE vs Training Steps")
    axes[1, 0].grid(True)

    # Plot e3 generalization vs object generalization
    if "e3_generalization_rmse" in df.columns and "object_generalization_rmse" in df.columns:
        axes[1, 1].plot(
            df["training_steps"], df["e3_generalization_rmse"], "o-", label="E3 Gen RMSE"
        )
        axes[1, 1].plot(
            df["training_steps"], df["object_generalization_rmse"], "s-", label="Object Gen RMSE"
        )
        axes[1, 1].set_xlabel("Training Steps")
        axes[1, 1].set_ylabel("RMSE")
        axes[1, 1].set_title("Generalization Performance")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{output_dir}/training_metrics.png")
    print(f"Plot saved to {output_dir}/training_metrics.png")

    # If you want to see the plot immediately
    plt.show()
