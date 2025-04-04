# This script is used to parse the giant log file and extract the metrics for each experiment

import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


def extract_metrics_from_log(log_text: str) -> List[Dict[str, Any]]:
    """
    Extract experiment metrics from the log text.

    Args:
        log_text:
            The complete log text from experiments

    Returns:
        A list of dictionaries containing the experiment parameters and metrics
    """
    # Find all experiment sections
    experiment_results: List[Dict[str, Any]] = []

    # Extract experiments with different training steps
    step_patterns = [
        r"Starting experiment with (\d+) training steps",
        r"\[.+\] Training for (\d+) steps",
    ]

    # Find all experiment starts
    for pattern in step_patterns:
        matches = re.finditer(pattern, log_text)

        for match in matches:
            steps = int(match.group(1))
            start_pos = match.start()

            # Find the experiment section
            experiment_section = log_text[start_pos:]
            end_marker = "Finished experiment with"
            end_pos = experiment_section.find(end_marker)

            if end_pos > 0:
                experiment_section = experiment_section[:end_pos]

            # Create a result dictionary with steps
            result = {"training_steps": steps}

            # Extract metrics for different datasets, excluding validation loop
            datasets = ["object_generalization", "e3_generalization", "eval"]

            for dataset in datasets:
                # Find the metrics section for this dataset
                dataset_pattern = rf"\[.+\] Ran evaluation on dataset {dataset}:"
                dataset_match = re.search(dataset_pattern, experiment_section)

                if dataset_match:
                    dataset_section = experiment_section[dataset_match.start() :]
                    end_section = dataset_section.find("[", dataset_match.end())
                    if end_section > 0:
                        dataset_section = dataset_section[:end_section]
                    else:
                        # If there's no next log entry, take the rest of the text
                        lines = dataset_section.strip().split("\n")
                        dataset_section = "\n".join(
                            lines[:6]
                        )  # Most metric sections have 5-6 lines

                    # Extract metrics
                    metrics = {
                        "loss": extract_metric(dataset_section, "loss"),
                        "mse": extract_metric(dataset_section, "mse"),
                        "rmse": extract_metric(dataset_section, "rmse"),
                        "output_reg": extract_metric(dataset_section, "output_reg"),
                        "mae": extract_metric(dataset_section, "mae"),
                    }

                    # Add metrics to result with dataset prefix
                    dataset_key = dataset.replace(" ", "_").lower()
                    for metric, value in metrics.items():
                        if value is not None:  # Only add metrics that were found
                            result[f"{dataset_key}_{metric}"] = value

            # Only add the result if we found metrics
            if len(result) > 1:  # More than just training_steps
                experiment_results.append(result)

    return experiment_results


def extract_metric(text: str, metric_name: str) -> Optional[float]:
    """
    Extract a specific metric value from text.

    Args:
        text:
            The text to search in
        metric_name:
            The name of the metric to extract

    Returns:
        The metric value as a float, or None if not found
    """
    pattern = rf"{metric_name} = ([\d.]+)"
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return None


def save_experiment_results(log_file: str, output_dir: str) -> Tuple[str, str]:
    """
    Process experiment logs and save results as JSON and CSV.

    Args:
        log_file:
            Path to the log file
        output_dir:
            Directory to save the results

    Returns:
        Tuple of paths to the saved JSON and CSV files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read log file
    with open(log_file, "r") as f:
        log_text = f.read()

    # Extract metrics
    results = extract_metrics_from_log(log_text)

    # Save as JSON
    json_path = output_path / "experiment_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save as CSV using pandas
    df = pd.DataFrame(results)
    csv_path = output_path / "experiment_results.csv"
    df.to_csv(csv_path, index=False)

    return str(json_path), str(csv_path)


# Example usage
if __name__ == "__main__":
    # Replace with your actual log file path and output directory
    log_file = "experiment_logs.txt"
    output_dir = "experiment_results"

    json_path, csv_path = save_experiment_results(log_file, output_dir)
    print(f"Results saved as JSON: {json_path}")
    print(f"Results saved as CSV: {csv_path}")

    # Load and display the DataFrame
    df = pd.read_csv(csv_path)
    print("\nExperiment Results DataFrame:")
    print(df)

    # Create visualizations if there's data
    if not df.empty and len(df) > 0:
        import matplotlib.pyplot as plt

        # Create a figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot RMSE for different datasets
        for dataset in ["object_generalization", "e3_generalization", "eval"]:
            rmse_col = f"{dataset}_rmse"
            if rmse_col in df.columns:
                axes[0, 0].plot(df["training_steps"], df[rmse_col], "o-", label=dataset)

        axes[0, 0].set_xlabel("Training Steps")
        axes[0, 0].set_ylabel("RMSE")
        axes[0, 0].set_title("RMSE Comparison")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot MSE for different datasets
        for dataset in ["object_generalization", "e3_generalization", "eval"]:
            mse_col = f"{dataset}_mse"
            if mse_col in df.columns:
                axes[0, 1].plot(df["training_steps"], df[mse_col], "o-", label=dataset)

        axes[0, 1].set_xlabel("Training Steps")
        axes[0, 1].set_ylabel("MSE")
        axes[0, 1].set_title("MSE Comparison")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot MAE for different datasets
        for dataset in ["object_generalization", "e3_generalization", "eval"]:
            mae_col = f"{dataset}_mae"
            if mae_col in df.columns:
                axes[1, 0].plot(df["training_steps"], df[mae_col], "o-", label=dataset)

        axes[1, 0].set_xlabel("Training Steps")
        axes[1, 0].set_ylabel("MAE")
        axes[1, 0].set_title("MAE Comparison")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Plot loss for different datasets
        for dataset in ["object_generalization", "e3_generalization", "eval"]:
            loss_col = f"{dataset}_loss"
            if loss_col in df.columns:
                axes[1, 1].plot(df["training_steps"], df[loss_col], "o-", label=dataset)

        axes[1, 1].set_xlabel("Training Steps")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].set_title("Loss Comparison")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plot_path = output_path / "metrics_comparison.png"
        plt.savefig(plot_path)
        print(f"Visualization saved to: {plot_path}")
