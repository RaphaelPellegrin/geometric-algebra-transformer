# This script is used to parse the individual log files and combine them into a single CSV file

import pandas as pd
import glob
from pathlib import Path

base_dir = "/path/to/base_dir/experiments/nbody"
results = []

# Find all experiment directories
experiment_dirs = glob.glob(f"{base_dir}/gatr_sample*")

for exp_dir in experiment_dirs:
    # Extract experiment parameters from directory name
    dir_name = Path(exp_dir).name
    # Parse parameters from name like "gatr_sample20_steps500_mv16_s128_b10"
    params = {}

    for param in dir_name.split("_"):
        if param.startswith("sample"):
            params["sample_size"] = float(param[6:])
        elif param.startswith("steps"):
            params["training_steps"] = int(param[5:])
        elif param.startswith("mv"):
            params["mv_channels"] = int(param[2:])
        elif param.startswith("s") and not param.startswith("sample"):
            params["s_channels"] = int(param[1:])
        elif param.startswith("b"):
            params["num_blocks"] = int(param[1:])

    # Get metrics from evaluation CSV files
    metrics_dir = Path(exp_dir) / "metrics"

    for metric_file in metrics_dir.glob("eval_*.csv"):
        dataset = metric_file.stem.replace("eval_", "")
        try:
            df = pd.read_csv(metric_file)

            # Add dataset metrics to params
            for col in df.columns:
                params[f"{dataset}_{col}"] = df[col].values[0]

        except Exception as e:
            print(f"Error reading {metric_file}: {e}")

    results.append(params)

# Create DataFrame with all results
all_results = pd.DataFrame(results)

# Save combined results
all_results.to_csv("combined_experiment_results.csv", index=False)
print(f"Combined {len(results)} experiment results.")
