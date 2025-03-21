#!/usr/bin/env python3
"""
Analyze training sample size vs. loss relationship from experiment logs.
"""

import os
import re
import glob
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class SampleSizeResult:
    """Data class to store sample size experiment results."""

    sample_size: float
    steps: int
    eval_loss: float
    eval_mse: float
    eval_rmse: float
    eval_mae: float
    object_gen_loss: float
    e3_gen_loss: float


def extract_sample_size_from_dirname(dirname: str) -> Optional[float]:
    """
    Extract sample size from directory name.

    Parameters
    ----------
    dirname : str
        Directory name like 'gatr_sample010_steps500'

    Returns
    -------
    Optional[float]
        Sample size as a float between 0 and 1, or None if parsing failed
    """
    match = re.search(r"gatr_sample(\d+)_steps", dirname)
    if match:
        sample_int = int(match.group(1))
        return sample_int / 100.0
    return None


def extract_metrics_from_log(log_path: str, sample_size: float) -> Optional[SampleSizeResult]:
    """
    Extract metrics from a sample size experiment log file.

    Parameters
    ----------
    log_path : str
        Path to the log file
    sample_size : float
        Sample size for this experiment

    Returns
    -------
    Optional[SampleSizeResult]
        Extracted metrics or None if parsing failed
    """
    try:
        with open(log_path, "r") as f:
            content = f.read()

        # Extract steps
        steps_match = re.search(r"Training for (\d+) steps", content)
        if not steps_match:
            return None
        steps = int(steps_match.group(1))

        # Extract eval dataset metrics
        eval_loss_match = re.search(
            r"Ran evaluation on dataset eval:\s+\[.*?\]\s+loss = ([\d\.]+)", content, re.DOTALL
        )
        eval_loss = float(eval_loss_match.group(1)) if eval_loss_match else float("nan")

        eval_mse_match = re.search(
            r"Ran evaluation on dataset eval:.*?\s+mse = ([\d\.]+)", content, re.DOTALL
        )
        eval_mse = float(eval_mse_match.group(1)) if eval_mse_match else float("nan")

        eval_rmse_match = re.search(
            r"Ran evaluation on dataset eval:.*?\s+rmse = ([\d\.]+)", content, re.DOTALL
        )
        eval_rmse = float(eval_rmse_match.group(1)) if eval_rmse_match else float("nan")

        eval_mae_match = re.search(
            r"Ran evaluation on dataset eval:.*?\s+mae = ([\d\.]+)", content, re.DOTALL
        )
        eval_mae = float(eval_mae_match.group(1)) if eval_mae_match else float("nan")

        # Extract object generalization loss
        obj_gen_match = re.search(
            r"Ran evaluation on dataset object_generalization:.*?\s+loss = ([\d\.]+)",
            content,
            re.DOTALL,
        )
        obj_gen_loss = float(obj_gen_match.group(1)) if obj_gen_match else float("nan")

        # Extract e3 generalization loss
        e3_gen_match = re.search(
            r"Ran evaluation on dataset e3_generalization:.*?\s+loss = ([\d\.]+)",
            content,
            re.DOTALL,
        )
        e3_gen_loss = float(e3_gen_match.group(1)) if e3_gen_match else float("nan")

        return SampleSizeResult(
            sample_size=sample_size,
            steps=steps,
            eval_loss=eval_loss,
            eval_mse=eval_mse,
            eval_rmse=eval_rmse,
            eval_mae=eval_mae,
            object_gen_loss=obj_gen_loss,
            e3_gen_loss=e3_gen_loss,
        )
    except Exception as e:
        print(f"Error processing {log_path}: {e}")
        return None


def collect_sample_size_results(base_dir: str) -> List[SampleSizeResult]:
    """
    Collect results from all sample size experiment directories.

    Parameters
    ----------
    base_dir : str
        Base directory containing experiment folders

    Returns
    -------
    List[SampleSizeResult]
        List of sample size experiment results
    """
    results = []

    # Find all sample size experiment directories
    exp_dirs = glob.glob(os.path.join(base_dir, "gatr_sample*_steps*"))

    for exp_dir in exp_dirs:
        dirname = os.path.basename(exp_dir)
        sample_size = extract_sample_size_from_dirname(dirname)

        if sample_size is not None:
            log_path = os.path.join(exp_dir, "output.log")
            if os.path.exists(log_path):
                result = extract_metrics_from_log(log_path, sample_size)
                if result:
                    results.append(result)

    # Sort by sample size
    results.sort(key=lambda x: x.sample_size)
    return results


def plot_sample_size_scaling(results: List[SampleSizeResult], output_dir: str) -> None:
    """
    Plot scaling curves for sample size experiments.

    Parameters
    ----------
    results : List[SampleSizeResult]
        List of sample size experiment results
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data for plotting
    sample_sizes = np.array([r.sample_size for r in results])
    eval_loss = np.array([r.eval_loss for r in results])
    eval_rmse = np.array([r.eval_rmse for r in results])
    obj_gen_loss = np.array([r.object_gen_loss for r in results])
    e3_gen_loss = np.array([r.e3_gen_loss for r in results])

    # 1. Plot loss vs. sample size (log-log scale)
    plt.figure(figsize=(10, 6))
    plt.loglog(sample_sizes, eval_loss, "o-", label="Eval Loss")
    plt.loglog(sample_sizes, obj_gen_loss, "^-", label="Object Generalization Loss")
    plt.loglog(sample_sizes, e3_gen_loss, "v-", label="E3 Generalization Loss")

    plt.xlabel("Training Sample Size (fraction)")
    plt.ylabel("Loss")
    plt.title("Scaling of Loss with Training Sample Size")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_vs_sample_size_loglog.png"), dpi=300)
    plt.close()

    # 2. Plot RMSE vs. sample size (log-log scale)
    plt.figure(figsize=(10, 6))
    plt.loglog(sample_sizes, eval_rmse, "o-", label="Eval RMSE")

    plt.xlabel("Training Sample Size (fraction)")
    plt.ylabel("RMSE")
    plt.title("Scaling of RMSE with Training Sample Size")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_vs_sample_size.png"), dpi=300)
    plt.close()

    # 3. Plot power law fit for eval loss
    valid_indices = ~np.isnan(eval_loss)
    if np.sum(valid_indices) > 2:  # Need at least 3 points for meaningful fit
        log_samples = np.log(sample_sizes[valid_indices])
        log_loss = np.log(eval_loss[valid_indices])

        # Linear fit in log-log space
        coeffs = np.polyfit(log_samples, log_loss, 1)
        slope, intercept = coeffs

        plt.figure(figsize=(10, 6))
        plt.loglog(sample_sizes[valid_indices], eval_loss[valid_indices], "o", label="Eval Loss")

        # Plot the power law fit
        fit_x = np.logspace(
            np.log10(sample_sizes[valid_indices].min()),
            np.log10(sample_sizes[valid_indices].max()),
            100,
        )
        fit_y = np.exp(intercept) * fit_x**slope
        plt.loglog(
            fit_x, fit_y, "r-", label=f"Power Law Fit: y = {np.exp(intercept):.4e} * x^{slope:.4f}"
        )

        plt.xlabel("Training Sample Size (fraction)")
        plt.ylabel("Eval Loss")
        plt.title("Power Law Scaling of Eval Loss with Training Sample Size")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eval_loss_sample_power_law.png"), dpi=300)
        plt.close()

        # Print the power law parameters
        print(
            f"Power Law Fit for Eval Loss vs Sample Size: L(N) = {np.exp(intercept):.4e} * N^{slope:.4f}"
        )


def main() -> None:
    """Main function to run the sample size analysis."""
    # Base directory containing experiment results
    base_dir = "tmp/gatr-experiments/experiments/nbody"

    # Output directory for plots
    output_dir = "tmp/gatr-experiments/analysis/sample_size"

    # Collect results
    results = collect_sample_size_results(base_dir)
    print(f"Collected results from {len(results)} sample size experiments")

    # Print summary
    for result in results:
        print(
            f"Sample Size: {result.sample_size:.4f}, Eval Loss: {result.eval_loss:.6f}, "
            f"Object Gen Loss: {result.object_gen_loss:.6f}"
        )

    # Plot scaling curves
    plot_sample_size_scaling(results, output_dir)
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
