#!/usr/bin/env python3

"""
Simplified N-body dataset generator that works with GATr.
"""

import os
import numpy as np
import h5py
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

# Disable xformers to avoid compatibility issues
os.environ["XFORMERS_DISABLED"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate N-body dataset")
    parser.add_argument(
        "--base_dir", type=str, default="./nbody_data", help="Base directory to save datasets"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n_bodies", type=int, default=5, help="Number of bodies in the simulation"
    )
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument(
        "--n_timesteps", type=int, default=100, help="Number of timesteps per sample"
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Time step size")
    return parser.parse_args()


def simulate_nbody(n_bodies, n_timesteps, dt, seed=None):
    """
    Simple N-body simulation using Newton's law of gravitation.

    Args:
        n_bodies: Number of bodies
        n_timesteps: Number of timesteps to simulate
        dt: Time step size
        seed: Random seed

    Returns:
        positions: Array of shape [n_timesteps, n_bodies, 3]
        velocities: Array of shape [n_timesteps, n_bodies, 3]
    """
    if seed is not None:
        np.random.seed(seed)

    # Constants
    G = 1.0  # Gravitational constant
    mass = np.ones(n_bodies)  # All bodies have the same mass

    # Initialize positions and velocities
    positions = np.random.randn(n_bodies, 3) * 2.0
    velocities = np.random.randn(n_bodies, 3) * 0.1

    # Center of mass correction
    positions -= np.mean(positions, axis=0)
    velocities -= np.mean(velocities, axis=0)

    # Arrays to store simulation results
    all_positions = np.zeros((n_timesteps, n_bodies, 3))
    all_velocities = np.zeros((n_timesteps, n_bodies, 3))

    # Run simulation
    pos = positions.copy()
    vel = velocities.copy()

    for t in range(n_timesteps):
        all_positions[t] = pos
        all_velocities[t] = vel

        # Compute accelerations
        acc = np.zeros_like(pos)
        for i in range(n_bodies):
            for j in range(n_bodies):
                if i != j:
                    r = pos[j] - pos[i]
                    r_norm = np.linalg.norm(r)
                    if r_norm < 0.1:  # Softening to avoid numerical instability
                        r_norm = 0.1
                    acc[i] += G * mass[j] * r / r_norm**3

        # Update velocities and positions using leapfrog integration
        vel += acc * dt
        pos += vel * dt

    return all_positions, all_velocities


def generate_dataset(args):
    """Generate and save the N-body dataset."""
    np.random.seed(args.seed)

    # Create directories
    os.makedirs(args.base_dir, exist_ok=True)

    # Generate samples
    all_positions = []
    all_velocities = []

    for i in tqdm(range(args.n_samples), desc="Generating samples"):
        # Use a different seed for each sample
        sample_seed = args.seed + i
        positions, velocities = simulate_nbody(
            args.n_bodies, args.n_timesteps, args.dt, seed=sample_seed
        )
        all_positions.append(positions)
        all_velocities.append(velocities)

    # Convert to arrays
    all_positions = np.array(all_positions)  # [n_samples, n_timesteps, n_bodies, 3]
    all_velocities = np.array(all_velocities)  # [n_samples, n_timesteps, n_bodies, 3]

    # Split into train, validation, and test sets
    n_train = int(0.8 * args.n_samples)
    n_val = int(0.1 * args.n_samples)

    train_positions = all_positions[:n_train]
    train_velocities = all_velocities[:n_train]

    val_positions = all_positions[n_train : n_train + n_val]
    val_velocities = all_velocities[n_train : n_train + n_val]

    test_positions = all_positions[n_train + n_val :]
    test_velocities = all_velocities[n_train + n_val :]

    # Save datasets
    for name, pos, vel in [
        ("train", train_positions, train_velocities),
        ("val", val_positions, val_velocities),
        ("test", test_positions, test_velocities),
    ]:
        filename = os.path.join(args.base_dir, f"nbody_{name}.h5")
        with h5py.File(filename, "w") as f:
            f.create_dataset("positions", data=pos)
            f.create_dataset("velocities", data=vel)
        print(f"Saved {name} dataset to {filename}")

    # Visualize a sample
    visualize_sample(train_positions[0], os.path.join(args.base_dir, "sample_visualization.png"))


def visualize_sample(positions, output_file):
    """Create a visualization of a sample trajectory."""
    n_timesteps, n_bodies, _ = positions.shape

    # Plot the trajectories
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.jet(np.linspace(0, 1, n_bodies))

    for i in range(n_bodies):
        ax.plot(
            positions[:, i, 0],
            positions[:, i, 1],
            positions[:, i, 2],
            c=colors[i],
            label=f"Body {i+1}",
        )

        # Mark start and end positions
        ax.scatter(
            positions[0, i, 0],
            positions[0, i, 1],
            positions[0, i, 2],
            c=colors[i],
            marker="o",
            s=50,
        )
        ax.scatter(
            positions[-1, i, 0],
            positions[-1, i, 1],
            positions[-1, i, 2],
            c=colors[i],
            marker="*",
            s=100,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("N-body Simulation Trajectories")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Visualization saved to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args)
    print("Dataset generation complete!")
