#!/usr/bin/env python3
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.

from pathlib import Path
import os
import sys
import numpy as np
from scipy.stats import special_ortho_group
import argparse

# Create xformers stub for Mac compatibility
if sys.platform == "darwin":  # Check if running on Mac
    import sys

    # Define stub classes/functions
    class AttentionBias:
        pass

    def memory_efficient_attention(*args, **kwargs):
        import torch
        import torch.nn.functional as F

        q, k, v = args[:3]
        # Fall back to standard PyTorch attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    # Create module structure
    class XformersOps:
        AttentionBias = AttentionBias
        memory_efficient_attention = memory_efficient_attention

    class XformersModule:
        ops = XformersOps()

    # Insert into sys.modules
    sys.modules["xformers"] = XformersModule()
    sys.modules["xformers.ops"] = XformersModule.ops

    # Set environment variable
    os.environ["XFORMERS_DISABLED"] = "1"


# Utility functions from gatr.utils.misc
def sample_log_uniform(min_val, max_val, size=None):
    """Samples from log-uniform distribution."""
    log_min, log_max = np.log(min_val), np.log(max_val)
    return np.exp(np.random.uniform(log_min, log_max, size=size))


def sample_uniform_in_circle(n, min_radius=0.0, max_radius=1.0):
    """Samples uniformly from a circle or annulus."""
    r = np.sqrt(np.random.uniform(min_radius**2, max_radius**2, size=n))
    theta = np.random.uniform(0, 2 * np.pi, size=n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)


# Custom NBodySimulator that doesn't use DGL
class NBodySimulator:
    """Simulator for the n-body dataset without DGL dependencies."""

    def __init__(
        self,
        time=0.1,
        time_steps=100,
        star_mass_range=(1.0, 10.0),
        planet_mass_range=(0.01, 0.1),
        radius_range=(0.1, 1.0),
        vel_std=0.01,
        shift_std=20.0,
        ood_shift=200.0,
        outlier_threshold=2.0,
    ):
        self.time = time
        self.delta_t = time / time_steps
        self.time_steps = time_steps
        self.star_mass_range = star_mass_range
        self.planet_mass_range = planet_mass_range
        self.radius_range = radius_range
        self.vel_std = vel_std
        self.shift_std = shift_std
        self.ood_shift = ood_shift
        self.outlier_threshold = outlier_threshold

    def sample(self, num_samples, num_planets=5, domain_shift=False):
        """Samples from gravity problem."""

        # Sample a bit more initially, to make up for outlier removal later
        num_candidates = int(round(1.1 * num_samples))

        # 1. Sample planet masses
        star_mass = sample_log_uniform(*self.star_mass_range, size=(num_candidates, 1))
        planet_masses = sample_log_uniform(
            *self.planet_mass_range, size=(num_candidates, num_planets)
        )
        masses = np.concatenate((star_mass, planet_masses), axis=1)

        # 2. Sample initial positions in x-y plane around origin
        planet_pos = np.zeros((num_candidates, num_planets, 3))
        planet_pos[..., :2] = sample_uniform_in_circle(
            num_candidates * num_planets,
            min_radius=self.radius_range[0],
            max_radius=self.radius_range[1],
        ).reshape((num_candidates, num_planets, 2))
        x_initial = np.concatenate((np.zeros((num_candidates, 1, 3)), planet_pos), axis=1)

        # 3. Sample initial velocities, approximately as stable circular orbits
        planet_vel = self._sample_planet_velocities(star_mass, planet_pos)
        v_initial = np.concatenate((np.zeros((num_candidates, 1, 3)), planet_vel), axis=1)

        # 4. Translate, rotate, permute
        masses, x_initial, v_initial = self._shift_and_rotate(
            masses, x_initial, v_initial, domain_shift=domain_shift
        )

        # 5. Evolve Newtonian gravity
        x_final, trajectory = self._simulate(masses, x_initial, v_initial)

        # 6. Remove outliers
        max_distance = np.max(np.linalg.norm(x_final - x_initial, axis=-1), axis=-1)
        mask = max_distance <= self.outlier_threshold
        masses, x_initial, v_initial, x_final, trajectory = (
            masses[mask],
            x_initial[mask],
            v_initial[mask],
            x_final[mask],
            trajectory[mask],
        )

        # Cut down to requested number
        masses, x_initial, v_initial, x_final, trajectory = (
            masses[:num_samples],
            x_initial[:num_samples],
            v_initial[:num_samples],
            x_final[:num_samples],
            trajectory[:num_samples],
        )

        return masses, x_initial, v_initial, x_final, trajectory

    def _sample_planet_velocities(self, star_mass, x):
        """Samples planet velocities around those that give stable circular orbits."""

        batchsize, num_planets, _ = x.shape

        # Defines rotation plane. Clockwise or counterclockwise is random for each planet.
        orientation = np.zeros((batchsize, num_planets, 3))
        orientation[:, :, 2] = np.random.choice(
            a=[-1.0, 1.0], size=(batchsize, num_planets), p=[0.5, 0.5]
        )

        # Compute stable velocities
        star_mass = star_mass[:, :, np.newaxis]  # (batchsize, 1, 1)
        radii = np.linalg.norm(x, axis=-1)[:, :, np.newaxis]  # (batchsize, num_planets, 1)
        v_stable = np.cross(orientation, x) * star_mass**0.5 / radii**1.5

        # Add noise
        v = v_stable + np.random.normal(scale=self.vel_std, size=(batchsize, num_planets, 3))

        return v

    def _shift_and_rotate(self, m, x, v, domain_shift=False):
        """Performs random E(3) transformations and permutations on given positions / velocities."""

        batchsize, num_objects, _ = x.shape

        # Permutations over objects
        for i in range(batchsize):
            perm = np.random.permutation(num_objects)
            m[i] = m[i][perm]
            x[i] = x[i][perm, :]
            v[i] = v[i][perm, :]

        # Rotations from Haar measure
        rotations = special_ortho_group(3).rvs(size=batchsize).reshape(batchsize, 1, 3, 3)
        x = np.einsum("bnij,bnj->bni", rotations, x)
        v = np.einsum("bnij,bnj->bni", rotations, v)

        # Translations
        shifts = np.random.normal(scale=self.shift_std, size=(batchsize, 1, 3))
        x = x + shifts

        # OOD shift
        if domain_shift:
            shifts = np.array([self.ood_shift, 0, 0]).reshape((1, 1, 3))
            x = x + shifts

        return m, x, v

    def _simulate(self, m, x_initial, v_initial):
        """Evolves an initial state under Newtonian equations of motions."""

        x, v = x_initial, v_initial
        trajectory = [x_initial]

        for _ in range(self.time_steps):
            a = self._compute_accelerations(m, x)
            v = v + self.delta_t * a
            x = x + self.delta_t * v
            trajectory.append(x)

        return x, np.array(trajectory).transpose([1, 2, 0, 3])

    @staticmethod
    def _compute_accelerations(m, x):
        """Computes accelerations for a set of point masses according to Newtonian gravity."""
        batchsize, num_objects, _ = x.shape
        mm = m.reshape((batchsize, 1, num_objects, 1)) * m.reshape(
            (batchsize, num_objects, 1, 1)
        )  # (b, n, n, 1)

        distance_vectors = x.reshape((batchsize, 1, num_objects, 3)) - x.reshape(
            (batchsize, num_objects, 1, 3)
        )  # (b, n, n, 3)
        distances = np.linalg.norm(distance_vectors, axis=-1)[:, :, :, np.newaxis]  # (b, n, n, 1)
        distances[np.abs(distances) < 1e-9] = 1.0

        forces = distance_vectors * mm / distances**3  # (b, n, n, 3)
        accelerations = np.sum(forces, axis=2) / m.reshape(batchsize, num_objects, 1)  # (b, n, 3)

        return accelerations  # (b, n, 3)


def generate_dataset(filename, simulator, num_samples, num_planets=5, domain_shift=False):
    """Samples from n-body simulator and stores the results at `filename`."""
    assert not Path(filename).exists()
    m, x_initial, v_initial, x_final, trajectories = simulator.sample(
        num_samples, num_planets=num_planets, domain_shift=domain_shift
    )
    np.savez(
        filename,
        m=m,
        x_initial=x_initial,
        v_initial=v_initial,
        x_final=x_final,
        trajectories=trajectories,
    )


def generate_datasets(path):
    """Generates a canonical set of datasets for the n-body problem, stores them in `path`."""
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)

    print(f"Creating gravity dataset in {str(path)}")

    simulator = NBodySimulator()

    # Generate smaller datasets for Mac to avoid memory issues
    generate_dataset(path / "train.npz", simulator, 10000, num_planets=3, domain_shift=False)
    generate_dataset(path / "val.npz", simulator, 1000, num_planets=3, domain_shift=False)
    generate_dataset(path / "eval.npz", simulator, 1000, num_planets=3, domain_shift=False)
    generate_dataset(
        path / "e3_generalization.npz", simulator, 1000, num_planets=3, domain_shift=True
    )
    generate_dataset(
        path / "object_generalization.npz", simulator, 1000, num_planets=5, domain_shift=False
    )

    print("Done, have a nice day!")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate N-body dataset")
    parser.add_argument(
        "--data_dir", type=str, default="./nbody_data", help="Directory to save datasets"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    """Entry point for n-body dataset generation."""
    args = parse_args()
    np.random.seed(args.seed)
    generate_datasets(args.data_dir)


if __name__ == "__main__":
    main()
