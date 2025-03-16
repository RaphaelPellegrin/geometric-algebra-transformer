#!/usr/bin/env python3

"""
Enhanced N-body experiment using GATr without problematic dependencies.
"""

import os
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Disable xformers to avoid compatibility issues
os.environ["XFORMERS_DISABLED"] = "1"

# Import only the core GATr components
from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_point


class SimpleNBodyDataset(Dataset):
    """Simplified dataset for N-body simulation data."""

    def __init__(self, file_path, context_size=10, target_size=1, subsample=1.0):
        """
        Args:
            file_path: Path to the HDF5 file
            context_size: Number of timesteps to use as context
            target_size: Number of timesteps to predict
            subsample: Fraction of data to use (for faster training)
        """
        with h5py.File(file_path, "r") as f:
            positions = f["positions"][:]
            velocities = f["velocities"][:]

        # Subsample if needed
        if subsample < 1.0:
            n_samples = positions.shape[0]
            n_keep = max(1, int(n_samples * subsample))
            indices = np.random.choice(n_samples, n_keep, replace=False)
            positions = positions[indices]
            velocities = velocities[indices]

        self.positions = torch.from_numpy(positions).float()
        self.velocities = torch.from_numpy(velocities).float()
        self.context_size = context_size
        self.target_size = target_size

        # Calculate valid indices
        self.valid_indices = []
        for i in range(len(self.positions)):
            max_start = self.positions.shape[1] - (context_size + target_size)
            if max_start >= 0:
                for start in range(max_start + 1):
                    self.valid_indices.append((i, start))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        sample_idx, start_idx = self.valid_indices[idx]

        # Get context and target positions
        context_pos = self.positions[sample_idx, start_idx : start_idx + self.context_size]
        target_pos = self.positions[
            sample_idx,
            start_idx + self.context_size : start_idx + self.context_size + self.target_size,
        ]

        # Get context and target velocities
        context_vel = self.velocities[sample_idx, start_idx : start_idx + self.context_size]
        target_vel = self.velocities[
            sample_idx,
            start_idx + self.context_size : start_idx + self.context_size + self.target_size,
        ]

        return {
            "context_pos": context_pos,
            "context_vel": context_vel,
            "target_pos": target_pos,
            "target_vel": target_vel,
        }


class GATrNBodyPredictor(nn.Module):
    """N-body predictor using GATr."""

    def __init__(self, hidden_dim=32, num_blocks=4):
        """
        Args:
            hidden_dim: Hidden dimension size
            num_blocks: Number of GATr blocks
        """
        super().__init__()

        # GATr model for processing the point cloud
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_dim,
            in_s_channels=3,  # Velocity as scalar features
            out_s_channels=3,  # Predict velocity
            hidden_s_channels=hidden_dim,
            num_blocks=num_blocks,
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
        )

    def forward(self, positions, velocities=None):
        """
        Forward pass.

        Args:
            positions: Tensor of shape [batch_size, num_points, 3]
            velocities: Tensor of shape [batch_size, num_points, 3] or None

        Returns:
            predicted_positions: Tensor of shape [batch_size, num_points, 3]
            predicted_velocities: Tensor of shape [batch_size, num_points, 3]
        """
        batch_size, num_points, _ = positions.shape

        # Embed positions in PGA
        embedded_positions = embed_point(positions).unsqueeze(-2)  # [batch_size, num_points, 1, 16]

        # Process through GATr
        if velocities is not None:
            # Use velocities as scalar features
            embedded_output, scalar_output = self.gatr(embedded_positions, velocities)
        else:
            # No scalar features
            embedded_output, scalar_output = self.gatr(embedded_positions, None)

        # Extract predicted positions and velocities
        predicted_positions = extract_point(
            embedded_output.squeeze(-2)
        )  # [batch_size, num_points, 3]
        predicted_velocities = scalar_output  # [batch_size, num_points, 3]

        return predicted_positions, predicted_velocities


class SimplifiedNBodyExperiment:
    """A simplified version of the NBodyExperiment class."""

    def __init__(self, args):
        """Initialize the experiment with command line arguments."""
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Create output directory
        os.makedirs(args.base_dir, exist_ok=True)

        # Initialize datasets and model
        self._init_datasets()
        self._init_model()

    def _init_datasets(self):
        """Initialize datasets."""
        # Create datasets
        self.train_dataset = SimpleNBodyDataset(
            os.path.join(self.args.base_dir, "nbody_train.h5"),
            context_size=self.args.context_size,
            target_size=self.args.target_size,
            subsample=self.args.subsample,
        )

        self.val_dataset = SimpleNBodyDataset(
            os.path.join(self.args.base_dir, "nbody_val.h5"),
            context_size=self.args.context_size,
            target_size=self.args.target_size,
            subsample=self.args.subsample,
        )

        self.test_dataset = SimpleNBodyDataset(
            os.path.join(self.args.base_dir, "nbody_test.h5"),
            context_size=self.args.context_size,
            target_size=self.args.target_size,
            subsample=self.args.subsample,
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4
        )

        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4
        )

        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4
        )

    def _init_model(self):
        """Initialize the model."""
        self.model = GATrNBodyPredictor(
            hidden_dim=self.args.hidden_dim, num_blocks=self.args.num_blocks
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def train(self):
        """Train the model."""
        best_val_loss = float("inf")

        for epoch in range(self.args.epochs):
            # Training
            self.model.train()
            train_pos_loss = 0.0
            train_vel_loss = 0.0

            for batch in tqdm(
                self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} (Train)"
            ):
                # Move data to device
                context_pos = batch["context_pos"].to(self.device)
                context_vel = batch["context_vel"].to(self.device)
                target_pos = batch["target_pos"].to(self.device)
                target_vel = batch["target_vel"].to(self.device)

                # Get the last positions and velocities from context
                current_pos = context_pos[:, -1]
                current_vel = context_vel[:, -1]

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                pred_pos, pred_vel = self.model(current_pos, current_vel)

                # Compute loss
                pos_loss = torch.nn.functional.mse_loss(pred_pos, target_pos[:, 0])
                vel_loss = torch.nn.functional.mse_loss(pred_vel, target_vel[:, 0])
                loss = pos_loss + vel_loss

                # Backward pass
                loss.backward()

                # Update weights
                self.optimizer.step()

                train_pos_loss += pos_loss.item()
                train_vel_loss += vel_loss.item()

            avg_train_pos_loss = train_pos_loss / len(self.train_loader)
            avg_train_vel_loss = train_vel_loss / len(self.train_loader)

            # Validation
            self.model.eval()
            val_pos_loss = 0.0
            val_vel_loss = 0.0

            with torch.no_grad():
                for batch in tqdm(
                    self.val_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} (Val)"
                ):
                    # Move data to device
                    context_pos = batch["context_pos"].to(self.device)
                    context_vel = batch["context_vel"].to(self.device)
                    target_pos = batch["target_pos"].to(self.device)
                    target_vel = batch["target_vel"].to(self.device)

                    # Get the last positions and velocities from context
                    current_pos = context_pos[:, -1]
                    current_vel = context_vel[:, -1]

                    # Forward pass
                    pred_pos, pred_vel = self.model(current_pos, current_vel)

                    # Compute loss
                    pos_loss = torch.nn.functional.mse_loss(pred_pos, target_pos[:, 0])
                    vel_loss = torch.nn.functional.mse_loss(pred_vel, target_vel[:, 0])

                    val_pos_loss += pos_loss.item()
                    val_vel_loss += vel_loss.item()

            avg_val_pos_loss = val_pos_loss / len(self.val_loader)
            avg_val_vel_loss = val_vel_loss / len(self.val_loader)
            avg_val_loss = avg_val_pos_loss + avg_val_vel_loss

            # Update scheduler
            self.scheduler.step(avg_val_loss)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    self.model.state_dict(), os.path.join(self.args.base_dir, "best_model.pt")
                )
                print(f"Saved new best model with validation loss: {best_val_loss:.6f}")

            # Print metrics
            print(f"Epoch {epoch+1}/{self.args.epochs}:")
            print(f"  Train Position Loss: {avg_train_pos_loss:.6f}")
            print(f"  Train Velocity Loss: {avg_train_vel_loss:.6f}")
            print(f"  Val Position Loss: {avg_val_pos_loss:.6f}")
            print(f"  Val Velocity Loss: {avg_val_vel_loss:.6f}")

    def evaluate(self):
        """Evaluate the model on the test set."""
        # Load best model
        self.model.load_state_dict(torch.load(os.path.join(self.args.base_dir, "best_model.pt")))

        # Evaluation
        self.model.eval()
        test_pos_loss = 0.0
        test_vel_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                # Move data to device
                context_pos = batch["context_pos"].to(self.device)
                context_vel = batch["context_vel"].to(self.device)
                target_pos = batch["target_pos"].to(self.device)
                target_vel = batch["target_vel"].to(self.device)

                # Get the last positions and velocities from context
                current_pos = context_pos[:, -1]
                current_vel = context_vel[:, -1]

                # Forward pass
                pred_pos, pred_vel = self.model(current_pos, current_vel)

                # Compute loss
                pos_loss = torch.nn.functional.mse_loss(pred_pos, target_pos[:, 0])
                vel_loss = torch.nn.functional.mse_loss(pred_vel, target_vel[:, 0])

                test_pos_loss += pos_loss.item()
                test_vel_loss += vel_loss.item()

        avg_pos_loss = test_pos_loss / len(self.test_loader)
        avg_vel_loss = test_vel_loss / len(self.test_loader)

        print(f"Test Position Loss: {avg_pos_loss:.6f}")
        print(f"Test Velocity Loss: {avg_vel_loss:.6f}")

        # Visualize predictions
        self._visualize_predictions()

    def _visualize_predictions(self):
        """Visualize model predictions."""
        # Get a batch from the test loader
        batch = next(iter(self.test_loader))

        # Move data to device
        context_pos = batch["context_pos"].to(self.device)
        context_vel = batch["context_vel"].to(self.device)
        target_pos = batch["target_pos"].to(self.device)

        # Get the last positions and velocities from context
        current_pos = context_pos[:, -1]
        current_vel = context_vel[:, -1]

        # Forward pass
        with torch.no_grad():
            pred_pos, _ = self.model(current_pos, current_vel)

        # Move to CPU for visualization
        context_pos = context_pos.cpu().numpy()
        target_pos = target_pos.cpu().numpy()
        pred_pos = pred_pos.cpu().numpy()

        # Visualize for the first sample in the batch
        sample_idx = 0

        # Plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection="3d")

        n_bodies = context_pos.shape[2]
        colors = plt.cm.jet(np.linspace(0, 1, n_bodies))

        for i in range(n_bodies):
            # Plot context trajectory
            ax.plot(
                context_pos[sample_idx, :, i, 0],
                context_pos[sample_idx, :, i, 1],
                context_pos[sample_idx, :, i, 2],
                c=colors[i],
                linestyle="-",
                label=f"Body {i+1} (Context)",
            )

            # Plot target position
            ax.scatter(
                target_pos[sample_idx, 0, i, 0],
                target_pos[sample_idx, 0, i, 1],
                target_pos[sample_idx, 0, i, 2],
                c=colors[i],
                marker="*",
                s=100,
                label=f"Body {i+1} (Target)",
            )

            # Plot predicted position
            ax.scatter(
                pred_pos[sample_idx, i, 0],
                pred_pos[sample_idx, i, 1],
                pred_pos[sample_idx, i, 2],
                c=colors[i],
                marker="o",
                s=100,
                label=f"Body {i+1} (Predicted)",
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("N-body Prediction Visualization")

        # Create a custom legend with one entry per body
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.savefig(os.path.join(self.args.base_dir, "prediction_visualization.png"))
        plt.close()

        print(
            f"Prediction visualization saved to {os.path.join(self.args.base_dir, 'prediction_visualization.png')}"
        )

    def __call__(self):
        """Run the experiment."""
        print(f"Starting experiment {self.args.run_name}")
        print(f"Training model with {sum(p.numel() for p in self.model.parameters())} parameters")

        self.train()
        self.evaluate()

        print(f"Experiment {self.args.run_name} completed successfully!")


def parse_args():
    parser = argparse.ArgumentParser(description="N-body experiment with GATr")
    parser.add_argument(
        "--base_dir", type=str, default="./nbody_data", help="Base directory with datasets"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--subsample", type=float, default=0.01, help="Fraction of data to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension size")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of GATr blocks")
    parser.add_argument("--context_size", type=int, default=10, help="Context size")
    parser.add_argument("--target_size", type=int, default=1, help="Target size")
    parser.add_argument("--run_name", type=str, default="gatr_nbody", help="Run name")
    return parser.parse_args()


def main():
    args = parse_args()
    experiment = SimplifiedNBodyExperiment(args)
    experiment()


if __name__ == "__main__":
    main()
