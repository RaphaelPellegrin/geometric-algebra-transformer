#!/usr/bin/env python3

print("Starting minimal GM-CNN test...")

# Test basic imports first
import torch

print("1. PyTorch imported OK")

import numpy as np

print("2. NumPy imported OK")

# Test GM-CNN model directly
print("3. Testing GM-CNN model import...")
from gatr.models.gmcnn_model import GMCNNModel

print("3. GM-CNN model imported OK")

# Test base wrapper
print("4. Testing BaseWrapper import...")
from gatr.experiments.base_wrapper import BaseWrapper

print("4. BaseWrapper imported OK")

# Now try to create the wrapper class manually without importing from wrappers module
print("5. Creating minimal GM-CNN wrapper...")


class MinimalGMCNNWrapper(BaseWrapper):
    def __init__(self, mv_channels=16, num_blocks=2, **kwargs):
        print("Creating GM-CNN model...")
        net = GMCNNModel(
            mv_channels=mv_channels,
            num_blocks=num_blocks,
            group="cyclic",
            order=8,
            nbr_size=3,
            input_channels=7,
            output_channels=3,
        )
        print("GM-CNN model created, calling super().__init__...")
        super().__init__(net)
        print("MinimalGMCNNWrapper initialized successfully!")

    def _forward(self, inputs):
        # Add extra dimension for GM-CNN compatibility
        inputs_4d = inputs.unsqueeze(-1)  # (batch, objects, features, 1)

        # Forward through GM-CNN
        outputs, _ = self.net(inputs_4d)  # Unpack the tuple

        # Remove extra dimension
        outputs_3d = outputs.squeeze(-1)  # (batch, objects, 3)

        return outputs_3d, torch.zeros(inputs.shape[0], device=inputs.device)


# Test the minimal wrapper
print("6. Testing minimal wrapper creation...")
wrapper = MinimalGMCNNWrapper()
print("SUCCESS: Minimal GM-CNN wrapper created!")

print("7. Testing forward pass with dummy data...")
dummy_input = torch.randn(2, 5, 7)  # batch=2, objects=5, features=7
output, reg = wrapper._forward(dummy_input)
print(f"SUCCESS: Forward pass completed! Output shape: {output.shape}")

print("All tests passed! GM-CNN is working.")
