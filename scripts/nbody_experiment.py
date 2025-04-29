#!/usr/bin/env python3
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.

import hydra

from gatr.experiments.nbody import NBodyExperiment

# Add at beginning of script
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")


@hydra.main(config_path="../config", config_name="nbody", version_base=None)
def main(cfg):
    """Entry point for n-body experiment."""
    exp = NBodyExperiment(cfg)
    exp()


if __name__ == "__main__":
    main()
