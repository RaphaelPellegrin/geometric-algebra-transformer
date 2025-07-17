#!/bin/bash

# Set base directory
export BASEDIR=/n/netscratch/mweber_lab/Everyone/rpellegrinext/tmp/gatr_experiments

echo "Setting up GM-CNN test..."
echo "Base directory: $BASEDIR"

# Create base directory if it doesn't exist
mkdir -p "${BASEDIR}"

nvidia-smi

module avail cuda

module load cuda/12.9.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01

# Check if dataset exists, if not generate it
if [ ! -f "${BASEDIR}/data/nbody/train.npz" ]; then
    echo "Dataset not found, generating..."
    python scripts/generate_nbody_dataset.py base_dir="${BASEDIR}" seed=42
else
    echo "Dataset already exists, skipping generation"
fi

# Check dataset files
echo "Checking dataset files..."
ls -la "${BASEDIR}/data/nbody/"

# Run GM-CNN experiment
echo "Running GM-CNN experiment..."
python scripts/nbody_experiment.py \
    base_dir="${BASEDIR}" \
    seed=42 \
    model=gm_cnn_nbody \
    data.subsample=0.01 \
    training.steps=100 \
    run_name=gmcnn_test \
    ++wandb.enabled=true \
    ++wandb.entity="weber-geoml-harvard-university" \
    ++wandb.project="gmcnn-nbody-test"

echo "GM-CNN test completed!"