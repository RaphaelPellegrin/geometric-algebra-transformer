#!/bin/bash

# Set base directory
export BASEDIR=/n/netscratch/mweber_lab/Everyone/rpellegrinext/tmp/gatr_experiments

echo "Setting up GM-CNN test..."
echo "Base directory: $BASEDIR"

# Create base directory if it doesn't exist
mkdir -p "${BASEDIR}"

nvidia-smi

module avail cuda

echo "Loading CUDA and cuDNN modules..."
module load cuda/12.9.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

# Check where cuDNN is installed
echo "CUDNN_PATH: $CUDNN_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Find the cuDNN library
find /n/sw -name "libcudnn.so*" 2>/dev/null | head -5

echo "Checking loaded modules..."
module list

echo "Checking if cuDNN library is available..."
ldconfig -p | grep cudnn

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

# Set library path to include cuDNN
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$LD_LIBRARY_PATH

# Run GM-CNN experiment
echo "Running GM-CNN experiment..."
python scripts/nbody_experiment.py \
    base_dir="${BASEDIR}" \
    seed=42 \
    model=gm_cnn_nbody \
    data.subsample=0.01 \
    training.steps=10 \
    training.batch_size=10 \
    device=cpu \
    run_name=gmcnn_test \
    ++wandb.enabled=false \
    ++wandb.entity="weber-geoml-harvard-university" \
    ++wandb.project="gmcnn-nbody-test"

echo "GM-CNN test completed!"