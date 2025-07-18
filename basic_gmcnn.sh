#!/bin/bash

# Set base directory
export BASEDIR=/n/netscratch/mweber_lab/Everyone/rpellegrinext/tmp/gatr_experiments

echo "Setting up GM-CNN test..."
echo "Base directory: $BASEDIR"

# Create base directory if it doesn't exist
mkdir -p "${BASEDIR}"

# Activate conda environment
echo "Activating conda environment..."
source /n/home04/rpellegrinext/miniconda3/etc/profile.d/conda.sh
conda activate /n/home04/rpellegrinext/miniconda3/envs/gatr

# Verify environment activation
echo "Conda environment activated: $CONDA_DEFAULT_ENV"

# Display system info (quick check)
nvidia-smi
echo ""

# Load required modules
echo "Loading CUDA and cuDNN modules..."
module load cuda/12.9.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

echo "Checking loaded modules..."
module list
echo ""

# # Set library path to include cuDNN
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$LD_LIBRARY_PATH
echo "Updated LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Make sure we're using the right conda environment
echo "Checking conda environment..."
python -c "
import sys
print(f'Python path: {sys.executable}')
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
print('Environment check successful!')
"
echo ""

# # Test PyTorch import first
# echo "Testing PyTorch import..."
# python -c "
# import torch
# print(f'PyTorch version: {torch.__version__}')
# print(f'CUDA available: {torch.cuda.is_available()}')
# print('PyTorch import successful!')
# "
# echo ""

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
echo ""

# Run minimal GM-CNN test first to verify it works
echo "Running minimal GM-CNN test..."
python test_gmcnn_minimal.py
echo ""

# If minimal test passed, run the full experiment
if [ $? -eq 0 ]; then
    echo "Minimal test passed! Running full GM-CNN experiment..."
    
    # Run GM-CNN experiment with conservative settings
    python scripts/nbody_experiment.py \
        base_dir="${BASEDIR}" \
        seed=42 \
        model=gm_cnn_nbody \
        data.subsample=0.01 \
        training.steps=100 \
        ++model.mv_channels=16 \
        ++model.num_blocks=2 \
        ++model.group="cyclic" \
        ++model.order=8 \
        ++model.nbr_size=3 \
        run_name=gmcnn_basic_test \
        ++wandb.enabled=false
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "SUCCESS: GM-CNN experiment completed successfully!"
    else
        echo ""
        echo "ERROR: GM-CNN experiment failed!"
        exit 1
    fi
else
    echo "ERROR: Minimal GM-CNN test failed! Check the implementation."
    exit 1
fi

echo "GM-CNN test completed!"