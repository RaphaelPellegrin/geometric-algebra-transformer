#!/bin/bash
#SBATCH --job-name=gatr_nbody       
#SBATCH --time=48:00:00         
#SBATCH --mem=32GB               
#SBATCH --output=logs/gatr_nbody_%j.log  # %j is job ID
#SBATCH --partition=gpu     
#SBATCH --gpus=4                   # One GPU per task

# Load modules and activate environment
module load anaconda/2023.07
source activate gatr

# Create log directory if it doesn't exist
mkdir -p logs
# Set the base directory for data and results
export BASEDIR='tmp/gatr-experiments'

# Create the data directory if it doesn't exist
mkdir -p ${BASEDIR}

# Print some information about the job
echo "Running GATr N-body experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Base directory: $BASEDIR"
echo "Current working directory: $(pwd)"
echo "Base directory relative path: $BASEDIR"
echo "Base directory absolute path: $(realpath $BASEDIR)"
echo "MLflow database location: $(realpath $BASEDIR)/tracking/mlflow.db"

python scripts/generate_nbody_dataset.py base_dir="${BASEDIR}" seed=42

# Run the experiment
python scripts/nbody_experiment.py \
    base_dir="${BASEDIR}" \
    seed=42 \
    model=gatr_nbody \
    data.subsample=0.01 \
    training.steps=5000 \
    run_name=gatr_${SLURM_JOB_ID}

echo "End time: $(date)"
echo "Done!"
