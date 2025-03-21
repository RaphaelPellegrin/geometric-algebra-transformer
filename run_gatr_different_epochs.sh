#!/bin/bash
#SBATCH --job-name=gatr_nbody       
#SBATCH --time=48:00:00         
#SBATCH --mem=32GB               
#SBATCH --output=logs/gatr_nbody_%j.log  # %j is job ID
#SBATCH --partition=gpu     
#SBATCH --gpus=4                   # One GPU per task

# Define the training steps to run
TRAINING_STEPS=(1 10 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 100 200 300 400 500 1000 10000 50000)

# Load modules and activate environment
module load anaconda/2023.07
source activate gatr

# Create log directory if it doesn't exist
mkdir -p logs

# Set the base directory for data and results
export BASEDIR="$(pwd)/tmp/gatr-experiments"

# Create the data directory if it doesn't exist
mkdir -p ${BASEDIR}

# Print some information about the job
echo "Running GATr N-body experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Base directory: $BASEDIR"
echo "Current working directory: $(pwd)"
echo "Base directory absolute path: $(realpath $BASEDIR)"
echo "MLflow database location: $(realpath $BASEDIR)/tracking/mlflow.db"

# Generate the dataset (only need to do this once)
python scripts/generate_nbody_dataset.py base_dir="${BASEDIR}" seed=42

# Run experiments with different training steps
for STEPS in "${TRAINING_STEPS[@]}"; do
    echo "Starting experiment with $STEPS training steps"
    echo "Start time: $(date)"
    
    # Create a unique run name that includes the number of steps
    RUN_NAME="gatr_steps${STEPS}"
    
    # Run the experiment
    python scripts/nbody_experiment.py \
        base_dir="${BASEDIR}" \
        seed=42 \
        model=gatr_nbody \
        data.subsample=0.01 \
        training.steps=${STEPS} \
        run_name=${RUN_NAME}
    
    echo "Finished experiment with $STEPS training steps"
    echo "End time: $(date)"
    echo "Results saved to: ${BASEDIR}/experiments/nbody/${RUN_NAME}"
    echo "-----------------------------------"
done

echo "All experiments completed"
echo "End time: $(date)"
echo "MLflow database location: $(realpath $BASEDIR)/tracking/mlflow.db"
echo "Done!"
