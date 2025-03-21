#!/bin/bash
#SBATCH --job-name=gatr_sample       
#SBATCH --time=48:00:00         
#SBATCH --mem=32GB               
#SBATCH --output=logs/gatr_sample_%j.log  # %j is job ID
#SBATCH --partition=gpu     
#SBATCH --gpus=4                   # One GPU per task

# Define the sample sizes to run (as percentages)
SAMPLE_SIZES=(1 2 5 10 20 30 40 50 60 70 80 90 100)

# Define a fixed number of training steps for all experiments
FIXED_STEPS=500

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
echo "Running GATr N-body sample size experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Base directory: $BASEDIR"
echo "Current working directory: $(pwd)"
echo "Base directory absolute path: $(realpath $BASEDIR)"
echo "MLflow database location: $(realpath $BASEDIR)/tracking/mlflow.db"

# Generate the dataset (only need to do this once)
python scripts/generate_nbody_dataset.py base_dir="${BASEDIR}" seed=42

# Run experiments with different sample sizes
for SIZE in "${SAMPLE_SIZES[@]}"; do
    # Convert percentage to decimal
    SUBSAMPLE=$(echo "scale=2; $SIZE/100" | bc)
    
    echo "Starting experiment with sample size $SIZE% (subsample=$SUBSAMPLE)"
    echo "Start time: $(date)"
    
    # Create a unique run name that includes the sample size and fixed steps
    RUN_NAME="gatr_sample${SIZE}_steps${FIXED_STEPS}"
    
    # Run the experiment
    python scripts/nbody_experiment.py \
        base_dir="${BASEDIR}" \
        seed=42 \
        model=gatr_nbody \
        data.subsample=${SUBSAMPLE} \
        training.steps=${FIXED_STEPS} \
        run_name=${RUN_NAME}
    
    echo "Finished experiment with sample size $SIZE%"
    echo "End time: $(date)"
    echo "Results saved to: ${BASEDIR}/experiments/nbody/${RUN_NAME}"
    echo "-----------------------------------"
done

echo "All sample size experiments completed"
echo "End time: $(date)"
echo "MLflow database location: $(realpath $BASEDIR)/tracking/mlflow.db"
echo "Done!"