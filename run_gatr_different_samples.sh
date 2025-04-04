#!/bin/bash
#SBATCH --job-name=gatr_3d_sweep       
#SBATCH --time=48:00:00         
#SBATCH --mem=32GB               
#SBATCH --output=logs/gatr_sweep_%j.log  # %j is job ID
#SBATCH --partition=gpu     
#SBATCH --gpus=4                   # One GPU per task

# Define the sample sizes to run (as percentages), we divide by 100 to get a decimal
SAMPLE_SIZES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 10 20 30 40 50 60 70 80 90 100)

# Define training steps to run
TRAINING_STEPS=(1 10 100 500 700 1000 5000 10000 50000 100000 500000 1000000 5000000)

# Define model sizes to run (as [hidden_mv_channels, hidden_s_channels, num_blocks])
# Format: "mv_channels,s_channels,blocks"
# default is blocks=20, hidden_mv_channels=16, hidden_s_channels=32
MODEL_SIZES=(
  "4,32,2"     # Tiny model
  "8,64,4"     # Small model
  "16,128,10"  # Medium model (default)
  "16,32,20"   # Deep and narrow model, default
  "32,256,16"  # Large model
  "64,512,20"  # X-Large model
  "8,256,8"    # Low MV, high S channels
  "32,64,8"    # High MV, low S channels
  "16,128,30"  # Very deep model
  "128,256,6"  # Very wide, shallow model
)

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
echo "Running GATr 3D parameter sweep experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Base directory: $BASEDIR"
echo "Current working directory: $(pwd)"
echo "Base directory absolute path: $(realpath $BASEDIR)"
echo "MLflow database location: $(realpath $BASEDIR)/tracking/mlflow.db"

# Generate the dataset (only need to do this once)
python scripts/generate_nbody_dataset.py base_dir="${BASEDIR}" seed=42

# Loop through all combinations of parameters
for SIZE in "${SAMPLE_SIZES[@]}"; do
    # Convert percentage to decimal
    SUBSAMPLE=$(echo "scale=2; $SIZE/100" | bc)
    
    for STEPS in "${TRAINING_STEPS[@]}"; do
        for MODEL_SIZE in "${MODEL_SIZES[@]}"; do
            # Parse model size parameters
            IFS=',' read -r MV_CHANNELS S_CHANNELS NUM_BLOCKS <<< "$MODEL_SIZE"
            
            # Create a unique name for this configuration
            MODEL_SIZE_NAME="mv${MV_CHANNELS}_s${S_CHANNELS}_b${NUM_BLOCKS}"
            RUN_NAME="gatr_sample${SIZE}_steps${STEPS}_${MODEL_SIZE_NAME}"
            
            echo "Starting experiment with:"
            echo "  Sample size: $SIZE% (subsample=$SUBSAMPLE)"
            echo "  Training steps: $STEPS"
            echo "  Model size: MV channels=$MV_CHANNELS, S channels=$S_CHANNELS, Blocks=$NUM_BLOCKS"
            echo "  Run name: $RUN_NAME"
            echo "Start time: $(date)"
            
            # Run the experiment
            python scripts/nbody_experiment.py \
                base_dir="${BASEDIR}" \
                seed=42 \
                model=gatr_nbody \
                data.subsample=${SUBSAMPLE} \
                training.steps=${STEPS} \
                model.hidden_mv_channels=${MV_CHANNELS} \
                model.hidden_s_channels=${S_CHANNELS} \
                model.num_blocks=${NUM_BLOCKS} \
                run_name=${RUN_NAME}
            
            echo "Finished experiment configuration: $RUN_NAME"
            echo "End time: $(date)"
            echo "Results saved to: ${BASEDIR}/experiments/nbody/${RUN_NAME}"
            echo "-----------------------------------"
        done
    done
done

echo "All experiments completed"
echo "End time: $(date)"
echo "MLflow database location: $(realpath $BASEDIR)/tracking/mlflow.db"
echo "Done!"