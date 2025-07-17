#!/bin/bash
#SBATCH --job-name=gmcnn_3d_sweep       
#SBATCH --time=48:00:00         
#SBATCH --mem=32GB               
#SBATCH --output=/n/netscratch/mweber_lab/Everyone/rpellegrinext/tmp/logs/gmcnn_sweep_%j.log
#SBATCH --partition=gpu     
#SBATCH --gpus=4

export CUDA_LAUNCH_BLOCKING=1

# Wandb configuration
export WANDB_API_KEY="ea7c6eeb5a095b531ef60cc784bfeb87d47ea0b0"
export WANDB_ENTITY="weber-geoml-harvard-university"
export WANDB_PROJECT="gmcnn-nbody-sweep"

# Set base directory
export BASEDIR=/n/netscratch/mweber_lab/Everyone/rpellegrinext/tmp/gmcnn_experiments

# Create directories
mkdir -p /n/netscratch/mweber_lab/Everyone/rpellegrinext/tmp/logs
mkdir -p ${BASEDIR}

# Create timestamped log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="/n/netscratch/mweber_lab/Everyone/rpellegrinext/tmp/logs/logging_gatr_with_gmcnn_${TIMESTAMP}.txt"

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to run command with logging (captures both stdout and stderr)
run_with_logging() {
    local cmd="$1"
    local description="$2"
    
    log_with_timestamp "STARTING: $description"
    log_with_timestamp "COMMAND: $cmd"
    
    # Run command and capture both stdout and stderr, while showing on terminal
    eval "$cmd" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        log_with_timestamp "SUCCESS: $description completed successfully"
    else
        log_with_timestamp "ERROR: $description failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Start logging
log_with_timestamp "=== GM-CNN SWEEP EXPERIMENT STARTED ==="
log_with_timestamp "Log file: $LOG_FILE"
log_with_timestamp "SLURM Job ID: $SLURM_JOB_ID"
log_with_timestamp "Node: $SLURMD_NODENAME"
log_with_timestamp "Base directory: $BASEDIR"

# Sample sizes (as percentages)
SAMPLE_SIZES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 10 20 30 40 50 60 70 80 90 100)

# Training steps
TRAINING_STEPS=(1 10 100 500 700 1000 5000 10000 50000 100000)

# Model configurations for GMCNN
# Format: "mv_channels,num_blocks,group,order"
MODEL_CONFIGS=(
    "4,2,cyclic,8"     # Tiny model
    "8,4,cyclic,8"     # Small model
    "16,10,cyclic,8"   # Medium model
    "32,16,cyclic,8"   # Large model
    "64,20,cyclic,8"   # X-Large model
    "16,10,dihedral,8" # Dihedral group model
    "32,16,dihedral,8" # Large dihedral model
)

# Load modules and activate environment
log_with_timestamp "Loading anaconda3 module..."
module load anaconda3

log_with_timestamp "Activating gatr environment..."
source activate gatr

# Log environment info
log_with_timestamp "Python version: $(python --version)"
log_with_timestamp "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
log_with_timestamp "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Log experiment parameters
log_with_timestamp "Sample sizes: ${SAMPLE_SIZES[*]}"
log_with_timestamp "Training steps: ${TRAINING_STEPS[*]}"
log_with_timestamp "Model configs: ${MODEL_CONFIGS[*]}"

# Calculate total experiments
TOTAL_EXPERIMENTS=$((${#SAMPLE_SIZES[@]} * ${#TRAINING_STEPS[@]} * ${#MODEL_CONFIGS[@]}))
log_with_timestamp "Total experiments to run: $TOTAL_EXPERIMENTS"

# Initialize counters
CURRENT_EXPERIMENT=0
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

# Loop through all combinations
for SIZE in "${SAMPLE_SIZES[@]}"; do
    SUBSAMPLE=$(echo "scale=2; $SIZE/100" | bc)
    
    for STEPS in "${TRAINING_STEPS[@]}"; do
        for MODEL_CONFIG in "${MODEL_CONFIGS[@]}"; do
            CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
            
            # Parse model configuration
            IFS=',' read -r MV_CHANNELS NUM_BLOCKS GROUP ORDER <<< "$MODEL_CONFIG"
            
            # Create unique name
            MODEL_NAME="mv${MV_CHANNELS}_b${NUM_BLOCKS}_${GROUP}${ORDER}"
            RUN_NAME="gmcnn_sample${SIZE}_steps${STEPS}_${MODEL_NAME}"
            
            log_with_timestamp "=== EXPERIMENT $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS ==="
            log_with_timestamp "Sample size: $SIZE% (subsample=$SUBSAMPLE)"
            log_with_timestamp "Training steps: $STEPS"
            log_with_timestamp "Model: MV channels=$MV_CHANNELS, Blocks=$NUM_BLOCKS, Group=$GROUP, Order=$ORDER"
            log_with_timestamp "Run name: $RUN_NAME"
            
            # Prepare the command
            EXPERIMENT_CMD="python scripts/nbody_experiment.py \
                base_dir=\"${BASEDIR}\" \
                seed=42 \
                model=gm_cnn_nbody \
                data.subsample=${SUBSAMPLE} \
                training.steps=${STEPS} \
                ++model.mv_channels=${MV_CHANNELS} \
                ++model.num_blocks=${NUM_BLOCKS} \
                ++model.group=${GROUP} \
                ++model.order=${ORDER} \
                ++model.nbr_size=3 \
                run_name=${RUN_NAME} \
                ++wandb.enabled=true \
                ++wandb.entity=\"weber-geoml-harvard-university\" \
                ++wandb.project=\"gmcnn-nbody-sweep\""
            
            # Run experiment with logging
            if run_with_logging "$EXPERIMENT_CMD" "Experiment $CURRENT_EXPERIMENT: $RUN_NAME"; then
                SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
                log_with_timestamp "Results saved to: ${BASEDIR}/experiments/nbody/${RUN_NAME}"
            else
                FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
                log_with_timestamp "WARNING: Experiment $CURRENT_EXPERIMENT failed!"
            fi
            
            # Log progress
            REMAINING=$((TOTAL_EXPERIMENTS - CURRENT_EXPERIMENT))
            log_with_timestamp "Progress: $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS completed, $REMAINING remaining"
            log_with_timestamp "Success rate: $SUCCESSFUL_EXPERIMENTS successful, $FAILED_EXPERIMENTS failed"
            log_with_timestamp "-----------------------------------"
        done
    done
done

# Final summary
log_with_timestamp "=== GM-CNN SWEEP EXPERIMENT COMPLETED ==="
log_with_timestamp "Total experiments: $TOTAL_EXPERIMENTS"
log_with_timestamp "Successful: $SUCCESSFUL_EXPERIMENTS"
log_with_timestamp "Failed: $FAILED_EXPERIMENTS"
log_with_timestamp "Success rate: $(echo "scale=2; $SUCCESSFUL_EXPERIMENTS * 100 / $TOTAL_EXPERIMENTS" | bc)%"
log_with_timestamp "End time: $(date)"
log_with_timestamp "Log file: $LOG_FILE"
log_with_timestamp "Done!"

echo ""
echo "=== EXPERIMENT SUMMARY ==="
echo "Log file saved to: $LOG_FILE"
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Successful: $SUCCESSFUL_EXPERIMENTS"
echo "Failed: $FAILED_EXPERIMENTS"

if [ $FAILED_EXPERIMENTS -gt 0 ]; then
    echo "WARNING: $FAILED_EXPERIMENTS experiments failed. Check the log file for details."
    exit 1
else
    echo "All experiments completed successfully!"
    exit 0
fi