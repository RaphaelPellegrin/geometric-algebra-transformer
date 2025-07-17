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
module load anaconda3
source activate gatr

# Print job information
echo "Running GMCNN 3D parameter sweep experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Base directory: $BASEDIR"

# Loop through all combinations
for SIZE in "${SAMPLE_SIZES[@]}"; do
    SUBSAMPLE=$(echo "scale=2; $SIZE/100" | bc)
    
    for STEPS in "${TRAINING_STEPS[@]}"; do
        for MODEL_CONFIG in "${MODEL_CONFIGS[@]}"; do
            # Parse model configuration
            IFS=',' read -r MV_CHANNELS NUM_BLOCKS GROUP ORDER <<< "$MODEL_CONFIG"
            
            # Create unique name
            MODEL_NAME="mv${MV_CHANNELS}_b${NUM_BLOCKS}_${GROUP}${ORDER}"
            RUN_NAME="gmcnn_sample${SIZE}_steps${STEPS}_${MODEL_NAME}"
            
            echo "Starting experiment:"
            echo "  Sample size: $SIZE% (subsample=$SUBSAMPLE)"
            echo "  Training steps: $STEPS"
            echo "  Model: MV channels=$MV_CHANNELS, Blocks=$NUM_BLOCKS, Group=$GROUP, Order=$ORDER"
            echo "  Run name: $RUN_NAME"
            
            # Run experiment
            python scripts/nbody_experiment.py \
                base_dir="${BASEDIR}" \
                seed=42 \
                model=gmcnn_nbody \
                data.subsample=${SUBSAMPLE} \
                data.use_gmcnn=true \
                training.steps=${STEPS} \
                ++model.mv_channels=${MV_CHANNELS} \
                ++model.num_blocks=${NUM_BLOCKS} \
                ++model.group=${GROUP} \
                ++model.order=${ORDER} \
                ++model.input_channels=7 \
                ++model.output_channels=3 \
                run_name=${RUN_NAME} \
                ++wandb.enabled=true \
                ++wandb.entity="weber-geoml-harvard-university" \
                ++wandb.project="gmcnn-nbody-sweep"
            
            echo "Finished: $RUN_NAME"
            echo "Results: ${BASEDIR}/experiments/nbody/${RUN_NAME}"
            echo "-----------------------------------"
        done
    done
done

echo "All experiments completed"
echo "End time: $(date)"
echo "Done!"