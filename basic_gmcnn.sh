python scripts/nbody_experiment.py \
    base_dir="${BASEDIR}" \
    seed=42 \
    model=gm_cnn_nbody \
    run_name=gmcnn_test \
    ++wandb.enabled=true \
    ++wandb.entity="weber-geoml-harvard-university" \
    ++wandb.project="gmcnn-nbody-test"