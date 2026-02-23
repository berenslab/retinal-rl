#!/bin/bash

cd ~/retinal-rl/
# Initialize sweep and extract ID
SWEEP_ID=$(apptainer run --nv --bind /etc/pki/tls/certs/ca-bundle.trust.crt:/etc/ssl/certs/ca-bundle.crt retinal-rl-scipy.sif python main.py +experiment="vae-classification-analysis" command=sweep +sweep="recon-weight-sweep" )

echo "Sweep ID: $SWEEP_ID"

ENTITY_PROJECT="harinijg2001-university of tuebingen/retinal-vae-classification"
SWEEP_ID="$SWEEP_ID"
NUM_GPUS=3

pids=()

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  echo "Starting agent on GPU $gpu"
  apptainer run --nv \
    --bind /etc/pki/tls/certs/ca-bundle.trust.crt:/etc/ssl/certs/ca-bundle.crt \
    retinal-rl.sif \
    CUDA_VISIBLE_DEVICES=$gpu \
    wandb agent $SWEEP_ID &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid" || {
    echo "One agent failed"
    exit 1
  }
done

wait

echo "All agents completed successfully"
