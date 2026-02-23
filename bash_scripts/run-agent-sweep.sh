#!/bin/bash

ENTITY_PROJECT="harinijg2001-university-of-tuebingen/retinal-vae-classification"
SWEEP_ID="1l24zxee"
NUM_GPUS=3

pids=()

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  echo "Starting agent on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu \
  apptainer run --nv \
    --bind /etc/pki/tls/certs/ca-bundle.trust.crt:/etc/ssl/certs/ca-bundle.crt \
    retinal-rl.sif \
    wandb agent "$ENTITY_PROJECT/$SWEEP_ID" &
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
