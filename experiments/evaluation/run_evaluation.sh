#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define arguments
EXPERIMENT_NAME="WANDB_EXPERIMENT_NAME"
CHECK_POINT_TYPE="TRIRL" # Options: "RL-PPO" or "TRIRL"
ENCODER_TYPE="SPR"  # Options: "SPR" or "FEATURES"
OBS_TYPE="TASK-BASED" # Options: "TASK-BASED" or "JOINT-TASK-BASED"
SEED=0
SUCCESS_THRESHOLD=0.05
OUTPUT_DIR="$(dirname "$0")/../evaluations"  # experiments/evaluations

# Run from anywhere: the pipeline is found next to this script
python3 "$(dirname "$0")/evaluation_pipeline.py" \
    "$EXPERIMENT_NAME" \
    "$CHECK_POINT_TYPE" \
    "$ENCODER_TYPE" \
    "$OBS_TYPE" \
    "$SEED" \
    "$SUCCESS_THRESHOLD" \
    "$OUTPUT_DIR"
