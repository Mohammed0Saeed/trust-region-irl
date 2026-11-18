#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define arguments
EXPERIMENT_NAME="spr_k5_cT_oTask_DoRaT_pt200_s300"
CHECK_POINT_TYPE="TRIRL" # Options: "RL-PPO" or "TRIRL"
ENCODER_TYPE="SPR"  # Options: "SPR" or "FEATURES"
OBS_TYPE="TASK-BASED" # Options: "TASK-BASED" or "JOINT-TASK-BASED"
SEED=0
SUCCESS_THRESHOLD=0.05
PARALLEL_MTP=true # Options: true (all MTP poses of a seed in one batched GPU call) or false (one pose after the other)
OUTPUT_DIR="$(dirname "$0")/../evaluations_02"  # experiments/evaluations

# Run from anywhere: the pipeline is found next to this script
python evaluation_pipeline.py \
    "$EXPERIMENT_NAME" \
    "$CHECK_POINT_TYPE" \
    "$ENCODER_TYPE" \
    "$OBS_TYPE" \
    "$SEED" \
    "$SUCCESS_THRESHOLD" \
    "$OUTPUT_DIR" \
    --parallel-mtp "$PARALLEL_MTP"

# Overall evaluation of this experiment alone -> OUTPUT_DIR/EXPERIMENT_NAME/overall_evaluation.xlsx
python overall_evaluation.py "$OUTPUT_DIR/$EXPERIMENT_NAME"

# Compare every experiment evaluated into OUTPUT_DIR so far -> OUTPUT_DIR/overall_evaluation.xlsx
python overall_evaluation.py "$OUTPUT_DIR"
