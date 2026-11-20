#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define arguments
EXPERIMENT_NAME="spr_k3_cT_oTask_DoRaT"
CHECK_POINT_TYPE="TRIRL" # Options: "RL-PPO" or "TRIRL"
ENCODER_TYPE="SPR"  # Options: "SPR" or "FEATURES"
OBS_TYPE="TASK-BASED" # Options: "TASK-BASED" or "JOINT-TASK-BASED"
SEED=0
SUCCESS_THRESHOLD=0.05
# Options: "AND" (pos_err < thr and ort_err < thr) or
# "SUM" (pos_err + ort_err < thr).
# The coverage of the goal T is always recorded and averaged in the overall evaluation.
SUCCESS_MODE="AND"
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
    --parallel-mtp "$PARALLEL_MTP" \
    --success-mode "$SUCCESS_MODE"

# Overall evaluation of this experiment alone -> OUTPUT_DIR/EXPERIMENT_NAME_SUCCESS_MODE/overall_evaluation.xlsx
python overall_evaluation.py "$OUTPUT_DIR/${EXPERIMENT_NAME}_${SUCCESS_MODE}"

# Compare every experiment evaluated into OUTPUT_DIR so far -> OUTPUT_DIR/overall_evaluation.xlsx
python overall_evaluation.py "$OUTPUT_DIR"
