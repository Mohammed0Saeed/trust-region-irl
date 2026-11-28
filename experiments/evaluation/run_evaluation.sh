#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

PROJECT_NAME="spr_comparison"
EXPERIMENT_NAME="pusht_ppo_spr_K3"
CHECK_POINT_TYPE="TRIRL" # "RL-PPO" | "TRIRL"
ENCODER_TYPE="SPR"  # "SPR" | "FEATURES"
OBS_TYPE="TASK-BASED" # "TASK-BASED" | "JOINT-TASK-BASED"
POSES_SEEDS="0,1,2,3,4,5,6,7,8,9"
SUCCESS_THRESHOLD=0.05
# "AND" (pos_err < thr and ort_err < thr) |
# "SUM" (pos_err + ort_err < thr).
SUCCESS_MODE="AND"
PARALLEL_MTP=true # true (all MTP poses of a seed in one batched GPU call) | false (one pose after the other)
OUTPUT_DIR="$(dirname "$0")/../evaluations_02"  # experiments/evaluations

python evaluation_pipeline.py \
    --project-name "$PROJECT_NAME" \
    --experiment-name "$EXPERIMENT_NAME" \
    --check-point-type "$CHECK_POINT_TYPE" \
    --encoder-type "$ENCODER_TYPE" \
    --obs-type "$OBS_TYPE" \
    --poses-seeds "$POSES_SEEDS" \
    --success-threshold "$SUCCESS_THRESHOLD" \
    --output-dir "$OUTPUT_DIR" \
    --parallel-mtp "$PARALLEL_MTP" \
    --success-mode "$SUCCESS_MODE"

# Overall evaluation of this experiment alone -> OUTPUT_DIR/PROJECT_NAME/EXPERIMENT_NAME_SUCCESS_MODE/overall_evaluation.xlsx
python overall_evaluation.py "$OUTPUT_DIR/$PROJECT_NAME/${EXPERIMENT_NAME}_${SUCCESS_MODE}"

# Compare every experiment of the project evaluated so far -> OUTPUT_DIR/PROJECT_NAME/overall_evaluation.xlsx
python overall_evaluation.py "$OUTPUT_DIR/$PROJECT_NAME"
