#!/usr/bin/env bash
# Run one (METHOD, SEED) of the PushT comparison. Used locally and by cluster/pusht.sbatch.
#   PROTOCOL=paper (default) | fixed | v3 selects data, start set and wandb group (see methods.sh); DRY_RUN=1 prints the command line only.
#   METHOD=S SEED=0 experiments/pusht/run.sh [extra --flags]                 # full run (wandb, saves models)
#   SMOKE=1 METHOD=S SEED=0 PYTHON=/path/to/python experiments/pusht/run.sh   # tiny sizes, no wandb, no saving
#   SMOKE_SAVE=1 ...                                                          # tiny sizes, no wandb, WITH saving (checks save/restore)
#   RUNNER_MODE=test LOAD_MODEL=<run>/models/latest.model.zip METHOD=S SEED=0 [NR_ENVS=1024] [EVAL_SEED=7] run.sh
#                                                                             # finite re-evaluation of a saved model, prints one JSON line
# Every full run gets a unique run_name {METHOD}_s{SEED}_{job id or timestamp}, so a resubmitted job never overwrites a model.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
METHOD="${METHOD:?set METHOD}"
export PROTOCOL="${PROTOCOL:-paper}"    # paper | fixed | v3, defined in methods.sh
SEED="${SEED:?set SEED}"
RUNNER_MODE="${RUNNER_MODE:-train}"
UNIQUE="${SLURM_JOB_ID:-$(date +%m%d_%H%M%S)}"
SEED_ARG="${SEED}"

if [[ "${RUNNER_MODE}" == "test" ]]; then
  : "${LOAD_MODEL:?set LOAD_MODEL=<.../models/latest.model.zip> for RUNNER_MODE=test}"
  SEED_ARG="${EVAL_SEED:-${SEED}}"   # the evaluation seed
  RUNNER_ARGS=(--runner.track_wandb=False --runner.track_tb=False --runner.save_model=False --runner.track_console=True
               --runner.load_model="${LOAD_MODEL}" --runner.nr_test_episodes=1)
elif [[ "${SMOKE:-0}" == "1" || "${SMOKE_SAVE:-0}" == "1" ]]; then
  # batch = 64 * 16 = 1024 transitions: 4 TRIRL iterations, eval every 2, 2 pre-training iterations
  export NR_ENVS="${NR_ENVS:-64}" NR_STEPS=16 MINIBATCH_SIZE=256 TOTAL_TIMESTEPS=4096 EVAL_FREQUENCY=2048 PRETRAIN_ENV_STEPS=2048 MILESTONE_EVERY=1
  SAVE=False
  RUN_NAME="smoke_${METHOD}_s${SEED}"
  if [[ "${SMOKE_SAVE:-0}" == "1" ]]; then SAVE=True; fi
  RUNNER_ARGS=(--runner.track_wandb=False --runner.track_tb=False --runner.save_model="${SAVE}" --runner.track_console=True
               --runner.run_name="${RUN_NAME}")
  EXP_NAME="${EXP_NAME:-pusht_smoke}"
else
  # wandb naming: group = exp_name, name = {METHOD}_s{SEED}_{job id}, tags = pusht,phase1,method:X,seed:N
  GIT_COMMIT="$(cat "${REPO_ROOT}/GIT_COMMIT" 2>/dev/null || echo unknown)"
  RUNNER_ARGS=(
    --runner.track_wandb=True --runner.track_tb=False --runner.save_model=True --runner.track_console=True
    --runner.wandb_entity="${WANDB_ENTITY:-yichen-feng-tu-darmstadt}"
    --runner.project_name="${WANDB_PROJECT:-role_ip}"
    --runner.run_name="${METHOD}_s${SEED}_${UNIQUE}"
    --runner.notes="PushT ${METHOD}: placeholder_desc | protocol ${PROTOCOL} | code ${GIT_COMMIT}"
  )
fi

source "${REPO_ROOT}/experiments/pusht/methods.sh"
set_method "${METHOD}"
export WANDB_TAGS="pusht,${PHASE:-${PHASE_DEFAULT}},method:${METHOD},seed:${SEED},protocol:${PROTOCOL}"
RUNNER_ARGS=("${RUNNER_ARGS[@]//placeholder_desc/$(method_desc "${METHOD}")}")
if [[ "${RUNNER_MODE}" != "test" ]]; then
  RUNNER_ARGS+=(--runner.exp_name="${EXP_NAME:-${DEFAULT_EXP_NAME}}")   # wandb group
fi

# Expert data: print the hash and refuse to start when it differs from the protocol's file.
if [[ -n "${DATA_SHA256}" ]]; then
  ACTUAL_SHA256="$(sha256sum "${PUSHT_DATA}" | cut -d' ' -f1)"
  echo "[pusht] expert data ${PUSHT_DATA} sha256=${ACTUAL_SHA256}"
  if [[ "${ACTUAL_SHA256}" != "${DATA_SHA256}" ]]; then
    echo "[pusht] ERROR: expert data hash differs from the ${PROTOCOL} protocol (expected ${DATA_SHA256})" >&2
    exit 2
  fi
fi

echo "[pusht] code=$(cat "${REPO_ROOT}/GIT_COMMIT" 2>/dev/null || echo unknown) METHOD=${METHOD} SEED=${SEED} MODE=${RUNNER_MODE} ALGO=${ALGO} SMOKE=${SMOKE:-0} SMOKE_SAVE=${SMOKE_SAVE:-0} host=$(hostname)"
cd "${REPO_ROOT}/experiments"
if [[ "${DRY_RUN:-0}" == "1" ]]; then   # print the command line instead of running it (checked in PLAN_phase1 step 3)
  PRINT=printf; "${PRINT}" '%s\n' "${PYTHON:-python} -u experiment.py --algorithm.name=${ALGO}" "${COMMON_ARGS[@]}" "${METHOD_ARGS[@]}" \
    "--environment.seed=${SEED_ARG}" "--runner.mode=${RUNNER_MODE}" "${RUNNER_ARGS[@]}" "$@"
  exit 0
fi
exec "${PYTHON:-python}" -u experiment.py \
  --algorithm.name="${ALGO}" \
  "${COMMON_ARGS[@]}" \
  "${METHOD_ARGS[@]}" \
  --environment.seed="${SEED_ARG}" \
  --runner.mode="${RUNNER_MODE}" \
  "${RUNNER_ARGS[@]}" \
  "$@"
