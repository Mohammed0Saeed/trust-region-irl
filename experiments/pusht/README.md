# PushT: self-supervised reward encoders for TRIRL

Trainer `trirl_ppo_spr`: feature-based TRIRL on the Franka PushT task with a learned reward encoder,
`r(s) = theta^T f(s)`. The self-supervised objective of the encoder is a config switch. Check out this branch and run
it as it is (it is based on `main` as of 2026-11-03).

## Before running

- A working environment for this repository (RL-X, JAX, MuJoCo MJX).
- The expert data at `trirl_dataset/rl_expert/trirl_expert_data_100_episodes.npz`
  (SHA-256 `d1374c0b605dbf22f2901e9846133a5864f850677770a4a90f8575ffdcf8fc9b`); `run.sh` checks the hash.
- `export WANDB_ENTITY=... WANDB_PROJECT=...` (the defaults in `run.sh` are ours).

## Commands

Run from the repository root. `METHOD` is `B` (1-step prediction, BYOL-style), `S` (SPR), `SG` (SPR + SIGReg),
`SC` (temporal InfoNCE), `L` (LeJEPA-style) or `N` (CPC-style). With `PROTOCOL=v3` add the reward-correction scheme:
`_TH` (theta-only mixing), `_FRZ` (encoder frozen after pre-training) or `_FN` (function-space mixing).

```bash
# quick checks
cd experiments/pusht && PYTHONPATH=../.. JAX_PLATFORMS=cpu python test_spr_components.py && cd ../..
SMOKE=1 PROTOCOL=v3 METHOD=SG_TH SEED=0 experiments/pusht/run.sh   # short GPU run, no W&B, nothing saved

# main comparison (v3, theta-only correction, full 26-dim reward input)
PROTOCOL=v3 METHOD=SG_TH SEED=1400 experiments/pusht/run.sh      # seeds 1400-1402
PROTOCOL=v3 METHOD=S_TH  SEED=2600 experiments/pusht/run.sh      # seeds 2600-2602
PROTOCOL=v3 METHOD=B_TH  SEED=1500 experiments/pusht/run.sh      # seeds 1500-1502
# no self-supervised loss and no pre-training (seeds 2610-2612)
PROTOCOL=v3 METHOD=S_TH  SEED=2610 experiments/pusht/run.sh \
    --algorithm.pretrain_env_steps=0 --algorithm.w_ssl=0.0 --algorithm.w_sigreg=0.0
# restricted reward input: task12 = block pose + end-effector position (seeds 2200-2202),
#                          object9 = block pose only (seeds 2300-2302)
PROTOCOL=v3 METHOD=SG_TH SEED=2300 experiments/pusht/run.sh --algorithm.reward_input_mode=object9
# reward-correction schemes: B_FRZ 1000-1002, B_FN 1100-1102, SG_FRZ 1200-1202, SG_FN 1300-1302
PROTOCOL=v3 METHOD=SG_FN SEED=1300 experiments/pusht/run.sh
# first round (20 Hz, quaternion observation), seeds 0-2: B S SG SC L N
PROTOCOL=paper METHOD=SG SEED=0 experiments/pusht/run.sh
```

`DRY_RUN=1` prints the full command line without running. A full run is 50M pre-training steps plus 150M TRIRL steps,
about 3 hours on an RTX A5000. The number to compare per run is the mean of `eval/is_success` over its last 15
evaluations (`v3`) or last 5 (`paper`).
