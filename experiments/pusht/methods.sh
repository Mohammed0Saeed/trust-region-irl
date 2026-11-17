# PushT comparison: the single place where the shared protocol and every method's flags are defined.
# Sourced by run.sh (local and cluster). Every size below can be overridden through the environment
# (run.sh's SMOKE=1 does that for quick local checks).

: "${REPO_ROOT:?REPO_ROOT must be set before sourcing methods.sh}"

# PROTOCOL=paper (default): 600 paper demonstrations (Table-1 starts x 100), training starts drawn from the six Table-1
#                           starts, evaluation per start (see PLAN_phase1.md section 3).
# PROTOCOL=fixed:           old protocol, 93 demonstrations of one fixed start (kept for reference only).
PROTOCOL="${PROTOCOL:-paper}"
case "${PROTOCOL}" in
  paper)
    DEFAULT_PUSHT_DATA="${REPO_ROOT}/trirl_dataset/rl_expert/trirl_expert_data_100_episodes.npz"
    DATA_SHA256="d1374c0b605dbf22f2901e9846133a5864f850677770a4a90f8575ffdcf8fc9b"
    RANDOM_TPOS=True
    PRESET_SET=paper
    DEFAULT_EXP_NAME=pusht_phase1_paper ;;
  fixed)
    DEFAULT_PUSHT_DATA="${REPO_ROOT}/trirl_dataset/rl_expert/expert_dataset_pusht_mtp_clean_93_episodes_trirl_f32abs.npz"
    DATA_SHA256=""
    RANDOM_TPOS=False
    PRESET_SET=legacy
    DEFAULT_EXP_NAME=pusht_phase1 ;;
  *)
    echo "unknown PROTOCOL '${PROTOCOL}' (known: paper fixed)" >&2
    return 1 2>/dev/null || exit 1 ;;
esac
PUSHT_DATA="${PUSHT_DATA:-${DEFAULT_PUSHT_DATA}}"
# The hash only applies to the protocol's own data file; overriding PUSHT_DATA drops the check (prints a warning).
if [[ "${PUSHT_DATA}" != "${DEFAULT_PUSHT_DATA}" && -n "${DATA_SHA256}" ]]; then
  echo "[pusht] WARNING: PUSHT_DATA overridden (${PUSHT_DATA}); the SHA-256 check of the ${PROTOCOL} data file is skipped" >&2
  DATA_SHA256=""
fi

# Shared protocol: same data, env, input, PPO/TRIRL hyper-parameters, budgets, eval starts and threshold.
COMMON_ARGS=(
  --algorithm.data_path="${PUSHT_DATA}"
  --algorithm.total_timesteps="${TOTAL_TIMESTEPS:-150000000}"
  --algorithm.entropy_coef=0.001
  --algorithm.clip_range=0.2
  --algorithm.env_reward_frac=0.0
  --algorithm.nr_steps="${NR_STEPS:-128}"
  --algorithm.nr_epochs=10
  --algorithm.nr_epochs_disc=10
  --algorithm.minibatch_size="${MINIBATCH_SIZE:-512}"
  --algorithm.learning_rate_disc=1e-04
  --algorithm.learning_rate=4e-04
  --algorithm.epsilon=0.2
  --algorithm.init_eta=30.0
  --algorithm.gp_lambda=0.5
  --algorithm.gae_lambda=0.95
  --algorithm.std_dev=0.4
  --algorithm.anneal_learning_rate=True
  --algorithm.evaluation_and_save_frequency="${EVAL_FREQUENCY:-2097152}"
  --environment.name=pusht_mjx
  --environment.random_TPos="${RANDOM_TPOS}"
  --environment.preset_set="${PRESET_SET}"
  --environment.horizon=250
  --environment.success_threshold=0.05
  --environment.eval_init_noise_xy=0.01
  --environment.eval_init_noise_yaw=0.0872665
  --environment.feature_fn=pos_ort
  --environment.nr_envs="${NR_ENVS:-4096}"
)

# Options that only the trirl_ppo_spr trainer knows (the teammate's trirl_ppo_byol_s and trirl_ppo_fb reject them);
# for R and B0 the data hash is checked by run.sh only.
SPR_ONLY_ARGS=(
  --algorithm.expected_data_sha256="${DATA_SHA256}"
  --algorithm.milestone_every_evals="${MILESTONE_EVERY:-12}"
)

# Encoder pre-training budget (env steps, not counted in total_timesteps); identical for every SSL method.
PRETRAIN_ENV_STEPS="${PRETRAIN_ENV_STEPS:-50000000}"

# set_method METHOD -> sets ALGO and METHOD_ARGS
set_method() {
  case "$1" in
    B0)  # old method: Boltzmann (DSM) encoder on pos_ort features
      ALGO="trirl_ppo_fb.flax_full_jit"
      METHOD_ARGS=(
        --algorithm.reward_type=boltzmann-feature-based
        --algorithm.boltzmann_hidden_dims=16,32,64
        --algorithm.boltzmann_latent_dim=16
        --algorithm.boltzmann_energy_hidden_dim=32
        --algorithm.dsm_alpha=0.001
        --algorithm.dsm_sigma=0.5
      ) ;;
    B|S|SG|SC)  # same code (trirl_ppo_spr), same encoder / budgets; they differ only in the flags below
      ALGO="trirl_ppo_spr.flax_full_jit"
      METHOD_ARGS=(--algorithm.pretrain_env_steps="${PRETRAIN_ENV_STEPS}" --algorithm.ssl_loss=cosine --algorithm.w_sigreg=0.0)
      case "$1" in
        B)  METHOD_ARGS+=(--algorithm.predictor_mode=byol_s) ;;                                  # BYOL-S: 1-step predictor
        S)  METHOD_ARGS+=(--algorithm.predictor_mode=spr --algorithm.spr_k=5) ;;                 # SPR: latent dynamics, K=5
        SG) METHOD_ARGS+=(--algorithm.predictor_mode=spr --algorithm.spr_k=5 --algorithm.w_sigreg=0.05) ;;   # SPR + SIGReg (LeJEPA)
        SC) METHOD_ARGS+=(--algorithm.predictor_mode=spr --algorithm.spr_k=5 --algorithm.ssl_loss=infonce --algorithm.infonce_temperature=0.1) ;;  # temporal InfoNCE control
      esac ;;
    L)   # LeJEPA-style (exploratory): ψ-space K-step prediction, MSE, no EMA / stop-grad / projector, SIGReg on ψ.
         # total = 0.95 * prediction + 0.05 * SIGReg (LeJEPA: (1-lam) * inv + lam * sigreg)
      ALGO="trirl_ppo_spr.flax_full_jit"
      METHOD_ARGS=(--algorithm.pretrain_env_steps="${PRETRAIN_ENV_STEPS}" --algorithm.predictor_mode=lejepa --algorithm.spr_k=5
                   --algorithm.ssl_loss=mse --algorithm.w_ssl=0.95 --algorithm.w_sigreg=0.05) ;;
    N)   # CPC-style bilinear InfoNCE (exploratory): ψ-space K-step rollout, score ψ(s_{t+k})^T W_k c_{t,k}, no EMA / stop-grad
      ALGO="trirl_ppo_spr.flax_full_jit"
      METHOD_ARGS=(--algorithm.pretrain_env_steps="${PRETRAIN_ENV_STEPS}" --algorithm.predictor_mode=bilinear --algorithm.spr_k=5
                   --algorithm.ssl_loss=bilinear_infonce --algorithm.w_sigreg=0.0) ;;
    B_NOGAE)  # B with the GAE recursion NOT cut at time-outs (= upstream behaviour): regression check for the GAE fix
      ALGO="trirl_ppo_spr.flax_full_jit"
      METHOD_ARGS=(--algorithm.pretrain_env_steps="${PRETRAIN_ENV_STEPS}" --algorithm.ssl_loss=cosine --algorithm.w_sigreg=0.0
                   --algorithm.predictor_mode=byol_s --algorithm.gae_cut_on_truncation=False) ;;
    R)   # teammate's original BYOL-S code (regression reference for B)
      ALGO="trirl_ppo_byol_s.flax_full_jit"
      METHOD_ARGS=(--algorithm.pretrain_env_steps="${PRETRAIN_ENV_STEPS}") ;;
    *)
      echo "unknown METHOD '$1' (known: B S SG SC L N B_NOGAE R B0)" >&2
      return 1 ;;
  esac
  if [[ "${ALGO}" == trirl_ppo_spr.* ]]; then METHOD_ARGS+=("${SPR_ONLY_ARGS[@]}"); fi
}

# one-line description of a method (wandb notes)
method_desc() {
  case "$1" in
    B)      echo "BYOL-S, 1-step predictor, cosine" ;;
    S)      echo "SPR, latent transition model, K=5, cosine" ;;
    SG)     echo "SPR (K=5) + SIGReg (w=0.05)" ;;
    SC)     echo "temporal InfoNCE control (SPR model, K=5, T=0.1)" ;;
    L)      echo "LeJEPA-style latent prediction + SIGReg (K=5, no EMA/stop-grad), exploratory" ;;
    N)      echo "CPC-style bilinear InfoNCE (K=5, no EMA/stop-grad), exploratory" ;;
    B_NOGAE) echo "B without the GAE truncation fix (regression check)" ;;
    R)      echo "teammate's original trirl_ppo_byol_s (regression reference)" ;;
    B0)     echo "Boltzmann (DSM) encoder" ;;
    *)      echo "$1" ;;
  esac
}
