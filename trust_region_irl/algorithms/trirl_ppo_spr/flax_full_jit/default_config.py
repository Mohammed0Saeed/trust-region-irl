from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.reward_input_mode = "full26"
    config.device = "gpu"  # cpu, gpu
    config.nr_parallel_seeds = 1
    config.total_timesteps = 2e9
    config.learning_rate = 4e-4
    config.anneal_learning_rate = True
    config.nr_steps = 128
    config.nr_epochs = 10
    config.minibatch_size = 32768
    config.gamma = 0.99
    config.gae_lambda = 0.9
    config.clip_range = 0.1
    config.entropy_coef = 0.001
    config.critic_coef = 1.0
    config.max_grad_norm = 5.0
    config.std_dev = 1.0
    config.action_clipping_and_rescaling = False
    config.evaluation_and_save_frequency = -1  # -1 to disable
    config.evaluation_active = True

    # TRIRL Params
    config.nr_hidden_units_disc = 256
    config.learning_rate_disc = 0.0002015531860350999
    config.nr_epochs_disc = 30
    config.env_reward_frac = 0.0
    config.data_path = "../trirl_dataset/rl_expert/Ant-v5_30_PPO.npz"
    config.expected_data_sha256 = ""  # if set, the run refuses to start when the SHA-256 of data_path differs (always logged)
    config.milestone_every_evals = 12  # besides latest/best: keep a full checkpoint at the end of pre-training, every N evaluations and at the end (0 = off)

    config.epsilon = 0.693139308915975
    config.init_eta = 100.0
    config.const_eta = False
    config.beta = float(1/config.entropy_coef)
    config.gp_lambda = 0.03267622091691947
    config.gp_alpha = 0.5

    # BYOL / SPR Params
    config.target_momentum = 0.99 # For EMA (SPR paper: 0.99 without augmentation)
    config.pretrain_env_steps = 50e6 # Environment steps of encoder-only pre-training before TRIRL (not counted in total_timesteps)
    config.warmup_env_steps = 0 # TRIRL steps during which the discriminator loss does not train the encoder
    config.w_disc_loss = 1.0

    config.predictor_mode = "byol_s"  # "byol_s": 1-step predictor q([g(z), a]) of BYOL-S | "spr": latent transition model h(z, a) unrolled spr_k steps | "lejepa": same unrolling in z space, no EMA / stop-grad / projector (needs ssl_loss=mse) | "bilinear": CPC-style W_k scores, no EMA / stop-grad (needs ssl_loss=bilinear_infonce)
    config.spr_k = 5  # prediction depth K of the "spr" mode (SPR paper: 5). The "byol_s" mode always uses 1.
    config.spr_residual = False  # transition model z_k = h(z_{k-1}, a) (False, SPR default) or z_{k-1} + h(...)
    config.ssl_loss = "cosine"  # "cosine": SPR/BYOL normalised-L2 | "infonce": temporal InfoNCE (SPR appendix B) | "mse": LeJEPA-style, mean over latent dims | "bilinear_infonce": CPC-style unnormalised bilinear InfoNCE
    config.infonce_temperature = 0.1
    config.infonce_mask_radius = 0  # same-episode samples within this many steps are not used as negatives (0: exact duplicates only)
    config.correction_mode = "theta"  # theta: mix theta only, newest encoder (Phase 1) | function: mix every past discriminator's reward (official trirl_ppo, full history)
    config.log_correction_residual = False  # also keep the full-history buffer in theta mode and log the per-iteration FN - theta difference
    config.fn_buffer_capacity = 0  # 0: every TRIRL iteration of the run (set automatically, stored with the checkpoint)
    config.fn_chunk_size = 5  # discriminators evaluated at once in the function-space correction (memory vs speed)
    config.freeze_encoder_online = False  # FRZ: the encoder keeps its pre-trained value during TRIRL; only theta is learned
    config.w_ssl = 1.0  # weight of the SSL loss (the sum over the K steps) relative to the BCE loss
    config.w_sigreg = 0.0  # weight of SIGReg (LeJEPA); 0 disables it. LeJEPA: (1-lam)*pred + lam*SIGReg with lam=0.05, i.e. w_sigreg/w_ssl ~ 0.053
    config.sigreg_slices = 256
    config.sigreg_on = "z"  # features SIGReg is applied to: "z" (the reward features) | "y" (the projector output, as in the official LeJEPA code)

    # Shared trainer fixes (the original trainers behave as False / False)
    config.gae_cut_on_truncation = True  # cut the GAE recursion at time-out resets too (originally only at terminations)
    config.fix_entropy_scaling = False  # True: the entropy bonus uses entropy_coef once; False (upstream): entropy_coef is applied twice (1e-6 effective)

    config.handle_absorbing_states = True
    config.reward_type = 'byol-observation-based'
    config.retraining = False
    config.curriculum_learning = False

    return config
