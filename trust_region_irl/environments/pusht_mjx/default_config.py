from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.seed = 1
    config.nr_envs = 64
    config.render = False
    config.device = "gpu"
    config.copy_train_env_for_eval = True
    config.feature_fn = "base"  # IRL feature basis: base | base_rbf | base_without_ctrl | pos_ort | state_action
    config.random_TPos = False
    config.preset_set = "legacy"  # start poses used by random_TPos / the fixed start: legacy | paper (Table 1, the 600-demo dataset)
    config.horizon = 250
    config.success_threshold = 0.1
    config.eval_init_noise_xy = 0.0   # eval-only uniform block xy perturbation of the start (m)
    config.eval_init_noise_yaw = 0.0  # eval-only uniform block yaw perturbation of the start (rad)
    config.control_freq = 20  # Hz: 20 = 2 physics steps of 0.025 s per action (Phase 1); 40 = 1 step (v3, expert timebase)
    config.orientation_repr = "quat"  # block orientation in the observation: quat (24 dims, Phase 1) | rot6d (26 dims, sign invariant)

    return config
