from trust_region_irl.environments.pusht_mjx.environment import PushT
from trust_region_irl.environments.pusht_mjx.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    env_kwargs = dict(
        render=config.environment.render,
        feature_fn=config.environment.feature_fn,
        random_TPos=config.environment.random_TPos,
        horizon=config.environment.horizon,
        success_threshold=config.environment.success_threshold,
        eval_init_noise_xy=config.environment.eval_init_noise_xy,
        eval_init_noise_yaw=config.environment.eval_init_noise_yaw,
        preset_set=config.environment.preset_set,
        orientation_repr=config.environment.get("orientation_repr", "quat"),
        control_freq=config.environment.get("control_freq", 20),
    )
    train_env = PushT(**env_kwargs)
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env

    eval_env = PushT(**env_kwargs)
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
