from trust_region_irl.environments.franka_pusht.environment import FrankaPushT
from trust_region_irl.environments.franka_pusht.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    """
    Create a train and eval environment.
    """
    train_env = FrankaPushT(config.environment.render)
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env

    eval_env = FrankaPushT(config.environment.render)
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
