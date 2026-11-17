from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.spr import SPR


def get_discriminator(config, env, reward_type='byol-observation-based'):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        if reward_type == 'byol-observation-based':
            return SPR(
                predictor_mode=config.algorithm.predictor_mode,
                reward_input_mode=config.algorithm.get("reward_input_mode", "full26"),
                transition_residual=config.algorithm.spr_residual,
                num_steps=int(config.algorithm.spr_k),
            )
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
