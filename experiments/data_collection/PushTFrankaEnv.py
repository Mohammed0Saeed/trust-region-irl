from trust_region_irl.environments.pusht_mjx.environment import PushT

class PushTFrankaVariant(PushT):
    def success(self, state):
        position_cost = self._get_position_err(state)
        orientation_cost = self._get_orientation_err(state)
        pos_err = jnp.sqrt(jnp.sum(jnp.square(position_cost)))
        orn_err = jnp.sqrt(jnp.sum(jnp.square(orientation_cost)))
        return pos_err + orn_err < 0.05


def get_obs(mj_data, ee_body_id, block_body_id, goal_pos, arm_qadr, arm_dofadr):
    block_pos = mj_data.xpos[block_body_id] - goal_pos
    bquat = mj_data.xquat[block_body_id]
    goal_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # quat conj
    g_conj = np.array([goal_quat[0], -goal_quat[1], -goal_quat[2], -goal_quat[3]])

    # quat mul
    w1, x1, y1, z1 = g_conj
    w2, x2, y2, z2 = bquat
    b_quat_rel = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

    ee_rel = mj_data.xpos[ee_body_id] - goal_pos
    arm_q = mj_data.qpos[arm_qadr]
    arm_v = mj_data.qvel[arm_dofadr]

    obs = np.concatenate([block_pos, b_quat_rel, ee_rel, arm_q, arm_v])
    obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
    return np.clip(obs, -100.0, 100.0)