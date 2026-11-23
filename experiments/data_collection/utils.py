import numpy as np

EE_GOAL_POS = [0.45, 0.1, 0.035]  # = PushT.GOAL_POS_EE

DET_INITS = {
    40: {"block_pos_x": 0.45, "block_pos_y": -0.04, "block_angle": np.pi / 2, "ee_goal_pos": [0.45, 0.1, 0.035]},
    41: {"block_pos_x": 0.45, "block_pos_y": 0.0, "block_angle": -np.pi / 2, "ee_goal_pos": [0.45, 0.1, 0.035]},
    42: {"block_pos_x": 0.45, "block_pos_y": 0.0, "block_angle": np.pi, "ee_goal_pos": [0.45, 0.1, 0.035]},
    43: {"block_pos_x": 0.45, "block_pos_y": -0.04, "block_angle": np.pi / 4, "ee_goal_pos": [0.45, 0.1, 0.035]},
    44: {"block_pos_x": 0.45, "block_pos_y": 0.0, "block_angle": -np.pi / 4, "ee_goal_pos": [0.45, 0.1, 0.035]},
    45: {"block_pos_x": 0.45, "block_pos_y": -0.06, "block_angle": np.pi / 3, "ee_goal_pos": [0.45, 0.1, 0.035]},
    46: {"block_pos_x": 0.45, "block_pos_y": 0.0, "block_angle": -np.pi / 3, "ee_goal_pos": [0.45, 0.1, 0.035]},
    47: {"block_pos_x": 0.45, "block_pos_y": -0.015, "block_angle": np.pi / 6, "ee_goal_pos": [0.45, 0.1, 0.035]},
    48: {"block_pos_x": 0.45, "block_pos_y": 0.0, "block_angle": -np.pi / 6, "ee_goal_pos": [0.45, 0.1, 0.035]},
    49: {"block_pos_x": 0.45, "block_pos_y": 0.2, "block_angle": 0.0, "ee_goal_pos": [0.45, 0.1, 0.035]},
    50: {"block_pos_x": 0.45, "block_pos_y": 0.3, "block_angle": 0.0, "ee_goal_pos": [0.45, 0.1, 0.035]},
    51: {"block_pos_x": 0.45, "block_pos_y": -0.1, "block_angle": 0.0, "ee_goal_pos": [0.45, 0.1, 0.035]},
    52: {"block_pos_x": 0.45, "block_pos_y": -0.2, "block_angle": 0.0, "ee_goal_pos": [0.45, 0.1, 0.035]},
    53: {"block_pos_x": 0.55, "block_pos_y": 0.1, "block_angle": 0.0, "ee_goal_pos": [0.45, 0.1, 0.035]},
    54: {"block_pos_x": 0.65, "block_pos_y": 0.1, "block_angle": 0.0, "ee_goal_pos": [0.45, 0.1, 0.035]},
    55: {"block_pos_x": 0.35, "block_pos_y": 0.1, "block_angle": 0.0, "ee_goal_pos": [0.45, 0.1, 0.035]},
    56: {"block_pos_x": 0.25, "block_pos_y": 0.1, "block_angle": 0.0, "ee_goal_pos": [0.45, 0.1, 0.035]},
    57: {"block_pos_x": 0.4, "block_pos_y": -0.1, "block_angle": np.pi / 2, "ee_goal_pos": [0.45, 0.1, 0.035]},
    58: {"block_pos_x": 0.65, "block_pos_y": 0.0, "block_angle": np.pi, "ee_goal_pos": [0.45, 0.1, 0.035]},
    59: {"block_pos_x": 0.6, "block_pos_y": 0.05, "block_angle": 5 * np.pi / 4, "ee_goal_pos": [0.45, 0.1, 0.035]},
    60: {"block_pos_x": 0.5, "block_pos_y": 0.2, "block_angle": -np.pi / 2, "ee_goal_pos": [0.45, 0.1, 0.035]},
    61: {"block_pos_x": 0.4, "block_pos_y": -0.05, "block_angle": -np.pi / 4, "ee_goal_pos": [0.45, 0.1, 0.035]},
    62: {"block_pos_x": 0.45, "block_pos_y": -0.15, "block_angle": 0 * np.pi, "ee_goal_pos": [0.45, 0.1, 0.035]},
}


def get_det_init_for_seed(seed: int) -> dict:
    if seed in DET_INITS:
        return DET_INITS[seed].copy()
    return {"ee_goal_pos": list(EE_GOAL_POS)}


# Random start poses: the same rule as PushT._reset with random_TPos
SPAWN_RADIUS = 0.20
EE_SPHERE_RADIUS = 0.015
EE_CLEARANCE = 0.02
T_BOX_CENTRES = np.array([[0.0, 0.0], [0.075, 0.0]])
T_BOX_HALF_SIZES = np.array([[0.05, 0.025], [0.025, 0.06]])


def ee_to_t_distance(ee_xy, xy, angle):
    """Distance in the table plane from the end-effector centre to the T surface (0 if inside)."""
    d = np.asarray(ee_xy) - np.asarray(xy)
    c, s = np.cos(angle), np.sin(angle)
    p = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    outside = np.maximum(np.abs(p - T_BOX_CENTRES) - T_BOX_HALF_SIZES, 0.0)
    return float(np.min(np.linalg.norm(outside, axis=-1)))


def sample_random_pose(rng, goal_xy, ee_xy=EE_GOAL_POS[0:2]):
    """Random start pose (x, y, angle) of the T in the world frame."""
    min_distance = EE_SPHERE_RADIUS + EE_CLEARANCE
    while True:
        rho = SPAWN_RADIUS * np.sqrt(rng.uniform())
        phi = rng.uniform(0.0, 2 * np.pi)
        xy = np.asarray(goal_xy) + rho * np.array([np.cos(phi), np.sin(phi)])
        angle = rng.uniform(0.0, 2 * np.pi)
        if ee_to_t_distance(ee_xy, xy, angle) >= min_distance:
            return float(xy[0]), float(xy[1]), float(angle)


def get_obs(mj_data, ee_body_id, block_body_id, goal_pos, arm_qadr, arm_dofadr):
    """24-D TRIRL observation, as PushT.get_observation."""
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
