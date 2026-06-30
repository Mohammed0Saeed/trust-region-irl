import math
from pathlib import Path
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from trust_region_irl.environments.franka_pusht.state import State
from trust_region_irl.environments.franka_pusht.box_space import BoxSpace
from trust_region_irl.environments.franka_pusht.viewer import MujocoViewer

# Path to the FR3 push-T model bundled with this environment (under ./data) (Taken from magnus code)
_MODEL_PATH = Path(__file__).resolve().parent / "data" / "fr3_pushT_vel" / "scene_mjx_free.xml"

EE_VEL_LIMIT = 0.35  # max planar End Effector speed (m/s) used to normalize actions

# Names for joints of 7-DOF
_JOINT_NAMES = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3",
    "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7",
]

# IK goal: EE at 0.3m in front of table, flat orientation
_GOAL_POS_EE = np.array([0.3, 0.0, 0.045])
_GOAL_QUAT_EE = np.array([0.0, 0.7071, 0.7071, 0.0])  # [w,x,y,z]

# Null-space home configuration (from magnus code)
# [q1, q2, q3, q4, q5, q6, q7]
_QHOME = np.array([0.51199203, 0.1014329, -0.36340348,
                   -2.9813132, 0.50339095, 3.06692214, -1.92271156])


# ----------------------------------------------------------------------
# JAX quaternion helpers (wxyz convention)
# ----------------------------------------------------------------------

def _quat_mul(a, b):
    """Hamilton product of two [w,x,y,z] quaternions."""
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return jnp.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _quat_to_rotvec(q):
    """Rotation vector (axis * angle) from a unit [w,x,y,z] quaternion."""
    # Take the shortest path (w >= 0).
    q = jnp.where(q[0] < 0.0, -q, q)
    w = jnp.clip(q[0], -1.0, 1.0)
    angle = 2.0 * jnp.arccos(w)
    s = jnp.sqrt(jnp.clip(1.0 - w * w, min=0.0))
    # For tiny angles the axis is ill-defined; fall back to the vector part.
    axis = jnp.where(s < 1e-8, q[1:4], q[1:4] / jnp.where(s < 1e-8, 1.0, s))
    return axis * angle


def _quat_error_body(qd, q):
    """Right-invariant orientation error as a rotvec. Inputs [w,x,y,z]."""
    q_inv = jnp.array([q[0], -q[1], -q[2], -q[3]])  # unit-quat inverse = conjugate
    rel = _quat_mul(qd, q_inv)
    return _quat_to_rotvec(rel)


def _euler_to_quat_yaw(angle):
    """[w,x,y,z] quaternion for a yaw-only rotation (numpy scalar)."""
    return np.array([math.cos(angle * 0.5), 0.0, 0.0, math.sin(angle * 0.5)])


class FrankaPushT:
    """
    Franka FR3 push-T environment using the hydrax model, ported to MuJoCo MJX
    so it can run fully inside JAX alongside the *_flax_full_jit algorithms.

    Observation (24D):
      [0:3]   block position relative to goal  (sensor "position")
      [3:7]   block quaternion relative to goal (sensor "orientation")
      [7:10]  EE position relative to goal      (sensor "safety")
      [10:17] FR3 joint positions
      [17:24] FR3 joint velocities

    Action (2D): normalised planar EE velocity [vx, vy] in [-1, 1]
      Actual EE velocity = action * EE_VEL_LIMIT (0.35 m/s). A differential-IK
      controller maps this planar twist (plus height/orientation hold and a
      null-space posture term) to the 7 FR3 velocity-actuator commands.
    """

    def __init__(self, render, horizon: int = 200, feature_fn: str = "base"):
        self.horizon = horizon
        self.feature_fn = feature_fn

        self.mj_model = mujoco.MjModel.from_xml_path(_MODEL_PATH.as_posix())
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.make_data(self.mjx_model)

        # Original env used 1 substep x 5 intermediate steps per env step.
        self.nr_intermediate_steps = 25

        # --- static indices (plain ints / numpy, resolved once) ---
        joint_ids = [self.mj_model.joint(n).id for n in _JOINT_NAMES]
        fr3_qadr = np.asarray(self.mj_model.jnt_qposadr[joint_ids])
        fr3_dadr = np.asarray(self.mj_model.jnt_dofadr[joint_ids])
        self._joint_limits = np.asarray(self.mj_model.jnt_range[joint_ids])
        self._ee_body_id = int(self.mj_model.body("ee_frame").id)

        block_jid = int(self.mj_model.body("block").jntadr[0])
        self._block_qadr = int(self.mj_model.jnt_qposadr[block_jid])

        def _sadr(name):
            sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            return int(self.mj_model.sensor_adr[sid])

        self._sadr_pos = _sadr("position")        # 3D
        self._sadr_quat = _sadr("orientation")    # 4D
        self._sadr_safety = _sadr("safety")       # 3D
        self._sadr_ee_t1 = _sadr("ee_t1")         # 3D
        self._sadr_ee_t2 = _sadr("ee_t2")         # 3D
        self._sadr_ee_t3 = _sadr("ee_t3")         # 3D
        self._sadr_ee_pos = _sadr("ee_frame_pos")    # 3D
        self._sadr_ee_quat = _sadr("ee_frame_quat")  # 4D

        # jnp versions for in-graph fancy indexing
        self._fr3_qadr = jnp.asarray(fr3_qadr, dtype=jnp.int32)
        self._fr3_dadr = jnp.asarray(fr3_dadr, dtype=jnp.int32)
        self._goal_quat_ee = jnp.asarray(_GOAL_QUAT_EE, dtype=jnp.float32)
        self._qhome = jnp.asarray(_QHOME, dtype=jnp.float32)

        # Base qpos/qvel; the FR3 home pose is solved once on CPU because the
        # IK goal is fixed, so every reset places the EE at the same point.
        initial_fr3_qpos = self._solve_home_ik(fr3_qadr, fr3_dadr)
        init_qpos = np.asarray(self.mj_model.qpos0).copy()
        init_qpos[fr3_qadr] = initial_fr3_qpos
        self.initial_qpos = jnp.asarray(init_qpos, dtype=jnp.float32)
        self.initial_qvel = jnp.zeros(self.mjx_model.nv, dtype=jnp.float32)
        self._initial_fr3_qpos = jnp.asarray(initial_fr3_qpos, dtype=jnp.float32)

        # --- spaces ---
        action_low = -jnp.ones(2, dtype=jnp.float32)
        action_high = jnp.ones(2, dtype=jnp.float32)
        self.single_action_space = BoxSpace(low=action_low, high=action_high, shape=(2,), dtype=jnp.float32)
        self.single_observation_space = BoxSpace(
            low=-jnp.inf, high=jnp.inf, shape=(24,), dtype=jnp.float32
        )
        if self.feature_fn == "state_action":
            feature_dim = self.single_observation_space.shape[0] + self.single_action_space.shape[0]
        else:
            feature_dim = 3
        self.single_features_shape = BoxSpace(low=-jnp.inf, high=jnp.inf, shape=(feature_dim,), dtype=jnp.float32)

        self.viewer = None
        if render:
            dt = self.mj_model.opt.timestep * self.nr_intermediate_steps
            self.viewer = MujocoViewer(self.mj_model, dt)

    # ------------------------------------------------------------------
    # One-time CPU IK to place the EE at the (fixed) goal pose
    # ------------------------------------------------------------------

    def _solve_home_ik(self, fr3_qadr, fr3_dadr):
        """Damped-least-squares IK on the CPU model (run once at construction)."""
        data = mujoco.MjData(self.mj_model)
        q = np.array([0.0, -np.pi / 4, 0.0, -9 * np.pi / 10, 0.0, 3 * np.pi / 4, np.pi / 4])

        max_iters = 1000
        tolerance = 1e-3
        damping = 0.1
        step_size = 1.0

        from scipy.spatial.transform import Rotation as R
        r_des = R.from_quat([_GOAL_QUAT_EE[1], _GOAL_QUAT_EE[2], _GOAL_QUAT_EE[3], _GOAL_QUAT_EE[0]])

        for _ in range(max_iters):
            data.qpos[fr3_qadr] = q
            mujoco.mj_forward(self.mj_model, data)

            current_pos = data.xpos[self._ee_body_id].copy()
            current_quat = data.xquat[self._ee_body_id].copy()  # w,x,y,z

            pos_err = _GOAL_POS_EE - current_pos
            r_cur = R.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
            orn_err = (r_des * r_cur.inv()).as_rotvec()

            err = np.concatenate([pos_err, orn_err])
            if np.linalg.norm(err) < tolerance:
                break

            J_pos = np.zeros((3, self.mj_model.nv))
            J_rot = np.zeros((3, self.mj_model.nv))
            mujoco.mj_jacBody(self.mj_model, data, J_pos, J_rot, self._ee_body_id)
            J = np.vstack([J_pos[:, fr3_dadr], J_rot[:, fr3_dadr]])

            H = J.T @ J + damping * np.eye(7)
            dq = np.linalg.solve(H, J.T @ err)
            q = q + step_size * dq
            q = np.clip(q, self._joint_limits[:, 0], self._joint_limits[:, 1])

        return q

    # ------------------------------------------------------------------
    # JAX environment interface (mirrors the *_mjx siblings)
    # ------------------------------------------------------------------

    @partial(jax.vmap, in_axes=(None, 0, None))
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, eval_mode):
        data = self.mjx_data

        next_observation = jnp.zeros(self.single_observation_space.shape, dtype=jnp.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "rollout/episode_return": reward,
            "rollout/episode_length": 0,
            "env_info/pos_dist": 0.0,
            "env_info/angle_dist": 0.0,
            "env_info/push_dist": 0.0,
            "env_info/is_success": 0.0,
        }
        info_episode_store = {
            "episode_return": reward,
            "episode_length": 0,
        }

        state = State(data, next_observation, next_observation, reward, terminated, truncated, info, info_episode_store, key)
        return self._reset(state)

    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        key, block_key = jax.random.split(state.key)
        k1, k2, k3, k4 = jax.random.split(block_key, 4)

        # Randomise the T-block pose (free joint stores absolute world coords).
        sign_x = jnp.where(jax.random.bernoulli(k1), 1.0, -1.0)
        offset_x = sign_x * jax.random.uniform(k2, minval=0.1, maxval=0.25)
        offset_y = jax.random.uniform(k3, minval=-0.1, maxval=0.15)
        angle = jax.random.uniform(k4, minval=jnp.pi / 4, maxval=jnp.pi)
        quat = jnp.array([jnp.cos(angle * 0.5), 0.0, 0.0, jnp.sin(angle * 0.5)])

        a = self._block_qadr
        qpos = self.initial_qpos
        qpos = qpos.at[a + 0].set(0.5 + offset_x)   # world x, board center at 0.5
        qpos = qpos.at[a + 1].set(0.0 + offset_y)   # world y, board center at 0.0
        qpos = qpos.at[a + 2].set(0.045)
        qpos = qpos.at[a + 3:a + 7].set(quat)
        qpos = qpos.at[self._fr3_qadr].set(self._initial_fr3_qpos)

        data = self.mjx_data.replace(qpos=qpos, qvel=self.initial_qvel, ctrl=jnp.zeros(self.mjx_model.nu))
        data = mjx.forward(self.mjx_model, data)

        next_observation = self.get_observation(data)
        reward = 0.0
        terminated = False
        truncated = False
        info_episode_store = {
            "episode_return": reward,
            "episode_length": 0,
        }

        return state.replace(
            data=data,
            next_observation=next_observation,
            actual_next_observation=next_observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info_episode_store=info_episode_store,
            key=key,
        )

    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        return self._step(state, action)

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        action = jnp.clip(action, -1.0, 1.0)
        world_vel = action.astype(jnp.float32) * EE_VEL_LIMIT  # planar EE velocity (m/s)

        def substep(data, _):
            dq = self._differential_ik(data, world_vel)
            data = mjx.step(self.mjx_model, data.replace(ctrl=dq))
            return data, None

        data, _ = jax.lax.scan(substep, state.data, xs=None, length=self.nr_intermediate_steps)

        state.info_episode_store["episode_length"] += 1

        next_observation = self.get_observation(data)
        reward, r_info = self.get_reward(data)
        terminated = r_info["env_info/is_success"] > 0.5
        truncated = state.info_episode_store["episode_length"] >= self.horizon
        done = terminated | truncated

        state.info.update(r_info)
        state.info_episode_store["episode_return"] += reward
        state.info["rollout/episode_return"] = jnp.where(done, state.info_episode_store["episode_return"], state.info["rollout/episode_return"])
        state.info["rollout/episode_length"] = jnp.where(done, state.info_episode_store["episode_length"], state.info["rollout/episode_length"])

        def when_done(_):
            start_state = self._reset(state)
            start_state = start_state.replace(
                actual_next_observation=next_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
            return start_state

        def when_not_done(_):
            return state.replace(
                data=data,
                next_observation=next_observation,
                actual_next_observation=next_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )

        return jax.lax.cond(done, when_done, when_not_done, None)

    # ------------------------------------------------------------------
    # Differential IK: planar EE twist -> 7 joint velocity commands
    # ------------------------------------------------------------------

    def _differential_ik(self, data, world_vel):
        ee_pos = data.xpos[self._ee_body_id]
        jacp, jacr = mjx.jac(self.mjx_model, data, ee_pos, self._ee_body_id)  # (nv, 3) each
        J = jnp.concatenate([jacp.T, jacr.T], axis=0)[:, self._fr3_dadr]      # (6, 7)
        # Damped least-squares (DLS) pseudo-inverse: J^T (J J^T + lambda^2 I)^-1.
        # The damping term keeps the inverse well-conditioned near kinematic
        # singularities, preventing the huge joint velocities that otherwise
        # diverge the MuJoCo integrator into NaN/Inf.
        dls_lambda = 0.1
        J_pinv = J.T @ jnp.linalg.inv(J @ J.T + (dls_lambda ** 2) * jnp.eye(6))  # (7, 6)

        ee_pos_s = data.sensordata[self._sadr_ee_pos:self._sadr_ee_pos + 3]
        ee_quat = data.sensordata[self._sadr_ee_quat:self._sadr_ee_quat + 4]
        goal_vec = _quat_error_body(self._goal_quat_ee, ee_quat)

        twist_err = jnp.concatenate([world_vel, jnp.array([0.045 - ee_pos_s[2]]), goal_vec])  # (6,)
        dq = J_pinv @ twist_err
        N = jnp.eye(7) - J_pinv @ J
        qnow = data.qpos[self._fr3_qadr]
        dq = dq + N @ (10.0 * (self._qhome - qnow))
        jnp.clip(dq, self._dq_low, self._dq_high)
        return dq

    # ------------------------------------------------------------------
    # Observation / reward
    # ------------------------------------------------------------------

    def get_observation(self, data):
        observation = jnp.concatenate([
            data.sensordata[self._sadr_pos:self._sadr_pos + 3],        # 0:3
            data.sensordata[self._sadr_quat:self._sadr_quat + 4],      # 3:7
            data.sensordata[self._sadr_safety:self._sadr_safety + 3],  # 7:10
            data.qpos[self._fr3_qadr],                                  # 10:17
            data.qvel[self._fr3_dadr],                                  # 17:24
        ])
        return jnp.nan_to_num(observation.astype(jnp.float32))

    def get_reward(self, data):
        pos_err = data.sensordata[self._sadr_pos:self._sadr_pos + 3]
        pos_dist = jnp.linalg.norm(pos_err)

        quat_rel = data.sensordata[self._sadr_quat:self._sadr_quat + 4]
        w = jnp.clip(jnp.abs(quat_rel[0]), 0.0, 1.0)
        orn_dist = 2.0 * jnp.arccos(w)

        ee_t1 = jnp.linalg.norm(data.sensordata[self._sadr_ee_t1:self._sadr_ee_t1 + 3])
        ee_t2 = jnp.linalg.norm(data.sensordata[self._sadr_ee_t2:self._sadr_ee_t2 + 3])
        ee_t3 = jnp.linalg.norm(data.sensordata[self._sadr_ee_t3:self._sadr_ee_t3 + 3])
        push_dist = ee_t1 + ee_t2 + ee_t3

        # Constraint to keep the ee near the block always
        d_min = 0.04
        d_max = 0.13

        zone_violation = jnp.maximum(0.0, d_min - push_dist) + jnp.maximum(0.0, push_dist - d_max)

        cost = pos_dist + orn_dist + push_dist + zone_violation
        # Map NaN->0 and clamp infinities to a finite range, then clip the reward
        # so a single diverged env cannot produce ~1e38 returns that explode the
        # critic. REWARD_MIN bounds the worst per-step penalty.
        REWARD_MIN = -50.0
        reward = jnp.nan_to_num(-cost, nan=0.0, posinf=0.0, neginf=REWARD_MIN)
        reward = jnp.clip(reward, REWARD_MIN, 0.0)

        is_success = (pos_dist + orn_dist < 0.05).astype(jnp.float32)

        info = {
            "env_info/pos_dist": pos_dist,
            "env_info/angle_dist": orn_dist,
            "env_info/push_dist": push_dist,
            "env_info/is_success": is_success,
        }
        return reward, info

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def render(self, state):
        env_id = 0
        data = mjx.get_data(self.mj_model, state.data)[env_id]
        self.viewer.render(data)
        return state

    def close(self):
        if self.viewer:
            self.viewer.close()

    # ------------------------------------------------------------------
    # Replay a saved policy with on-screen rendering
    # ------------------------------------------------------------------

    @staticmethod
    def find_latest_model(search_dir="."):
        """Return the path to the most recently modified ``latest.model.zip``.

        Walks ``search_dir`` recursively (the runner saves checkpoints under
        ``runs/<project>/<exp>/<run_name>/models/``) and picks the newest one
        by modification time. Returns ``None`` if nothing is found.
        """
        from pathlib import Path
        candidates = list(Path(search_dir).rglob("latest.model.zip"))
        if not candidates:
            return None
        return str(max(candidates, key=lambda p: p.stat().st_mtime))

    def replay_policy(self, model_path=None, search_dir=".", nr_steps=1000,
                      deterministic=True, seed=0):
        """Load a saved policy checkpoint and roll it out here with rendering.

        Convenience wrapper so the last trained agent can be watched directly,
        without going through the runner's ``test`` mode. The policy network is
        reconstructed from whichever algorithm produced the checkpoint (read
        from the bundled ``config_algorithm.json``), so this stays in sync if a
        network definition changes.

        Args:
            model_path: Path to a ``*.model.zip`` checkpoint. If ``None``, the
                most recently saved ``latest.model.zip`` under ``search_dir`` is
                used.
            search_dir: Root to search when ``model_path`` is ``None``.
            nr_steps: Number of environment steps to roll out.
            deterministic: If ``True`` use the policy mean; otherwise sample.
            seed: PRNG seed for the reset/sampling keys.

        Returns:
            The mean episode return over the rollout (float).
        """
        import os
        import json
        import shutil
        import tempfile
        import importlib
        from types import SimpleNamespace

        import orbax.checkpoint as ocp
        from flax.training import orbax_utils

        if model_path is None:
            model_path = self.find_latest_model(search_dir)
            if model_path is None:
                raise FileNotFoundError(
                    f"No 'latest.model.zip' found under '{os.path.abspath(search_dir)}'. "
                    f"Pass model_path explicitly."
                )
        print(f"Replaying policy: {model_path}")

        # The viewer is only created at construction time when render=True; make
        # sure one exists so the rollout is actually visible.
        if self.viewer is None:
            dt = self.mj_model.opt.timestep * self.nr_intermediate_steps
            self.viewer = MujocoViewer(self.mj_model, dt)

        # Unpack the zipped orbax checkpoint into a temp dir.
        tmp_dir = tempfile.mkdtemp(prefix="franka_pusht_replay_")
        try:
            shutil.unpack_archive(model_path, tmp_dir, "zip")

            algo_config = json.load(open(os.path.join(tmp_dir, "config_algorithm.json"), "r"))
            algo_name = algo_config["name"]  # e.g. "trirl_ppo_fb.flax_full_jit"

            # Make sure the env exposes the rl_x general properties get_policy needs.
            if not hasattr(self, "general_properties"):
                from trust_region_irl.environments.franka_pusht.general_properties import GeneralProperties
                self.general_properties = GeneralProperties

            # Reconstruct the exact policy network from the producing algorithm.
            policy_module = importlib.import_module(
                f"trust_region_irl.algorithms.{algo_name}.policy"
            )
            cfg = SimpleNamespace(algorithm=SimpleNamespace(
                std_dev=algo_config.get("std_dev", 1.0),
                action_clipping_and_rescaling=algo_config.get("action_clipping_and_rescaling", False),
            ))
            policy, get_processed_action = policy_module.get_policy(cfg, self)

            # Build a target with the policy params only and partial-restore it.
            dummy_obs = jnp.zeros((1,) + self.single_observation_space.shape, dtype=jnp.float32)
            init_params = policy.init(jax.random.PRNGKey(0), dummy_obs)
            target = {"policy": {"params": init_params}}
            restore_args = orbax_utils.restore_args_from_target(target)
            checkpointer = ocp.PyTreeCheckpointer()
            restored = checkpointer.restore(
                tmp_dir,
                args=ocp.args.PyTreeRestore(item=target, restore_args=restore_args, partial_restore=True),
            )
            policy_params = restored["policy"]["params"]
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Single-env rollout (the public reset/step are vmapped over a leading
        # axis, so we use nr_envs=1) with the viewer driven each step.
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        reset_keys = jax.random.split(subkey, 1)
        env_state = self.reset(reset_keys, True)

        episode_return = jnp.zeros((1,))
        for _ in range(nr_steps):
            observation = env_state.next_observation
            action_mean, action_logstd = policy.apply(policy_params, observation)
            if deterministic:
                action = action_mean
            else:
                key, noise_key = jax.random.split(key)
                action_std = jnp.exp(action_logstd)
                action = action_mean + action_std * jax.random.normal(noise_key, shape=action_mean.shape)
            processed_action = get_processed_action(action)
            env_state = self.step(env_state, processed_action)
            episode_return += env_state.reward
            env_state = self.render(env_state)

        mean_return = float(episode_return.mean())
        print(f"Mean episode return: {mean_return}")
        return mean_return

    def feature_from_transition(self, observation, action, eps=1e-6):
        """
        f1 = T Block to Goal
        f2 = EE to Block
        f3 = Orientation Err
        Add a constraint ||EE - Block|| < e
        """
        def feature_base(observation, action):
            observation = jnp.asarray(observation, dtype=jnp.float32)
            action = jnp.asarray(action, dtype=jnp.float32)

            squeeze = observation.ndim == 1
            if squeeze:
                observation = observation[None, :]
                action = action[None, :]

            pos_err = jnp.linalg.norm(observation[:, 0:3], axis=-1)
            w = jnp.clip(jnp.abs(observation[:, 3]), 0.0, 1.0)
            orn_err = 2.0 * jnp.arccos(w)
            ee_to_block_dist = jnp.linalg.norm(observation[:, 0:3] - observation[:, 7:10], axis=-1)

            features = jnp.stack([pos_err, orn_err, ee_to_block_dist], axis=-1).astype(jnp.float32)
            return features[0] if squeeze else features

        def feature_state_action(observation, action):
            observation = jnp.asarray(observation, dtype=jnp.float32)
            action = jnp.asarray(action, dtype=jnp.float32)

            squeeze = observation.ndim == 1
            if squeeze:
                observation = observation[None, :]
                action = action[None, :]

            features = jnp.concatenate([observation, action], axis=-1)
            return features[0] if squeeze else features

        if self.feature_fn == "state_action":
            return feature_state_action(observation, action)
        return feature_base(observation, action)
