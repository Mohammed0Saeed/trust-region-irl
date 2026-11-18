"""Use a TRIRL reward as the cost of the MTP controller and roll it out from the random T poses (t_config).

The controller and the Push-T Franka task come from pusht_franka_eval_mtp; the parts needed are copied into
mtp_dependencies/hydrax/ (the robot meshes are shared with trust_region_irl/environments/pusht_mjx/data/assets).
The cost is -reward of the checkpoint:
    SPR       r = theta . z(obs), z = online SPR encoder (7-D task-based or 24-D observation, read from the checkpoint)
    FEATURES  r = theta . phi(s), phi = pos_ort (2-D) or base_without_ctrl (3-D) features as in pusht_mjx,
              or the Boltzmann feature encoder for boltzmann-feature-based runs

Every pose is rolled out once (controller seeded with `seed`), stopping early on success. Success uses the same
definition as PushT: position error < success_threshold (m) and orientation error < success_threshold (rad),
both measured from the body frames of the block and the goal.

Outputs in output_dir:
    mtp_results.csv    final position / orientation error, success, status and steps of every pose
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

# the copied hydrax package (MTP controller + Push-T Franka task)
sys.path.insert(0, str(Path(__file__).resolve().parent / "mtp_dependencies"))

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.algs import MTP
from hydrax.ik import differential_IK, gravity_comp_torque
from hydrax.tasks.pusht_franka_eval import PushTFranka

# Controller and simulation settings of pusht_franka_eval_mtp/pusht_mtp_eval.py
NUM_SAMPLES = 128
NUM_RANDOMIZATIONS = 1
PLANNING_FREQUENCY = 25
PLANNING_HORIZON = 24
SIM_STEPS_PER_CONTROL_STEP = 3
MAX_SPEED = 0.35  # m/s
SIGMA = 0.2
ALPHA = 0.0
TEMPERATURE = 0.01
NUM_ELITES = 48
KEEP_ELITES = 1
BETA = 0.4
MAX_STEP = 250
EE_GOAL_POS = [0.45, 0.1, 0.035]  # end-effector start position, = PushT.GOAL_POS_EE

SPR_INPUT_DIM = {"TASK-BASED": 7, "JOINT-TASK-BASED": 24}


def build_task(checkpoint_path):
    return PushTFranka(
        ik_type="pinv",
        planning_horizon=PLANNING_HORIZON,
        sim_steps_per_control_step=SIM_STEPS_PER_CONTROL_STEP,
        ctrl_limits={"u_min": jnp.array([-MAX_SPEED, -MAX_SPEED]), "u_max": jnp.array([MAX_SPEED, MAX_SPEED])},
        trace_sites=["ee_site"],
        actuation_type="velocity",
        sampling_space="velocity",
        det_init={"ee_goal_pos": list(EE_GOAL_POS)},
        block_type="free",
        reward_path=str(checkpoint_path),
    )


def build_controller(task, seed):
    return MTP(
        task,
        temperature=TEMPERATURE,
        num_samples=NUM_SAMPLES,
        M=3,
        N=64,
        sigma_min=SIGMA,
        sigma_max=SIGMA,
        sigma_start=SIGMA,
        num_elites=NUM_ELITES,
        keep_elites=KEEP_ELITES,
        beta=BETA,
        alpha=ALPHA,
        interpolation="akima",
        num_randomizations=NUM_RANDOMIZATIONS,
        seed=seed,
        update_cov=False,
        shift=True,
        savgol_filter=False,
        planning_freq=PLANNING_FREQUENCY,
    )


def check_reward(task, encoder_type, obs_type):
    """Fail early if the checkpoint's reward does not match ENCODER_TYPE / OBS_TYPE."""
    if encoder_type == "SPR":
        if task.reward_kind != "spr":
            raise ValueError(f"ENCODER_TYPE is SPR, but the checkpoint holds a '{task.reward_kind}' reward")
        if task.spr_input_dim != SPR_INPUT_DIM[obs_type]:
            raise ValueError(f"The checkpoint's SPR encoder takes {task.spr_input_dim}-D input, "
                             f"but OBS_TYPE {obs_type} feeds {SPR_INPUT_DIM[obs_type]}-D input")
    elif task.reward_kind not in ("linear", "fb"):
        raise ValueError(f"ENCODER_TYPE is FEATURES, but the checkpoint holds a '{task.reward_kind}' reward")


def goal_errors(task, mj_data):
    """Position error (m) and orientation error (rad) of the block, as PushT._goal_errors computes them."""
    pos_err = float(np.linalg.norm(mj_data.xpos[task.T_bid] - mj_data.xpos[task.goal_bid]))
    rel_quat = np.zeros(4)
    goal_conj = np.zeros(4)
    mujoco.mju_negQuat(goal_conj, mj_data.xquat[task.goal_bid])
    mujoco.mju_mulQuat(rel_quat, goal_conj, mj_data.xquat[task.T_bid])
    w = abs(float(rel_quat[0]))  # w >= 0: the shorter of the two equivalent rotations
    orn_err = float(2.0 * np.arccos(np.clip(w, -1.0, 1.0)))
    return pos_err, orn_err


def roll_out_pose(task, ctrl, jit_optimize, pose, seed, success_threshold):
    """One MTP episode from the block pose [x, y, angle]; returns its final errors and status."""
    x, y, angle = (float(v) for v in pose)
    task.det_init = {"block_pos_x": x, "block_pos_y": y, "block_angle": angle, "ee_goal_pos": list(EE_GOAL_POS)}
    mj_model, mj_data = task.reset(seed=seed)
    mujoco.mj_forward(mj_model, mj_data)
    ctrl.set_seed(seed)

    sim_steps_per_replan = max(int((1.0 / PLANNING_FREQUENCY) / mj_model.opt.timestep), 1)
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat)
    policy_params = ctrl.init_params(seed)

    status = "timeout"
    step = 0
    for step in range(1, MAX_STEP + 1):
        mjx_data = mjx_data.replace(
            qpos=jnp.array(mj_data.qpos),
            qvel=jnp.array(mj_data.qvel),
            mocap_pos=jnp.array(mj_data.mocap_pos),
            mocap_quat=jnp.array(mj_data.mocap_quat),
            time=mj_data.time,
        )
        policy_params, _ = jit_optimize(mjx_data, policy_params)

        for i in range(sim_steps_per_replan):
            u = ctrl.get_action(policy_params, i * mj_model.opt.timestep)
            if ctrl.control_mapper is not None and task.actuation_type == "velocity":
                u = differential_IK(model=mj_model, data=mj_data, world_site_vel_desired=u)
            if ctrl.gravity_compensator:
                tau_g = gravity_comp_torque(mj_model, mj_data)
                mj_data.qfrc_applied[:] = 0.0
                mj_data.xfrc_applied[:] = 0.0
                mj_data.qfrc_applied[task.actuator_joint_idxs] = tau_g[task.actuator_joint_idxs]
            if np.isnan(u).any():
                status = "nan"
                break
            mj_data.ctrl[:] = np.array(u)
            mujoco.mj_step(mj_model, mj_data)
        if status == "nan":
            break

        pos_err, orn_err = goal_errors(task, mj_data)
        if pos_err < success_threshold and orn_err < success_threshold:
            status = "success"
            break

    pos_err, orn_err = goal_errors(task, mj_data)
    return {"pos_err": pos_err, "orient_err": orn_err, "success": int(status == "success"),
            "status": status, "steps": step}


def run_mtp(t_pos, checkpoint_path, encoder_type, obs_type, seed, success_threshold, output_dir):
    """Roll out MTP with the checkpoint's reward as cost from every pose in t_pos ((nr, 3) [x, y, angle], world frame).

    t_pos=None loads output_dir/t_config.npy (saved by rollout_policy). Returns one result dict per pose.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if t_pos is None:
        t_pos = np.load(output_dir / "t_config.npy")
    t_pos = np.asarray(t_pos, dtype=np.float64)

    task = build_task(checkpoint_path)
    check_reward(task, encoder_type, obs_type)
    ctrl = build_controller(task, seed)
    jit_optimize = jax.jit(ctrl.optimize, donate_argnums=(1,))

    results = []
    for i, pose in enumerate(t_pos):
        start = time.time()
        r = roll_out_pose(task, ctrl, jit_optimize, pose, seed, success_threshold)
        r = {"pose": i + 1, "block_x_m": pose[0], "block_y_m": pose[1], "block_angle_rad": pose[2], **r}
        results.append(r)
        print(f"MTP pose {i + 1}/{len(t_pos)}: pos {r['pos_err']:.3f} m, ort {r['orient_err']:.3f} rad, "
              f"{r['status']} after {r['steps']} steps ({time.time() - start:.1f} s)")

    csv_path = output_dir / "mtp_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {csv_path}")
    print(f"MTP success {sum(r['success'] for r in results)}/{len(results)} (threshold {success_threshold})")
    return results
