"""Use a TRIRL reward as the cost of the MTP controller and roll it out from the random T poses (t_config).

The controller and the Push-T Franka task come from pusht_franka_eval_mtp; the parts needed are copied into
mtp_dependencies/hydrax/ (the robot meshes are shared with trust_region_irl/environments/pusht_mjx/data/assets).
The cost is -reward of the checkpoint:
    SPR       r = theta . z(obs), z = online SPR encoder (7-D task-based or 24-D observation, read from the checkpoint)
    FEATURES  r = theta . phi(s), phi = pos_ort (2-D) or base_without_ctrl (3-D) features as in pusht_mjx,
              or the Boltzmann feature encoder for boltzmann-feature-based runs

Every pose is rolled out once, its controller seeded with the seed the pose was drawn from, stopping early on
success. Success uses the same rule as the policy rollouts (success.py: AND or SUM), with errors and IoU measured
from the body frames of the block and the goal. The IoU is only recorded, it does not decide success.

Outputs in output_dir:
    mtp_results.csv    final position / orientation error, IoU, success, status and steps of every pose
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

from success import describe, iou, is_success

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
MTP_BATCH_SIZE = 10  # poses per batched planner call with parallel=True (lower it if the GPU runs out of memory)

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
    """Position error (m) and orientation error (rad) of the block, as PushT._goal_errors computes them,
    and the IoU of the block T and the goal T (success.iou), all from the body frames of the block and the goal."""
    rel_pos = mj_data.xpos[task.T_bid] - mj_data.xpos[task.goal_bid]
    pos_err = float(np.linalg.norm(rel_pos))
    rel_quat = np.zeros(4)
    goal_conj = np.zeros(4)
    mujoco.mju_negQuat(goal_conj, mj_data.xquat[task.goal_bid])
    mujoco.mju_mulQuat(rel_quat, goal_conj, mj_data.xquat[task.T_bid])
    w = abs(float(rel_quat[0]))  # w >= 0: the shorter of the two equivalent rotations
    orn_err = float(2.0 * np.arccos(np.clip(w, -1.0, 1.0)))
    return pos_err, orn_err, iou(rel_pos[0:2], rel_quat)


def roll_out_pose(task, ctrl, jit_optimize, pose, seed, success_threshold, success_mode="AND"):
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

        if is_success(success_mode, success_threshold, *goal_errors(task, mj_data)[0:2]):
            status = "success"
            break

    pos_err, orn_err, block_iou = goal_errors(task, mj_data)
    return {"pos_err": pos_err, "orient_err": orn_err, "iou": block_iou, "success": int(status == "success"),
            "status": status, "steps": step}


def stack_trees(trees):
    """Stack a list of identical pytrees into one pytree with a leading batch axis."""
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *trees)


def roll_out_poses_parallel(task, ctrl, batch_optimize, batch_get_action, poses, seeds, success_threshold,
                            success_mode="AND"):
    """MTP episodes from several block poses at once: one batched (vmapped) planner call per control step for
    all poses, the cheap CPU part (IK, gravity compensation, mj_step) per pose. Each pose gets the same reset,
    controller seed and success / stop rule as roll_out_pose; a pose that has finished is no longer stepped."""
    nr = len(poses)
    mj_datas, params = [], []
    for pose, seed in zip(poses, seeds):
        x, y, angle = (float(v) for v in pose)
        task.det_init = {"block_pos_x": x, "block_pos_y": y, "block_angle": angle, "ee_goal_pos": list(EE_GOAL_POS)}
        mj_model, mj_data = task.reset(seed=seed)
        mujoco.mj_forward(mj_model, mj_data)
        mj_datas.append(mj_data)
        params.append(ctrl.init_params(seed))
    policy_params = stack_trees(params)

    sim_steps_per_replan = max(int((1.0 / PLANNING_FREQUENCY) / mj_model.opt.timestep), 1)
    # batched mjx state: built once, then only the fields that change are replaced every control step
    mjx_data = stack_trees([mjx.put_data(mj_model, d).replace(mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat)
                            for d in mj_datas])

    status = ["running"] * nr
    steps = [0] * nr
    for step in range(1, MAX_STEP + 1):
        mjx_data = mjx_data.replace(
            qpos=jnp.array(np.stack([d.qpos for d in mj_datas])),
            qvel=jnp.array(np.stack([d.qvel for d in mj_datas])),
            mocap_pos=jnp.array(np.stack([d.mocap_pos for d in mj_datas])),
            mocap_quat=jnp.array(np.stack([d.mocap_quat for d in mj_datas])),
            time=jnp.array([d.time for d in mj_datas]),
        )
        policy_params = batch_optimize(mjx_data, policy_params)

        running = [i for i in range(nr) if status[i] == "running"]
        for k in range(sim_steps_per_replan):
            actions = np.asarray(batch_get_action(policy_params, k * mj_model.opt.timestep))  # (nr, nu)
            for i in running:
                if status[i] != "running":
                    continue
                mj_data = mj_datas[i]
                u = actions[i]
                if ctrl.control_mapper is not None and task.actuation_type == "velocity":
                    u = differential_IK(model=mj_model, data=mj_data, world_site_vel_desired=u)
                if ctrl.gravity_compensator:
                    tau_g = gravity_comp_torque(mj_model, mj_data)
                    mj_data.qfrc_applied[:] = 0.0
                    mj_data.xfrc_applied[:] = 0.0
                    mj_data.qfrc_applied[task.actuator_joint_idxs] = tau_g[task.actuator_joint_idxs]
                if np.isnan(u).any():
                    status[i], steps[i] = "nan", step
                    continue
                mj_data.ctrl[:] = np.array(u)
                mujoco.mj_step(mj_model, mj_data)

        for i in running:
            if status[i] != "running":
                continue
            steps[i] = step
            if is_success(success_mode, success_threshold, *goal_errors(task, mj_datas[i])[0:2]):
                status[i] = "success"
        if all(s != "running" for s in status):
            break

    results = []
    for i in range(nr):
        if status[i] == "running":
            status[i] = "timeout"
        pos_err, orn_err, block_iou = goal_errors(task, mj_datas[i])
        results.append({"pos_err": pos_err, "orient_err": orn_err, "iou": block_iou,
                        "success": int(status[i] == "success"), "status": status[i], "steps": steps[i]})
    return results


def run_mtp(t_pos, checkpoint_path, encoder_type, obs_type, seeds, success_threshold, output_dir, pose_indices=None,
            parallel=False, batch_size=MTP_BATCH_SIZE, success_mode="AND"):
    """Roll out MTP with the checkpoint's reward as cost from poses of t_pos ((nr, 3) [x, y, angle], world frame).

    seeds: one seed per pose of t_pos (the poses seeds of rollout_policy); pose i's controller uses seeds[i].
    pose_indices: which poses of t_pos to run (0-based); None runs all of them.
    parallel: roll the poses out together, batch_size poses per batched planner call (lower it if the GPU runs out
              of memory); False rolls them out one after the other.
    success_mode / success_threshold: see success.py (AND or SUM).
    t_pos=None loads output_dir/t_config.npy (saved by rollout_policy). Returns one result dict per pose.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if t_pos is None:
        t_pos = np.load(output_dir / "t_config.npy")
    t_pos = np.asarray(t_pos, dtype=np.float64)
    seeds = [int(s) for s in seeds]
    if len(seeds) != len(t_pos):
        raise ValueError(f"{len(seeds)} seeds for {len(t_pos)} poses, expected one seed per pose")
    if pose_indices is None:
        pose_indices = range(len(t_pos))
    pose_indices = [int(i) for i in pose_indices]

    task = build_task(checkpoint_path)
    check_reward(task, encoder_type, obs_type)
    ctrl = build_controller(task, seeds[0])  # the random key lives in the params (init_params(seed) per pose)

    results = []
    if parallel:
        # only the updated params are returned, so XLA drops the sampled rollouts instead of keeping them for 10 poses
        batch_optimize = jax.jit(jax.vmap(lambda data, params: ctrl.optimize(data, params)[0]), donate_argnums=(1,))
        batch_get_action = jax.jit(jax.vmap(ctrl.get_action, in_axes=(0, None)))
        for chunk_start in range(0, len(pose_indices), batch_size):
            chunk = pose_indices[chunk_start:chunk_start + batch_size]
            start = time.time()
            chunk_results = roll_out_poses_parallel(task, ctrl, batch_optimize, batch_get_action,
                                                    t_pos[chunk], [seeds[i] for i in chunk], success_threshold,
                                                    success_mode)
            print(f"MTP poses {chunk[0] + 1}-{chunk[-1] + 1} in parallel ({time.time() - start:.1f} s)")
            for i, r in zip(chunk, chunk_results):
                pose = t_pos[i]
                r = {"pose": i + 1, "pose_seed": seeds[i], "block_x_m": pose[0], "block_y_m": pose[1], "block_angle_rad": pose[2], **r}
                results.append(r)
                print(f"MTP pose {i + 1}/{len(t_pos)}: pos {r['pos_err']:.3f} m, ort {r['orient_err']:.3f} rad, "
                      f"IoU {r['iou']:.2f}, {r['status']} after {r['steps']} steps")
        pose_indices = []  # done; skip the sequential loop below

    jit_optimize = jax.jit(ctrl.optimize, donate_argnums=(1,))
    for i in pose_indices:
        pose = t_pos[i]
        start = time.time()
        r = roll_out_pose(task, ctrl, jit_optimize, pose, seeds[i], success_threshold, success_mode)
        # "pose" is the 1-based row of t_config, so it matches the policy rollout from the same pose
        r = {"pose": i + 1, "pose_seed": seeds[i], "block_x_m": pose[0], "block_y_m": pose[1], "block_angle_rad": pose[2], **r}
        results.append(r)
        print(f"MTP pose {i + 1}/{len(t_pos)}: pos {r['pos_err']:.3f} m, ort {r['orient_err']:.3f} rad, "
              f"IoU {r['iou']:.2f}, {r['status']} after {r['steps']} steps ({time.time() - start:.1f} s)")

    csv_path = output_dir / "mtp_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {csv_path}")
    print(f"MTP success {sum(r['success'] for r in results)}/{len(results)} ({describe(success_mode, success_threshold)})")
    return results
