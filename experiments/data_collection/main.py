"""Collect successful MTP expert trajectories for Push-T (Franka) in the TRIRL dataset format (settings in config.py).

DETERMINISTIC: NUM_TRAJECTORIES_SAMPLES successful trajectories from every pose of utils.DET_INITS.
RANDOM:        NUM_TRAJECTORIES_SAMPLES in total, a new start pose per episode drawn as PushT does.
Episodes run BATCH_SIZE at a time with one batched planner call per control step, as in evaluation/mtp_runner.py.

Saves trirl_dataset/rl_expert/pusht_mtp_<mode>_..._episodes.npz (+ .csv with the start pose of every episode).
Run in the conda environment that has hydrax installed: python main.py
"""
import csv
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from hydrax.algs import MTP
from hydrax.tasks.pusht_franka import PushTFranka
from hydrax.simulation.deterministic_beta import differential_IK, gravity_comp_torque

import config
from utils import DET_INITS, EE_GOAL_POS, get_obs, sample_random_pose

DATASET_DIR = Path(__file__).resolve().parents[2] / "trirl_dataset" / "rl_expert"
KEYS = ("states", "actions", "next_states", "rewards", "absorbing")


def build_task():
    return PushTFranka(
        ik_type="pinv",
        planning_horizon=config.PLANNING_HORIZON,
        sim_steps_per_control_step=config.SIM_STEPS_PER_CONTROL_STEP,
        ctrl_limits={
            "u_min": jnp.array([-config.MAX_SPEED, -config.MAX_SPEED]),
            "u_max": jnp.array([config.MAX_SPEED, config.MAX_SPEED]),
        },
        trace_sites=["ee_site"],
        actuation_type="velocity",
        sampling_space="velocity",
        det_init={"ee_goal_pos": list(EE_GOAL_POS)},
        block_type="free",
        mode="error",
    )


def build_controller(task):
    # the random key lives in the planner params (init_params(trial_seed)), not here
    return MTP(
        task,
        temperature=config.TEMPERATURE,
        num_samples=config.NUM_SAMPLES,
        M=3,
        N=64,
        sigma_min=config.SIGMA,
        sigma_max=config.SIGMA,
        sigma_start=config.SIGMA,
        num_elites=config.NUM_ELITES,
        keep_elites=config.KEEP_ELITES,
        beta=config.BETA,
        alpha=config.ALPHA,
        interpolation="akima",
        num_randomizations=config.NUM_RANDOMIZATIONS,
        seed=0,
        update_cov=config.UPDATE_COV,
        shift=config.SHIFT,
        savgol_filter=config.SAVGOL_FILTER,
        planning_freq=config.PLANNING_FREQUENCY,
    )


class BodyIds:
    def __init__(self, mj_model):
        arm_joint_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"fr3_joint{i}") for i in range(1, 8)]
        self.arm_qadr = mj_model.jnt_qposadr[arm_joint_ids]
        self.arm_dofadr = mj_model.jnt_dofadr[arm_joint_ids]
        self.ee = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "ee_frame")
        self.block = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "block")
        self.goal = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "goal")

    def obs(self, mj_data, goal_pos):
        return get_obs(mj_data, self.ee, self.block, goal_pos, self.arm_qadr, self.arm_dofadr)


def stack_trees(trees):
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *trees)


def run_batch(task, ctrl, batch_optimize, batch_get_action, ids, episodes):
    """Roll the episodes ({"pose", "trial_seed"}) out together; returns (status, trajectory) per episode."""
    nr = len(episodes)
    mj_datas, params, goal_pos = [], [], []
    for ep in episodes:
        x, y, angle = ep["pose"]
        task.det_init = {"block_pos_x": x, "block_pos_y": y, "block_angle": angle, "ee_goal_pos": list(EE_GOAL_POS)}
        mj_model, mj_data = task.reset(seed=ep["trial_seed"])
        mujoco.mj_forward(mj_model, mj_data)
        mj_datas.append(mj_data)
        goal_pos.append(np.array(mj_data.xpos[ids.goal]).copy())
        params.append(ctrl.init_params(ep["trial_seed"]))
    policy_params = stack_trees(params)

    sim_steps_per_replan = max(int((1.0 / config.PLANNING_FREQUENCY) / mj_model.opt.timestep), 1)
    mjx_data = stack_trees([mjx.put_data(mj_model, d).replace(mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat)
                            for d in mj_datas])

    status = ["running"] * nr
    trajectories = [{key: [] for key in KEYS} for _ in range(nr)]
    for _ in range(config.MAX_STEP):
        running = [i for i in range(nr) if status[i] == "running"]
        if not running:
            break
        obs = {}
        for i in running:
            mujoco.mj_forward(mj_model, mj_datas[i])
            obs[i] = ids.obs(mj_datas[i], goal_pos[i])

        mjx_data = mjx_data.replace(
            qpos=jnp.array(np.stack([d.qpos for d in mj_datas])),
            qvel=jnp.array(np.stack([d.qvel for d in mj_datas])),
            mocap_pos=jnp.array(np.stack([d.mocap_pos for d in mj_datas])),
            mocap_quat=jnp.array(np.stack([d.mocap_quat for d in mj_datas])),
            time=jnp.array([d.time for d in mj_datas]),
        )
        policy_params = batch_optimize(mjx_data, policy_params)

        macro_actions = None
        for k in range(sim_steps_per_replan):
            actions = np.asarray(batch_get_action(policy_params, k * mj_model.opt.timestep))  # (nr, 2)
            if k == 0:
                macro_actions = actions
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
                    status[i] = "nan"
                    continue
                mj_data.ctrl[:] = np.array(u)
                mujoco.mj_step(mj_model, mj_data)

        for i in running:
            if status[i] != "running":
                continue
            mujoco.mj_forward(mj_model, mj_datas[i])
            success = bool(task.success(mj_datas[i]))
            trajectory = trajectories[i]
            trajectory["states"].append(obs[i])
            trajectory["actions"].append(macro_actions[i] / config.MAX_SPEED)
            trajectory["next_states"].append(ids.obs(mj_datas[i], goal_pos[i]))
            trajectory["rewards"].append(1.0 if success else 0.0)
            trajectory["absorbing"].append(success)
            if success:
                status[i] = "success"

    return [("timeout" if s == "running" else s, t) for s, t in zip(status, trajectories)]


def save(dataset, rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **{key: np.array(dataset[key], dtype=np.float32) for key in KEYS})
    with open(out_path.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "seed", "trial_seed", "block_x_m", "block_y_m",
                                               "block_angle_rad", "steps"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved {len(rows)} episodes ({len(dataset['states'])} transitions) to {out_path}")


def collect(mode=config.RESPOWNING_MODE, num_trajectories=config.NUM_TRAJECTORIES_SAMPLES,
            max_trials=config.MAX_TRIALS, batch_size=config.BATCH_SIZE, out_dir=DATASET_DIR):
    task = build_task()
    ctrl = build_controller(task)
    ids = BodyIds(task.mj_model)
    goal_xy = np.array(task.mj_model.body_pos[ids.goal][0:2])

    batch_optimize = jax.jit(jax.vmap(lambda data, params: ctrl.optimize(data, params)[0]), donate_argnums=(1,))
    batch_get_action = jax.jit(jax.vmap(ctrl.get_action, in_axes=(0, None)))

    if mode == "DETERMINISTIC":
        groups = [(seed, lambda trial, seed=seed: (
            (DET_INITS[seed]["block_pos_x"], DET_INITS[seed]["block_pos_y"], DET_INITS[seed]["block_angle"]),
            seed * 1000 + trial)) for seed in sorted(DET_INITS)]
        out_path = Path(out_dir) / f"pusht_mtp_deterministic_{len(groups)}x{num_trajectories}_episodes.npz"
    elif mode == "RANDOM":
        groups = [(None, lambda trial: (sample_random_pose(np.random.default_rng(trial), goal_xy), trial))]
        out_path = Path(out_dir) / f"pusht_mtp_random_{num_trajectories}_episodes.npz"
    else:
        raise ValueError(f"Unknown RESPOWNING_MODE {mode}, expected DETERMINISTIC or RANDOM")

    budget = max_trials * num_trajectories
    print(f"{mode}: {num_trajectories} successful trajectories per {'seed' if mode == 'DETERMINISTIC' else 'run'} "
          f"({len(groups)} group(s)), at most {budget} attempts each, {batch_size} episodes in parallel")

    dataset = {key: [] for key in KEYS}
    rows = []
    shortfalls = {}
    for seed, make_episode in groups:
        label = f"seed {seed}" if seed is not None else "random poses"
        kept, attempts = 0, 0
        while kept < num_trajectories and attempts < budget:
            # always a full batch, so the planner is compiled only once
            episodes = []
            for _ in range(batch_size):
                attempts += 1
                pose, trial_seed = make_episode(attempts)
                episodes.append({"pose": pose, "trial_seed": trial_seed})
            start = time.time()
            results = run_batch(task, ctrl, batch_optimize, batch_get_action, ids, episodes)

            for ep, (status, trajectory) in zip(episodes, results):
                if status != "success" or kept >= num_trajectories:
                    continue
                kept += 1
                for key in KEYS:
                    dataset[key].extend(trajectory[key])
                rows.append({"episode": len(rows) + 1, "seed": seed if seed is not None else "",
                             "trial_seed": ep["trial_seed"], "block_x_m": ep["pose"][0], "block_y_m": ep["pose"][1],
                             "block_angle_rad": ep["pose"][2], "steps": len(trajectory["states"])})
            nr_success = sum(status == "success" for status, _ in results)
            print(f"{label}: batch {nr_success}/{len(results)} successful ({time.time() - start:.1f} s), "
                  f"kept {kept}/{num_trajectories}, attempts {attempts}/{budget}")
            if mode == "RANDOM":
                save(dataset, rows, out_path)
        if kept < num_trajectories:
            shortfalls[label] = kept
            print(f"WARNING {label}: only {kept}/{num_trajectories} successful trajectories in {attempts} attempts")
        if mode == "DETERMINISTIC":
            save(dataset, rows, out_path)

    print(f"Done: {len(rows)} episodes, {len(dataset['states'])} transitions -> {out_path}")
    if shortfalls:
        print(f"Short of {num_trajectories}: " + ", ".join(f"{k} ({v})" for k, v in shortfalls.items()))
    return out_path


if __name__ == "__main__":
    collect()
