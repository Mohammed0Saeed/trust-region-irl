"""Roll out the policy of a checkpoint from random T poses and save one video per rollout.

The start poses are drawn by PushT itself (the same sampler as training with random_TPos: disk around the goal,
any angle, clear of the end effector), from PRNG keys built from `seed`. The same seed therefore gives the same
poses for every checkpoint and every experiment.

Rollouts use the deterministic policy (its mean). The scene lights are scaled up and every frame is
gamma-corrected so the robot, the T and the goal are clearly visible.

Outputs in output_dir:
    rollout_<k>.mp4      one video per rollout (k = 1..nr_rollouts), captioned with its final error and status
    initial_poses.csv    start pose [x, y, angle] of every rollout
    t_config.npy         the same poses as a (nr_rollouts, 3) float array, loaded by MTP
    policy_results.csv   final position / orientation error, coverage, success, status and steps of every rollout

A rollout stops at its first success under the chosen success mode (success.py: AND or SUM). The coverage of
the goal T is recorded for every rollout, but it does not decide success.
"""
import csv
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# GLFW renders offscreen through the desktop session; EGL is for headless machines (e.g. the cluster)
os.environ.setdefault("MUJOCO_GL", "glfw" if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") else "egl")

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from trust_region_irl.environments.pusht_mjx.environment import PushT
from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.policy import Policy, get_processed_action_function
from plot_rewards import load_checkpoint
from success import coverage, describe, is_success

NR_ROLLOUTS = 10
LIGHT_FACTOR = 2.5 # scale of the scene light intensities (> 1 brightens)
GAMMA = 0.7 # gamma correction of the rendered pixels (< 1 brightens)
CAMERA = "comparison"
WIDTH, HEIGHT = 640, 480
FPS = 20 # the env's control frequency, so the videos play in real time
CAPTION_HEIGHT = 44


# ------------------------------------------------------------- start poses #
def initial_states(env, seed, nr_rollouts):
    """Reset nr_rollouts envs to random T poses drawn by PushT._reset from keys built from seed."""
    keys = jax.random.split(jax.random.PRNGKey(seed), nr_rollouts)
    return jax.vmap(lambda s: env._reset(s, force_random=True))(env.reset(keys, True))


def block_poses(state):
    """(nr, 3) block pose [x, y, angle] of every env, angle in [0, 2 pi)."""
    qpos = np.asarray(state.data.qpos)
    angle = np.mod(2.0 * np.arctan2(qpos[:, 6], qpos[:, 3]), 2.0 * np.pi)  # quat (w, 0, 0, z) about the vertical axis
    return np.stack([qpos[:, 0], qpos[:, 1], angle], axis=1)


# --------------------------------------------------------------- rendering #
def brighten_scene(model, factor):
    """Scale the headlight and scene light intensities."""
    model.vis.headlight.ambient[:] = np.clip(model.vis.headlight.ambient * factor, 0.0, 1.0)
    model.vis.headlight.diffuse[:] = np.clip(model.vis.headlight.diffuse * factor, 0.0, 1.0)
    model.light_diffuse[:] = np.clip(model.light_diffuse * factor, 0.0, 1.0)
    model.light_ambient[:] = np.clip(model.light_ambient * factor + 0.05 * (factor - 1.0), 0.0, 1.0)


def brighten_image(img, gamma):
    """Gamma-correct an already rendered uint8 RGB image."""
    corrected = np.power(img.astype(np.float32) / 255.0, gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def load_font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def caption(frame, text, font):
    """Put a dark caption bar with text above the frame."""
    h, w, _ = frame.shape
    panel = Image.new("RGB", (w, h + CAPTION_HEIGHT), color=(20, 20, 20))
    panel.paste(Image.fromarray(frame), (0, CAPTION_HEIGHT))
    draw = ImageDraw.Draw(panel)
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text(((w - (bbox[2] - bbox[0])) / 2, (CAPTION_HEIGHT - (bbox[3] - bbox[1])) / 2 - bbox[1]),
              text, fill=(255, 255, 255), font=font)
    return np.array(panel)


# ----------------------------------------------------------------- rollout #
def make_step_fn(env, config, policy_params):
    """Jitted state -> state that steps every env with the policy mean."""
    policy = Policy(env.single_action_space.shape, config["std_dev"], jnp.arange(env.single_observation_space.shape[0]))
    get_processed_action = get_processed_action_function(
        config["action_clipping_and_rescaling"],
        jnp.array(env.single_action_space.low),
        jnp.array(env.single_action_space.high),
    )

    @jax.jit
    def step_fn(state):
        action_mean, _ = policy.apply(policy_params, state.next_observation)
        return env.step(state, get_processed_action(action_mean))
    return step_fn


def roll_out(env, step_fn, renderer, state, success_mode, success_threshold):
    """Roll every env out for one episode; return per-rollout frames and final results.

    Success is checked here after every step with success.is_success (the env's own success termination is
    switched off, see rollout_policy). The errors come from the env info and the coverage from
    actual_next_observation (block pose relative to the goal, dims 0..6), which both still describe the final
    state on the step an episode ends (state.data has already been reset then)."""
    nr = state.next_observation.shape[0]
    # reset leaves some info entries weakly typed and step returns them strongly typed, which would compile step twice
    state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype), state)

    def render(data):
        renderer.update_scene(data, camera=CAMERA)
        return brighten_image(renderer.render(), GAMMA)

    frames = [[] for _ in range(nr)]
    results = [None] * nr
    for step in range(env.horizon):
        datas = mjx.get_data(env.mj_model, state.data)
        for i in range(nr):
            # once an episode ended the env auto-resets, so hold its last frame
            running = results[i] is None or results[i]["status"] == "running"
            frames[i].append(render(datas[i]) if running else frames[i][-1])

        state = step_fn(state)
        truncated = np.asarray(state.truncated)
        pos_err, orient_err = np.asarray(state.info["env_info/pos_err"]), np.asarray(state.info["env_info/orient_err"])
        block_rel_goal = np.asarray(state.actual_next_observation)[:, 0:7]
        for i in range(nr):
            if results[i] is not None and results[i]["status"] != "running":
                continue
            cov = coverage(block_rel_goal[i, 0:2], block_rel_goal[i, 3:7])
            status = "running"
            if is_success(success_mode, success_threshold, float(pos_err[i]), float(orient_err[i])):
                status = "success"
            elif truncated[i]:
                status = "timeout" if step + 1 >= env.horizon else "diverged"
            results[i] = {"pos_err": float(pos_err[i]), "orient_err": float(orient_err[i]), "coverage": cov,
                          "steps": step + 1, "status": status}
    return frames, results


def rollout_policy(checkpoint_path, seed, success_threshold, output_dir, nr_rollouts=NR_ROLLOUTS, success_mode="AND"):
    """Roll the checkpoint's policy out from nr_rollouts random T poses (drawn from seed), save one video each.

    success_mode / success_threshold: see success.py (AND or SUM).
    Returns the (nr_rollouts, 3) start poses [x, y, angle], so the same poses can be reused (e.g. for MTP).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt, config = load_checkpoint(checkpoint_path)
    # success_threshold=0 switches off the env's own success termination (pos_err < 0 never holds);
    # success is checked in roll_out with the chosen mode instead
    env = PushT(render=False, success_threshold=0.0)

    start_state = initial_states(env, seed, nr_rollouts)
    poses = block_poses(start_state)
    poses_path = output_dir / "initial_poses.csv"
    with open(poses_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rollout", "block_x_m", "block_y_m", "block_angle_rad"])
        for i, (x, y, angle) in enumerate(poses):
            writer.writerow([i + 1, f"{x:.5f}", f"{y:.5f}", f"{angle:.5f}"])
    print(f"Saved {poses_path} (seed {seed})")
    # the same poses as an array, so MTP can load them directly: (nr_rollouts, 3) [x, y, angle] in the world frame
    t_config_path = output_dir / "t_config.npy"
    np.save(t_config_path, poses)
    print(f"Saved {t_config_path}")

    brighten_scene(env.mj_model, LIGHT_FACTOR)
    renderer = mujoco.Renderer(env.mj_model, HEIGHT, WIDTH)
    step_fn = make_step_fn(env, config, ckpt["policy"]["params"])
    print(f"Rolling out the policy from {nr_rollouts} random start poses ...")
    frames, results = roll_out(env, step_fn, renderer, start_state, success_mode, success_threshold)
    renderer.close()

    font = load_font(18)
    for i in range(nr_rollouts):
        r = results[i]
        label = (f"rollout {i + 1} | pos {r['pos_err']:.3f} m  ort {r['orient_err']:.3f} rad  "
                 f"cov {r['coverage']:.2f} | {r['status']}")
        out_path = output_dir / f"rollout_{i + 1}.mp4"
        with imageio.get_writer(out_path, fps=FPS, macro_block_size=None) as writer:
            for frame in frames[i]:
                writer.append_data(caption(frame, label, font))
        print(f"Saved {out_path}")

    # final errors of every rollout, same columns as mtp_results.csv (read by overall_evaluation.py)
    results_path = output_dir / "policy_results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pose", "block_x_m", "block_y_m", "block_angle_rad", "pos_err", "orient_err", "coverage",
                         "success", "status", "steps"])
        for i, (r, (x, y, angle)) in enumerate(zip(results, poses)):
            writer.writerow([i + 1, x, y, angle, r["pos_err"], r["orient_err"], r["coverage"],
                             int(r["status"] == "success"), r["status"], r["steps"]])
    print(f"Saved {results_path}")

    nr_success = sum(r["status"] == "success" for r in results)
    print(f"Success {nr_success}/{nr_rollouts} ({describe(success_mode, success_threshold)})")
    return poses
