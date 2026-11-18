"""Plot the reward learned by a TRIRL checkpoint (latest.model.zip) as reward_plot.png.

ENCODER_TYPE = FEATURES (trirl_ppo_fb): reward = theta . phi(s), phi from PushT.feature_from_transition.
    The feature dimension is read from the checkpoint:
        2 -> feature_fn "pos_ort"            [-pos_err, -orient_err]            -> position + orientation panels
        3 -> feature_fn "base_without_ctrl"  [-pos_err, -orient_err, -ee_block] -> + end-effector panel
ENCODER_TYPE = SPR (trirl_ppo_spr): reward = theta . z(s), z = online SPR encoder stored in the discriminator.
    OBS_TYPE = TASK-BASED       -> the encoder sees the block pose only (observation dims 0..6)
    OBS_TYPE = JOINT-TASK-BASED -> the encoder sees the full 24-D observation (+ end-effector panel)

Every sweep is built from real 24-D observations: the block is placed in the MJX simulator, the robot is held in
its start configuration, and PushT.get_observation is read after mjx.forward (same as during training).
Each panel is normalized to [0, 1], because the absolute scale of theta is arbitrary.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import jax.numpy as jnp
from mujoco import mjx
import orbax.checkpoint as ocp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trust_region_irl.environments.pusht_mjx.environment import PushT
from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.byol import BYOL as SPR
from trust_region_irl.algorithms.trirl_ppo_fb.flax_full_jit.discriminator import BoltzmannDiscriminatorFeatureBased

FEATURE_FN = {2: "pos_ort", 3: "base_without_ctrl"}
SPR_INPUT_INDICES = {"TASK-BASED": (0, 1, 2, 3, 4, 5, 6),  # block position + quaternion
                     "JOINT-TASK-BASED": ()}                # empty = full 24-D observation
PUSHT_OBSERVATION_DIM = 24
BLOCK_Z = 0.045      # block height the env spawns it at (PushT._reset)
GRID = 100           # resolution of every sweep
POS_RANGE = 0.4      # position heatmap spans [-POS_RANGE, POS_RANGE]^2 around the goal (m)
EE_RANGE = 0.4       # end-effector distance sweep spans [0, EE_RANGE] (m)


# ------------------------------------------------------------------ loading #
def load_checkpoint(model_zip):
    """Restore the checkpoint as numpy arrays (shapes read from disk) plus its algorithm config."""
    tmp = tempfile.mkdtemp()
    try:
        shutil.unpack_archive(model_zip, tmp, "zip")
        checkpointer = ocp.PyTreeCheckpointer()
        metadata = checkpointer.metadata(tmp)
        restore_args = jax.tree_util.tree_map(
            lambda m: ocp.RestoreArgs(restore_type=np.ndarray),
            metadata.item_metadata.tree,
            is_leaf=lambda x: hasattr(x, "shape"),
        )
        ckpt = checkpointer.restore(tmp, restore_args=restore_args)
        with open(f"{tmp}/config_algorithm.json", "r") as f:
            config = json.load(f)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ckpt, config


def make_feature_encoder(ckpt, config):
    """FEATURES: return (encode, theta, features_dim). encode maps hand-crafted features to the reward latent."""
    disc_params = ckpt["discriminator"]["params"]
    theta = np.asarray(disc_params["params"]["theta"], dtype=np.float32)

    if config.get("reward_type") == "boltzmann-feature-based":
        features_dim = disc_params["params"]["feat_denses_0"]["kernel"].shape[0]
        defaults = BoltzmannDiscriminatorFeatureBased()
        disc = BoltzmannDiscriminatorFeatureBased(
            hidden_dims=config.get("boltzmann_hidden_dims", defaults.hidden_dims),
            latent_dim=config.get("boltzmann_latent_dim", defaults.latent_dim),
            energy_hidden_dim=config.get("boltzmann_energy_hidden_dim", defaults.energy_hidden_dim),
        )
        encode = jax.jit(lambda f: disc.apply(disc_params, f, method=BoltzmannDiscriminatorFeatureBased.encode_feature))
    else:
        # linear feature-based reward: r = theta . phi, so the feature dimension is the size of theta
        features_dim = theta.shape[0]
        encode = lambda f: f

    if features_dim not in FEATURE_FN:
        raise ValueError(f"Feature dimension {features_dim} is not supported, expected one of {list(FEATURE_FN)}")
    return encode, theta, features_dim


def make_spr_encoder(ckpt, obs_type):
    """SPR: return (encode, theta). encode maps the 24-D observation to the 64-D SPR latent."""
    disc_params = ckpt["discriminator"]["params"]
    if "online_enc1" not in disc_params["params"]:
        raise ValueError("ENCODER_TYPE is SPR, but the checkpoint's discriminator has no online SPR encoder")
    theta = np.asarray(disc_params["params"]["theta"], dtype=np.float32)

    spr = SPR(input_indices=SPR_INPUT_INDICES[obs_type])
    input_dim = disc_params["params"]["online_enc1"]["kernel"].shape[0]
    expected_dim = len(spr.input_indices) or PUSHT_OBSERVATION_DIM
    if input_dim != expected_dim:
        raise ValueError(f"The checkpoint's SPR encoder takes {input_dim}-D input, "
                         f"but OBS_TYPE {obs_type} feeds {expected_dim}-D input")

    # select the input dimensions first (as _online_network does), then encode
    encode = jax.jit(lambda obs: spr.apply(disc_params, obs,
                                           method=lambda module, o: module._online_encoder(module.input(o))))
    return encode, theta


# ------------------------------------------------------------------ sweeps #
def env_observations(env, block_xy, block_yaw):
    """Real 24-D observations with the block at goal + block_xy (nr, 2) and yaw block_yaw (nr,), robot at its start pose."""
    goal_xy = env.goal_pos[0:2]

    def observe(xy, yaw):
        block_quat = jnp.array([jnp.cos(yaw / 2.0), 0.0, 0.0, jnp.sin(yaw / 2.0)])
        block_qpos = jnp.concatenate([goal_xy + xy, jnp.array([BLOCK_Z]), block_quat])
        data = env.mjx_data.replace(qpos=env.initial_qpos.at[0:7].set(block_qpos), qvel=env.initial_qvel,
                                    ctrl=jnp.zeros(env.mjx_model.nu),
                                    mocap_pos=env.mocap_pos, mocap_quat=env.mocap_quat)
        return env.get_observation(mjx.forward(env.mjx_model, data))

    return np.array(jax.jit(jax.vmap(observe))(jnp.asarray(block_xy, jnp.float32),
                                                 jnp.asarray(block_yaw, jnp.float32)))


def build_sweeps(env, to_input, with_ee):
    """{name: (axis, reward input)} for the orientation, position and (optionally) end-effector sweeps."""
    sweeps = {}

    # orientation: block at the goal, rotated about z
    qs = np.linspace(-np.pi, np.pi, GRID)
    sweeps["orientation"] = (qs, to_input(env_observations(env, np.zeros((GRID, 2)), qs)))

    # position: block aligned with the goal and shifted in x / y
    xs = np.linspace(-POS_RANGE, POS_RANGE, GRID)
    XX, YY = np.meshgrid(xs, xs)
    obs = env_observations(env, np.stack([XX.ravel(), YY.ravel()], axis=1), np.zeros(GRID * GRID))
    sweeps["position"] = (xs, to_input(obs))

    # end effector: block at the goal; the ee position in the observation (dims 7..9) is moved away along x
    if with_ee:
        ds = np.linspace(0.0, EE_RANGE, GRID)
        obs = env_observations(env, np.zeros((GRID, 2)), np.zeros(GRID))
        obs[:, 7:10] = obs[:, 0:3]
        obs[:, 7] += ds
        sweeps["ee"] = (ds, to_input(obs))
    return sweeps


def normalize_01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


# -------------------------------------------------------------------- plot #
def draw_reward_plot(rewards, axes, out_path, title):
    has_ee = "ee" in rewards
    nr_cols = 3 if has_ee else 2
    fig, axs = plt.subplots(1, nr_cols, figsize=(6.5 * nr_cols, 5.5))

    ax = axs[0]
    ax.plot(axes["orientation"], rewards["orientation"], color="#e11d48")
    ax.set_title("Orientation Reward")
    ax.set_xlabel("block yaw relative to goal")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

    ax = axs[1]
    xs = axes["position"]
    image = ax.imshow(rewards["position"], origin="lower", cmap="Blues", vmin=0.0, vmax=1.0,
                      extent=[xs[0], xs[-1], xs[0], xs[-1]])
    ax.plot(0, 0, marker="*", color="red", markersize=12, linestyle="None", zorder=5, label="goal")
    ax.set_title("Position Reward")
    ax.set_xlabel("block x relative to goal (m)")
    ax.set_ylabel("block y relative to goal (m)")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, ticks=[0, 1], fraction=0.046, pad=0.04)

    if has_ee:
        ax = axs[2]
        ax.plot(axes["ee"], rewards["ee"], color="#e11d48")
        ax.set_title("End-Effector Reward")
        ax.set_xlabel("ee-block distance (m)")
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reward(check_point_type, encoder_type, obs_type, checkpoint_path, output_dir):
    """Plot the reward of a TRIRL checkpoint into output_dir/reward_plot.png and return that path."""
    if check_point_type != "TRIRL":
        raise ValueError(f"Only TRIRL checkpoints have a learned reward, got {check_point_type}")

    ckpt, config = load_checkpoint(checkpoint_path)

    if encoder_type == "FEATURES":
        encode, theta, features_dim = make_feature_encoder(ckpt, config)
        env = PushT(render=False, feature_fn=FEATURE_FN[features_dim])
        # zero action: pos_ort / base_without_ctrl don't read it
        to_input = lambda obs: np.asarray(env.feature_from_transition(obs, np.zeros((obs.shape[0], 2), np.float32)))
        with_ee = features_dim == 3
        title = f"TRIRL reward, {features_dim}-D features ({FEATURE_FN[features_dim]})"
    elif encoder_type == "SPR":
        encode, theta = make_spr_encoder(ckpt, obs_type)
        env = PushT(render=False)
        to_input = lambda obs: obs  # the encoder selects its input dimensions itself
        with_ee = obs_type == "JOINT-TASK-BASED"  # the task-based encoder never sees the end effector
        title = f"TRIRL reward, SPR encoder ({obs_type})"
    else:
        raise ValueError(f"Unknown ENCODER_TYPE {encoder_type}")

    sweeps = build_sweeps(env, to_input, with_ee)
    axes = {name: axis for name, (axis, _) in sweeps.items()}
    rewards = {name: normalize_01(np.asarray(encode(jnp.asarray(x))) @ theta) for name, (_, x) in sweeps.items()}
    rewards["position"] = rewards["position"].reshape(GRID, GRID)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "reward_plot.png"
    draw_reward_plot(rewards, axes, out_path, title)
    print(f"Saved {out_path}")
    return out_path
