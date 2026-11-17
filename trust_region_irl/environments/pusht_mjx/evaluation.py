"""PushT evaluation: roll out the deterministic policy and summarise the first episode of each env.

How success is counted:
  * Only the first episode of each parallel env is used. The env resets itself after the episode ends,
    and everything after that first `done` is ignored.
  * An episode is a success if at any step (block position error in m) + (block orientation error in rad) < threshold.
    The env stops the episode at that step. So success means "touched the goal pose once", not "stayed there".
    The threshold is not a 5 cm limit: metres and radians are just added (2 degrees is 0.035 rad,
    which leaves 1.5 cm for the position).
  * Diverged episodes and episodes with NaN/inf count as failures and stay in the denominator.
  * The policy acts with its mean action (no noise), so this measures the mean policy, not the noisy
    policy that produces the agent data during training.
  * The final number is the mean over the six start poses of the per-start success rate.
  * Scores over a whole run (last 5 evals, mean over seeds) are computed afterwards from the logs, not here.

Two start sets:
  * legacy (`env.preset_set == "legacy"`): the env's own reset (random or fixed start), as before.
  * paper  (`env.preset_set == "paper"`): the six Table-1 starts, evaluated explicitly and independently of the
    training seed (fixed key). The perturbed set assigns env i to start i % 6 (4096 envs: 683/683/683/683/682/682);
    the nominal set runs every start once. Metrics are reported per start and as the equal-weight mean over starts.
"""

import jax
import jax.numpy as jnp

# The evaluation start set must not depend on the training seed: every method, seed and evaluation uses this key.
# So all runs are scored on the same 4096 perturbed start states.
PAPER_EVAL_KEY_SEED = 20241104
NR_PAPER_STARTS = 6
# Metrics also reported per start (prefix/p<k>/<name>); the equal-weight mean over starts keeps the plain name.
PER_START_METRICS = ("is_success", "geometric_success_rate", "reached_0.1_rate", "final_err", "pos_err",
                     "orient_err", "divergence_rate", "valid_episode_rate")


def goal_errors(observations):
    """Block position / orientation error w.r.t. the goal, from (..., 24) quaternion or (..., 26) rot6d
    observations. The quaternion branch is the original Phase-1 expression.

    obs[0:3] is the block position relative to the goal (m), obs[3:7] the block quaternion relative to the goal
    (w, x, y, z). Position error is the norm of the position (m), orientation error is the rotation angle (rad)."""
    position_error = jnp.linalg.norm(observations[..., :3], axis=-1)
    if observations.shape[-1] == 24:
        # Rotation angle of a unit quaternion is 2*arccos(w). abs() makes q and -q give the same angle,
        # so the sign problem in the observation does not affect the evaluation (it does affect the reward).
        orientation_error = 2.0 * jnp.arccos(jnp.clip(jnp.abs(observations[..., 3]), 0.0, 1.0))
    else:
        from trust_region_irl.environments.pusht_mjx.environment import orientation_sin2_half
        # theta = 2 asin(sqrt(sin^2(theta/2))), well conditioned near 0 (unlike arccos of the trace)
        orientation_error = 2.0 * jnp.arcsin(jnp.sqrt(orientation_sin2_half(observations)))
    return position_error, orientation_error


def episode_metrics(observations, terminated, truncated, rewards, diverged,
                    success_threshold, reach_threshold=0.1):
    """Per-env metrics of the first episode. All inputs are time-major (T, nr_envs); outputs are (nr_envs,).

    `observations` has to be the observation from before the reset (env_state.actual_next_observation).
    After an auto-reset, next_observation is already the next episode's first state.
    """
    done = terminated | truncated
    # active[t, i] is True while env i is still in its first episode, i.e. no `done` before step t.
    # The step where `done` fires counts as active.
    active = (jnp.cumsum(done, axis=0) - done.astype(jnp.int32)) == 0
    lengths = jnp.sum(active, axis=0)
    returns = jnp.sum(jnp.where(active, rewards, 0.0), axis=0)
    # Last step of the first episode: the first `done`, or the last step of the rollout if there was none.
    last = jnp.where(jnp.any(done, axis=0), jnp.argmax(done, axis=0),
                     observations.shape[0] - 1)
    final_observation = observations[last, jnp.arange(observations.shape[1])]
    position_error, orientation_error = goal_errors(final_observation)
    # Diverged = the env's divergence flag, or a NaN/inf reward or observation, during the first episode.
    finite_steps = jnp.isfinite(rewards) & jnp.all(jnp.isfinite(observations), axis=-1)
    did_diverge = jnp.any(active & (diverged | ~finite_steps), axis=0)
    valid = ~did_diverge & (lengths > 0) & jnp.isfinite(returns)
    # Main success: the env ended the first episode because error < env.success_threshold at some step.
    # This is `is_success`.
    event_success = jnp.any(active & terminated, axis=0)
    # Cross-check: the same condition on the final observation only. It differs from event_success only if the
    # episode ended for another reason, or the env and this file compute the error differently
    # (see termination_geometry_mismatch_rate).
    geometric_success = (position_error + orientation_error) < success_threshold
    # Looser check: error below 0.1 at any step (default). Diagnostic only, not the success rate.
    step_position_error, step_orientation_error = goal_errors(observations)
    reached = jnp.any(active & ((step_position_error + step_orientation_error) < reach_threshold), axis=0)

    as_float = lambda x: x.astype(jnp.float32)
    # Success metrics are multiplied by `valid`: invalid episodes count as failures but stay in the average.
    return {
        "episode_return": returns,
        "episode_length": as_float(lengths),
        "is_success": as_float(event_success & valid),
        "geometric_success_rate": as_float(geometric_success & valid),
        "termination_geometry_mismatch_rate": as_float((event_success != geometric_success) & valid),
        f"reached_{reach_threshold:g}_rate": as_float(reached & valid),
        "divergence_rate": as_float(did_diverge),
        "valid_episode_rate": as_float(valid),
        "truncated_rate": as_float(jnp.any(active & truncated, axis=0)),
        "pos_err": position_error,
        "orient_err": orientation_error,
        "final_err": position_error + orientation_error,
    }


def summarize_episodes(observations, terminated, truncated, rewards, diverged,
                       success_threshold, reach_threshold=0.1, prefix="eval"):
    """All inputs are time-major (T, nr_envs). Only the first episode of every env is evaluated."""
    per_env = episode_metrics(observations, terminated, truncated, rewards, diverged,
                              success_threshold, reach_threshold)
    return {f"{prefix}/{name}": jnp.mean(value) for name, value in per_env.items()}


def summarize_by_start(per_env, start_index, nr_starts, prefix="eval"):
    """Equal-weight mean over starts (plain names) and the PER_START_METRICS of every start (prefix/p<k>/name).

    Every start has the same weight, whatever its number of envs (683 vs 682).
    `eval/is_success` in the logs is this mean.
    """
    onehot = (start_index[None, :] == jnp.arange(nr_starts)[:, None]).astype(jnp.float32)   # (starts, envs)
    count = jnp.sum(onehot, axis=1)
    present = count > 0
    weights = onehot / jnp.maximum(count, 1.0)[:, None]
    metrics = {}
    for name, value in per_env.items():
        per_start = weights @ value                                                          # (starts,)
        metrics[f"{prefix}/{name}"] = jnp.sum(jnp.where(present, per_start, 0.0)) / jnp.sum(present)
        if name in PER_START_METRICS:
            for start in range(nr_starts):
                metrics[f"{prefix}/p{start}/{name}"] = per_start[start]
    metrics[f"{prefix}/nr_episodes"] = jnp.float32(start_index.shape[0])
    return metrics


def _rollout(env, action_fn, env_state, horizon):
    """Run `horizon` steps for all envs and collect what `episode_metrics` needs (time-major)."""
    def step(env_state, _):
        env_state = env.step(env_state, action_fn(env_state.next_observation))
        transition = (
            env_state.actual_next_observation,   # observation before the auto-reset
            env_state.terminated,
            env_state.truncated,
            env_state.reward,
            env_state.info["env_info/diverged"] > 0.5,
        )
        return env_state, transition

    _, transitions = jax.lax.scan(step, env_state, None, horizon)
    return transitions


def run_pusht_eval(env, action_fn, key, nr_envs, horizon, eval_mode, prefix="eval"):
    """Roll out `action_fn(observation) -> processed action` for `horizon` steps and summarise.

    eval_mode=True applies the env's eval-only start perturbation (eval_init_noise_*);
    eval_mode=False starts from the nominal (training) start state.

    With env.preset_set == "paper" the start set is the six Table-1 starts (see module docstring). `key` is then
    ignored (the fixed PAPER_EVAL_KEY_SEED is used), and the nominal set always has exactly one env per start
    (`nr_envs` is ignored for eval_mode=False).
    """
    if getattr(env, "preset_set", "legacy") == "paper":
        return run_pusht_eval_paper(env, action_fn, nr_envs, horizon, perturb=eval_mode, prefix=prefix)

    env_state = env.reset(jax.random.split(key, nr_envs), eval_mode)
    transitions = _rollout(env, action_fn, env_state, horizon)
    return summarize_episodes(*transitions, success_threshold=env.success_threshold, prefix=prefix)


def run_pusht_eval_paper(env, action_fn, nr_envs, horizon, perturb, prefix="eval"):
    """Six-start evaluation. perturb=True: nr_envs episodes, env i at start i % 6, start perturbed by
    eval_init_noise_*; perturb=False: one unperturbed episode per start.

    The perturbed set is logged as `eval/*`. The nominal set (`eval_nominal/*`) has only 6 episodes, so its
    success can only be k/6. In Phase 1 the perturbation is +-1 cm in x and y and +-5 deg in yaw, and the starts
    are the paper's Table 1 poses (P0-P5).
    """
    if not perturb:
        nr_envs = NR_PAPER_STARTS
    start_index = jnp.arange(nr_envs) % NR_PAPER_STARTS
    key = jax.random.fold_in(jax.random.PRNGKey(PAPER_EVAL_KEY_SEED), int(perturb))
    env_state = env.reset_eval(jax.random.split(key, nr_envs), start_index, perturb)
    transitions = _rollout(env, action_fn, env_state, horizon)
    per_env = episode_metrics(*transitions, success_threshold=env.success_threshold)
    return summarize_by_start(per_env, start_index, NR_PAPER_STARTS, prefix)
