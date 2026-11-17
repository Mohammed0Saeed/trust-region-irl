"""PushT evaluation: roll out the deterministic policy and summarise the first episode of each env.

Two start sets:
  * legacy (`env.preset_set == "legacy"`): the env's own reset (random or fixed start), as before.
  * paper  (`env.preset_set == "paper"`): the six Table-1 starts, evaluated explicitly and independently of the
    training seed (fixed key). The perturbed set assigns env i to start i % 6 (4096 envs: 683/683/683/683/682/682);
    the nominal set runs every start once. Metrics are reported per start and as the equal-weight mean over starts.
"""

import jax
import jax.numpy as jnp

# The evaluation start set must not depend on the training seed: every method, seed and evaluation uses this key.
PAPER_EVAL_KEY_SEED = 20241104
NR_PAPER_STARTS = 6
# Metrics also reported per start (prefix/p<k>/<name>); the equal-weight mean over starts keeps the plain name.
PER_START_METRICS = ("is_success", "geometric_success_rate", "reached_0.1_rate", "final_err", "pos_err",
                     "orient_err", "divergence_rate", "valid_episode_rate")


def goal_errors(observations):
    """Block position / orientation error w.r.t. the goal, from (..., 24) observations."""
    position_error = jnp.linalg.norm(observations[..., :3], axis=-1)
    orientation_error = 2.0 * jnp.arccos(jnp.clip(jnp.abs(observations[..., 3]), 0.0, 1.0))
    return position_error, orientation_error


def episode_metrics(observations, terminated, truncated, rewards, diverged,
                    success_threshold, reach_threshold=0.1):
    """Per-env metrics of the first episode. All inputs are time-major (T, nr_envs); outputs are (nr_envs,)."""
    done = terminated | truncated
    active = (jnp.cumsum(done, axis=0) - done.astype(jnp.int32)) == 0
    lengths = jnp.sum(active, axis=0)
    returns = jnp.sum(jnp.where(active, rewards, 0.0), axis=0)
    last = jnp.where(jnp.any(done, axis=0), jnp.argmax(done, axis=0),
                     observations.shape[0] - 1)
    final_observation = observations[last, jnp.arange(observations.shape[1])]
    position_error, orientation_error = goal_errors(final_observation)
    finite_steps = jnp.isfinite(rewards) & jnp.all(jnp.isfinite(observations), axis=-1)
    did_diverge = jnp.any(active & (diverged | ~finite_steps), axis=0)
    valid = ~did_diverge & (lengths > 0) & jnp.isfinite(returns)
    event_success = jnp.any(active & terminated, axis=0)
    geometric_success = (position_error + orientation_error) < success_threshold
    step_position_error, step_orientation_error = goal_errors(observations)
    reached = jnp.any(active & ((step_position_error + step_orientation_error) < reach_threshold), axis=0)

    as_float = lambda x: x.astype(jnp.float32)
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
    """Equal-weight mean over starts (plain names) and the PER_START_METRICS of every start (prefix/p<k>/name)."""
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
    def step(env_state, _):
        env_state = env.step(env_state, action_fn(env_state.next_observation))
        transition = (
            env_state.actual_next_observation,
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
    eval_init_noise_*; perturb=False: one unperturbed episode per start."""
    if not perturb:
        nr_envs = NR_PAPER_STARTS
    start_index = jnp.arange(nr_envs) % NR_PAPER_STARTS
    key = jax.random.fold_in(jax.random.PRNGKey(PAPER_EVAL_KEY_SEED), int(perturb))
    env_state = env.reset_eval(jax.random.split(key, nr_envs), start_index, perturb)
    transitions = _rollout(env, action_fn, env_state, horizon)
    per_env = episode_metrics(*transitions, success_threshold=env.success_threshold)
    return summarize_by_start(per_env, start_index, NR_PAPER_STARTS, prefix)
