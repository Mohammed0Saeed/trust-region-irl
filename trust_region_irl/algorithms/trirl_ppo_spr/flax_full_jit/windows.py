"""K-step windows (actions a_t..a_{t+K-1}, next states s_{t+1}..s_{t+K}) with episode-boundary masks.

mask[:, k] = 1 iff the (k+1)-th target s_{t+k+1} exists in the same episode, i.e. the window has not left the
available data and none of the transitions t .. t+k-1 ended an episode. The target of the transition that itself
ends the episode (the terminal state) is valid. (The official SPR code masks with 1 - sign(cumsum(done)),
which additionally drops that last step.)
"""
import jax.numpy as jnp


def agent_windows(idx, flat_actions, flat_next_states, flat_dones, nr_envs, k):
    """Windows over a rollout stored time-major and flattened: flat index = t * nr_envs + env.

    idx: (M,) flat indices.  flat_*: (T * nr_envs, ...).  flat_dones = terminated | truncated, float32.
    Returns a_seq (M, k, A), x_next (M, k, obs), mask (M, k)."""
    total = flat_actions.shape[0]
    alive = jnp.ones(idx.shape, jnp.float32)
    actions, next_states, masks = [], [], []
    for step in range(k):
        ids = idx + step * nr_envs
        in_range = ids < total
        safe_ids = jnp.where(in_range, ids, idx)
        actions.append(flat_actions[safe_ids])
        next_states.append(flat_next_states[safe_ids])
        masks.append(alive * in_range)
        alive = alive * in_range * (1.0 - flat_dones[safe_ids])
    return jnp.stack(actions, 1), jnp.stack(next_states, 1), jnp.stack(masks, 1)


def expert_episode_end(absorbing):
    """Index of the last transition of the episode each transition belongs to. The expert data is stored
    contiguously per episode and every episode ends with absorbing=1."""
    import jax
    n = absorbing.shape[0]
    marked = jnp.where(absorbing > 0.5, jnp.arange(n), n - 1)
    return jax.lax.cummin(marked, reverse=True)


def expert_windows(idx, actions, next_states, episode_end, k):
    """Windows over contiguous expert episodes. idx: (M,) dataset indices."""
    last = episode_end[idx]
    action_list, state_list, masks = [], [], []
    for step in range(k):
        ids = idx + step
        safe_ids = jnp.minimum(ids, actions.shape[0] - 1)
        action_list.append(actions[safe_ids])
        state_list.append(next_states[safe_ids])
        masks.append((ids <= last).astype(jnp.float32))
    return jnp.stack(action_list, 1), jnp.stack(state_list, 1), jnp.stack(masks, 1)
