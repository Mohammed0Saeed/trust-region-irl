import jax
import jax.numpy as jnp


def gae_from_deltas(delta, episode_ends, gamma, gae_lambda):
    """GAE recursion A_t = delta_t + gamma * lambda * (1 - end_t) * A_{t+1} over a (T, nr_envs) rollout.

    `episode_ends[t]` must be 1 whenever the env was reset after step t. The TD errors `delta` bootstrap from the
    *actual* next state, so they only treat true terminations as terminal; the recursion, however, has to be cut at
    every reset, including time-limit truncations, otherwise the advantages of the next episode leak into this one."""
    def step(next_advantage, inputs):
        delta_t, end_t = inputs
        advantage = delta_t + gamma * gae_lambda * (1.0 - end_t) * next_advantage
        return advantage, advantage

    _, advantages = jax.lax.scan(step, jnp.zeros_like(delta[-1]), (delta, episode_ends.astype(delta.dtype)), reverse=True)
    return advantages
