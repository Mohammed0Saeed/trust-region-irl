"""Function-space reward correction with the full history of discriminators (PLAN_phase2.md, "FN").

Port of the official trirl_ppo correction (flax_full_jit/reward_correction.py + buffer.py), restricted to what the
reward needs. Per TRIRL iteration t (same order as the official trainer):
  1. after the discriminator update, append D_t (its online encoder and theta) to the buffer;
  2. the reward of the batch is the recursion, from r = 0 (the official initial term),
         r <- (1 - s_i) * r + s_i * beta * D_i(x),   i = 1..t,   s_i = epsilon / (1 + max(eta_i, 0)),
     where eta_i is the eta of iteration i's PPO update; the newest entry has no eta yet (0, as the official
     zero-padded eta buffer), so its step is epsilon;
  3. after the PPO update, eta_t is written into the buffer.
Differences from the official code: the buffer keeps every discriminator of the run (no capacity truncation),
eta is a scalar per entry (the official stores a (nr_steps, nr_envs) array and, with maximum_eta=True, uses its max,
which is the same scalar for our schedule), and only the reward parameters are stored.

With a fixed encoder, D_i(x) = theta_i . psi(x) and the recursion equals mixing theta with the same steps
(the theta-only correction started from theta = 0); test_fn_correction.py checks this.
"""
import jax
import jax.numpy as jnp

REWARD_KEYS = ("online_encoder", "theta")


def reward_params(discriminator_params):
    """The part of the discriminator parameters that D(x) = online_encoder(x) . theta uses."""
    p = discriminator_params["params"]
    return {k: p[k] for k in REWARD_KEYS}


def create(template_reward_params, capacity):
    return {
        "params": jax.tree_util.tree_map(lambda x: jnp.zeros((capacity,) + x.shape, x.dtype), template_reward_params),
        "etas": jnp.zeros((capacity,), jnp.float32),
        "level": jnp.zeros((), jnp.int32),
    }


def add(buffer, new_reward_params):
    """Append one discriminator (its eta stays 0 until set_last_eta). Raises at trace time only if shapes differ;
    the caller sizes the buffer for every iteration of the run."""
    i = buffer["level"]
    params = jax.tree_util.tree_map(lambda b, x: b.at[i].set(x), buffer["params"], new_reward_params)
    return {**buffer, "params": params, "level": i + 1}


def set_last_eta(buffer, eta):
    return {**buffer, "etas": buffer["etas"].at[buffer["level"] - 1].set(eta)}


def mixture_reward(apply_reward, buffer, x, epsilon, beta, chunk):
    """apply_reward(reward_params, x) -> (N,). Returns the corrected reward (N,) over the buffer's valid entries.
    `chunk` discriminators are evaluated at a time; the buffer capacity must be a multiple of chunk."""
    level = buffer["level"]
    n_chunks = (level + chunk - 1) // chunk

    def chunk_body(c, r):
        p = jax.tree_util.tree_map(lambda a: jax.lax.dynamic_slice_in_dim(a, c * chunk, chunk), buffer["params"])
        etas = jax.lax.dynamic_slice_in_dim(buffer["etas"], c * chunk, chunk)
        out = jax.vmap(lambda pp: apply_reward(pp, x))(p)  # (chunk, N)

        def step(i, r):
            valid = c * chunk + i < level
            s = jnp.where(valid, epsilon / (1.0 + jnp.maximum(etas[i], 0.0)), 0.0)
            return (1.0 - s) * r + s * beta * out[i]

        return jax.lax.fori_loop(0, chunk, step, r)

    return jax.lax.fori_loop(0, n_chunks, chunk_body, jnp.zeros(x.shape[0], jnp.float32))
