"""Stand-alone viewer for the Franka push-T environment.

While ``viewer.MujocoViewer`` is the low-level GLFW window used for rendering a
single ``mjx`` data frame, ``FrankaPushTViewer`` is a small convenience wrapper
that builds the environment and drives an on-screen rollout, so the scene can be
inspected without a trained policy.

Run it directly to watch the environment with zero (default), random, or a
custom action policy::

    python -m trust_region_irl.environments.franka_pusht.env_viewer
    python -m trust_region_irl.environments.franka_pusht.env_viewer --actions random --steps 500
"""
import argparse

import jax
import jax.numpy as jnp

from trust_region_irl.environments.franka_pusht.environment import FrankaPushT


class FrankaPushTViewer:
    """Build the Franka push-T env and render a rollout on screen.

    Args:
        horizon: Episode length passed through to :class:`FrankaPushT`.
        seed: PRNG seed for the reset key and (optionally) action sampling.
    """

    def __init__(self, horizon: int = 200, seed: int = 0):
        self.env = FrankaPushT(render=True, horizon=horizon)
        self.key = jax.random.PRNGKey(seed)

    def reset(self):
        """Reset the single rendered environment and return its state."""
        self.key, subkey = jax.random.split(self.key)
        # The public reset/step are vmapped over a leading axis, so use nr_envs=1.
        reset_keys = jax.random.split(subkey, 1)
        return self.env.reset(reset_keys, True)

    def _action(self, observation, actions):
        action_shape = (1,) + self.env.single_action_space.shape
        if actions == "zero":
            return jnp.zeros(action_shape, dtype=jnp.float32)
        if actions == "random":
            self.key, noise_key = jax.random.split(self.key)
            return jax.random.uniform(
                noise_key, shape=action_shape, minval=-1.0, maxval=1.0, dtype=jnp.float32
            )
        if callable(actions):
            return jnp.asarray(actions(observation), dtype=jnp.float32).reshape(action_shape)
        raise ValueError(f"Unknown action mode: {actions!r} (use 'zero', 'random', or a callable)")

    def view(self, nr_steps: int = 1000, actions="zero"):
        """Roll out the environment with rendering.

        Args:
            nr_steps: Number of environment steps to render.
            actions: ``"zero"``, ``"random"``, or a callable mapping the current
                observation to a ``(2,)`` planar EE velocity action in ``[-1, 1]``.
        """
        state = self.reset()
        try:
            for _ in range(nr_steps):
                action = self._action(state.next_observation, actions)
                state = self.env.step(state, action)
                state = self.env.render(state)
        finally:
            self.close()

    def close(self):
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="View the Franka push-T environment.")
    parser.add_argument("--actions", default="zero", choices=["zero", "random"],
                        help="Action policy used to drive the rollout.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of environment steps to render.")
    parser.add_argument("--horizon", type=int, default=200, help="Episode length before auto-reset.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    args = parser.parse_args()

    viewer = FrankaPushTViewer(horizon=args.horizon, seed=args.seed)
    viewer.view(nr_steps=args.steps, actions=args.actions)


if __name__ == "__main__":
    main()
