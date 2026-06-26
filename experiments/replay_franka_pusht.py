"""Watch the last trained policy in the Franka push-T sim.

Usage (from the repo root, in the `trirl` conda env):

    python experiments/replay_franka_pusht.py
    python experiments/replay_franka_pusht.py --model_path path/to/latest.model.zip
    python experiments/replay_franka_pusht.py --search_dir experiments/runs --steps 500 --sample
"""
import argparse

from trust_region_irl.environments.franka_pusht.environment import FrankaPushT


def main():
    parser = argparse.ArgumentParser(description="Replay a saved Franka push-T policy with rendering.")
    parser.add_argument("--model_path", default=None,
                        help="Path to a *.model.zip checkpoint. Defaults to the newest one found under --search_dir.")
    parser.add_argument("--search_dir", default="experiments/runs",
                        help="Where to look for the latest checkpoint when --model_path is not given.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of environment steps to roll out.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    parser.add_argument("--sample", action="store_true",
                        help="Sample actions from the policy instead of using the deterministic mean.")
    args = parser.parse_args()

    env = FrankaPushT(render=True)
    try:
        env.replay_policy(
            model_path=args.model_path,
            search_dir=args.search_dir,
            nr_steps=args.steps,
            deterministic=not args.sample,
            seed=args.seed,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
