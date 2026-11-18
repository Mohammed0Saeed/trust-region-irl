"""
1. Check the type of the training using script config CHECK_POINT_TYPE = RL-PPO | TRIRL
2. Check the type of the encoder using script config ENCODER_TYPE = SPR | FEATURES
3. Check the type of observation in the spr encoder using script config OBS_TYPE = TASK-BASED | JOINT-TASK-BASED

Outputs go to OUTPUT_DIR/<EXPERIMENT_NAME>/seed<N>/, one folder per training seed of the experiment.
"""
import argparse
import sys
from pathlib import Path

# repository root, so the trust_region_irl package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plot_rewards import plot_reward
from utils import find_checkpoints_paths, seed_name
from env_policy import rollout_policy
from mtp_runner import run_mtp

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "evaluations"  # experiments/evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("EXPERIMENT_NAME", type=str, help="Name of the experiment in wandb")
    parser.add_argument("CHECK_POINT_TYPE", type=str, choices=["RL-PPO", "TRIRL"], help="Type of the algorithm the checkpoint is trained on")
    parser.add_argument("ENCODER_TYPE", type=str, choices=["SPR", "FEATURES"], help="Type of the encoder trained on TRIRL")
    parser.add_argument("OBS_TYPE", type=str, choices=["TASK-BASED", "JOINT-TASK-BASED"], help="Type of the observations fed into SPR encoder")
    parser.add_argument("SEED", type=int, help="Seed for the random T configuration generator")
    parser.add_argument("SUCCESS_THRESHOLD", type=float, help="Threshold to determine if a policy is successful")
    parser.add_argument("OUTPUT_DIR", type=Path, nargs="?", default=DEFAULT_OUTPUT_DIR, help="Path to the output directory")
    args = parser.parse_args()

    paths = find_checkpoints_paths(experiment_name=args.EXPERIMENT_NAME, check_point_type=args.CHECK_POINT_TYPE)
    for path in paths:
        print("FOUND CHECKPOINT PATH: ", path)

    for index, path in enumerate(paths):
        # Build output dir, named after the training seed of the checkpoint
        seed = seed_name(path, fallback=index)
        output_dir = args.OUTPUT_DIR / args.EXPERIMENT_NAME / seed
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot the rewards based on ENCODER_TYPE and OBS_TYPE
        if args.CHECK_POINT_TYPE == "TRIRL":
            print(f"PLOTTING REWARD {args.EXPERIMENT_NAME}_{seed}")
            plot_reward(args.CHECK_POINT_TYPE, args.ENCODER_TYPE, args.OBS_TYPE, checkpoint_path=path, output_dir=output_dir)

        # Roll out the policy and save the videos, use the seed in the configuration for the keys
        print(f"ROLLING OUT POLICY {args.EXPERIMENT_NAME}_{seed}")
        t_config = rollout_policy(path, seed=args.SEED, success_threshold=args.SUCCESS_THRESHOLD, output_dir=output_dir)

        # Using the same T configs, run MTP and set the cost function to be the reward function if CHECK_POINT_TYPE = TRIRL
        if args.CHECK_POINT_TYPE == "TRIRL":
            print(f"RUNNING MTP {args.EXPERIMENT_NAME}_{seed}")
            run_mtp(t_config, checkpoint_path=path, encoder_type=args.ENCODER_TYPE, obs_type=args.OBS_TYPE,
                    seed=args.SEED, success_threshold=args.SUCCESS_THRESHOLD, output_dir=output_dir)


if __name__ == "__main__":
    main()
