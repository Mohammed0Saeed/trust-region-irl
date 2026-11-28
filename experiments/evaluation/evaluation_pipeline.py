"""
1. Check the type of the training using script config CHECK_POINT_TYPE = RL-PPO | TRIRL
2. Check the type of the encoder using script config ENCODER_TYPE = SPR | FEATURES
3. Check the type of observation in the spr encoder using script config OBS_TYPE = TASK-BASED | JOINT-TASK-BASED

The checkpoints are read from runs/<PROJECT_NAME>/<EXPERIMENT_NAME>/<N>/models/latest.model.zip, one run per seed
(PPO retraining: <N>_ppo_retraining/), see utils.py.

Outputs go to OUTPUT_DIR/<PROJECT_NAME>/<EXPERIMENT_NAME>_<SUCCESS_MODE>/seed_<N>/, one folder per training seed of
the experiment, so evaluations of the same experiment with different success modes are kept apart.
"""
import argparse
import json
import sys
from pathlib import Path

# repository root, so the trust_region_irl package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plot_rewards import plot_reward
from utils import find_checkpoints_paths, seed_name
from env_policy import rollout_policy
from mtp_runner import run_mtp
from success import SUCCESS_MODES, describe

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "evaluations"  # experiments/evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-name", type=str, required=True, help="Project folder inside runs/ (e.g. spr_comparison)")
    parser.add_argument("--experiment-name", type=str, required=True, help="Experiment folder inside the project (e.g. pusht_ppo_spr_K3)")
    parser.add_argument("--check-point-type", type=str, required=True, choices=["RL-PPO", "TRIRL"], help="Type of the algorithm the checkpoint is trained on")
    parser.add_argument("--encoder-type", type=str, required=True, choices=["SPR", "FEATURES"], help="Type of the encoder trained on TRIRL")
    parser.add_argument("--obs-type", type=str, required=True, choices=["TASK-BASED", "JOINT-TASK-BASED"], help="Type of the observations fed into SPR encoder")
    parser.add_argument("--poses-seeds", type=str, required=True, help="Comma-separated seeds for the random T configurations (e.g. 0,1,2)")
    parser.add_argument("--success-threshold", type=float, required=True, help="Threshold to determine if a policy is successful")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Path to the output directory")
    parser.add_argument("--success-mode", type=str, default="AND", choices=list(SUCCESS_MODES),
                        help="AND: pos_err < thr and ort_err < thr; SUM: pos_err + ort_err < thr "
                             "(the IoU with the goal T is always recorded, but never decides success)")
    parser.add_argument("--parallel-mtp", type=str, default="false", choices=["true", "false"],
                        help="Roll the MTP poses of a seed out in parallel (one batched planner call) instead of one by one")
    args = parser.parse_args()
    parallel_mtp = args.parallel_mtp == "true"

    evaluation_seeds = [int(i) for i in args.poses_seeds.split(",")]

    # Settings of this evaluation, read by overall_evaluation.py (e.g. the algorithm type column)
    experiment_dir = args.output_dir / args.project_name / f"{args.experiment_name}_{args.success_mode}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with open(experiment_dir / "evaluation_info.json", "w") as f:
        json.dump({"project_name": args.project_name, "experiment_name": args.experiment_name,
                   "check_point_type": args.check_point_type, "encoder_type": args.encoder_type,
                   "obs_type": args.obs_type, "poses_seeds": evaluation_seeds, "success_threshold": args.success_threshold,
                   "success_mode": args.success_mode,
                   "parallel_mtp": parallel_mtp}, f, indent=2)

    print(f"SUCCESS: {describe(args.success_mode, args.success_threshold)}")

    paths = find_checkpoints_paths(project_name=args.project_name, experiment_name=args.experiment_name,
                                   check_point_type=args.check_point_type)
    for path in paths:
        print("FOUND CHECKPOINT PATH: ", path)

    for path in paths:
        # Build output dir, named after the training seed of the checkpoint (seed_<N>)
        seed = seed_name(path)
        output_dir = experiment_dir / seed
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot the rewards based on ENCODER_TYPE and OBS_TYPE
        if args.check_point_type == "TRIRL":
            print(f"PLOTTING REWARD {args.experiment_name}_{seed}")
            plot_reward(args.check_point_type, args.encoder_type, args.obs_type, checkpoint_path=path, output_dir=output_dir)

        # Roll out the policy from one pose per poses seed and save the videos
        print(f"ROLLING OUT POLICY {args.experiment_name}_{seed}")
        t_config = rollout_policy(path, seeds=evaluation_seeds, success_threshold=args.success_threshold, output_dir=output_dir,
                                  success_mode=args.success_mode)

        # Using the same T configs, run MTP and set the cost function to be the reward function if CHECK_POINT_TYPE = TRIRL.
        # Every seed's reward is rolled out from all poses of t_config (the same poses as the policy, one per poses seed).
        if args.check_point_type == "TRIRL":
            print(f"RUNNING MTP {args.experiment_name}_{seed} FROM {len(t_config)} POSES"
                  f"{' IN PARALLEL' if parallel_mtp else ''}")
            run_mtp(t_config, checkpoint_path=path, encoder_type=args.encoder_type, obs_type=args.obs_type,
                    seeds=evaluation_seeds, success_threshold=args.success_threshold, output_dir=output_dir,
                    parallel=parallel_mtp, success_mode=args.success_mode)


if __name__ == "__main__":
    main()
