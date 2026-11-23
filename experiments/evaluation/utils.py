import re
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"  # experiments/runs
RETRAINING_SUFFIX = "_ppo_retraining"

# Run layout (runs/<project_name>/<exp_name>/<run_name>/ as written by the trainers), with one run per seed:
#   runs/<project>/<experiment>/<N>/models/latest.model.zip                   TRIRL run of seed N (or seed_<N>/)
#   runs/<project>/<experiment>/<N>_ppo_retraining/models/latest.model.zip    PPO retrained on that run's reward
# At most 5 digits, so a run id timestamp folder (older layout runs/<experiment>/<prefix>_seed<N>/<timestamp>/) is
# never taken for a seed.
SEED_FOLDER = re.compile(r"(?:seed_)?(\d{1,5})")
RETRAINING_FOLDER = re.compile(r"(?:seed_)?(\d{1,5})" + RETRAINING_SUFFIX)


def find_checkpoints_paths(project_name: str, experiment_name: str, check_point_type: str) -> list[Path]:
    """Return the latest.model.zip of every seed of runs/<project_name>/<experiment_name>, sorted by seed number.

    TRIRL:  the seed folders <N>/ (or seed_<N>/)
    RL-PPO: the retraining folders <N>_ppo_retraining/ (or seed_<N>_ppo_retraining/)
    """
    experiment_dir = RUNS_DIR / project_name / experiment_name
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    pattern = RETRAINING_FOLDER if check_point_type == "RL-PPO" else SEED_FOLDER
    checkpoints = []
    for run_dir in experiment_dir.iterdir():
        match = pattern.fullmatch(run_dir.name)
        checkpoint = run_dir / "models" / "latest.model.zip"
        if run_dir.is_dir() and match and checkpoint.is_file():
            checkpoints.append((int(match.group(1)), checkpoint))
    if not checkpoints:
        raise FileNotFoundError(f"No {check_point_type} <seed>/models/latest.model.zip found under {experiment_dir}")
    return [checkpoint for _, checkpoint in sorted(checkpoints)]


def seed_name(checkpoint_path: Path) -> str:
    """seed_<N> from the run folder of a checkpoint (.../<N>[_ppo_retraining]/models/latest.model.zip)."""
    run_folder = checkpoint_path.parent.parent.name
    match = SEED_FOLDER.fullmatch(run_folder) or RETRAINING_FOLDER.fullmatch(run_folder)
    if not match:
        raise ValueError(f"Run folder {run_folder} of {checkpoint_path} is not a seed folder (<N> or seed_<N>)")
    return f"seed_{match.group(1)}"
