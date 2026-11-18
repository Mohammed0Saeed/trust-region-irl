import re
from pathlib import Path

RETRAINING_SUFFIX = "_ppo_retraining"  # PPO retraining runs live in runs/<experiment>/<prefix>_seed<N>/<id>_ppo_retraining/


def find_checkpoints_paths(experiment_name: str, check_point_type: str) -> list[Path]:
    """Return every latest.model.zip under runs/<experiment_name> of the given type, sorted by path.

    TRIRL:  the TRIRL checkpoints (runs/<experiment>/<prefix>_seed<N>/<id>/models/), retraining runs excluded
    RL-PPO: the PPO retraining checkpoints (runs/<experiment>/<prefix>_seed<N>/<id>_ppo_retraining/models/)
    """
    runs_dir = Path(__file__).resolve().parents[1] / "runs"
    experiment_dir = runs_dir / experiment_name
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    def is_retraining(path):
        return any(part.endswith(RETRAINING_SUFFIX) for part in path.relative_to(experiment_dir).parts)

    want_retraining = check_point_type == "RL-PPO"
    checkpoints = sorted(p for p in experiment_dir.rglob("latest.model.zip") if is_retraining(p) == want_retraining)
    if not checkpoints:
        raise FileNotFoundError(f"No {check_point_type} latest.model.zip found under {experiment_dir}")
    return checkpoints


def seed_name(checkpoint_path: Path, fallback: int) -> str:
    """seed<N> from the <prefix>_seed<N> folder of a checkpoint path, or seed<fallback> if there is none."""
    for part in reversed(checkpoint_path.parts):
        match = re.fullmatch(r".*_seed(\d+)", part)
        if match:
            return f"seed{match.group(1)}"
    return f"seed{fallback}"
