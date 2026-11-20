"""Compare every evaluated experiment: policy rollouts and MTP with the learned reward as cost.

Reads the results evaluation_pipeline.py writes into <evaluations>/<experiment>/seed<N>/:
    policy_results.csv   the policy rolled out from the random T poses
    mtp_results.csv      MTP with the reward as cost, from the same poses

For every experiment and method (policy / mtp) it reports the success rate over all seeds and poses, the mean
final errors (total = position + orientation, as in the old evaluation scripts), the mean final coverage of the
goal T (Diffusion Policy's measure, recorded only, it does not decide success) and the best seed (highest success
rate, lowest mean total error as tie-breaker).

Prints the overview and writes <evaluations>/overall_evaluation.xlsx with three sheets:
    Overview       per experiment and method
    Per seed       per experiment, method and seed
    All rollouts   every row of every results file, tagged with experiment, method and seed

Usage (from anywhere):
    python overall_evaluation.py                          # every experiment in experiments/evaluations
    python overall_evaluation.py ../evaluations_02        # another evaluations folder
    python overall_evaluation.py ../evaluations/exp1 ../evaluations/exp2   # selected experiments only
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

DEFAULT_EVALUATIONS = Path(__file__).resolve().parents[1] / "evaluations"  # experiments/evaluations
RESULT_FILES = {"policy": "policy_results.csv", "mtp": "mtp_results.csv"}
OUT_NAME = "overall_evaluation.xlsx"

HEADER_FONT = Font(bold=True, color="FFFFFF")
HEADER_FILL = PatternFill("solid", fgColor="305496")


# ------------------------------------------------------------------ reading #
def read_results(csv_path):
    with open(csv_path, newline="") as f:
        rows = []
        for r in csv.DictReader(f):
            pos_err, orient_err = float(r["pos_err"]), float(r["orient_err"])
            cov = r.get("coverage")  # older results files have no coverage column
            rows.append({"pose": int(r["pose"]), "pos_err": pos_err, "orient_err": orient_err,
                         "total_err": pos_err + orient_err, "coverage": float(cov) if cov else None,
                         "success": int(r["success"]),
                         "status": r["status"], "steps": int(r["steps"])})
        return rows


def seed_number(name):
    match = re.fullmatch(r"seed(\d+)", name)
    return int(match.group(1)) if match else name


def experiment_dirs(paths):
    """Experiment folders: a path holding seed<N>/ folders is an experiment, otherwise its subfolders are."""
    found = []
    for path in paths:
        if not path.is_dir():
            print(f"Skipping {path}: not a directory", file=sys.stderr)
            continue
        if any(d.is_dir() and re.fullmatch(r"seed\d+", d.name) for d in path.iterdir()):
            found.append(path)
        else:
            found.extend(sorted(d for d in path.iterdir() if d.is_dir()))
    return found


def collect(exp_dirs):
    """{(experiment, method): {seed: rows}} for every results file found."""
    data = {}
    for exp_dir in exp_dirs:
        seed_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and re.fullmatch(r"seed\d+", d.name)]
        for seed_dir in sorted(seed_dirs, key=lambda d: seed_number(d.name)):
            for method, file_name in RESULT_FILES.items():
                csv_path = seed_dir / file_name
                if csv_path.is_file():
                    data.setdefault((exp_dir.name, method), {})[seed_number(seed_dir.name)] = read_results(csv_path)
    return data


# -------------------------------------------------------------- statistics #
def mean_coverage(rows):
    """Mean final coverage (fraction 0..1) over the rows that recorded it, None if none did."""
    covs = [r["coverage"] for r in rows if r["coverage"] is not None]
    return float(np.mean(covs)) if covs else None


def seed_stats(rows):
    return {
        "success_rate": 100.0 * np.mean([r["success"] for r in rows]),
        "mean_pos_err": float(np.mean([r["pos_err"] for r in rows])),
        "mean_orient_err": float(np.mean([r["orient_err"] for r in rows])),
        "mean_total_err": float(np.mean([r["total_err"] for r in rows])),
        "mean_coverage": mean_coverage(rows),
        "successes": sum(r["success"] for r in rows),
        "rollouts": len(rows),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
    }


def evaluation_settings(exp_dir):
    """(algorithm, success rule) of the evaluation, from evaluation_info.json written by the pipeline.
    Older evaluations have no such file: only TRIRL evaluations plot a reward (reward_plot.png), and they used
    the AND rule (threshold not recorded)."""
    info_path = exp_dir / "evaluation_info.json"
    info = {}
    if info_path.is_file():
        with open(info_path) as f:
            info = json.load(f)
    algorithm = info.get("check_point_type") or ("TRIRL" if any(exp_dir.glob("seed*/reward_plot.png")) else "RL-PPO")
    threshold = info.get("success_threshold")
    rule = f"{info.get('success_mode', 'AND')} {threshold if threshold is not None else '?'}"
    return algorithm, rule


def overview_row(experiment, algorithm, rule, policy_seeds, mtp_seeds):
    """One line per experiment: policy and MTP side by side. Errors are mean final total errors (pos + orient);
    the best seed is chosen on the policy rollouts (highest success rate, lowest mean total error as tie-breaker)."""
    policy = seed_stats([r for rows in policy_seeds.values() for r in rows]) if policy_seeds else None
    mtp = seed_stats([r for rows in mtp_seeds.values() for r in rows]) if mtp_seeds else None
    best_seed = None
    if policy_seeds:
        per_seed = {seed: seed_stats(rows) for seed, rows in policy_seeds.items()}
        best_seed = min(per_seed, key=lambda s: (-per_seed[s]["success_rate"], per_seed[s]["mean_total_err"]))
    seeds = policy_seeds or mtp_seeds
    return {
        "experiment": experiment,
        "algorithm": algorithm,
        "success_rule": rule,
        "seeds": len(seeds),
        "rollouts": (policy or mtp)["rollouts"],
        "mean_err": policy["mean_total_err"] if policy else None,
        "mtp_mean_err": mtp["mean_total_err"] if mtp else None,
        "mean_coverage": policy["mean_coverage"] if policy else None,
        "mtp_mean_coverage": mtp["mean_coverage"] if mtp else None,
        "success_rate": policy["success_rate"] if policy else None,
        "successes": policy["successes"] if policy else None,
        "mtp_success_rate": mtp["success_rate"] if mtp else None,
        "mtp_successes": mtp["successes"] if mtp else None,
        "best_seed": f"seed{best_seed}" if best_seed is not None else None,
    }


# ------------------------------------------------------------------ output #
def print_overview(rows):
    def num(value, fmt):
        return "-" if value is None else format(value, fmt)

    def pct(fraction):
        return "-" if fraction is None else f"{100.0 * fraction:.1f}%"

    header = (f"{'Experiment':<40} | {'Algorithm':<9} | {'Success rule':<14} | {'Seeds':>5} | {'Rollouts':>8} | {'Mean err':>8} | {'MTP mean err':>12} | "
              f"{'Cov (%)':>7} | {'MTP cov (%)':>11} | {'Succ (%)':>8} | {'Succ':>4} | {'MTP succ (%)':>12} | {'MTP succ':>8} | Best seed")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['experiment']:<40} | {r['algorithm']:<9} | {r['success_rule']:<14} | {r['seeds']:>5} | {r['rollouts']:>8} | {num(r['mean_err'], '.4f'):>8} | "
              f"{num(r['mtp_mean_err'], '.4f'):>12} | {pct(r['mean_coverage']):>7} | {pct(r['mtp_mean_coverage']):>11} | "
              f"{num(r['success_rate'], '.1f'):>7}% | "
              f"{num(r['successes'], 'd'):>4} | {num(r['mtp_success_rate'], '.1f'):>11}% | "
              f"{num(r['mtp_successes'], 'd'):>8} | {r['best_seed'] or '-'}")
    print("-" * len(header))
    print("Mean err = mean final total error (position + orientation) of the policy rollouts; "
          "Cov = mean final coverage of the goal T; best seed is chosen on the policy rollouts.")
    print("Success rule: AND = pos_err < thr and ort_err < thr, SUM = pos_err + ort_err < thr "
          "(rates under different rules are not comparable).")


def add_sheet(wb, title, rows, percent_cols=(), error_cols=(), fraction_cols=()):
    """percent_cols hold 0..100 values, fraction_cols 0..1 values; both are shown as percentages."""
    ws = wb.create_sheet(title)
    header = list(rows[0])
    ws.append(header)
    for r in rows:
        ws.append([r[h] / 100.0 if h in percent_cols and r[h] is not None else r[h] for h in header])
    for col, name in enumerate(header, start=1):
        cell = ws.cell(row=1, column=col)
        cell.font, cell.fill = HEADER_FONT, HEADER_FILL
        cell.alignment = Alignment(horizontal="center")
        ws.column_dimensions[get_column_letter(col)].width = max(len(name) + 4, 12)
        number_format = ("0.0%" if name in percent_cols or name in fraction_cols
                         else "0.0000" if name in error_cols else None)
        if number_format:
            for row in ws.iter_rows(min_row=2, min_col=col, max_col=col):
                row[0].number_format = number_format
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path, default=[DEFAULT_EVALUATIONS],
                        help="Evaluations folder(s) or experiment folder(s) (default: experiments/evaluations)")
    parser.add_argument("--out-xlsx", type=Path, default=None,
                        help=f"Output workbook (default: <first path>/{OUT_NAME})")
    args = parser.parse_args()

    exp_dirs = experiment_dirs(args.paths)
    data = collect(exp_dirs)
    settings = {d.name: evaluation_settings(d) for d in exp_dirs}
    if not data:
        sys.exit(f"No {' or '.join(RESULT_FILES.values())} found under {', '.join(map(str, args.paths))}")

    experiments = sorted({exp for exp, _ in data})
    overview = [overview_row(exp, *settings[exp], data.get((exp, "policy"), {}), data.get((exp, "mtp"), {}))
                for exp in experiments]
    per_seed = [{"experiment": exp, "method": method, "seed": seed, **seed_stats(rows)}
                for (exp, method), seeds in sorted(data.items()) for seed, rows in seeds.items()]
    all_rollouts = [{"experiment": exp, "method": method, "seed": seed, **r}
                    for (exp, method), seeds in sorted(data.items()) for seed, rows in seeds.items() for r in rows]

    print_overview(overview)

    errors = ("mean_err", "mtp_mean_err", "mean_pos_err", "mean_orient_err", "mean_total_err",
              "pos_err", "orient_err", "total_err")
    wb = Workbook()
    wb.remove(wb.active)
    coverages = ("mean_coverage", "mtp_mean_coverage", "coverage")
    add_sheet(wb, "Overview", overview, ("success_rate", "mtp_success_rate"), errors, coverages)
    add_sheet(wb, "Per seed", per_seed, ("success_rate",), errors, coverages)
    add_sheet(wb, "All rollouts", all_rollouts, (), errors, coverages)
    out_path = args.out_xlsx or args.paths[0] / OUT_NAME
    wb.save(out_path)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
