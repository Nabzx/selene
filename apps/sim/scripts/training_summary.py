"""CLI tool to print a formatted summary table from a MAPPO CSV training log."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


METRICS = [
    "episode_reward",
    "crashes",
    "craters_visited",
    "crater_overlap",
    "relay_conflicts",
    "entropy",
    "value_loss",
    "wm_error",
]


def format_training_summary(csv_path: str | Path) -> str:
    """Read a MAPPO training CSV file and format a clean terminal summary table."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Training log CSV not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return f"No rows found in {path}"

    k = min(3, len(rows))
    first_rows = rows[:k]
    last_rows = rows[-k:]

    def mean_col(subset: list[dict[str, str]], col: str) -> float:
        vals = [float(r[col]) for r in subset if col in r and r[col] != ""]
        return sum(vals) / len(vals) if vals else 0.0

    total_updates = len(rows)
    total_steps = int(float(rows[-1].get("env_steps", 0)))
    wall_time_s = float(rows[-1].get("wall_time_s", 0.0))

    lines = [
        f"=== Selene MAPPO Training Summary ({path.name}) ===",
        f"Updates: {total_updates} | Env Steps: {total_steps} | Wall Time: {wall_time_s:.2f}s",
        "-" * 68,
        f"{'Metric':<20} | {'First ' + str(k):>12} | {'Last ' + str(k):>12} | {'Delta':>12}",
        "-" * 68,
    ]

    for col in METRICS:
        v_start = mean_col(first_rows, col)
        v_end = mean_col(last_rows, col)
        delta = v_end - v_start
        lines.append(f"{col:<20} | {v_start:>12.4f} | {v_end:>12.4f} | {delta:>+12.4f}")

    lines.append("-" * 68)

    r_start = mean_col(first_rows, "episode_reward")
    r_end = mean_col(last_rows, "episode_reward")
    c_start = mean_col(first_rows, "crashes")
    c_end = mean_col(last_rows, "crashes")
    o_start = mean_col(first_rows, "crater_overlap")
    o_end = mean_col(last_rows, "crater_overlap")
    w_start = mean_col(first_rows, "wm_error")
    w_end = mean_col(last_rows, "wm_error")

    stage2_pass = (r_end - r_start > 5.0) and (c_end <= c_start + 0.05)
    stage3_pass = o_end <= o_start
    stage4_pass = w_end <= w_start

    lines.append(f"Stage 2 Check (Reward Up & Crashes Low): {'PASS' if stage2_pass else 'FAIL'}")
    lines.append(f"Stage 3 Check (Crater Overlap Down):     {'PASS' if stage3_pass else 'FAIL'}")
    lines.append(f"Stage 4 Check (World Model Error Down):  {'PASS' if stage4_pass else 'FAIL'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print summary of Selene MAPPO training_log.csv")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/training_log.csv",
        help="Path to training_log.csv (default: data/training_log.csv)",
    )
    args = parser.parse_args(argv)
    print(format_training_summary(args.csv_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
