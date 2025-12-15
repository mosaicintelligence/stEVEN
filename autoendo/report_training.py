"""
report_training.py

Aggregate SB3 Monitor logs and plot episode rewards/lengths over training.
Reads monitor.csv files under a log directory and writes a PNG report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import glob

import matplotlib

# Use non-interactive backend for headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from tensorboard.backend.event_processing import event_accumulator  # type: ignore


def load_monitor_file(path: Path) -> pd.DataFrame:
    """Load a single SB3 Monitor CSV."""
    df = pd.read_csv(path, comment="#")
    # Monitor columns: r (reward), l (length), t (time)
    return df


def load_all_monitors(log_dir: Path) -> pd.DataFrame:
    files = glob.glob(str(log_dir / "**" / "monitor*.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No monitor*.csv found under {log_dir}")

    frames: List[pd.DataFrame] = []
    for idx, f in enumerate(sorted(files)):
        df = load_monitor_file(Path(f))
        df["env_id"] = idx
        # Approximate cumulative steps within each env file
        df["steps"] = df["l"].cumsum()
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    # Rebase steps to be monotonic across all envs by ordering on t then steps.
    combined = combined.sort_values(["t", "env_id"]).reset_index(drop=True)
    combined["global_step"] = combined["l"].cumsum()
    return combined


def load_tensorboard_scalars(log_dir: Path) -> pd.DataFrame:
    """Fallback: load ep reward/length scalars from TensorBoard event files."""

    event_files = glob.glob(str(log_dir / "**" / "events.out.tfevents.*"), recursive=True)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {log_dir}")

    def _acc_scalar(path: str, tag: str):
        ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            return None
        scalars = ea.Scalars(tag)
        return pd.DataFrame({"global_step": [s.step for s in scalars], tag: [s.value for s in scalars]})

    dfs = []
    for ef in sorted(event_files):
        rew = _acc_scalar(ef, "rollout/ep_rew_mean")
        length = _acc_scalar(ef, "rollout/ep_len_mean")
        frames = []
        if rew is not None:
            frames.append(rew)
        if length is not None:
            frames.append(length)
        if frames:
            merged = frames[0]
            for f in frames[1:]:
                merged = merged.merge(f, on="global_step", how="outer")
            dfs.append(merged)
    if not dfs:
        raise FileNotFoundError("TensorBoard events found, but no rollout scalars available.")
    combined = pd.concat(dfs, ignore_index=True).sort_values("global_step")
    combined = combined.reset_index(drop=True)
    # Align names to monitor-style columns if possible
    if "rollout/ep_rew_mean" in combined.columns:
        combined["r"] = combined["rollout/ep_rew_mean"]
    if "rollout/ep_len_mean" in combined.columns:
        combined["l"] = combined["rollout/ep_len_mean"]
    return combined


def plot_report(df: pd.DataFrame, out_path: Path, window: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    steps = df["global_step"]
    rewards = df["r"]
    lengths = df["l"]

    rewards_roll = rewards.rolling(window=window, min_periods=1).mean()
    lengths_roll = lengths.rolling(window=window, min_periods=1).mean()

    axes[0].plot(steps, rewards, alpha=0.3, label="Episode reward")
    axes[0].plot(steps, rewards_roll, color="C1", label=f"Reward (rolling {window})")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, lengths, alpha=0.3, label="Episode length")
    axes[1].plot(steps, lengths_roll, color="C2", label=f"Length (rolling {window})")
    axes[1].set_xlabel("Global steps (approx)")
    axes[1].set_ylabel("Episode length")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[report] saved to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("outputs/autoendo/sb3_logs"),
        help="Directory containing SB3 monitor*.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the report PNG.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling window (episodes) for smoothing plots.",
    )
    parser.add_argument(
        "--prefer-monitor",
        action="store_true",
        help="Prefer monitor CSV files; otherwise fall back to TensorBoard scalars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        df = load_all_monitors(args.log_dir)
    except FileNotFoundError:
        df = load_tensorboard_scalars(args.log_dir)
    output = args.output if args.output is not None else args.log_dir / "training_report.png"
    plot_report(df, output, args.window)


if __name__ == "__main__":
    main()
