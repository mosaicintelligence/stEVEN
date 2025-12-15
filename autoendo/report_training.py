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
        comp_target = _acc_scalar(ef, "rollout/rew_target")
        comp_path = _acc_scalar(ef, "rollout/rew_path")
        comp_tip = _acc_scalar(ef, "rollout/rew_tip")
        frames = []
        if rew is not None:
            frames.append(rew)
        if length is not None:
            frames.append(length)
        for comp in (comp_target, comp_path, comp_tip):
            if comp is not None:
                frames.append(comp)
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


def merge_monitor_and_tb(monitor_df: pd.DataFrame, tb_df: pd.DataFrame) -> pd.DataFrame:
    """Merge monitor data with TensorBoard scalars (primarily component rewards)."""
    if monitor_df is None:
        return tb_df
    if tb_df is None:
        return monitor_df

    # Keep monitor r/l/t as ground truth; add/align component scalars from TB.
    monitor_df = monitor_df.sort_values("global_step")
    tb_df = tb_df.sort_values("global_step")

    # Ensure numeric steps
    monitor_df["global_step"] = pd.to_numeric(monitor_df["global_step"], errors="coerce")
    tb_df["global_step"] = pd.to_numeric(tb_df["global_step"], errors="coerce")
    monitor_df = monitor_df.dropna(subset=["global_step"])
    tb_df = tb_df.dropna(subset=["global_step"])

    comp_cols = [c for c in tb_df.columns if c.startswith("rollout/rew_")]
    if not comp_cols:
        return monitor_df

    tb_keep = tb_df[["global_step"] + comp_cols].drop_duplicates(subset="global_step")
    merged = pd.merge_asof(
        monitor_df,
        tb_keep,
        on="global_step",
        direction="nearest",
    )
    return merged


def plot_report(df: pd.DataFrame, log_dir: Path, out_path: Path, window: int) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].set_title(f"Training Report: {log_dir}")

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
    axes[1].set_ylabel("Episode length")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Reward components if present
    comp_tags = {
        "rollout/rew_target": ("Target", "C3"),
        "rollout/rew_path": ("PathDelta", "C4"),
        "rollout/rew_tip": ("TipProgress", "C5"),
    }
    plotted = False
    for tag, (label, color) in comp_tags.items():
        if tag in df:
            axes[2].plot(steps, df[tag], alpha=0.4, label=label, color=color)
            axes[2].plot(
                steps, df[tag].rolling(window=window, min_periods=1).mean(), color=color, linestyle="--"
            )
            plotted = True
    if plotted:
        axes[2].set_ylabel("Reward components")
        axes[2].grid(alpha=0.3)
        axes[2].legend()
    axes[2].set_xlabel("Global steps (approx)")

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
    monitor_df = None
    tb_df = None
    try:
        monitor_df = load_all_monitors(args.log_dir)
    except FileNotFoundError:
        monitor_df = None
    try:
        tb_df = load_tensorboard_scalars(args.log_dir)
    except FileNotFoundError:
        tb_df = None

    if args.prefer_monitor and monitor_df is not None:
        df = merge_monitor_and_tb(monitor_df, tb_df)
    elif monitor_df is None and tb_df is not None:
        df = tb_df
    elif monitor_df is not None:
        df = merge_monitor_and_tb(monitor_df, tb_df)
    else:
        raise FileNotFoundError("No monitor CSV or TensorBoard event files found.")
    output = args.output if args.output is not None else args.log_dir / "training_report.png"
    plot_report(df, args.log_dir, output, args.window)


if __name__ == "__main__":
    main()
