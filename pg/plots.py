"""Policy-gradient figures.

    python pg/plots.py    # re-plot every run in results/pg and the Part 1 vs Part 2 comparisons
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")  # no display on the server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def moving_average(x, window=100):
    """Mean of the last `window` episodes (fewer at the start, so the curve begins at episode 1)."""
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rolling(window, min_periods=1).mean().values


def run_title(run_dir):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    game = "Pong-v5" if "Pong" in cfg["env"] else cfg["env"]
    return f"{game} ({'Part 2, value baseline' if cfg['baseline'] else 'Part 1, no baseline'})"


def plot_run(run_dir):
    """Training curve with 100-episode moving average, and histogram of evaluation rewards."""
    df = pd.read_csv(os.path.join(run_dir, "train_log.csv"))
    title = run_title(run_dir)
    pong = "Pong" in title

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(df.episode, df.reward, color="C0", alpha=0.35, lw=0.7, label="episode reward")
    ax.plot(df.episode, moving_average(df.reward), color="C3", lw=2, label="100-episode moving average")
    ax.set(xlabel="Training episode", ylabel="Episode reward", title=f"Training: {title}")
    ax.legend(loc="upper left" if pong else "lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, "training_curve.png"), dpi=160)

    eval_path = os.path.join(run_dir, "eval_rewards.txt")
    if os.path.exists(eval_path):
        r = np.atleast_1d(np.loadtxt(eval_path))
        bins = np.arange(-21.5, 22.5, 1) if pong else np.linspace(0, 510, 52)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(r, bins=bins, color="C2", edgecolor="white")
        std = r.std(ddof=1) if len(r) > 1 else 0.0
        ax.axvline(r.mean(), color="k", ls="--", lw=1.2, label=f"mean {r.mean():.2f}, std {std:.2f}")
        ax.set(xlabel="Episode reward", ylabel="Count", title=f"{len(r)} evaluation episodes: {title}")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(run_dir, "eval_histogram.png"), dpi=160)
    plt.close("all")


def plot_comparison(game, root="results/pg"):
    """Part 1 vs Part 2: learning curves and spread of the policy-gradient norm."""
    runs = [(os.path.join(root, f"{game}_part1"), "C0", "Part 1: no baseline"),
            (os.path.join(root, f"{game}_part2"), "C3", "Part 2: value baseline")]
    if not all(os.path.exists(os.path.join(d, "train_log.csv")) for d, _, _ in runs):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run_dir, color, label in runs:
        df = pd.read_csv(os.path.join(run_dir, "train_log.csv"))
        axes[0].plot(df.episode, moving_average(df.reward), color=color, lw=2, label=label)
        per_update = df.drop_duplicates("update", keep="last")
        axes[1].plot(per_update.episode, per_update.grad_norm.rolling(20, min_periods=2).std(),
                     color=color, lw=1.5, label=label)
    name = "Pong-v5" if game == "pong" else "CartPole-v1"
    axes[0].set(xlabel="Training episode", ylabel="100-episode moving average", title=f"{name}: learning curves")
    axes[1].set(xlabel="Training episode", ylabel="std of gradient norm (20 updates)", yscale="log",
                title=f"{name}: gradient variability")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(root, f"{game}_part1_vs_part2.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    for run_dir in sorted(glob.glob("results/pg/*_part[12]")):
        plot_run(run_dir)
    for game in ("cartpole", "pong"):
        plot_comparison(game)
