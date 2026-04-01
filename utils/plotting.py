"""
All matplotlib plotting functions for CS7IS2 Assignment 3.
Call setup_style() once at the top of your notebook/script.
All functions save to `save_path` and close the figure to avoid memory leaks.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (works in Colab and scripts)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.metrics import rolling_average


# ------------------------------------------------------------------ #
# Style                                                               #
# ------------------------------------------------------------------ #

PALETTE = {
    "Minimax":    "#2196F3",
    "Minimax+AB": "#4CAF50",
    "Q-Learning": "#FF9800",
    "DQN":        "#E91E63",
    "Default":    "#9C27B0",
    "Random":     "#607D8B",
    "win":        "#2ecc71",
    "draw":       "#3498db",
    "loss":       "#e74c3c",
}


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.figsize": (10, 5),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
    })


def _save(fig, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Training curves                                                     #
# ------------------------------------------------------------------ #

def plot_training_curve(win_history, title, save_path, window=500):
    """2-panel: (1) rolling win/draw/loss rates, (2) cumulative win rate."""
    n = len(win_history)
    wins   = rolling_average([1.0 if x ==  1 else 0.0 for x in win_history], window)
    draws  = rolling_average([1.0 if x ==  0 else 0.0 for x in win_history], window)
    losses = rolling_average([1.0 if x == -1 else 0.0 for x in win_history], window)
    episodes = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: rolling rates
    ax = axes[0]
    ax.plot(episodes, wins,   color=PALETTE["win"],  label="Win")
    ax.plot(episodes, draws,  color=PALETTE["draw"], label="Draw")
    ax.plot(episodes, losses, color=PALETTE["loss"], label="Loss")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Rate (rolling avg, w={window})")
    ax.set_title(f"{title}\nRolling W/D/L Rates")
    ax.legend()

    # Panel 2: cumulative win rate
    ax2 = axes[1]
    cum_wins = np.cumsum([1 if x == 1 else 0 for x in win_history])
    cum_rate = cum_wins / (episodes + 1)
    ax2.plot(episodes, cum_rate, color=PALETTE["win"])
    ax2.axhline(cum_rate[-1], color="grey", linestyle="--", alpha=0.6,
                label=f"Final: {cum_rate[-1]:.1%}")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Cumulative Win Rate")
    ax2.set_title(f"{title}\nCumulative Win Rate")
    ax2.legend()

    _save(fig, save_path)


def plot_dqn_loss(loss_history, title, save_path, window=200):
    """Plot DQN training loss curve."""
    if not loss_history:
        return
    smoothed = rolling_average(loss_history, min(window, len(loss_history)))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loss_history, alpha=0.3, color="#999", linewidth=0.8, label="Raw loss")
    ax.plot(smoothed, color="#e74c3c", label=f"Smoothed (w={window})")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Huber Loss")
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)


# ------------------------------------------------------------------ #
# Evaluation charts                                                   #
# ------------------------------------------------------------------ #

def plot_win_rate_bar(results, game_name, save_path):
    """
    Grouped bar chart of W/D/L rates for each agent vs. a baseline.
    results: dict[agent_name -> {"win_rate", "draw_rate", "loss_rate"}]
    """
    names = list(results.keys())
    win_r  = [results[n]["win_rate"]  for n in names]
    draw_r = [results[n]["draw_rate"] for n in names]
    loss_r = [results[n]["loss_rate"] for n in names]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 5))
    ax.bar(x - w,   win_r,  w, label="Win",  color=PALETTE["win"])
    ax.bar(x,       draw_r, w, label="Draw", color=PALETTE["draw"])
    ax.bar(x + w,   loss_r, w, label="Loss", color=PALETTE["loss"])

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title(f"{game_name}: Algorithm Comparison vs Default Opponent")
    ax.legend()
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
    _save(fig, save_path)


def plot_heatmap(matrix, labels, title, save_path):
    """
    Win-rate heatmap for round-robin results.
    matrix[i,j] = win rate of agents[i] when playing against agents[j].
    """
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n + 1), max(5, n)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Agent")
    ax.set_title(title)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val < 0.25 or val > 0.75 else "black"
            if i == j:
                text = "—"
            else:
                text = f"{val:.0%}"
            ax.text(j, i, text, ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Win Rate", shrink=0.8)
    _save(fig, save_path)


# ------------------------------------------------------------------ #
# Parameter sweep plots                                               #
# ------------------------------------------------------------------ #

def plot_param_sweep(sweep_results, param_name, title, save_path, window=500):
    """
    Overlay of training curves for different param configs.
    sweep_results: dict[label -> list[int]]  (win_history per config)
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(sweep_results)))

    for (label, history), color in zip(sweep_results.items(), colors):
        w = min(window, len(history) // 4 or 1)
        smoothed = rolling_average([1 if x == 1 else 0 for x in history], w)
        ax.plot(smoothed, label=label, color=color, alpha=0.85)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Win Rate (rolling avg)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.4)
    _save(fig, save_path)


def plot_param_bar(param_values, metric_values, param_name, metric_name, title, save_path):
    """Bar chart of a single metric across param values (e.g. final win rate vs alpha)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(param_values))
    bars = ax.bar(x, metric_values, color="#3498db", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in param_values])
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(metric_values) * 1.15 + 0.05)
    _save(fig, save_path)


# ------------------------------------------------------------------ #
# Depth / scalability analysis                                        #
# ------------------------------------------------------------------ #

def plot_depth_analysis(depth_results, save_path):
    """
    3-panel plot: win rate, avg time (ms), avg nodes vs depth limit.
    depth_results: dict[depth -> {"win_rate", "avg_time_ms", "avg_nodes"}]
    """
    depths    = sorted(depth_results.keys())
    win_rates = [depth_results[d]["win_rate"]    for d in depths]
    times     = [depth_results[d]["avg_time_ms"] for d in depths]
    nodes     = [depth_results[d]["avg_nodes"]   for d in depths]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(depths, win_rates, color="#2ecc71")
    axes[0].set_xlabel("Depth Limit")
    axes[0].set_ylabel("Win Rate")
    axes[0].set_title("Win Rate vs Depth")
    axes[0].set_ylim(0, 1.1)
    for d, v in zip(depths, win_rates):
        axes[0].text(d, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)

    axes[1].bar(depths, times, color="#3498db")
    axes[1].set_xlabel("Depth Limit")
    axes[1].set_ylabel("Avg Move Time (ms)")
    axes[1].set_title("Move Time vs Depth")
    if max(times) > 100:
        axes[1].set_yscale("log")

    axes[2].bar(depths, nodes, color="#e74c3c")
    axes[2].set_xlabel("Depth Limit")
    axes[2].set_ylabel("Avg Nodes Visited")
    axes[2].set_title("Nodes Visited vs Depth")
    if max(nodes) > 1000:
        axes[2].set_yscale("log")

    fig.suptitle("Connect 4: Minimax Depth Limit Analysis", fontweight="bold", y=1.01)
    _save(fig, save_path)


def plot_node_comparison(results, save_path):
    """
    Bar chart comparing Minimax vs Minimax+AB: avg nodes visited and avg time.
    results: dict[algo_name -> {"avg_nodes", "avg_time_ms"}]
    """
    names = list(results.keys())
    nodes = [results[n]["avg_nodes"]   for n in names]
    times = [results[n]["avg_time_ms"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(names, nodes, color=["#2196F3", "#4CAF50"])
    axes[0].set_ylabel("Avg Nodes Visited")
    axes[0].set_title("Nodes Visited: Minimax vs Alpha-Beta")
    for i, v in enumerate(nodes):
        axes[0].text(i, v * 1.02, f"{v:,.0f}", ha="center", fontsize=9)

    axes[1].bar(names, times, color=["#2196F3", "#4CAF50"])
    axes[1].set_ylabel("Avg Move Time (ms)")
    axes[1].set_title("Move Time: Minimax vs Alpha-Beta")
    for i, v in enumerate(times):
        axes[1].text(i, v * 1.02, f"{v:.1f}ms", ha="center", fontsize=9)

    _save(fig, save_path)
