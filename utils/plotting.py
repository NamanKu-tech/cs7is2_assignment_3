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


def plot_dqn_loss(loss_history, title, save_path, window=200, ylabel="Loss / TD Error"):
    """Plot training loss or TD error curve. Works for both DQN and Q-Learning."""
    if not loss_history:
        return
    w = min(window, max(1, len(loss_history) // 4))
    smoothed = rolling_average(loss_history, w)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loss_history, alpha=0.25, color="#999", linewidth=0.6, label="Raw")
    ax.plot(smoothed, color="#e74c3c", linewidth=1.8, label=f"Smoothed (w={w})")
    ax.set_xlabel("Update Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)


# ------------------------------------------------------------------ #
# Evaluation charts                                                   #
# ------------------------------------------------------------------ #

def plot_win_rate_bar(results, game_name, save_path, opponent_name="Default"):
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
    bars_w = ax.bar(x - w, win_r,  w, label="Win",  color=PALETTE["win"])
    bars_d = ax.bar(x,     draw_r, w, label="Draw", color=PALETTE["draw"])
    bars_l = ax.bar(x + w, loss_r, w, label="Loss", color=PALETTE["loss"])

    # Annotate each bar with its value
    for bars in (bars_w, bars_d, bars_l):
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Rate")
    ax.set_title(f"{game_name}: Win / Draw / Loss Rate vs {opponent_name} Opponent\n"
                 f"(each agent evaluated over equal games as Player 1 and Player 2)")
    ax.legend(loc="upper right")
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5, label="_nolegend_")
    _save(fig, save_path)


def _draw_heatmap_ax(ax, matrix, labels, title, cmap, cbar_label):
    """Helper: draw a single heatmap onto an existing Axes."""
    from matplotlib.colors import LinearSegmentedColormap
    n = len(labels)
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Opponent  →", fontsize=9)
    ax.set_ylabel("←  Agent", fontsize=9)
    ax.set_title(title, fontsize=10, pad=8)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="#aaaaaa", fontsize=11)
            else:
                val = matrix[i, j]
                color = "#333333" if 0.15 <= val <= 0.85 else "#555555"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8, pad=0.02)


def plot_heatmap(matrix, labels, title, save_path, draw_matrix=None):
    """
    Round-robin heatmap.  If draw_matrix is provided, saves two side-by-side
    heatmaps in one image: left = win rate, right = draw rate.
    Each game is played swap_sides=True so win rates are averaged over P1 and P2.
    """
    from matplotlib.colors import LinearSegmentedColormap
    win_cmap  = LinearSegmentedColormap.from_list(
        "pastel_rg", ["#f4a8a8", "#fdf6f0", "#a8d5b5"], N=256)
    draw_cmap = LinearSegmentedColormap.from_list(
        "pastel_bu", ["#fdf6f0", "#b8d4f0", "#5a9fd4"], N=256)

    n = len(labels)
    ncols = 2 if draw_matrix is not None else 1
    fig, axes = plt.subplots(1, ncols,
                             figsize=(max(6, n * 1.5) * ncols + 1, max(5, n * 1.2)))
    if ncols == 1:
        axes = [axes]

    _draw_heatmap_ax(axes[0], matrix, labels,
                     "Win Rate\n(row agent goes first, col agent goes second)",
                     win_cmap, "Win Rate")

    if draw_matrix is not None:
        _draw_heatmap_ax(axes[1], draw_matrix, labels,
                         "Draw Rate\n(row agent goes first, col agent goes second)",
                         draw_cmap, "Draw Rate")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
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
    ax.set_ylabel("Win Rate (rolling avg)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, title=param_name)
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


# ------------------------------------------------------------------ #
# Additional analysis plots                                           #
# ------------------------------------------------------------------ #

def plot_epsilon_decay(n_episodes, epsilon_start, epsilon_min, epsilon_decay, title, save_path):
    """Show how epsilon decays over episodes."""
    eps = epsilon_start
    values = []
    for _ in range(n_episodes):
        values.append(eps)
        eps = max(epsilon_min, eps * epsilon_decay)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(values, color="#9C27B0", linewidth=1.5)
    ax.axhline(epsilon_min, color="grey", linestyle="--", alpha=0.6,
               label=f"Min ε = {epsilon_min}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon (ε)")
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)


def plot_ql_qtable_growth(q_size_history, title, save_path):
    """Plot how many unique states Q-Learning has discovered over training."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(q_size_history, color="#FF9800", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Unique States in Q-Table")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _save(fig, save_path)


def plot_combined_loss(ql_loss, dqn_loss, game_name, save_path):
    """Side-by-side loss curves: QL TD error vs DQN Huber loss."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Q-Learning TD error
    if ql_loss:
        w = min(500, len(ql_loss) // 4 or 1)
        smoothed = rolling_average(ql_loss, w)
        axes[0].plot(ql_loss, alpha=0.2, color="#FF9800", linewidth=0.6, label="Raw")
        axes[0].plot(smoothed, color="#FF9800", linewidth=1.8, label=f"Smoothed (w={w})")
        axes[0].set_title(f"{game_name}: Q-Learning TD Error")
        axes[0].set_xlabel("Update Step")
        axes[0].set_ylabel("|TD Error|")
        axes[0].legend()

    # DQN Huber loss
    if dqn_loss:
        w = min(500, len(dqn_loss) // 4 or 1)
        smoothed = rolling_average(dqn_loss, w)
        axes[1].plot(dqn_loss, alpha=0.2, color="#E91E63", linewidth=0.6, label="Raw")
        axes[1].plot(smoothed, color="#E91E63", linewidth=1.8, label=f"Smoothed (w={w})")
        axes[1].set_title(f"{game_name}: DQN Huber Loss")
        axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel("Huber Loss")
        axes[1].legend()

    fig.suptitle(f"{game_name}: Training Loss Comparison", fontweight="bold")
    _save(fig, save_path)


def plot_winrate_over_time(histories, labels, title, save_path, window=500):
    """
    Overlay win-rate curves for multiple agents on one chart.
    histories: list of win_history lists
    labels: list of agent names
    """
    colors = [PALETTE.get(l, "#607D8B") for l in labels]
    fig, ax = plt.subplots(figsize=(12, 5))

    for history, label, color in zip(histories, labels, colors):
        w = min(window, len(history) // 4 or 1)
        smoothed = rolling_average([1 if x == 1 else 0 for x in history], w)
        # Pad shorter histories so they align
        ax.plot(smoothed, label=label, color=color, alpha=0.85, linewidth=1.8)

    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.4, label="50% baseline")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Win Rate (rolling avg, w={window})")
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)


def plot_game_length_dist(game_lengths_dict, title, save_path):
    """
    Histogram of game lengths per agent.
    game_lengths_dict: dict[agent_name -> list[int]]
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(game_lengths_dict)))
    for (name, lengths), color in zip(game_lengths_dict.items(), colors):
        ax.hist(lengths, bins=20, alpha=0.5, label=name, color=color, edgecolor="white")
    ax.set_xlabel("Game Length (moves)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)


def plot_first_player_bias(results_dict, title, save_path, opponent_name="Default"):
    """
    Grouped bar chart showing win+draw rate as P1 vs P2 for each agent.
    results_dict: {agent_name -> stats} where stats has p1/p2 win/draw rates.
    """
    names = [n for n, s in results_dict.items()
             if "p1_win_rate" in s and (s["p1_win_rate"] + s["p2_win_rate"]) > 0]
    if not names:
        return

    p1_wr = [results_dict[n]["p1_win_rate"]  for n in names]
    p2_wr = [results_dict[n]["p2_win_rate"]  for n in names]
    p1_dr = [results_dict[n]["p1_draw_rate"] for n in names]
    p2_dr = [results_dict[n]["p2_draw_rate"] for n in names]

    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(9, len(names) * 2.5), 6))

    b1w = ax.bar(x - 1.5*w, p1_wr, w, label="P1 Win",  color="#7BB8F5", alpha=0.95)
    b1d = ax.bar(x - 0.5*w, p1_dr, w, label="P1 Draw", color="#B8D8F5", alpha=0.95)
    b2w = ax.bar(x + 0.5*w, p2_wr, w, label="P2 Win",  color="#FFB347", alpha=0.95)
    b2d = ax.bar(x + 1.5*w, p2_dr, w, label="P2 Draw", color="#FFD9A0", alpha=0.95)

    for bars in (b1w, b1d, b2w, b2d):
        for bar in bars:
            h = bar.get_height()
            if h > 0.03:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Rate")
    ax.set_title(f"{title}\nOpponent: {opponent_name}  |  P1 = goes first, P2 = goes second")
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", ncol=2)
    _save(fig, save_path)


def plot_curriculum_training(stage_histories, title, save_path, window=500):
    """
    Training curve across curriculum stages with vertical stage-boundary lines.
    stage_histories: list of {"label", "win_history"}
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(stage_histories)))

    offset = 0
    boundaries = []

    for sh, color in zip(stage_histories, colors):
        hist = sh["win_history"]
        w = min(window, max(1, len(hist) // 4))
        smoothed = rolling_average([1 if x == 1 else 0 for x in hist], w)
        episodes = np.arange(offset, offset + len(hist))
        ax.plot(episodes, smoothed, color=color, linewidth=1.8, label=sh["label"])
        if offset > 0:
            boundaries.append(offset)
        offset += len(hist)

    for b in boundaries:
        ax.axvline(b, color="grey", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Episode (across all stages)")
    ax.set_ylabel(f"Win Rate (rolling avg, w={window})")
    ax.set_title(title)
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.3)
    ax.legend(title="Stage")
    _save(fig, save_path)


def plot_move_limit_comparison(results_dict, title, save_path):
    """
    Bar chart comparing win/draw/loss rates under different move limits.
    results_dict: {label -> {"win_rate", "draw_rate", "loss_rate"}}
    """
    labels = list(results_dict.keys())
    win_r  = [results_dict[l]["win_rate"]  for l in labels]
    draw_r = [results_dict[l]["draw_rate"] for l in labels]
    loss_r = [results_dict[l]["loss_rate"] for l in labels]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 5))
    ax.bar(x - w, win_r,  w, label="Win",  color=PALETTE["win"])
    ax.bar(x,     draw_r, w, label="Draw", color=PALETTE["draw"])
    ax.bar(x + w, loss_r, w, label="Loss", color=PALETTE["loss"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)
