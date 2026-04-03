"""
experiments.py — All experiment runners for CS7IS2 Assignment 3.

Run everything:   python experiments.py
Run one section:  python experiments.py --phase ttt
"""

import os
import sys
import time
import numpy as np

# ---------------------------------------------------------------------------
# Colab path helper (auto-detected; no-op when running locally)
# ---------------------------------------------------------------------------
if "google.colab" in sys.modules:
    sys.path.insert(0, "/content/assignment_3")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from games.tictactoe import TicTacToe
from games.connect4 import Connect4
from agents.random_agent import RandomAgent
from agents.default_agent import DefaultAgent
from agents.minimax_agent import MinimaxAgent, MinimaxABAgent
from agents.qlearning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from training.trainer import Trainer, CurriculumTrainer
from training.evaluator import Evaluator
from utils.plotting import (
    setup_style, plot_training_curve, plot_win_rate_bar,
    plot_heatmap, plot_param_sweep, plot_param_bar,
    plot_dqn_loss, plot_depth_analysis, plot_node_comparison,
    plot_epsilon_decay, plot_ql_qtable_growth, plot_combined_loss,
    plot_winrate_over_time, plot_first_player_bias, plot_curriculum_training,
    plot_move_limit_comparison,
)
from utils.metrics import final_stats

setup_style()

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
FIG_DIR   = "results/figures"
MODEL_DIR = "results/models"
CSV_DIR   = "results/csv"
os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CSV_DIR,   exist_ok=True)


# ===========================================================================
# Helper
# ===========================================================================

def _fig(name):
    return os.path.join(FIG_DIR, name)

def _model(name):
    return os.path.join(MODEL_DIR, name)

def _csv(name):
    return os.path.join(CSV_DIR, name)

def _save_eval_csv(results_dict, path):
    """Save evaluation stats dict to CSV. results_dict: {agent_name -> stats}"""
    import csv
    rows = []
    for agent_name, stats in results_dict.items():
        rows.append({
            "agent": agent_name,
            "wins": stats.get("wins", ""),
            "draws": stats.get("draws", ""),
            "losses": stats.get("losses", ""),
            "win_rate": f"{stats['win_rate']:.4f}",
            "draw_rate": f"{stats['draw_rate']:.4f}",
            "loss_rate": f"{stats['loss_rate']:.4f}",
            "avg_game_length": f"{stats['avg_game_length']:.2f}",
            "avg_move_time_ms": f"{stats['avg_move_time_ms']:.3f}",
            "avg_nodes": f"{stats['avg_nodes']:.1f}",
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved: {path}")

def _save_training_csv(win_history, loss_history, path, window=1000):
    """Save per-episode training metrics to CSV (sampled every `window` episodes)."""
    import csv
    n = len(win_history)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "win_rate", "draw_rate", "loss_rate", "avg_loss"])
        for i in range(window - 1, n, window):
            chunk = win_history[max(0, i - window + 1): i + 1]
            wr = sum(1 for x in chunk if x ==  1) / len(chunk)
            dr = sum(1 for x in chunk if x ==  0) / len(chunk)
            lr = sum(1 for x in chunk if x == -1) / len(chunk)
            if loss_history:
                # sample corresponding loss window
                lstart = int(i / n * len(loss_history))
                lend   = int((i + 1) / n * len(loss_history))
                avg_l  = float(np.mean(loss_history[lstart:lend])) if lstart < lend else ""
            else:
                avg_l = ""
            writer.writerow([i + 1, f"{wr:.4f}", f"{dr:.4f}", f"{lr:.4f}", avg_l])
    print(f"  CSV saved: {path}")

def _save_rr_csv(matrix, labels, path):
    """Save round-robin win-rate matrix to CSV."""
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent"] + labels)
        for i, label in enumerate(labels):
            writer.writerow([label] + [f"{matrix[i,j]:.4f}" for j in range(len(labels))])
    print(f"  CSV saved: {path}")

def _print_header(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

def _print_results(stats):
    print(f"  Win: {stats['win_rate']:.1%}  Draw: {stats['draw_rate']:.1%}  "
          f"Loss: {stats['loss_rate']:.1%}  "
          f"Avg length: {stats['avg_game_length']:.1f}  "
          f"Avg time: {stats['avg_move_time_ms']:.1f}ms")


# ===========================================================================
# Phase 0: Connect 4 Scalability Analysis
# ===========================================================================

def run_scalability_test(time_limit=300):
    """
    Run full-depth minimax and minimax+AB on an empty C4 board until time_limit.
    Counts nodes explored in the time window.
    This is expected to NOT finish — that's the point.
    """
    _print_header("Phase 0: C4 Scalability Test")
    print(f"  Running each agent for {time_limit}s on empty Connect 4 board…")

    results = {}
    for AgentCls, label in [(MinimaxAgent, "Minimax"), (MinimaxABAgent, "Minimax+AB")]:
        agent = AgentCls(player=1, depth_limit=None)
        game = Connect4()
        game.reset()

        # Probe: run for time_limit seconds and count nodes
        start = time.time()

        # We won't finish — just run the first move search until timeout
        import threading
        result_box = {}

        def search_worker():
            result_box["action"] = agent.choose_move(game)
            result_box["nodes"] = agent.nodes_visited

        t = threading.Thread(target=search_worker, daemon=True)
        t.start()
        t.join(timeout=time_limit)

        elapsed = time.time() - start
        if t.is_alive():
            # Still running — estimate nodes from what we know
            nodes_so_far = agent.nodes_visited
            print(f"  [{label}] Still running after {elapsed:.0f}s | "
                  f"Nodes so far: {nodes_so_far:,}")
            results[label] = {
                "completed": False,
                "nodes": nodes_so_far,
                "time_s": elapsed,
            }
        else:
            nodes = result_box.get("nodes", agent.nodes_visited)
            print(f"  [{label}] Completed in {elapsed:.1f}s | Nodes: {nodes:,}")
            results[label] = {
                "completed": True,
                "nodes": nodes,
                "time_s": elapsed,
            }

    return results


# ===========================================================================
# Phase 1: Tic Tac Toe
# ===========================================================================

def run_ttt_experiments(
    ql_episodes=60_000,
    dqn_episodes=40_000,
    eval_games=500,
    verbose=True,
):
    _print_header("Phase 1a: Tic Tac Toe — Training RL Agents")

    default_opp = DefaultAgent()

    # --- Q-Learning ---
    print("\n[TTT] Training Q-Learning vs Default Opponent…")
    ttt_ql = QLearningAgent(player=1, alpha=0.1, gamma=0.99, epsilon=1.0,
                             epsilon_min=0.05, epsilon_decay=0.9998)
    trainer = Trainer(TicTacToe, ttt_ql, default_opp,
                      n_episodes=ql_episodes, swap_sides=True, verbose=verbose)
    ql_metrics = trainer.train()
    _save_training_csv(ql_metrics["win_history"], ql_metrics["loss_history"],
                       _csv("ttt_ql_training.csv"))
    plot_training_curve(ql_metrics["win_history"],
                        "Q-Learning Training — Tic Tac Toe vs Default",
                        _fig("ttt_ql_training.png"))
    plot_dqn_loss(ql_metrics["loss_history"],
                  "Q-Learning TD Error — Tic Tac Toe",
                  _fig("ttt_ql_loss.png"), ylabel="|TD Error|")
    plot_ql_qtable_growth(ql_metrics["q_size_history"],
                          "Q-Learning Q-Table Growth — Tic Tac Toe",
                          _fig("ttt_ql_qtable.png"))
    plot_epsilon_decay(ql_episodes, 1.0, 0.05, 0.9998,
                       "Epsilon Decay — TTT Q-Learning",
                       _fig("ttt_ql_epsilon.png"))
    ttt_ql.save(_model("ttt_ql.pkl"))
    print(f"  Final stats: {final_stats(ql_metrics['win_history'])}")

    # --- DQN ---
    print("\n[TTT] Training DQN vs Default Opponent…")
    ttt_dqn = DQNAgent(player=1, state_size=9, action_size=9,
                        hidden_sizes=(128, 128), lr=1e-3, gamma=0.99,
                        epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9992,
                        batch_size=64, memory_size=20_000,
                        target_update_freq=200, game_name="tictactoe")
    trainer = Trainer(TicTacToe, ttt_dqn, default_opp,
                      n_episodes=dqn_episodes, swap_sides=True,
                      train_every=1, verbose=verbose)
    dqn_metrics = trainer.train()
    _save_training_csv(dqn_metrics["win_history"], dqn_metrics["loss_history"],
                       _csv("ttt_dqn_training.csv"))
    plot_training_curve(dqn_metrics["win_history"],
                        "DQN Training — Tic Tac Toe vs Default",
                        _fig("ttt_dqn_training.png"))
    if dqn_metrics["loss_history"]:
        plot_dqn_loss(dqn_metrics["loss_history"],
                      "DQN Loss — Tic Tac Toe",
                      _fig("ttt_dqn_loss.png"))
    plot_epsilon_decay(dqn_episodes, 1.0, 0.05, 0.9992,
                       "Epsilon Decay — TTT DQN",
                       _fig("ttt_dqn_epsilon.png"))
    ttt_dqn.save(_model("ttt_dqn.pkl"))
    print(f"  Final stats: {final_stats(dqn_metrics['win_history'])}")

    # --- Combined loss comparison ---
    plot_combined_loss(ql_metrics["loss_history"], dqn_metrics["loss_history"],
                       "Tic Tac Toe", _fig("ttt_combined_loss.png"))

    # --- Evaluation: all agents vs Default ---
    _print_header("Phase 1b: Tic Tac Toe — All Agents vs Default Opponent")
    ev = Evaluator(TicTacToe, n_games=eval_games, swap_sides=True, verbose=verbose)
    agents = {
        "Minimax":    MinimaxAgent(player=1),
        "Minimax+AB": MinimaxABAgent(player=1),
        "Q-Learning": ttt_ql,
        "DQN":        ttt_dqn,
    }
    vs_default = {}
    for name, agent in agents.items():
        stats = ev.evaluate(agent, DefaultAgent())
        vs_default[name] = stats
        print(f"  {name:12s}", end="  ")
        _print_results(stats)

    plot_win_rate_bar(vs_default, "Tic Tac Toe", _fig("ttt_vs_default.png"),
                      opponent_name="Default")
    _save_eval_csv(vs_default, _csv("ttt_vs_default.csv"))
    plot_first_player_bias(vs_default,
                           "Tic Tac Toe: First-Player Bias (P1 vs P2 Win/Draw Rate)",
                           _fig("ttt_first_player_bias.png"),
                           opponent_name="Default")

    # --- Avg game length per agent ---
    names_gl = list(vs_default.keys())
    avg_lengths = [vs_default[n]["avg_game_length"] for n in names_gl]
    plot_param_bar(names_gl, avg_lengths, "Agent", "Avg Game Length (moves)",
                   "Tic Tac Toe: Avg Game Length vs Default",
                   _fig("ttt_game_length.png"))

    # --- Overlaid RL win-rate curves ---
    plot_winrate_over_time(
        [ql_metrics["win_history"], dqn_metrics["win_history"]],
        ["Q-Learning", "DQN"],
        "TTT: Q-Learning vs DQN Win Rate During Training",
        _fig("ttt_rl_comparison.png"),
    )

    # --- Minimax node comparison ---
    _print_header("Phase 1c: Tic Tac Toe — Minimax vs Alpha-Beta Nodes")
    node_ev = Evaluator(TicTacToe, n_games=50, swap_sides=False, verbose=False)
    mm  = MinimaxAgent(player=1)
    ab  = MinimaxABAgent(player=1)
    mm_stats  = node_ev.evaluate(mm,  RandomAgent())
    ab_stats  = node_ev.evaluate(ab,  RandomAgent())
    node_results = {
        "Minimax":    {"avg_nodes": mm_stats["avg_nodes"],  "avg_time_ms": mm_stats["avg_move_time_ms"]},
        "Minimax+AB": {"avg_nodes": ab_stats["avg_nodes"],  "avg_time_ms": ab_stats["avg_move_time_ms"]},
    }
    print(f"  Minimax     avg nodes: {mm_stats['avg_nodes']:,.0f}  time: {mm_stats['avg_move_time_ms']:.2f}ms")
    print(f"  Minimax+AB  avg nodes: {ab_stats['avg_nodes']:,.0f}  time: {ab_stats['avg_move_time_ms']:.2f}ms")
    plot_node_comparison(node_results, _fig("ttt_node_comparison.png"))
    _save_eval_csv(
        {"Minimax": mm_stats, "Minimax+AB": ab_stats},
        _csv("ttt_node_comparison.csv")
    )

    # --- Round-robin ---
    _print_header("Phase 1d: Tic Tac Toe — Round Robin")
    rr_ev = Evaluator(TicTacToe, n_games=60, swap_sides=False, verbose=True)
    matrix, draw_matrix, labels, rr_results = rr_ev.round_robin(agents)
    np.fill_diagonal(matrix, 0)
    np.fill_diagonal(draw_matrix, 0)
    plot_heatmap(matrix, labels, "Tic Tac Toe: Head-to-Head Win Rates",
                 _fig("ttt_round_robin.png"), draw_matrix=draw_matrix)
    _save_rr_csv(matrix, labels, _csv("ttt_round_robin.csv"))
    print("  Win rate matrix (row beats col):")
    print("  " + "  ".join(f"{l[:5]:>5}" for l in labels))
    for i, label in enumerate(labels):
        print(f"  {label[:5]:>5}  " + "  ".join(f"{matrix[i,j]:.0%}" if i != j else "  —  "
                                                   for j in range(len(labels))))

    return {
        "ql": ttt_ql, "dqn": ttt_dqn,
        "vs_default": vs_default,
        "rr_matrix": matrix, "rr_labels": labels,
    }


# ===========================================================================
# Phase 2: Connect 4
# ===========================================================================

def run_c4_experiments(
    ql_episodes=100_000,
    dqn_episodes=60_000,
    eval_games=200,
    minimax_depth=4,
    verbose=True,
):
    _print_header("Phase 2a: Connect 4 — Training RL Agents (vs Random)")
    random_opp = RandomAgent()

    # Q-Learning (train vs Random — easier target to beat with limited training)
    print("\n[C4] Training Q-Learning vs Random Opponent…")
    c4_ql = QLearningAgent(player=1, alpha=0.1, gamma=0.99, epsilon=1.0,
                            epsilon_min=0.05, epsilon_decay=0.9999)
    trainer = Trainer(Connect4, c4_ql, random_opp,
                      n_episodes=ql_episodes, swap_sides=True, verbose=verbose)
    ql_metrics = trainer.train()
    _save_training_csv(ql_metrics["win_history"], ql_metrics["loss_history"],
                       _csv("c4_ql_training.csv"))
    plot_training_curve(ql_metrics["win_history"],
                        "Q-Learning Training — Connect 4 vs Random",
                        _fig("c4_ql_training.png"), window=1000)
    plot_dqn_loss(ql_metrics["loss_history"],
                  "Q-Learning TD Error — Connect 4",
                  _fig("c4_ql_loss.png"), ylabel="|TD Error|")
    plot_ql_qtable_growth(ql_metrics["q_size_history"],
                          "Q-Learning Q-Table Growth — Connect 4",
                          _fig("c4_ql_qtable.png"))
    plot_epsilon_decay(ql_episodes, 1.0, 0.05, 0.9999,
                       "Epsilon Decay — C4 Q-Learning",
                       _fig("c4_ql_epsilon.png"))
    c4_ql.save(_model("c4_ql.pkl"))
    print(f"  Final stats: {final_stats(ql_metrics['win_history'])}")

    # DQN (GPU if available)
    print("\n[C4] Training DQN vs Random Opponent…")
    c4_dqn = DQNAgent(player=1, state_size=42, action_size=7,
                       hidden_sizes=(256, 256, 128), lr=5e-4, gamma=0.99,
                       epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.99993,
                       batch_size=128, memory_size=100_000,
                       target_update_freq=500, game_name="connect4")
    trainer = Trainer(Connect4, c4_dqn, random_opp,
                      n_episodes=dqn_episodes, swap_sides=True,
                      train_every=1, verbose=verbose)
    dqn_metrics = trainer.train()
    _save_training_csv(dqn_metrics["win_history"], dqn_metrics["loss_history"],
                       _csv("c4_dqn_training.csv"))
    plot_training_curve(dqn_metrics["win_history"],
                        "DQN Training — Connect 4 vs Random",
                        _fig("c4_dqn_training.png"), window=1000)
    if dqn_metrics["loss_history"]:
        plot_dqn_loss(dqn_metrics["loss_history"],
                      "DQN Loss — Connect 4", _fig("c4_dqn_loss.png"))
    plot_epsilon_decay(dqn_episodes, 1.0, 0.05, 0.99993,
                       "Epsilon Decay — C4 DQN",
                       _fig("c4_dqn_epsilon.png"))
    c4_dqn.save(_model("c4_dqn.pkl"))
    print(f"  Final stats: {final_stats(dqn_metrics['win_history'])}")

    # --- Combined loss + win-rate overlay ---
    plot_combined_loss(ql_metrics["loss_history"], dqn_metrics["loss_history"],
                       "Connect 4", _fig("c4_combined_loss.png"))
    plot_winrate_over_time(
        [ql_metrics["win_history"], dqn_metrics["win_history"]],
        ["Q-Learning", "DQN"],
        "C4: Q-Learning vs DQN Win Rate During Training",
        _fig("c4_rl_comparison.png"),
        window=1000,
    )

    # --- Evaluation vs Default ---
    _print_header("Phase 2b: Connect 4 — All Agents vs Default Opponent")
    ev = Evaluator(Connect4, n_games=eval_games, swap_sides=True, verbose=verbose)
    agents = {
        "Minimax":    MinimaxAgent(player=1, depth_limit=minimax_depth),
        "Minimax+AB": MinimaxABAgent(player=1, depth_limit=minimax_depth),
        "Q-Learning": c4_ql,
        "DQN":        c4_dqn,
    }
    vs_default = {}
    for name, agent in agents.items():
        stats = ev.evaluate(agent, DefaultAgent())
        vs_default[name] = stats
        print(f"  {name:12s}", end="  ")
        _print_results(stats)

    plot_win_rate_bar(vs_default, "Connect 4", _fig("c4_vs_default.png"),
                      opponent_name="Default")
    _save_eval_csv(vs_default, _csv("c4_vs_default.csv"))
    plot_first_player_bias(vs_default,
                           "Connect 4: First-Player Bias (P1 vs P2 Win/Draw Rate)",
                           _fig("c4_first_player_bias.png"),
                           opponent_name="Default")

    # --- Game length bar (data already collected in vs_default) ---
    names_gl = list(vs_default.keys())
    avg_lengths = [vs_default[n]["avg_game_length"] for n in names_gl]
    plot_param_bar(names_gl, avg_lengths, "Agent", "Avg Game Length (moves)",
                   "Connect 4: Avg Game Length vs Default",
                   _fig("c4_game_length.png"))

    # --- Round-robin ---
    _print_header("Phase 2c: Connect 4 — Round Robin")
    rr_ev = Evaluator(Connect4, n_games=60, swap_sides=False, verbose=True)
    matrix, draw_matrix, labels, rr_results = rr_ev.round_robin(agents)
    np.fill_diagonal(matrix, 0)
    np.fill_diagonal(draw_matrix, 0)
    plot_heatmap(matrix, labels, "Connect 4: Head-to-Head Win Rates",
                 _fig("c4_round_robin.png"), draw_matrix=draw_matrix)
    _save_rr_csv(matrix, labels, _csv("c4_round_robin.csv"))

    return {
        "ql": c4_ql, "dqn": c4_dqn,
        "vs_default": vs_default,
        "rr_matrix": matrix, "rr_labels": labels,
    }


# ===========================================================================
# Phase 3: Depth limit analysis (Connect 4)
# ===========================================================================

def run_depth_analysis(depths=(2, 3, 4, 5, 6), games_per_depth=50):
    _print_header("Phase 3: C4 Depth Limit Analysis")
    ev = Evaluator(Connect4, n_games=games_per_depth, swap_sides=True, verbose=False)
    depth_results = {}

    for d in depths:
        agent = MinimaxABAgent(player=1, depth_limit=d)
        stats = ev.evaluate(agent, DefaultAgent())
        depth_results[d] = {
            "win_rate":    stats["win_rate"],
            "avg_time_ms": stats["avg_move_time_ms"],
            "avg_nodes":   stats["avg_nodes"],
        }
        print(f"  Depth {d}: win={stats['win_rate']:.0%}  "
              f"time={stats['avg_move_time_ms']:.1f}ms  "
              f"nodes={stats['avg_nodes']:,.0f}")

    plot_depth_analysis(depth_results, _fig("c4_depth_analysis.png"))

    import csv
    with open(_csv("c4_depth_analysis.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["depth", "win_rate", "avg_time_ms", "avg_nodes"])
        for d, r in depth_results.items():
            writer.writerow([d, f"{r['win_rate']:.4f}",
                             f"{r['avg_time_ms']:.3f}", f"{r['avg_nodes']:.1f}"])
    print(f"  CSV saved: {_csv('c4_depth_analysis.csv')}")
    return depth_results


# ===========================================================================
# Phase 4: Hyperparameter sweeps
# ===========================================================================

def run_ql_param_sweep(game_name="ttt", episodes=20_000):
    _print_header(f"Phase 4a: Q-Learning Hyperparameter Sweep ({game_name.upper()})")
    GameCls = TicTacToe if game_name == "ttt" else Connect4
    opponent = DefaultAgent() if game_name == "ttt" else RandomAgent()

    configs = [
        {"alpha": 0.01, "gamma": 0.99, "label": "α=0.01"},
        {"alpha": 0.05, "gamma": 0.99, "label": "α=0.05"},
        {"alpha": 0.10, "gamma": 0.99, "label": "α=0.10 (default)"},
        {"alpha": 0.30, "gamma": 0.99, "label": "α=0.30"},
        {"alpha": 0.10, "gamma": 0.80, "label": "α=0.10, γ=0.80"},
        {"alpha": 0.10, "gamma": 0.50, "label": "α=0.10, γ=0.50"},
    ]

    sweep = {}
    for cfg in configs:
        print(f"  {cfg['label']}…")
        agent = QLearningAgent(player=1, alpha=cfg["alpha"], gamma=cfg["gamma"],
                                epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995)
        trainer = Trainer(GameCls, agent, opponent,
                          n_episodes=episodes, swap_sides=True, verbose=False)
        metrics = trainer.train()
        sweep[cfg["label"]] = metrics["win_history"]
        fs = final_stats(metrics["win_history"])
        print(f"    Final win: {fs['win_rate']:.1%}")

    plot_param_sweep(sweep, "alpha / gamma",
                     f"Q-Learning Param Sweep — {game_name.upper()}",
                     _fig(f"{game_name}_ql_param_sweep.png"))

    # Bar: final win rate per config
    labels = list(sweep.keys())
    final_wins = [final_stats(sweep[l])["win_rate"] for l in labels]
    plot_param_bar(labels, final_wins, "Config", "Final Win Rate",
                   f"Q-Learning Final Win Rate by Config ({game_name.upper()})",
                   _fig(f"{game_name}_ql_param_bar.png"))

    import csv
    with open(_csv(f"{game_name}_ql_sweep.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "final_win_rate", "final_draw_rate", "final_loss_rate"])
        for l in labels:
            fs = final_stats(sweep[l])
            writer.writerow([l, f"{fs['win_rate']:.4f}",
                             f"{fs['draw_rate']:.4f}", f"{fs['loss_rate']:.4f}"])
    print(f"  CSV saved: {_csv(f'{game_name}_ql_sweep.csv')}")
    return sweep


def run_dqn_param_sweep(game_name="ttt", episodes=15_000):
    _print_header(f"Phase 4b: DQN Hyperparameter Sweep ({game_name.upper()})")
    GameCls = TicTacToe if game_name == "ttt" else Connect4
    opponent = DefaultAgent() if game_name == "ttt" else RandomAgent()
    is_ttt = game_name == "ttt"
    state_size  = 9  if is_ttt else 42
    action_size = 9  if is_ttt else 7

    configs = [
        {"lr": 1e-3, "hidden": (64, 64),    "label": "lr=1e-3, 64x64"},
        {"lr": 1e-3, "hidden": (128, 128),  "label": "lr=1e-3, 128x128 (default)"},
        {"lr": 1e-4, "hidden": (128, 128),  "label": "lr=1e-4, 128x128"},
        {"lr": 1e-3, "hidden": (256, 256),  "label": "lr=1e-3, 256x256"},
    ]

    sweep = {}
    for cfg in configs:
        print(f"  {cfg['label']}…")
        agent = DQNAgent(player=1, state_size=state_size, action_size=action_size,
                          hidden_sizes=cfg["hidden"], lr=cfg["lr"],
                          game_name=game_name)
        trainer = Trainer(GameCls, agent, opponent,
                          n_episodes=episodes, swap_sides=True,
                          train_every=1, verbose=False)
        metrics = trainer.train()
        sweep[cfg["label"]] = metrics["win_history"]
        fs = final_stats(metrics["win_history"])
        print(f"    Final win: {fs['win_rate']:.1%}")

    plot_param_sweep(sweep, "lr / hidden_size",
                     f"DQN Param Sweep — {game_name.upper()}",
                     _fig(f"{game_name}_dqn_param_sweep.png"))

    labels = list(sweep.keys())
    final_wins = [final_stats(sweep[l])["win_rate"] for l in labels]
    plot_param_bar(labels, final_wins, "Config", "Final Win Rate",
                   f"DQN Final Win Rate by Config ({game_name.upper()})",
                   _fig(f"{game_name}_dqn_param_bar.png"))

    import csv
    with open(_csv(f"{game_name}_dqn_sweep.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "final_win_rate", "final_draw_rate", "final_loss_rate"])
        for l in labels:
            fs = final_stats(sweep[l])
            writer.writerow([l, f"{fs['win_rate']:.4f}",
                             f"{fs['draw_rate']:.4f}", f"{fs['loss_rate']:.4f}"])
    print(f"  CSV saved: {_csv(f'{game_name}_dqn_sweep.csv')}")
    return sweep


# ===========================================================================
# Phase 5: Curriculum Training (random → default → self-play → minimax)
# ===========================================================================

def run_curriculum_experiment(
    game_name="c4",
    total_episodes=120_000,
    eval_games=100,
    minimax_depth=3,
    verbose=True,
):
    _print_header(f"Phase 5: Curriculum Training — {game_name.upper()}")
    GameCls = TicTacToe if game_name == "ttt" else Connect4
    is_c4   = game_name == "c4"
    state_size  = 42 if is_c4 else 9
    action_size = 7  if is_c4 else 9

    # Build a fresh DQN agent
    agent = DQNAgent(
        player=1,
        state_size=state_size, action_size=action_size,
        hidden_sizes=(256, 256, 128) if is_c4 else (128, 128),
        lr=5e-4, gamma=0.99,
        epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9997,
        batch_size=128, memory_size=100_000,
        target_update_freq=500, game_name=game_name,
    )

    stages = [
        (RandomAgent(),                                  "Random"),
        (DefaultAgent(),                                 "Default"),
        (None,                                           "self"),   # self-play snapshot
        (MinimaxABAgent(player=-1, depth_limit=minimax_depth), "Minimax+AB"),
    ]

    ct = CurriculumTrainer(
        GameCls, agent, stages,
        n_episodes=total_episodes,
        swap_sides=True, train_every=1,
        verbose=verbose, print_every=1000,
    )
    result = ct.train()

    # Plot curriculum training curve with stage boundaries
    plot_curriculum_training(
        result["stage_histories"],
        f"Curriculum Training — {game_name.upper()} (Random→Default→Self→Minimax)",
        _fig(f"{game_name}_curriculum.png"),
    )

    # Save combined CSV
    _save_training_csv(result["win_history"], result["loss_history"],
                       _csv(f"{game_name}_curriculum_training.csv"))

    # Evaluate trained agent vs default
    ev = Evaluator(GameCls, n_games=eval_games, swap_sides=True, verbose=verbose)
    stats = ev.evaluate(agent, DefaultAgent())
    print(f"\n  Curriculum DQN vs Default:")
    _print_results(stats)
    _save_eval_csv({"Curriculum-DQN": stats}, _csv(f"{game_name}_curriculum_eval.csv"))

    agent.save(_model(f"{game_name}_dqn_curriculum.pkl"))
    return {"agent": agent, "stage_histories": result["stage_histories"], "eval": stats}


# ===========================================================================
# Phase 6: Move-Limit vs Depth-Limit Comparison (Connect 4)
# ===========================================================================

def run_move_limit_experiment(
    move_limits=(10, 16, 22, 42),
    depth=4,
    games_per_config=60,
    verbose=True,
):
    _print_header("Phase 6: Move-Limit vs Depth-Limit Comparison — Connect 4")

    # --- Move-limit: how does limiting total moves (forced draw) affect agent ---
    print("\n  Move-limit analysis: Minimax+AB (depth=4) vs Default")
    move_results = {}
    for ml in move_limits:
        label = f"limit={ml}" if ml < 42 else "no limit"
        # Factory that returns a game with this move limit
        factory = lambda lim=ml: Connect4(move_limit=lim)
        ev = Evaluator(factory, n_games=games_per_config, swap_sides=True, verbose=verbose)
        agent = MinimaxABAgent(player=1, depth_limit=depth)
        stats = ev.evaluate(agent, DefaultAgent())
        move_results[label] = stats
        print(f"  {label:12s}: win={stats['win_rate']:.0%}  "
              f"draw={stats['draw_rate']:.0%}  "
              f"avg_len={stats['avg_game_length']:.1f}")

    plot_move_limit_comparison(
        move_results,
        "Connect 4: Effect of Move Limit on Minimax+AB (depth=4) vs Default",
        _fig("c4_move_limit.png"),
    )

    import csv
    with open(_csv("c4_move_limit.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["move_limit", "win_rate", "draw_rate", "loss_rate", "avg_game_length"])
        for label, stats in move_results.items():
            writer.writerow([label,
                             f"{stats['win_rate']:.4f}",
                             f"{stats['draw_rate']:.4f}",
                             f"{stats['loss_rate']:.4f}",
                             f"{stats['avg_game_length']:.2f}"])
    print(f"  CSV saved: {_csv('c4_move_limit.csv')}")
    return move_results


# ===========================================================================
# Master runner
# ===========================================================================

def run_all(
    ttt_ql_eps=60_000,
    ttt_dqn_eps=40_000,
    c4_ql_eps=100_000,
    c4_dqn_eps=60_000,
    eval_games_ttt=500,
    eval_games_c4=200,
    minimax_depth=4,
    depths=(2, 3, 4, 5, 6),
    sweep_eps=15_000,
    verbose=True,
    skip_scalability=True,
):
    """
    Run all experiments end-to-end.
    Returns a dict of all results for use in the notebook.
    """
    all_results = {}

    if not skip_scalability:
        all_results["scalability"] = run_scalability_test(time_limit=300)

    all_results["ttt"] = run_ttt_experiments(
        ql_episodes=ttt_ql_eps,
        dqn_episodes=ttt_dqn_eps,
        eval_games=eval_games_ttt,
        verbose=verbose,
    )

    all_results["depth"] = run_depth_analysis(depths=depths)

    all_results["c4"] = run_c4_experiments(
        ql_episodes=c4_ql_eps,
        dqn_episodes=c4_dqn_eps,
        eval_games=eval_games_c4,
        minimax_depth=minimax_depth,
        verbose=verbose,
    )

    all_results["ql_sweep_ttt"]  = run_ql_param_sweep("ttt",  episodes=sweep_eps)
    all_results["dqn_sweep_ttt"] = run_dqn_param_sweep("ttt", episodes=sweep_eps)

    all_results["curriculum"] = run_curriculum_experiment(
        n_episodes=100_000, eval_games=eval_games_c4, verbose=verbose
    )

    all_results["move_limit"] = run_move_limit_experiment(
        move_limits=(10, 16, 22, 42), depth=minimax_depth,
        games_per_config=eval_games_c4, verbose=verbose,
    )

    _print_header("ALL EXPERIMENTS COMPLETE")
    print(f"\n  Figures saved to: {os.path.abspath(FIG_DIR)}")
    print(f"  Models saved to:  {os.path.abspath(MODEL_DIR)}")
    return all_results


# ===========================================================================
# CLI entry point
# ===========================================================================

def _ask_int(prompt, default):
    try:
        val = input(f"  {prompt} [{default}]: ").strip()
        return int(val) if val else default
    except (ValueError, EOFError):
        return default


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CS7IS2 A3 — Experiment Runner")
    print("=" * 60)
    print("  Phases:")
    print("    1  ttt         — Train + evaluate on Tic Tac Toe")
    print("    2  depth       — Connect 4 depth-limit analysis")
    print("    3  c4          — Train + evaluate on Connect 4")
    print("    4  sweep       — Hyperparameter sweeps (TTT)")
    print("    5  curriculum  — Curriculum training (random→default→self→minimax)")
    print("    6  movelimit   — Move-limit vs depth-limit comparison")
    print("    7  scalability — C4 full-depth feasibility test")
    print("    8  all         — Run everything (1→7)")
    print()

    try:
        phase_input = input("  Choose phase (1-8 or name) [8]: ").strip().lower()
    except EOFError:
        phase_input = "8"

    phase_map = {
        "1": "ttt", "2": "depth", "3": "c4", "4": "sweep",
        "5": "curriculum", "6": "movelimit",
        "7": "scalability", "8": "all", "": "all",
    }
    phase = phase_map.get(phase_input, phase_input)

    if phase == "ttt":
        ql_eps  = _ask_int("Q-Learning episodes", 60_000)
        dqn_eps = _ask_int("DQN episodes", 40_000)
        ev_games = _ask_int("Eval games vs Default", 500)
        run_ttt_experiments(ql_eps, dqn_eps, ev_games, verbose=True)

    elif phase == "depth":
        max_d = _ask_int("Max depth to test (min=2)", 6)
        games = _ask_int("Games per depth", 50)
        run_depth_analysis(depths=tuple(range(2, max_d + 1)), games_per_depth=games)

    elif phase == "c4":
        ql_eps   = _ask_int("Q-Learning episodes", 100_000)
        dqn_eps  = _ask_int("DQN episodes", 60_000)
        ev_games = _ask_int("Eval games vs Default", 200)
        depth    = _ask_int("Minimax depth", 4)
        run_c4_experiments(ql_eps, dqn_eps, ev_games, depth, verbose=True)

    elif phase == "sweep":
        eps = _ask_int("Episodes per config", 20_000)
        run_ql_param_sweep("ttt", eps)
        run_dqn_param_sweep("ttt", eps)

    elif phase == "curriculum":
        eps = _ask_int("Total curriculum episodes", 100_000)
        ev_games = _ask_int("Eval games", 200)
        run_curriculum_experiment(n_episodes=eps, eval_games=ev_games, verbose=True)

    elif phase == "movelimit":
        depth = _ask_int("Minimax depth", 4)
        games = _ask_int("Games per config", 60)
        run_move_limit_experiment(depth=depth, games_per_config=games, verbose=True)

    elif phase == "scalability":
        t = _ask_int("Time limit in seconds", 300)
        run_scalability_test(time_limit=t)

    else:  # all
        print("\n  Configure experiments (press Enter to use defaults):\n")
        ttt_ql  = _ask_int("TTT Q-Learning episodes", 60_000)
        ttt_dqn = _ask_int("TTT DQN episodes", 40_000)
        c4_ql   = _ask_int("C4 Q-Learning episodes", 100_000)
        c4_dqn  = _ask_int("C4 DQN episodes", 60_000)
        ev_ttt  = _ask_int("TTT eval games", 500)
        ev_c4   = _ask_int("C4 eval games", 200)
        depth   = _ask_int("C4 minimax depth", 4)
        sw_eps  = _ask_int("Sweep episodes per config", 20_000)
        run_all(
            ttt_ql_eps=ttt_ql,
            ttt_dqn_eps=ttt_dqn,
            c4_ql_eps=c4_ql,
            c4_dqn_eps=c4_dqn,
            eval_games_ttt=ev_ttt,
            eval_games_c4=ev_c4,
            minimax_depth=depth,
            sweep_eps=sw_eps,
            verbose=True,
        )
