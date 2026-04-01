"""
experiments.py — All experiment runners for CS7IS2 Assignment 3.

Run everything:   python experiments.py
Run one section:  python experiments.py --phase ttt
"""

import os
import sys
import time
import argparse
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
from training.trainer import Trainer
from training.evaluator import Evaluator
from utils.plotting import (
    setup_style, plot_training_curve, plot_win_rate_bar,
    plot_heatmap, plot_param_sweep, plot_param_bar,
    plot_dqn_loss, plot_depth_analysis, plot_node_comparison,
)
from utils.metrics import final_stats

setup_style()

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
FIG_DIR   = "results/figures"
MODEL_DIR = "results/models"
os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ===========================================================================
# Helper
# ===========================================================================

def _fig(name):
    return os.path.join(FIG_DIR, name)

def _model(name):
    return os.path.join(MODEL_DIR, name)

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
        nodes_total = 0
        moves_completed = 0

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
    ql_episodes=30_000,
    dqn_episodes=20_000,
    eval_games=200,
    verbose=True,
):
    _print_header("Phase 1a: Tic Tac Toe — Training RL Agents")

    default_opp = DefaultAgent()
    random_opp  = RandomAgent()

    # --- Q-Learning ---
    print("\n[TTT] Training Q-Learning vs Default Opponent…")
    ttt_ql = QLearningAgent(player=1, alpha=0.1, gamma=0.99, epsilon=1.0,
                             epsilon_min=0.05, epsilon_decay=0.9995)
    trainer = Trainer(TicTacToe, ttt_ql, default_opp,
                      n_episodes=ql_episodes, swap_sides=True, verbose=verbose)
    ql_metrics = trainer.train()
    plot_training_curve(ql_metrics["win_history"],
                        "Q-Learning Training — Tic Tac Toe vs Default",
                        _fig("ttt_ql_training.png"))
    ttt_ql.save(_model("ttt_ql.pkl"))
    print(f"  Final stats: {final_stats(ql_metrics['win_history'])}")

    # --- DQN ---
    print("\n[TTT] Training DQN vs Default Opponent…")
    ttt_dqn = DQNAgent(player=1, state_size=9, action_size=9,
                        hidden_sizes=(128, 128), lr=1e-3, gamma=0.99,
                        epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                        batch_size=64, memory_size=20_000,
                        target_update_freq=200, game_name="tictactoe")
    trainer = Trainer(TicTacToe, ttt_dqn, default_opp,
                      n_episodes=dqn_episodes, swap_sides=True,
                      train_every=1, verbose=verbose)
    dqn_metrics = trainer.train()
    plot_training_curve(dqn_metrics["win_history"],
                        "DQN Training — Tic Tac Toe vs Default",
                        _fig("ttt_dqn_training.png"))
    if dqn_metrics["loss_history"]:
        plot_dqn_loss(dqn_metrics["loss_history"],
                      "DQN Loss — Tic Tac Toe",
                      _fig("ttt_dqn_loss.png"))
    ttt_dqn.save(_model("ttt_dqn.pth"))
    print(f"  Final stats: {final_stats(dqn_metrics['win_history'])}")

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

    plot_win_rate_bar(vs_default, "Tic Tac Toe", _fig("ttt_vs_default.png"))

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

    # --- Round-robin ---
    _print_header("Phase 1d: Tic Tac Toe — Round Robin")
    rr_ev = Evaluator(TicTacToe, n_games=100, swap_sides=True, verbose=False)
    matrix, labels, rr_results = rr_ev.round_robin(agents)
    np.fill_diagonal(matrix, 0)  # zero out self-play diagonal
    plot_heatmap(matrix, labels, "Tic Tac Toe: Head-to-Head Win Rates",
                 _fig("ttt_round_robin.png"))
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
    ql_episodes=50_000,
    dqn_episodes=30_000,
    eval_games=100,
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
    plot_training_curve(ql_metrics["win_history"],
                        "Q-Learning Training — Connect 4 vs Random",
                        _fig("c4_ql_training.png"), window=1000)
    c4_ql.save(_model("c4_ql.pkl"))
    print(f"  Final stats: {final_stats(ql_metrics['win_history'])}")

    # DQN (GPU if available)
    print("\n[C4] Training DQN vs Random Opponent…")
    c4_dqn = DQNAgent(player=1, state_size=42, action_size=7,
                       hidden_sizes=(256, 128, 64), lr=5e-4, gamma=0.99,
                       epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9998,
                       batch_size=128, memory_size=50_000,
                       target_update_freq=500, game_name="connect4")
    trainer = Trainer(Connect4, c4_dqn, random_opp,
                      n_episodes=dqn_episodes, swap_sides=True,
                      train_every=1, verbose=verbose)
    dqn_metrics = trainer.train()
    plot_training_curve(dqn_metrics["win_history"],
                        "DQN Training — Connect 4 vs Random",
                        _fig("c4_dqn_training.png"), window=1000)
    if dqn_metrics["loss_history"]:
        plot_dqn_loss(dqn_metrics["loss_history"],
                      "DQN Loss — Connect 4", _fig("c4_dqn_loss.png"))
    c4_dqn.save(_model("c4_dqn.pth"))
    print(f"  Final stats: {final_stats(dqn_metrics['win_history'])}")

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

    plot_win_rate_bar(vs_default, "Connect 4", _fig("c4_vs_default.png"))

    # --- Round-robin ---
    _print_header("Phase 2c: Connect 4 — Round Robin")
    rr_ev = Evaluator(Connect4, n_games=60, swap_sides=True, verbose=False)
    matrix, labels, rr_results = rr_ev.round_robin(agents)
    np.fill_diagonal(matrix, 0)
    plot_heatmap(matrix, labels, "Connect 4: Head-to-Head Win Rates",
                 _fig("c4_round_robin.png"))

    return {
        "ql": c4_ql, "dqn": c4_dqn,
        "vs_default": vs_default,
        "rr_matrix": matrix, "rr_labels": labels,
    }


# ===========================================================================
# Phase 3: Depth limit analysis (Connect 4)
# ===========================================================================

def run_depth_analysis(depths=(2, 3, 4, 5, 6), games_per_depth=30):
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
    return depth_results


# ===========================================================================
# Phase 4: Hyperparameter sweeps
# ===========================================================================

def run_ql_param_sweep(game_name="ttt", episodes=10_000):
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
    return sweep


def run_dqn_param_sweep(game_name="ttt", episodes=8_000):
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
    return sweep


# ===========================================================================
# Master runner
# ===========================================================================

def run_all(
    ttt_ql_eps=30_000,
    ttt_dqn_eps=20_000,
    c4_ql_eps=50_000,
    c4_dqn_eps=30_000,
    eval_games_ttt=200,
    eval_games_c4=100,
    minimax_depth=4,
    depths=(2, 3, 4, 5),
    sweep_eps=8_000,
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

    _print_header("ALL EXPERIMENTS COMPLETE")
    print(f"\n  Figures saved to: {os.path.abspath(FIG_DIR)}")
    print(f"  Models saved to:  {os.path.abspath(MODEL_DIR)}")
    return all_results


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS7IS2 A3 Experiments")
    parser.add_argument("--phase", choices=["ttt", "c4", "depth", "sweep", "scalability", "all"],
                        default="all")
    parser.add_argument("--ttt-ql-eps",  type=int, default=30_000)
    parser.add_argument("--ttt-dqn-eps", type=int, default=20_000)
    parser.add_argument("--c4-ql-eps",   type=int, default=50_000)
    parser.add_argument("--c4-dqn-eps",  type=int, default=30_000)
    parser.add_argument("--eval-games",  type=int, default=200)
    parser.add_argument("--depth",       type=int, default=4)
    parser.add_argument("--sweep-eps",   type=int, default=8_000)
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.phase == "scalability":
        run_scalability_test(time_limit=300)
    elif args.phase == "ttt":
        run_ttt_experiments(args.ttt_ql_eps, args.ttt_dqn_eps, args.eval_games, verbose)
    elif args.phase == "c4":
        run_c4_experiments(args.c4_ql_eps, args.c4_dqn_eps, args.eval_games,
                           args.depth, verbose)
    elif args.phase == "depth":
        run_depth_analysis()
    elif args.phase == "sweep":
        run_ql_param_sweep("ttt", args.sweep_eps)
        run_dqn_param_sweep("ttt", args.sweep_eps)
    else:
        run_all(
            ttt_ql_eps=args.ttt_ql_eps,
            ttt_dqn_eps=args.ttt_dqn_eps,
            c4_ql_eps=args.c4_ql_eps,
            c4_dqn_eps=args.c4_dqn_eps,
            eval_games_ttt=args.eval_games,
            eval_games_c4=args.eval_games // 2,
            minimax_depth=args.depth,
            sweep_eps=args.sweep_eps,
            verbose=verbose,
        )
