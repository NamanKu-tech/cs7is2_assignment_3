"""
main.py — CLI entry point for CS7IS2 Assignment 3.

Commands:
  play     Interactive human vs agent game
  train    Train a RL agent and save weights
  evaluate Evaluate agents and print results
  demo     Quick demo: watch two agents play with board display
"""

import argparse
import sys
import time
import os

from games.tictactoe import TicTacToe
from games.connect4 import Connect4
from agents.random_agent import RandomAgent
from agents.default_agent import DefaultAgent
from agents.minimax_agent import MinimaxAgent, MinimaxABAgent
from agents.qlearning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from training.trainer import Trainer
from training.evaluator import Evaluator
from utils.metrics import final_stats


# ------------------------------------------------------------------ #
# Agent factory                                                        #
# ------------------------------------------------------------------ #

def make_agent(name, player, game_name, depth=None, model_path=None,
               alpha=0.1, gamma=0.99, lr=1e-3, hidden=(128, 128)):
    state_size  = 9  if game_name == "ttt" else 42
    action_size = 9  if game_name == "ttt" else 7

    if name == "random":
        return RandomAgent(player=player)
    elif name == "default":
        return DefaultAgent(player=player)
    elif name == "minimax":
        return MinimaxAgent(player=player, depth_limit=depth)
    elif name == "alphabeta":
        return MinimaxABAgent(player=player, depth_limit=depth)
    elif name == "ql":
        agent = QLearningAgent(player=player, alpha=alpha, gamma=gamma)
        if model_path and os.path.exists(model_path):
            agent.load(model_path)
            print(f"  Loaded Q-table from {model_path}  "
                  f"(states: {agent.q_table_size})")
        return agent
    elif name == "dqn":
        agent = DQNAgent(player=player, state_size=state_size,
                          action_size=action_size, hidden_sizes=hidden,
                          lr=lr, game_name=game_name)
        if model_path and os.path.exists(model_path):
            agent.load(model_path)
            print(f"  Loaded DQN weights from {model_path}")
        return agent
    else:
        print(f"Unknown agent: {name}")
        sys.exit(1)


def make_game(game_name):
    return TicTacToe if game_name == "ttt" else Connect4


# ------------------------------------------------------------------ #
# Human play                                                           #
# ------------------------------------------------------------------ #

def play_human(args):
    GameCls = make_game(args.game)
    agent   = make_agent(args.agent, player=-1 if args.human_player == "1" else 1,
                          game_name=args.game, depth=args.depth,
                          model_path=args.model_path)
    human_player = 1 if args.human_player == "1" else -1
    agent.player = -human_player

    game = GameCls()
    print(f"\n  You are {'X (Player 1)' if human_player == 1 else 'O (Player -1)'}")
    print(f"  Agent: {agent.name}\n")
    game.print_board(show_indices=(args.game == "ttt"))

    while not game.is_terminal():
        if game.current_player == human_player:
            # Human move
            valid = game.get_valid_actions()
            print(f"  Valid actions: {valid}")
            while True:
                try:
                    action = int(input("  Your move: ").strip())
                    if action in valid:
                        break
                    print(f"  Invalid — choose from {valid}")
                except (ValueError, KeyboardInterrupt):
                    print("\n  Exiting.")
                    return
            game.make_move(action)
        else:
            # Agent move
            print(f"  {agent.name} is thinking…")
            t0 = time.perf_counter()
            action = agent.choose_move(game, training=False)
            ms = (time.perf_counter() - t0) * 1000
            print(f"  Agent plays: {action}  ({ms:.1f}ms)")
            game.make_move(action)

        game.print_board()

    winner = game.get_winner()
    if winner == human_player:
        print("  You win!")
    elif winner == 0:
        print("  Draw!")
    else:
        print(f"  {agent.name} wins!")


# ------------------------------------------------------------------ #
# Agent demo (watch two agents play)                                   #
# ------------------------------------------------------------------ #

def demo(args):
    GameCls = make_game(args.game)
    agent1  = make_agent(args.agent1, player=1,  game_name=args.game,
                          depth=args.depth, model_path=args.model_path)
    agent2  = make_agent(args.agent2, player=-1, game_name=args.game,
                          depth=args.depth)

    game = GameCls()
    print(f"\n  {agent1.name} (X) vs {agent2.name} (O)\n")
    game.print_board()

    delay = args.delay
    while not game.is_terminal():
        agent = agent1 if game.current_player == 1 else agent2
        action = agent.choose_move(game, training=False)
        symbol = "X" if game.current_player == 1 else "O"
        print(f"  {agent.name} ({symbol}) → action {action}")
        game.make_move(action)
        game.print_board()
        if delay > 0:
            time.sleep(delay)

    winner = game.get_winner()
    if winner == 1:
        print(f"  {agent1.name} wins!")
    elif winner == -1:
        print(f"  {agent2.name} wins!")
    else:
        print("  Draw!")


# ------------------------------------------------------------------ #
# Training                                                             #
# ------------------------------------------------------------------ #

def train(args):
    GameCls  = make_game(args.game)
    agent    = make_agent(args.agent, player=1, game_name=args.game,
                           alpha=args.alpha, gamma=args.gamma,
                           lr=args.lr, hidden=tuple(args.hidden))
    opponent = make_agent(args.opponent, player=-1, game_name=args.game)

    print(f"\n  Training {agent.name} on {args.game.upper()} "
          f"vs {opponent.name} for {args.episodes:,} episodes…\n")

    trainer = Trainer(
        GameCls, agent, opponent,
        n_episodes=args.episodes,
        swap_sides=True,
        verbose=True,
        print_every=max(1000, args.episodes // 20),
    )
    metrics = trainer.train()
    fs = final_stats(metrics["win_history"])
    print(f"\n  Final (last 2k): win={fs['win_rate']:.1%}  "
          f"draw={fs['draw_rate']:.1%}  loss={fs['loss_rate']:.1%}")

    # Save
    save_path = args.save_path or f"results/models/{args.game}_{args.agent}.{'pkl' if args.agent == 'ql' else 'pth'}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"  Saved to {save_path}")


# ------------------------------------------------------------------ #
# Evaluation                                                           #
# ------------------------------------------------------------------ #

def evaluate(args):
    GameCls = make_game(args.game)
    a1 = make_agent(args.agent1, player=1,  game_name=args.game,
                     depth=args.depth, model_path=args.model_path)
    a2 = make_agent(args.agent2, player=-1, game_name=args.game,
                     depth=args.depth)

    ev = Evaluator(GameCls, n_games=args.n_games, swap_sides=True, verbose=True)
    stats = ev.evaluate(a1, a2)

    print(f"\n  {a1.name} vs {a2.name}  ({args.n_games} games)")
    print(f"  Win: {stats['win_rate']:.1%}  Draw: {stats['draw_rate']:.1%}  "
          f"Loss: {stats['loss_rate']:.1%}")
    print(f"  Avg game length: {stats['avg_game_length']:.1f} moves")
    print(f"  Avg move time:   {stats['avg_move_time_ms']:.2f}ms")
    if stats["avg_nodes"] > 0:
        print(f"  Avg nodes:       {stats['avg_nodes']:,.0f}")


# ------------------------------------------------------------------ #
# CLI parser                                                           #
# ------------------------------------------------------------------ #

def build_parser():
    parser = argparse.ArgumentParser(
        description="CS7IS2 A3 — Minimax & RL for Tic Tac Toe / Connect 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Play against minimax as Player 1:
  python main.py play --game ttt --agent minimax

  # Play Connect 4 against a trained DQN:
  python main.py play --game c4 --agent dqn --model-path results/models/c4_dqn.pth

  # Watch Minimax+AB vs Q-Learning:
  python main.py demo --game ttt --agent1 alphabeta --agent2 ql --delay 0.5

  # Train Q-Learning on TTT:
  python main.py train --game ttt --agent ql --episodes 30000

  # Evaluate alphabeta vs default (100 games):
  python main.py evaluate --game c4 --agent1 alphabeta --agent2 default --depth 4

  # Run all experiments:
  python experiments.py
        """,
    )
    sub = parser.add_subparsers(dest="command")

    # --- play ---
    p = sub.add_parser("play", help="Human vs agent interactive game")
    p.add_argument("--game",         choices=["ttt","c4"], required=True)
    p.add_argument("--agent",        choices=["random","default","minimax","alphabeta","ql","dqn"], default="default")
    p.add_argument("--human-player", choices=["1","2"], default="1")
    p.add_argument("--depth",        type=int, default=None, help="Depth limit for minimax (None=full)")
    p.add_argument("--model-path",   type=str, default=None)

    # --- demo ---
    d = sub.add_parser("demo", help="Watch two agents play")
    d.add_argument("--game",       choices=["ttt","c4"], required=True)
    d.add_argument("--agent1",     choices=["random","default","minimax","alphabeta","ql","dqn"], default="alphabeta")
    d.add_argument("--agent2",     choices=["random","default","minimax","alphabeta","ql","dqn"], default="default")
    d.add_argument("--depth",      type=int, default=4)
    d.add_argument("--model-path", type=str, default=None)
    d.add_argument("--delay",      type=float, default=0.3, help="Seconds between moves")

    # --- train ---
    t = sub.add_parser("train", help="Train a RL agent")
    t.add_argument("--game",      choices=["ttt","c4"], required=True)
    t.add_argument("--agent",     choices=["ql","dqn"], required=True)
    t.add_argument("--opponent",  choices=["random","default"], default="default")
    t.add_argument("--episodes",  type=int, default=30_000)
    t.add_argument("--save-path", type=str, default=None)
    t.add_argument("--alpha",     type=float, default=0.1,  help="Q-Learning alpha")
    t.add_argument("--gamma",     type=float, default=0.99, help="Discount factor")
    t.add_argument("--lr",        type=float, default=1e-3, help="DQN learning rate")
    t.add_argument("--hidden",    type=int,   nargs="+", default=[128,128], help="DQN hidden layer sizes")

    # --- evaluate ---
    e = sub.add_parser("evaluate", help="Evaluate two agents")
    e.add_argument("--game",       choices=["ttt","c4"], required=True)
    e.add_argument("--agent1",     choices=["random","default","minimax","alphabeta","ql","dqn"], default="alphabeta")
    e.add_argument("--agent2",     choices=["random","default","minimax","alphabeta","ql","dqn"], default="default")
    e.add_argument("--depth",      type=int, default=4)
    e.add_argument("--n-games",    type=int, default=100)
    e.add_argument("--model-path", type=str, default=None)

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "play":
        play_human(args)
    elif args.command == "demo":
        demo(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
