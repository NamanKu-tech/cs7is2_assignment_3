CS7IS2 - Artificial Intelligence - Assignment 3
Naman Kumar | 25340053
Trinity College Dublin

================================================================================
REQUIREMENTS
================================================================================

Python 3.10 - 3.12 (PyTorch does not support Python 3.14+)
Install dependencies:

    pip install -r requirements.txt


================================================================================
PROJECT STRUCTURE
================================================================================

games/          - TicTacToe and Connect4 game environments
agents/         - RandomAgent, DefaultAgent, MinimaxAgent, MinimaxABAgent,
                  QLearningAgent, DQNAgent
training/       - Trainer (RL training loop) and Evaluator
utils/          - Plotting and metrics helpers
experiments.py  - All experiments (training, evaluation, sweeps)
main.py         - Interactive CLI (play, demo, train, evaluate)
results/        - Saved figures and model weights (generated on run)
docs/           - LaTeX report source


================================================================================
RUNNING THE EXPERIMENTS (generates all figures and results for the report)
================================================================================

Run everything at once:
    python experiments.py

Run individual phases:
    python experiments.py --phase ttt        # Tic Tac Toe only
    python experiments.py --phase c4         # Connect 4 only
    python experiments.py --phase depth      # C4 depth-limit analysis
    python experiments.py --phase sweep      # Hyperparameter sweeps

Run C4 scalability test (runs for 5 minutes, terminates early by design):
    python -c "from experiments import run_scalability_test; run_scalability_test(300)"


================================================================================
INTERACTIVE DEMO COMMANDS (for video demo)
================================================================================

Play Tic Tac Toe against Minimax (as Player 1):
    python main.py play --game ttt --agent minimax

Play Tic Tac Toe against Alpha-Beta Minimax:
    python main.py play --game ttt --agent alphabeta

Play Tic Tac Toe against trained Q-Learning agent:
    python main.py play --game ttt --agent ql --model-path results/models/ttt_ql.pkl

Play Tic Tac Toe against trained DQN agent:
    python main.py play --game ttt --agent dqn --model-path results/models/ttt_dqn.pth

Play Connect 4 against Alpha-Beta Minimax (depth 4):
    python main.py play --game c4 --agent alphabeta --depth 4

Play Connect 4 against trained DQN:
    python main.py play --game c4 --agent dqn --model-path results/models/c4_dqn.pth

Watch two agents play Tic Tac Toe (demo mode):
    python main.py demo --game ttt --agent1 alphabeta --agent2 ql --model-path results/models/ttt_ql.pkl

Watch two agents play Connect 4 (demo mode):
    python main.py demo --game c4 --agent1 alphabeta --agent2 dqn --depth 4 --model-path results/models/c4_dqn.pth

Train Q-Learning on Tic Tac Toe:
    python main.py train --game ttt --agent ql --episodes 30000

Train DQN on Connect 4:
    python main.py train --game c4 --agent dqn --episodes 30000

Evaluate Alpha-Beta vs Default (100 games, Connect 4, depth 4):
    python main.py evaluate --game c4 --agent1 alphabeta --agent2 default --depth 4 --n-games 100


================================================================================
ALGORITHMS IMPLEMENTED
================================================================================

1. Minimax (no pruning)        - agents/minimax_agent.py  MinimaxAgent
2. Minimax + Alpha-Beta        - agents/minimax_agent.py  MinimaxABAgent
3. Q-Learning (tabular)        - agents/qlearning_agent.py
4. DQN (deep Q-network)        - agents/dqn_agent.py
5. Default opponent            - agents/default_agent.py
6. Random agent (baseline)     - agents/random_agent.py


================================================================================
NOTES
================================================================================

- Full-depth Minimax is only feasible for Tic Tac Toe.
  For Connect 4, depth-limited search (default depth=4) is used.
- RL agents for Connect 4 train against a random opponent (as per spec).
- Saved models are not included in the zip due to size; regenerate by running
  experiments.py or the train command above.
- Figures are saved to results/figures/ after running experiments.py.
