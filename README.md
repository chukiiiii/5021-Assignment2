# Super Tic-Tac-Toe Reinforcement Learning Project

## 📁 Project Structure

```
Assignment2/
├── src/                    # Source code
│   ├── env/               # Game environment & UI
│   │   ├── super_tictactoe.py   # Core game logic
│   │   ├── match.py             # Game runner & statistics
│   │   └── game_ui.py           # Browser-based UI
│   ├── agents/            # Agent implementations
│   │   ├── agents.py            # Random, heuristic, MCTS, Q-table, DQN, human agents
│   │   └── play_game.py         # Terminal replay & human-vs-agent
│   └── models/            # Deep learning models
│       ├── dqn_model.py         # DQN architecture
│       └── ppo_model.py         # PPO actor-critic model
├── train/                 # Training scripts
│   ├── train_q_learning.py      # Tabular Q-learning
│   ├── train_dqn.py             # DQN training
│   ├── train_dqn_multiseed.py   # Multi-seed DQN
│   ├── train_ppo.py             # PPO training
│   ├── train_ppo_multiseed.py   # Multi-seed PPO
│   ├── train_bc.py              # Behavior cloning
│   └── generate_expert_data.py  # Expert data generation
├── eval/                  # Evaluation & experiments
│   ├── evaluate_agents.py       # Agent vs-agent evaluation
│   └── run_experiments.py       # Experiment orchestration
├── tests/                 # Unit tests
│   ├── test_super_tictactoe.py
│   ├── test_match.py
│   ├── test_game_ui.py
│   ├── test_dqn.py
│   └── test_ppo.py
├── docs/                  # Documentation
│   ├── README.md                # Main project documentation
│   ├── HANDOFF.md               # Handoff notes
│   ├── EXPERIMENT_PLAN.md       # Experiment roadmap
│   └── report.md                # Assignment report
├── results/               # Experimental results
│   ├── checkpoints/     # Trained model checkpoints
│   ├── evaluations/     # Evaluation reports (JSON/HTML)
│   ├── history/         # Training history (CSV/HTML)
│   └── summary/         # Summary reports
└── utils/                 # Utility scripts (future use)
```

## 🚀 Quick Start

### Run Tests
```bash
python3 -m pytest tests/ -v
# or
python3 -m unittest discover -s tests -v
```

### Train Agents
```bash
# Q-Learning
python3 train/train_q_learning.py

# DQN
python3 train/train_dqn.py --episodes 100

# PPO
python3 train/train_ppo.py --episodes 100

# Behavior Cloning
python3 train/generate_expert_data.py
python3 train/train_bc.py
```

### Evaluate Agents
```bash
python3 eval/evaluate_agents.py --agent-a dqn:results/checkpoints/dqn.pt --agent-b random --games 20
```

### Run Experiments
```bash
python3 eval/run_experiments.py --preset full --seeds 1 2 3
```

### Play Game
```bash
# Terminal mode
python3 src/agents/play_game.py

# Browser UI
python3 src/env/game_ui.py --host 127.0.0.1 --port 8765
```

## 📋 Common Commands

See [docs/README.md](docs/README.md) for detailed command examples.

## 📊 Results

All experimental results are stored in the `results/` directory:
- `checkpoints/`: Model weights and Q-tables
- `evaluations/`: Agent performance metrics
- `history/`: Training curves and logs
- `summary/`: Aggregated experiment summaries
