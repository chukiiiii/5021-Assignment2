# Super Tic-Tac-Toe Assignment

## Structure

- `super_tictactoe.py`: game rules and environment.
- `agents.py`: reusable agents: random, heuristic, MCTS, Q-table, DQN, human.
- `match.py`: reusable game runner and match statistics.
- `evaluate_agents.py`: agent-vs-agent evaluation with JSON/CSV/HTML reports.
- `play_game.py`: terminal replay or human-vs-agent play.
- `game_ui.py`: local browser UI.
- `train_q_learning.py`: tabular Q-learning baseline.
- `dqn_model.py`: DQN model and tensor helpers.
- `train_dqn.py`: DQN training script.
- `test_*.py`: unit and smoke tests.
- `results/checkpoints/`: trained Q-table and DQN checkpoints.
- `results/history/`: reward history CSV/HTML files.
- `results/evaluations/`: evaluation JSON/CSV/HTML reports.
- `HANDOFF.md`: full handoff log for the next agent.
- `report.md`: assignment report draft.
- `EXPERIMENT_PLAN.md`: experiment roadmap.

## Common Commands

Run tests:

```bash
python3 -m unittest -v
```

Start the browser UI:

```bash
python3 game_ui.py --host 127.0.0.1 --port 8765
```

Evaluate an agent:

```bash
python3 evaluate_agents.py --agent-a dqn:results/checkpoints/dqn_stochastic_smoke.pt --agent-b random --games 10 --progress-every 5 --json-output results/evaluations/eval_example.json --csv-output results/evaluations/eval_example.csv --html-output results/evaluations/eval_example.html
```

Train a short DQN smoke run:

```bash
python3 train_dqn.py --episodes 30 --eval-games 10 --output results/checkpoints/dqn_checkpoint.pt --history-csv results/history/dqn_history.csv --history-html results/history/dqn_history.html
```
