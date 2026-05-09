# Super Tic-Tac-Toe Reinforcement Learning Report

## 1. Problem Definition

This assignment asks us to train a reinforcement-learning agent to play a stochastic variant of tic-tac-toe called super tic-tac-toe.

The board is a triangle made from six 4x4 square regions. I model it as a 12x12 grid with invalid cells masked out:

- Level 1: rows 0-3, columns 4-7.
- Level 2: rows 4-7, columns 2-9.
- Level 3: rows 8-11, columns 0-11.

This gives 96 playable cells. Player one uses X and player two uses O.

On each turn, a player chooses an empty valid cell. The move is stochastic:

- With probability 1/2, the mark is placed on the chosen cell.
- Otherwise, one of the eight adjacent coordinates is selected with probability 1/16 each.
- If the redirected coordinate is outside the board or already occupied, the move is forfeited.

The implemented win conditions are:

- Four contiguous marks in a row.
- Four contiguous marks in a column, with the four cells spanning at least two levels.
- Five contiguous marks along either diagonal direction.

The column rule is the main ambiguity in the problem statement. I interpret "at least one move must be in a different level" as requiring a vertical four-in-a-column line to cross at least one level boundary.

## 2. Environment Design

The environment is implemented in `super_tictactoe.py`.

The environment provides:

- `reset()`
- `state()`
- `canonical_state()`
- `available_actions()`
- `step(action)`
- `render()`

The action space has 96 discrete actions, one for each playable cell. Invalid cells are not actions. Occupied cells are removed from `available_actions()`.

For learning agents, `canonical_state()` returns the board from the current player's perspective. This allows both players to share one policy or value function.

## 3. Baselines and Agents

### Random Agent

The random baseline chooses uniformly from currently available actions. It is useful as the lowest comparison point and as a smoke test for match evaluation.

### Tabular Q-Learning

The first learning baseline is a tabular Q-learning agent in `train_q_learning.py`.

It uses:

- Self-play.
- Canonical state from the current player's perspective.
- A shared Q-table for both players.
- A simple zero-sum update where the next player's best value is subtracted.

This baseline is easy to understand and verifies that the environment can support RL training. However, it is not expected to be a strong final solution because the 96-cell board creates a very large state space.

### Heuristic Agent

The next optimization step adds a heuristic agent in `agents.py`.

The heuristic agent is not trained. It evaluates each intended action by averaging over the actual stochastic placement outcomes:

- The chosen cell has total probability 8/16.
- Each adjacent coordinate has probability 1/16.
- Invalid or occupied redirected cells are treated as forfeits.

The scoring function prioritizes:

- Immediate wins.
- Blocking opponent threats through line-potential scoring.
- Creating promising lines.
- Playing near friendly stones.
- Avoiding high-risk forfeits unless the tactical gain is large.

This agent is intended to become:

- A stronger baseline than random.
- A rollout policy for future MCTS.
- A data generator for possible SFT pretraining.

## 4. Evaluation Framework

Reusable evaluation code was added before the heuristic agent:

- `agents.py`: common agent interface and built-in agents.
- `match.py`: single-game and multi-game match runner.
- `evaluate_agents.py`: command-line win-rate evaluation with terminal, JSON, CSV, and HTML output.
- `play_game.py`: single-game display and human-vs-agent play.

Every future method should expose the same shape:

```python
select_action(env) -> int
```

This makes PPO, MCTS, DQN, or SFT-initialized policies directly reusable in the existing evaluator.

## 5. Current Experiments

### Initial Tabular Q-Learning Smoke Test

The earlier stochastic Q-learning smoke test used 200 training episodes and 50 evaluation games.

Observed result:

| Agent | Opponent | Eval env | Win rate | Draw rate | Avg turns |
| --- | --- | --- | ---: | ---: | ---: |
| Tabular Q | Random | Stochastic | 0.44 | 0.00 | 60.26 |

This shows that the training loop works, but the agent is not yet strong.

### Heuristic Baseline

The heuristic baseline is now implemented. Its expected role is to provide a stronger, interpretable comparison point and to support future MCTS and SFT work.

The exact win-rate results should be updated after running:

I evaluated the heuristic agent against both random play and the current tabular Q-learning checkpoint in the stochastic environment.

Commands:

```bash
python3 evaluate_agents.py --agent-a heuristic --agent-b random --games 100 --seed 21 --json-output eval_heuristic_random.json --csv-output eval_heuristic_random.csv --html-output eval_heuristic_random.html
python3 evaluate_agents.py --agent-a heuristic --agent-b qtable:q_table.json --games 100 --seed 31 --json-output eval_heuristic_qtable.json --csv-output eval_heuristic_qtable.csv --html-output eval_heuristic_qtable.html
```

Observed results:

| Agent | Opponent | Eval env | Win rate | Draw rate | Avg turns | Avg forfeits | Avg redirects |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Heuristic | Random | Stochastic | 1.00 | 0.00 | 11.70 | 1.14 | 4.67 |
| Heuristic | Tabular Q | Stochastic | 1.00 | 0.00 | 12.14 | 1.30 | 4.80 |

The heuristic baseline is much stronger than the current tabular learner. It is therefore a useful baseline for future comparisons and a good rollout policy candidate for MCTS.

One caveat is that these are still smoke-test-sized evaluations. Final reporting should repeat the experiment across multiple seeds and larger game counts.

## 6. Next Steps

The next recommended steps are:

1. Run longer heuristic-vs-random and heuristic-vs-Q evaluations across multiple seeds.
2. Add MCTS using the heuristic as the rollout policy.
3. Add a neural method such as DQN or PPO.
4. Optionally generate SFT data from heuristic or MCTS games, pretrain a policy network, then fine-tune with RL.

## 7. Limitations

The current implementation still has several limitations:

- The column win interpretation should be confirmed against the course expectation.
- Tabular Q-learning is too small for the full game.
- The heuristic is hand-designed and may overfit the current interpretation of the rules.
- No neural network method has been trained yet.
- No MCTS has been implemented yet.
