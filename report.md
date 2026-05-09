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

### DQN Agent

I added a pure neural DQN baseline. Unlike the heuristic and MCTS agents, DQN does not use hand-written immediate-win or immediate-block rules.

The DQN input is a `3 x 12 x 12` tensor from the current player's perspective:

- Channel 0: current player's stones.
- Channel 1: opponent stones.
- Channel 2: empty legal cells.

The network outputs 96 Q-values, one for each playable action. Illegal occupied actions are masked during action selection. Training uses self-play with replay buffer, target network, and epsilon-greedy exploration.

The zero-sum target is:

```text
target = reward - gamma * max_a Q(next_state, a)
```

where `next_state` is viewed from the opponent's perspective. This mirrors the tabular Q-learning setup but replaces the table with a CNN.

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
python3 evaluate_agents.py --agent-a heuristic --agent-b random --games 100 --seed 21 --json-output results/evaluations/eval_heuristic_random.json --csv-output results/evaluations/eval_heuristic_random.csv --html-output results/evaluations/eval_heuristic_random.html
python3 evaluate_agents.py --agent-a heuristic --agent-b qtable:results/checkpoints/q_table.json --games 100 --seed 31 --json-output results/evaluations/eval_heuristic_qtable.json --csv-output results/evaluations/eval_heuristic_qtable.csv --html-output results/evaluations/eval_heuristic_qtable.html
```

Observed results:

| Agent | Opponent | Eval env | Win rate | Draw rate | Avg turns | Avg forfeits | Avg redirects |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Heuristic | Random | Stochastic | 1.00 | 0.00 | 11.70 | 1.14 | 4.67 |
| Heuristic | Tabular Q | Stochastic | 1.00 | 0.00 | 12.14 | 1.30 | 4.80 |

The heuristic baseline is much stronger than the current tabular learner. It is therefore a useful baseline for future comparisons and a good rollout policy candidate for MCTS.

One caveat is that these are still smoke-test-sized evaluations. Final reporting should repeat the experiment across multiple seeds and larger game counts.

## 6. MCTS and Reward Tracking

The next advanced agent is MCTS, implemented as `mcts` or `mcts:N` through the shared agent interface.

MCTS design:

- The root action is the intended move selected by the player.
- Stochastic placement is handled by sampling the real environment transition during simulations.
- Rollouts default to random actions for speed.
- The final value is from the root player's perspective:
  - win: `+1`
  - loss: `-1`
  - draw: `0`
  - depth-limited non-terminal state: bounded heuristic evaluation

This is a correctness-first MCTS implementation. Early smoke testing showed that larger settings, such as `mcts:20` over 20 stochastic games, can be slow when rollouts repeatedly call the heuristic policy. The default rollout depth was therefore reduced and the default rollout policy was changed to random actions for practical smoke testing.

A tiny command-line smoke test completed successfully:

```bash
python3 evaluate_agents.py --agent-a mcts:1 --agent-b random --games 1 --seed 61 --deterministic --json-output results/evaluations/eval_mcts1_random_smoke.json --csv-output results/evaluations/eval_mcts1_random_smoke.csv --html-output results/evaluations/eval_mcts1_random_smoke.html
```

Observed result:

| Agent | Opponent | Eval env | Games | Win rate | Avg reward | Avg turns |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| MCTS-1 | Random | Deterministic | 1 | 1.00 | 1.00 | 53.00 |

This one-game result only verifies the MCTS command path and reporting outputs. It is not a statistically meaningful strength estimate.

The longer `mcts:20` stochastic evaluation eventually completed:

```bash
python3 evaluate_agents.py --agent-a mcts:20 --agent-b random --games 20 --seed 51 --json-output results/evaluations/eval_mcts20_random.json --csv-output results/evaluations/eval_mcts20_random.csv --html-output results/evaluations/eval_mcts20_random.html
```

Observed result:

| Agent | Opponent | Eval env | Games | Win rate | Avg reward | Avg turns | Avg forfeits | Avg redirects |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MCTS-20 | Random | Stochastic | 20 | 0.40 | -0.20 | 57.60 | 10.80 | 17.30 |

This result is weaker than expected. The likely reason is that MCTS is currently using sampled stochastic transitions and a shallow, speed-oriented rollout policy. With only 20 simulations per move, the search is noisy and does not yet reliably outperform random play. The experiment is still useful because it confirms that the MCTS pipeline, stochastic simulation, reward output, and HTML/CSV/JSON reporting all work end-to-end.

### MCTS Optimization

The first MCTS version expanded actions almost uniformly from the full legal action set. This was inefficient because the board can have up to 96 legal actions, so a small simulation budget was spent mostly on poor or irrelevant moves. It also used root-player value at every node without explicitly accounting for the opponent's adversarial choices.

I optimized MCTS in three ways:

1. Candidate pruning:
   - Each node now ranks legal actions with a fast tactical prior.
   - MCTS expands only the top candidate actions, for example `mcts:20:6` means 20 simulations and top 6 candidate actions.
2. Adversarial tree selection:
   - Root-player nodes maximize root value.
   - Opponent nodes minimize root value.
3. Prior-guided selection:
   - The tactical prior is included in UCB-style tree selection and final root action selection.
   - This prevents low-simulation MCTS from drifting too far from strong tactical moves.

Optimized evaluation results:

| Agent | Opponent | Eval env | Games | Win rate | Avg reward | Avg turns | Avg forfeits | Avg redirects |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MCTS-20 top12 | Random | Stochastic | 20 | 1.00 | 1.00 | 15.80 | 1.70 | 6.55 |
| MCTS-20 top6 | Random | Stochastic | 20 | 1.00 | 1.00 | 16.00 | 2.05 | 6.45 |
| MCTS-20 top6 + prior | Random | Stochastic | 20 | 1.00 | 1.00 | 12.00 | 1.30 | 4.65 |
| MCTS-20 top6 + prior | Heuristic | Stochastic | 10 | 0.40 | -0.20 | 10.70 | 1.10 | 4.10 |
| MCTS-50 top6 + prior | Heuristic | Stochastic | 10 | 0.60 | 0.20 | 10.30 | 0.80 | 4.40 |

The optimized MCTS clearly improves over the first version. Against random, it reaches 100% win rate in these small evaluations and finishes games much faster. Against the stronger heuristic baseline, `mcts:50:6` wins 6 out of 10 games, suggesting that additional simulations can add value beyond the hand-designed heuristic.

These results are still small-sample experiments. Final evaluation should use more seeds and more games, but the direction is promising.

Reward tracking was also added:

- `evaluate_agents.py` now records per-game reward, cumulative reward, and rolling reward from agent A's perspective.
- The HTML match report includes a cumulative reward line chart.
- `train_q_learning.py` can now save reward history using `--history-csv` and `--history-html`.

Example training reward command:

```bash
python3 train_q_learning.py --episodes 60 --eval-games 20 --seed 17 --output results/checkpoints/q_table_reward_smoke.json --history-csv results/history/q_reward_history.csv --history-html results/history/q_reward_history.html
```

Observed Q-learning reward smoke test:

| Training run | Episodes | X win rate | O win rate | Eval learner win rate | Eval random win rate | Avg eval turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q reward smoke | 60 | 0.483 | 0.517 | 0.500 | 0.500 | 55.50 |

The reward curve is saved in `results/history/q_reward_history.html`.

### DQN Smoke Tests

DQN was trained in both deterministic and stochastic settings for short smoke tests. These runs are not intended as final results, but they verify the full neural RL loop.

Deterministic command:

```bash
python3 train_dqn.py --episodes 30 --eval-games 10 --seed 101 --batch-size 16 --train-after 32 --target-update 50 --output results/checkpoints/dqn_smoke.pt --history-csv results/history/dqn_smoke_history.csv --history-html results/history/dqn_smoke_history.html --deterministic
```

Stochastic command:

```bash
python3 train_dqn.py --episodes 30 --eval-games 10 --seed 103 --batch-size 16 --train-after 32 --target-update 50 --output results/checkpoints/dqn_stochastic_smoke.pt --history-csv results/history/dqn_stochastic_history.csv --history-html results/history/dqn_stochastic_history.html
```

Observed smoke results:

| Agent | Training env | Eval env | Games | Win rate vs random | Avg eval turns |
| --- | --- | --- | ---: | ---: | ---: |
| DQN smoke | Deterministic | Deterministic | 10 | 0.80 | 44.50 |
| DQN smoke | Stochastic | Stochastic | 10 | 0.60 | 51.90 |

The same checkpoints were also evaluated through the generic evaluator:

| Agent | Opponent | Eval env | Games | Win rate | Avg reward | Avg turns |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `dqn:results/checkpoints/dqn_smoke.pt` | Random | Deterministic | 10 | 0.80 | 0.60 | 43.30 |
| `dqn:results/checkpoints/dqn_stochastic_smoke.pt` | Random | Stochastic | 10 | 0.90 | 0.80 | 49.80 |

The difference between the trainer's built-in evaluation and the generic evaluator comes from different random seeds and game samples. Longer multi-seed evaluation is needed before making a reliable strength claim.

## 7. Interactive Game UI

I added a local browser interface in `game_ui.py` so that the command-line interaction can be used as a visual game board.

The interface supports:

- Human-vs-agent play.
- Agent-vs-agent stepping.
- Agent-vs-agent auto-play.
- Stochastic or deterministic placement.
- Built-in agent specs including `random`, `heuristic`, `mcts:20:6`, `mcts:50:6`, and `qtable:results/checkpoints/q_table.json`.

The UI uses the same Python environment and agent interface as the evaluation scripts, so future agents such as DQN or PPO can be exposed in the browser by adding their spec to `make_agent()`.

After visual inspection, I added tactical highlighting and explicit threat response logic:

- Cells where the current player can win immediately are highlighted.
- Cells where the opponent can win immediately are highlighted as block targets.
- Once a game is over, the full winning line is highlighted.
- The heuristic and MCTS prior now explicitly identify immediate wins and immediate blocks before applying softer line-shape scoring.

This addresses the concern that agents may look as if they are playing independently. In a deterministic reproduction where O has three in a row at `(4,2), (4,3), (4,4)` and X must respond, `HeuristicAgent` selects `(4,5)` to block.

Run command:

```bash
python3 game_ui.py --host 127.0.0.1 --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

Verification:

- Unit tests pass: `22/22`.
- `GET /api/state` returns the 96 playable cells.
- `POST /api/new`, `POST /api/move`, and `POST /api/agent-step` update the game state correctly.

## 8. Next Steps

The next recommended steps are:

1. Run longer heuristic-vs-random and heuristic-vs-Q evaluations across multiple seeds.
2. Add MCTS using the heuristic as the rollout policy.
3. Add a neural method such as DQN or PPO.
4. Optionally generate SFT data from heuristic or MCTS games, pretrain a policy network, then fine-tune with RL.

## 9. Limitations

The current implementation still has several limitations:

- The column win interpretation should be confirmed against the course expectation.
- Tabular Q-learning is too small for the full game.
- The heuristic is hand-designed and may overfit the current interpretation of the rules.
- MCTS is currently correctness-oriented and can be slow at higher simulation counts.
- No neural network method has been trained yet.
