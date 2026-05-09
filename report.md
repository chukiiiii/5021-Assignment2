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

### PPO Smoke Tests

I added a PPO actor-critic baseline as the second neural RL method. Like DQN, PPO uses the `3 x 12 x 12` current-player-perspective tensor and a 96-action legal mask. PPO does not use the hand-written immediate-win or immediate-block heuristic.

The first PPO version uses self-play rollouts and terminal game outcome as the return for each action from that acting player's perspective. This keeps the baseline simple and comparable with DQN, but it also means the learning signal is still sparse.

Command:

```bash
python3 train_ppo.py --episodes 30 --eval-games 10 --seed 403 --batch-size 32 --rollout-episodes 5 --update-epochs 3 --output results/checkpoints/ppo_stochastic_smoke.pt --history-csv results/history/ppo_stochastic_smoke_history.csv --history-html results/history/ppo_stochastic_smoke_history.html
```

Observed PPO smoke training result:

| Agent | Training env | Eval env | Games | Win rate vs random | Avg eval turns |
| --- | --- | --- | ---: | ---: | ---: |
| PPO smoke | Stochastic | Stochastic | 10 | 0.70 | 54.20 |

Generic evaluator results:

| Agent | Opponent | Eval env | Games | Win rate | Avg reward | Avg turns |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `ppo:results/checkpoints/ppo_stochastic_smoke.pt` | Random | Stochastic | 10 | 0.60 | 0.20 | 52.30 |
| `ppo:results/checkpoints/ppo_stochastic_smoke.pt` | Heuristic | Stochastic | 10 | 0.00 | -1.00 | 12.70 |

The result is similar to early DQN: PPO can beat random in a small smoke evaluation, but it is still far behind the tactical heuristic. This supports using PPO as a second RL baseline, while keeping heuristic/MCTS expert trajectories as the likely next path for improvement.

## 7. Multi-Seed Experiment Summary

I added `run_experiments.py` to run reusable multi-seed comparisons and write one detail CSV, one summary CSV, and one summary HTML report.

Command used for the current report-sized run:

```bash
python3 run_experiments.py --preset full --seeds 1 2 3 --games 10 --mcts-games 5 --progress
```

Generated outputs:

- `results/summary/full_s3_g10_detail.csv`
- `results/summary/full_s3_g10_summary.csv`
- `results/summary/full_s3_g10_summary.html`

Observed summary:

| Experiment | Agent A | Agent B | Games | Mean win rate | Mean reward | Mean turns |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| random_vs_random | Random | Random | 30 | 0.433 | -0.133 | 62.20 |
| qtable_vs_random | Tabular Q | Random | 30 | 0.433 | -0.133 | 62.20 |
| heuristic_vs_random | Heuristic | Random | 30 | 1.000 | 1.000 | 10.70 |
| mcts20c6_vs_random | MCTS-20 top6 | Random | 15 | 1.000 | 1.000 | 13.13 |
| mcts50c6_vs_heuristic | MCTS-50 top6 | Heuristic | 15 | 0.467 | -0.067 | 11.93 |
| dqn_stochastic_vs_random | DQN stochastic smoke | Random | 30 | 0.800 | 0.600 | 46.63 |
| dqn_stochastic_vs_heuristic | DQN stochastic smoke | Heuristic | 30 | 0.000 | -1.000 | 11.90 |

Interpretation:

- The tabular Q-table is still essentially random at this scale.
- The heuristic baseline is very strong against random and wins quickly.
- Optimized MCTS is reliable against random, but only roughly competitive against the heuristic at the current simulation budget.
- The DQN smoke checkpoint has learned something useful against random, but it is clearly not yet strong enough against the tactical heuristic.
- This supports the next training direction: longer DQN training, stronger self-play opponents, or expert-data pretraining before RL fine-tuning.

## 8. Longer DQN Training and Diagnostics

I added `train_dqn_multiseed.py` to train DQN across several seeds, save per-seed reward curves, and evaluate each checkpoint against stronger opponents.

Command used:

```bash
python3 train_dqn_multiseed.py --seeds 201 202 203 --episodes 120 --diagnostic-games 10 --mcts-games 4 --batch-size 32 --train-after 128 --target-update 200 --tag dqn_long --progress
```

Generated outputs:

- `results/checkpoints/dqn_long_seed201.pt`
- `results/checkpoints/dqn_long_seed202.pt`
- `results/checkpoints/dqn_long_seed203.pt`
- `results/history/dqn_long_seed201_history.html`
- `results/history/dqn_long_seed202_history.html`
- `results/history/dqn_long_seed203_history.html`
- `results/summary/dqn_long_s3_e120_reward_curves.html`
- `results/summary/dqn_long_s3_e120_training.csv`
- `results/summary/dqn_long_s3_e120_diagnostics_summary.csv`
- `results/summary/dqn_long_s3_e120_diagnostics.html`

Training summary:

| Seed | Episodes | X win rate | O win rate | Avg turns | Final rolling X reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| 201 | 120 | 0.575 | 0.425 | 56.01 | 0.200 |
| 202 | 120 | 0.475 | 0.525 | 50.44 | 0.080 |
| 203 | 120 | 0.467 | 0.533 | 56.88 | -0.240 |

Diagnostic evaluation:

| Opponent | Seeds | Games | Mean DQN win rate | Mean reward | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 3 | 30 | 0.767 | 0.533 | 44.13 |
| Heuristic | 3 | 30 | 0.000 | -1.000 | 11.57 |
| MCTS-20 top6 | 3 | 12 | 0.000 | -1.000 | 11.33 |

Interpretation:

- Longer DQN training remains clearly better than random overall, but the result has high seed variance.
- The reward curves do not show stable monotonic improvement yet; seed 203 ends with negative rolling reward.
- The agent still loses quickly to heuristic and MCTS, which means it has not learned reliable immediate-win or blocking tactics from self-play alone.
- The next improvement should not just be "more episodes"; it should also improve the learning signal, for example by using curriculum opponents, prioritized replay, shaped auxiliary rewards, or supervised pretraining from heuristic/MCTS trajectories.

## 9. Interactive Game UI

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

## 10. Next Steps

The next recommended steps are:

1. Generate heuristic or MCTS expert trajectories for possible supervised pretraining.
2. Add a behavior-cloning/SFT policy baseline and compare it with pure DQN.
3. Fine-tune the pretrained policy with RL, then rerun the same diagnostic suite.
4. Add PPO as a second neural RL method after the DQN loop is better understood.
5. Repeat the multi-seed table with larger game counts for final reporting.

Experiment artifacts are now organized under `results/`:

- `results/checkpoints/`
- `results/history/`
- `results/evaluations/`

## 11. Limitations

The current implementation still has several limitations:

- The column win interpretation should be confirmed against the course expectation.
- Tabular Q-learning is too small for the full game.
- The heuristic is hand-designed and may overfit the current interpretation of the rules.
- MCTS is currently correctness-oriented and can be slow at higher simulation counts.
- DQN has now been trained beyond smoke-test scale, but it is still sample-inefficient and weak against tactical opponents.
- PPO is currently only implemented and smoke-tested; it needs the same multi-seed diagnostic treatment as DQN before making stronger claims.
