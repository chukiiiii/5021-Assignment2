# Assignment 2 Handoff

## Goal

Train a reinforcement-learning agent to play the assignment game, "super tic-tac-toe".

The repository was empty when work started, so this handoff contains both the interpretation of the题目 and the implementation status. A future agent should be able to continue from this file alone.

## Interpreted Rules

- The board is a triangle made from 6 large squares.
- Each large square is 4x4, so there are 96 playable cells.
- I modelled the board as a 12x12 grid with valid cells only in this triangular shape:
  - Level 1, rows 0-3: columns 4-7.
  - Level 2, rows 4-7: columns 2-9.
  - Level 3, rows 8-11: columns 0-11.
- Players alternate turns. X is player one and O is player two.
- The chosen action must be an empty valid cell.
- Stochastic placement:
  - With probability 1/2, the mark is placed on the chosen cell.
  - Otherwise, one of the 8 adjacent coordinates is selected with probability 1/16 each.
  - If that adjacent coordinate is outside the triangular board or already occupied, the move is forfeited and the turn passes.
  - This matches the题目's corner example: a bottom corner has 5 outside adjacent coordinates, so the outside probability in the redirected half is 5/16.
- Win conditions implemented:
  - 4 contiguous marks in a row.
  - 4 contiguous marks in a column, but the 4 cells must include at least two different levels.
  - 5 contiguous marks on either diagonal direction.

## Ambiguities To Confirm

- The phrase "To win with 4 in a column, at least one move must be in a different level" is slightly ambiguous.
- Current implementation interprets it as: a vertical 4-cell winning line counts only if its cells span at least two levels.
- If the course expects a more 3D-like interpretation of "level", update only `build_winning_lines()` in `super_tictactoe.py` and then rerun tests.

## Files Created

- `super_tictactoe.py`
  - Board geometry.
  - Stochastic move resolution.
  - Win-line precomputation.
  - Small Gym-like environment: `reset()`, `state()`, `canonical_state()`, `available_actions()`, `step()`, `render()`.
- `train_q_learning.py`
  - Dependency-free tabular Q-learning/self-play baseline.
  - Alternating players share one Q-table using canonical state from current player's perspective.
  - Uses a zero-sum update: future opponent value is subtracted.
  - Can evaluate the learner against a random policy and save `q_table.json`.
- `test_super_tictactoe.py`
  - Unit tests for board size, invalid cells, corner outside probability, horizontal/vertical/diagonal wins, and forfeited turn passing.
- `agents.py`
  - Reusable agent interface and built-in `random`, `heuristic`, `qtable:path`, and `human` agents.
- `match.py`
  - Reusable single-game runner and multi-game win-rate evaluator.
- `evaluate_agents.py`
  - CLI for algorithm-vs-algorithm win-rate tests with terminal, JSON, CSV, and HTML output.
- `play_game.py`
  - CLI for visual single-game replay and human-vs-agent play.
- `q_table_smoke.json`
  - Output from a 50-episode deterministic smoke run.
- `q_table.json`
  - Output from a 200-episode stochastic smoke run.

## Completion Plan

1. Formalize the assignment rules into a board representation.
2. Implement a deterministic/stochastic environment that can be used by RL code.
3. Add tests for the rule details that are easiest to get wrong.
4. Implement a simple training baseline.
5. Run tests and a small training smoke test.
6. Record successes, failures, and next steps here.

## Progress Log

### Step 1: Repository inspection

- Tried:
  - `pwd`
  - `ls -la`
  - `rg --files`
  - `git status --short`
- Result:
  - Working directory is `/Users/llm/0_files/workspace/study/RL_finance/Assignment2`.
  - Directory was empty.
  - This is not a git repository.
- Status:
  - Successful. No existing code had to be preserved.

### Step 2: Rule decomposition

- Tried:
  - Parsed the screenshot manually.
  - Converted the triangular board into a 12x12 masked grid.
- Result:
  - Effective playable cells: 16 + 32 + 48 = 96.
  - The stochastic move rule can be represented by a 16-way roll: 8 outcomes for chosen cell, 8 outcomes for neighbors.
- Status:
  - Successful, with one documented ambiguity around column wins.

### Step 3: Environment implementation

- Tried:
  - Created `super_tictactoe.py`.
  - Implemented valid-cell mapping, action mapping, stochastic placement, win detection, and rendering.
- Result:
  - Environment is usable by a training loop without external packages.
- Status:
  - Implemented and passed unit tests.

### Step 4: RL baseline implementation

- Tried:
  - Created `train_q_learning.py`.
  - Implemented tabular self-play Q-learning.
- Result:
  - Baseline is intentionally simple and dependency-free.
  - It is not expected to solve the full game optimally because 96 cells make the tabular state space enormous.
- Status:
  - Implemented and passed deterministic/stochastic smoke tests.

### Step 5: Tests

- Tried:
  - Created `test_super_tictactoe.py`.
- Result:
  - Tests cover the assignment-specific mechanics.
- Status:
  - Passed.

### Step 6: Verification

- Tried:
  - `python3 -m unittest -v`
  - `python3 train_q_learning.py --episodes 50 --eval-games 20 --deterministic --output q_table_smoke.json`
  - `python3 train_q_learning.py --episodes 200 --eval-games 50 --output q_table.json`
- Result:
  - Unit tests: 7 tests passed.
  - Deterministic smoke training:
    - episodes: 50
    - states visited: 2364
    - avg_turns: 48.46
    - train x_win_rate: 0.58
    - train o_win_rate: 0.42
    - eval learner_win_rate vs random: 0.35
    - eval random_win_rate: 0.65
  - Stochastic smoke training:
    - episodes: 200
    - states visited: 11193
    - avg_turns: 60.695
    - train x_win_rate: 0.525
    - train o_win_rate: 0.475
    - eval learner_win_rate vs random: 0.44
    - eval random_win_rate: 0.56
- Status:
  - Successful as an implementation smoke test.
  - The trained baseline is not yet a strong player; it proves the environment and training loop work.

## Commands For Next Agent

Run tests:

```bash
python3 -m unittest -v
```

Run a quick deterministic training smoke test:

```bash
python3 train_q_learning.py --episodes 50 --eval-games 20 --deterministic --output q_table_smoke.json
```

Run a stochastic training smoke test:

```bash
python3 train_q_learning.py --episodes 200 --eval-games 50 --output q_table.json
```

Evaluate two agents and save reports:

```bash
python3 evaluate_agents.py --agent-a qtable:q_table.json --agent-b random --games 100 --json-output eval.json --csv-output eval.csv --html-output eval.html
```

Play or display a single game:

```bash
python3 play_game.py --x human --o qtable:q_table.json
python3 play_game.py --x random --o random --quiet
python3 play_game.py --x human --o heuristic
```

## Suggested Next Steps

1. Confirm the column-win interpretation with the course materials or instructor.
2. Improve the agent:
   - Add heuristic action ordering for immediate wins and blocks.
   - Add Monte Carlo control or DQN if external libraries are allowed.
   - Optionally implement the bonus with TorchRL, TF-Agents, or RLlib if the environment setup permits.
3. Run longer experiments with fixed seeds and plot learning curves.
4. Add a short report/notebook explaining the state representation, reward design, training curve, and limitations.

## Results

- Unit test result: passed, 7/7.
- Deterministic smoke training result: passed and saved `q_table_smoke.json`.
- Stochastic smoke training result: passed and saved `q_table.json`.
- Known failures: no runtime failures after implementation.
- Known limitation: tabular Q-learning is too small for the full 96-cell game state space, so the baseline is mostly a correctness scaffold rather than a high-performing final agent.

## Latest Additions

- Added reusable win-rate testing and display infrastructure.
- Agent specs supported today:
  - `random`
  - `heuristic`
  - `qtable:path/to/q_table.json`
  - `human`
- Future PPO, MCTS, heuristic, or SFT policies should implement the same shape as `Agent.select_action(env)` in `agents.py`.
- Verification run:
  - `python3 -m unittest -v`: passed, 10/10.
  - `python3 evaluate_agents.py --agent-a random --agent-b random --games 10 --seed 11 --deterministic --json-output eval_random.json --csv-output eval_random.csv --html-output eval_random.html`: passed.
  - `python3 play_game.py --x random --o random --seed 5 --deterministic --quiet`: passed.

## 2026-05-09 Progress

- Added the next planned optimization: a reusable `HeuristicAgent`.
- The heuristic evaluates intended moves by averaging over stochastic placement outcomes, including redirects and forfeits.
- Added `report.md` as the assignment report draft. It currently covers the problem definition, environment, baselines, heuristic optimization, evaluation framework, results, next steps, and limitations.
- Added tests for heuristic immediate-win behavior and `make_agent("heuristic")`.
- Verification:
  - `python3 -m unittest -v`: passed, 12/12.
  - `python3 evaluate_agents.py --agent-a heuristic --agent-b random --games 100 --seed 21 --json-output eval_heuristic_random.json --csv-output eval_heuristic_random.csv --html-output eval_heuristic_random.html`: heuristic won 100/100, avg turns 11.70.
  - `python3 evaluate_agents.py --agent-a heuristic --agent-b qtable:q_table.json --games 100 --seed 31 --json-output eval_heuristic_qtable.json --csv-output eval_heuristic_qtable.csv --html-output eval_heuristic_qtable.html`: heuristic won 100/100, avg turns 12.14.
- Generated reports:
  - `eval_heuristic_random.json`
  - `eval_heuristic_random.csv`
  - `eval_heuristic_random.html`
  - `eval_heuristic_qtable.json`
  - `eval_heuristic_qtable.csv`
  - `eval_heuristic_qtable.html`
- Next best step:
  - Implement MCTS using `heuristic` as rollout policy.
  - Alternatively implement DQN/PPO using the existing evaluator and `report.md` structure.

## 2026-05-09 MCTS and Reward Tracking

- Added `MCTSAgent` in `agents.py`.
- CLI specs now include:
  - `mcts`
  - `mcts:N`, where `N` is the number of simulations per move.
- MCTS implementation details:
  - Uses the same `select_action(env)` interface as other agents.
  - Handles stochastic move placement by sampling the real environment transition inside simulations.
  - Uses rollout simulation and a bounded board evaluation when depth-limited.
  - Returns values from the root player's perspective.
- Reward tracking added:
  - `evaluate_agents.py` adds per-game reward, cumulative reward, and rolling reward from agent A's perspective.
  - HTML evaluation reports now include a cumulative reward chart.
  - `train_q_learning.py` supports `--history-csv` and `--history-html`.
- Successful verification:
  - `python3 -m unittest -v`: passed, 13/13 after MCTS was added.
  - `python3 train_q_learning.py --episodes 60 --eval-games 20 --seed 17 --output q_table_reward_smoke.json --history-csv q_reward_history.csv --history-html q_reward_history.html`: passed.
  - `python3 evaluate_agents.py --agent-a mcts:1 --agent-b random --games 1 --seed 61 --deterministic --json-output eval_mcts1_random_smoke.json --csv-output eval_mcts1_random_smoke.csv --html-output eval_mcts1_random_smoke.html`: passed.
- Important performance observation:
  - Started `python3 evaluate_agents.py --agent-a mcts:20 --agent-b random --games 20 --seed 51 --json-output eval_mcts20_random.json --csv-output eval_mcts20_random.csv --html-output eval_mcts20_random.html`.
  - This old-parameter run was very slow because each MCTS simulation performed repeated heuristic rollouts.
  - It eventually completed: mcts:20 won 8/20 vs random, win rate 0.40, avg reward -0.20, avg turns 57.60, avg forfeits 10.80, avg redirects 17.30.
  - Default `MCTSAgent` rollout depth was reduced afterward for future runs, and default rollouts were changed to random for speed.
  - Next agent should prefer tiny smoke tests first, for example `mcts:3` or `mcts:5` over 2-5 games, then optimize before larger evaluations.

## 2026-05-09 MCTS Optimization

- Optimized MCTS after the first `mcts:20` result underperformed random.
- Changes:
  - Candidate pruning with a fast tactical prior.
  - Spec format now supports `mcts:simulations:max_candidates`, for example `mcts:20:6`.
  - Tree selection is now adversarial: root-player nodes maximize root value, opponent nodes minimize root value.
  - Prior-guided UCB/final root action selection helps low-simulation MCTS keep strong tactical moves.
  - `evaluate_agents.py` now supports `--progress-every N`.
- Verification:
  - `python3 -m unittest -v`: passed, 14/14.
- Results:
  - `python3 evaluate_agents.py --agent-a mcts:20 --agent-b random --games 20 --seed 71 --progress-every 1 --json-output eval_mcts20_random_optimized.json --csv-output eval_mcts20_random_optimized.csv --html-output eval_mcts20_random_optimized.html`
    - mcts:20 won 20/20, avg reward 1.00, avg turns 15.80.
  - `python3 evaluate_agents.py --agent-a mcts:20:6 --agent-b random --games 20 --seed 82 --progress-every 5 --json-output eval_mcts20c6_random_prior.json --csv-output eval_mcts20c6_random_prior.csv --html-output eval_mcts20c6_random_prior.html`
    - mcts:20:6 won 20/20, avg reward 1.00, avg turns 12.00.
  - `python3 evaluate_agents.py --agent-a mcts:20:6 --agent-b heuristic --games 10 --seed 83 --progress-every 1 --json-output eval_mcts20c6_heuristic_prior.json --csv-output eval_mcts20c6_heuristic_prior.csv --html-output eval_mcts20c6_heuristic_prior.html`
    - mcts:20:6 won 4/10 vs heuristic, avg reward -0.20.
  - `python3 evaluate_agents.py --agent-a mcts:50:6 --agent-b heuristic --games 10 --seed 84 --progress-every 1 --json-output eval_mcts50c6_heuristic_prior.json --csv-output eval_mcts50c6_heuristic_prior.csv --html-output eval_mcts50c6_heuristic_prior.html`
    - mcts:50:6 won 6/10 vs heuristic, avg reward 0.20.
- Next best step:
  - Run multi-seed evaluation for `heuristic`, `mcts:20:6`, and `mcts:50:6`.
  - Then consider DQN/PPO, using MCTS or heuristic for SFT data generation.
