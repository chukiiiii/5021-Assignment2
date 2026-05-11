# Super Tic-Tac-Toe — Assignment 2

## Group Member: YU XIAOYA, CAI XINYI

Reinforcement learning agents for a stochastic variant of tic-tac-toe on a triangular board of 96 cells.

For full experimental details, results, and analysis, see **[report.md](report.md)**.

---

## Code Structure

```text
super_tictactoe.py           Game rules and environment.
agents.py                    Built-in agents: random, heuristic, MCTS, human.
match.py                     Game runner and match statistics.
evaluate_agents.py           Agent-vs-agent evaluation (JSON/CSV/HTML reports).
game_ui.py                   Local browser UI for human play.
run_experiments.py           Multi-seed experiment summary runner.

dqn_model.py                 DQN network and tensor helpers.
train_dqn.py                 DQN training script.
train_dqn_multiseed.py       Multi-seed DQN training with diagnostics.

ppo_model.py                 PPO actor-critic network and checkpoint helpers.
train_ppo.py                 PPO training script (supports --init-checkpoint).
train_ppo_multiseed.py       Multi-seed PPO training with diagnostics.

generate_expert_data.py      Heuristic/MCTS expert trajectory generator.
train_bc.py                  Behavior cloning (SFT) training script.

results/checkpoints/         Trained model checkpoints.
results/expert/              Expert datasets for behavior cloning.
results/history/             Training reward history CSV/HTML.
results/evaluations/         Match evaluation reports.
results/summary/             Multi-seed experiment summaries.
```

---

## Core Method: SFT + Conservative PPO Fine-Tuning

The best-performing approach uses a three-stage pipeline: expert data generation → behavior cloning (SFT) → conservative PPO fine-tuning.

### 1. Generate Expert Data

Heuristic self-play (400 games) + MCTS vs Heuristic (80 games), producing 5685 state-action pairs.

### Run Tests
```bash
python generate_expert_data.py \
  --heuristic-games 400 --mcts-games 80 \
  --output results/expert/mixed_expert_large.pt \
  --summary results/expert/mixed_expert_large_summary.csv
```

### 2. Behavior Cloning (Supervised Fine-Tuning)

Train the PPOActorCritic network via masked cross-entropy to predict expert actions. Achieves 64.6% validation top-1 accuracy with 100% legal action compliance.

```bash
python train_bc.py \
  --data results/expert/mixed_expert_large.pt \
  --epochs 12 --batch-size 128 --lr 1e-3 \
  --output results/checkpoints/bc_mixed_large.pt \
  --history-csv results/history/bc_mixed_large_history.csv \
  --history-html results/history/bc_mixed_large_history.html
```

### 3. Conservative PPO Fine-Tuning

Load the BC-pretrained checkpoint and refine via PPO with a conservative learning rate. Runs 3 seeds (901, 902, 903) for 120 episodes each.

```bash
python train_ppo_multiseed.py \
  --seeds 901 902 903 --episodes 120 \
  --init-checkpoint results/checkpoints/bc_mixed_large.pt \
  --tag sft_ppo_conservative --progress
```

### 4. Evaluate

Evaluate the fine-tuned checkpoints against baseline opponents.

### Evaluate Agents
```bash
# vs random
python evaluate_agents.py \
  --agent-a ppo:results/checkpoints/sft_ppo_conservative_seed901.pt \
  --agent-b random --games 30 \
  --html-output results/evaluations/eval_sft_ppo_conservative_random.html

# vs heuristic
python evaluate_agents.py \
  --agent-a ppo:results/checkpoints/sft_ppo_conservative_seed901.pt \
  --agent-b heuristic --games 30 \
  --html-output results/evaluations/eval_sft_ppo_conservative_heuristic.html

# vs MCTS
python evaluate_agents.py \
  --agent-a ppo:results/checkpoints/sft_ppo_conservative_seed901.pt \
  --agent-b mcts:20:6 --games 12 \
  --html-output results/evaluations/eval_sft_ppo_conservative_mcts.html
```

---

## Other Commands

### Run tests

```bash
python -m unittest -v
```

### Start browser UI

```bash
python game_ui.py --host 127.0.0.1 --port 8765
```

### Train baselines

```bash
# DQN multi-seed
python train_dqn_multiseed.py --seeds 201 202 203 --episodes 120 --tag dqn_long --progress

# PPO from scratch multi-seed
python train_ppo_multiseed.py --seeds 501 502 503 --episodes 120 --tag ppo_long --progress
```

### Run full experiment summary
```bash
python run_experiments.py --preset full --seeds 1 2 3 --games 10 --mcts-games 5 --progress
```
