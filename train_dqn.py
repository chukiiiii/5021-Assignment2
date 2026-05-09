"""Train a pure neural DQN baseline for super tic-tac-toe.

This agent intentionally does not use the hand-written heuristic or MCTS
tactical rules. It only receives board tensors, legal-action masks, and game
rewards.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from pathlib import Path
import random
from statistics import mean
from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F

from dqn_model import DQN, action_mask, masked_argmax, save_checkpoint, state_tensor
from super_tictactoe import SuperTicTacToeEnv, player_name


class Transition(NamedTuple):
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    next_mask: torch.Tensor
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.random = random.Random(seed)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return self.random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def select_action(
    model: DQN,
    env: SuperTicTacToeEnv,
    epsilon: float,
    rng: random.Random,
    device: torch.device,
) -> int:
    actions = env.available_actions()
    if rng.random() < epsilon:
        return rng.choice(actions)
    with torch.no_grad():
        state = state_tensor(env).unsqueeze(0).to(device)
        mask = action_mask(env).to(device)
        q_values = model(state)[0]
        return masked_argmax(q_values, mask)


def optimize(
    model: DQN,
    target_model: DQN,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float:
    batch = replay.sample(batch_size)
    states = torch.stack([item.state for item in batch]).to(device)
    actions = torch.tensor([item.action for item in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=device)
    next_states = torch.stack([item.next_state for item in batch]).to(device)
    next_masks = torch.stack([item.next_mask for item in batch]).to(device)
    dones = torch.tensor([item.done for item in batch], dtype=torch.bool, device=device)

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_model(next_states)
        next_q[~next_masks] = -1e9
        opponent_value = next_q.max(dim=1).values
        targets = rewards - gamma * opponent_value
        targets[dones] = rewards[dones]

    loss = F.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.item())


def epsilon_for_episode(
    episode: int,
    episodes: int,
    start: float,
    end: float,
) -> float:
    if episodes <= 1:
        return end
    fraction = episode / (episodes - 1)
    return max(end, start + fraction * (end - start))


def train(
    episodes: int,
    seed: int,
    stochastic: bool,
    lr: float,
    gamma: float,
    batch_size: int,
    replay_size: int,
    train_after: int,
    target_update: int,
    epsilon_start: float,
    epsilon_end: float,
    device: torch.device,
) -> tuple[DQN, list[dict[str, float | int | str]], dict[str, float]]:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = DQN().to(device)
    target_model = DQN().to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    replay = ReplayBuffer(replay_size, seed=seed)
    env = SuperTicTacToeEnv(seed=seed, stochastic=stochastic)

    history: list[dict[str, float | int | str]] = []
    episode_lengths: list[int] = []
    losses: list[float] = []
    recent_x_rewards: list[float] = []
    x_wins = 0
    o_wins = 0
    draws = 0
    global_step = 0

    for episode in range(episodes):
        env.reset()
        turns = 0
        episode_losses: list[float] = []
        epsilon = epsilon_for_episode(episode, episodes, epsilon_start, epsilon_end)

        while not env.done:
            state = state_tensor(env)
            action = select_action(model, env, epsilon, rng, device)
            _, reward, done, _ = env.step(action)
            next_state = state_tensor(env) if not done else torch.zeros_like(state)
            next_mask = action_mask(env) if not done else torch.zeros_like(action_mask(env))
            replay.append(
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    next_mask=next_mask,
                    done=done,
                )
            )

            if len(replay) >= train_after and len(replay) >= batch_size:
                loss = optimize(
                    model=model,
                    target_model=target_model,
                    optimizer=optimizer,
                    replay=replay,
                    batch_size=batch_size,
                    gamma=gamma,
                    device=device,
                )
                episode_losses.append(loss)
                losses.append(loss)

            global_step += 1
            if global_step % target_update == 0:
                target_model.load_state_dict(model.state_dict())
            turns += 1

        episode_lengths.append(turns)
        if env.winner == 1:
            x_wins += 1
            x_reward = 1.0
        elif env.winner == -1:
            o_wins += 1
            x_reward = -1.0
        else:
            draws += 1
            x_reward = 0.0

        recent_x_rewards.append(x_reward)
        if len(recent_x_rewards) > 50:
            recent_x_rewards.pop(0)
        history.append(
            {
                "episode": episode + 1,
                "turns": turns,
                "winner": player_name(env.winner),
                "x_reward": x_reward,
                "rolling_x_reward_50": sum(recent_x_rewards) / len(recent_x_rewards),
                "epsilon": epsilon,
                "replay_size": len(replay),
                "avg_loss": mean(episode_losses) if episode_losses else 0.0,
            }
        )

    metrics = {
        "episodes": float(episodes),
        "avg_turns": mean(episode_lengths) if episode_lengths else 0.0,
        "x_win_rate": x_wins / episodes if episodes else 0.0,
        "o_win_rate": o_wins / episodes if episodes else 0.0,
        "draw_rate": draws / episodes if episodes else 0.0,
        "avg_loss": mean(losses) if losses else 0.0,
    }
    return model, history, metrics


def evaluate(
    model: DQN,
    games: int,
    seed: int,
    stochastic: bool,
    device: torch.device,
) -> dict[str, float]:
    env = SuperTicTacToeEnv(seed=seed, stochastic=stochastic)
    rng = random.Random(seed + 1)
    learner_wins = 0
    random_wins = 0
    draws = 0
    turns: list[int] = []

    for game in range(games):
        env.reset()
        learner_is_x = game % 2 == 0
        game_turns = 0
        while not env.done:
            learner_turn = (env.current_player == 1 and learner_is_x) or (
                env.current_player == -1 and not learner_is_x
            )
            if learner_turn:
                action = select_action(model, env, epsilon=0.0, rng=rng, device=device)
            else:
                action = rng.choice(env.available_actions())
            env.step(action)
            game_turns += 1

        turns.append(game_turns)
        if env.winner is None:
            draws += 1
        elif (env.winner == 1 and learner_is_x) or (env.winner == -1 and not learner_is_x):
            learner_wins += 1
        else:
            random_wins += 1

    return {
        "games": float(games),
        "learner_win_rate": learner_wins / games if games else 0.0,
        "random_win_rate": random_wins / games if games else 0.0,
        "draw_rate": draws / games if games else 0.0,
        "avg_turns": mean(turns) if turns else 0.0,
    }


def save_history_csv(path: str, history: list[dict[str, float | int | str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "turns",
        "winner",
        "x_reward",
        "rolling_x_reward_50",
        "epsilon",
        "replay_size",
        "avg_loss",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_history_html(path: str, history: list[dict[str, float | int | str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    values = [float(row["rolling_x_reward_50"]) for row in history]
    points = svg_line_chart(values, width=760, height=220)
    rows = "\n".join(
        "<tr>"
        f"<td>{row['episode']}</td>"
        f"<td>{row['winner']}</td>"
        f"<td>{row['turns']}</td>"
        f"<td>{float(row['x_reward']):.0f}</td>"
        f"<td>{float(row['rolling_x_reward_50']):.3f}</td>"
        f"<td>{float(row['epsilon']):.3f}</td>"
        f"<td>{row['replay_size']}</td>"
        f"<td>{float(row['avg_loss']):.4f}</td>"
        "</tr>"
        for row in history
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DQN Reward History</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; color: #222; }}
    svg {{ width: 100%; max-width: 780px; height: auto; border: 1px solid #ddd; background: #fff; }}
    .axis {{ stroke: #bbb; stroke-width: 1; }}
    .line {{ fill: none; stroke: #1565c0; stroke-width: 3; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 24px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>DQN Reward History</h1>
  <p>Rolling reward is measured from X's perspective over the latest 50 episodes.</p>
  {points}
  <table>
    <thead><tr><th>Episode</th><th>Winner</th><th>Turns</th><th>X Reward</th><th>Rolling X Reward</th><th>Epsilon</th><th>Replay</th><th>Avg Loss</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)


def svg_line_chart(values: list[float], width: int, height: int) -> str:
    if not values:
        return f"<svg viewBox='0 0 {width} {height}'></svg>"
    pad = 28
    min_value = min(-1.0, min(values))
    max_value = max(1.0, max(values))

    def point(index: int, value: float) -> tuple[float, float]:
        if len(values) == 1:
            x = width / 2
        else:
            x = pad + index * (width - 2 * pad) / (len(values) - 1)
        y = height - pad - (value - min_value) * (height - 2 * pad) / (max_value - min_value)
        return x, y

    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(i, v) for i, v in enumerate(values)))
    zero_y = point(0, 0.0)[1]
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img'>"
        f"<line class='axis' x1='{pad}' y1='{zero_y:.1f}' x2='{width - pad}' y2='{zero_y:.1f}' />"
        f"<polyline class='line' points='{points}' />"
        f"<text x='{pad}' y='18'>+1</text>"
        f"<text x='{pad}' y='{height - 8}'>-1</text>"
        "</svg>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=20_000)
    parser.add_argument("--train-after", type=int, default=256)
    parser.add_argument("--target-update", type=int, default=250)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", default="results/checkpoints/dqn_checkpoint.pt")
    parser.add_argument("--history-csv")
    parser.add_argument("--history-html")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stochastic = not args.deterministic
    model, history, train_metrics = train(
        episodes=args.episodes,
        seed=args.seed,
        stochastic=stochastic,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        train_after=args.train_after,
        target_update=args.target_update,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        device=device,
    )
    eval_metrics = evaluate(
        model=model,
        games=args.eval_games,
        seed=args.seed + 10_000,
        stochastic=stochastic,
        device=device,
    )
    metadata = {
        "train": train_metrics,
        "eval": eval_metrics,
        "stochastic": stochastic,
        "seed": args.seed,
        "episodes": args.episodes,
        "note": "Pure DQN baseline with no hand-written tactical rules.",
    }
    save_checkpoint(args.output, model.cpu(), metadata)
    if args.history_csv:
        save_history_csv(args.history_csv, history)
    if args.history_html:
        save_history_html(args.history_html, history)

    print("Training metrics")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("Evaluation metrics")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"Saved DQN checkpoint to {args.output}")
    if args.history_csv:
        print(f"Saved reward history CSV to {args.history_csv}")
    if args.history_html:
        print(f"Saved reward history HTML to {args.history_html}")


if __name__ == "__main__":
    main()
