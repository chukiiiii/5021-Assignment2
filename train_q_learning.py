"""Train a simple tabular Q-learning baseline for super tic-tac-toe."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
from statistics import mean
from typing import DefaultDict

from super_tictactoe import SuperTicTacToeEnv, player_name


QTable = DefaultDict[str, dict[int, float]]


def state_key(state: tuple[int, ...]) -> str:
    """Compact JSON-safe encoding for values -1, 0, 1."""
    return "".join(str(value + 1) for value in state)


class QLearner:
    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 0.25,
        seed: int | None = None,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.random = random.Random(seed)
        self.q: QTable = defaultdict(dict)
        self.training_history: list[dict[str, float | int | str]] = []

    def choose_action(
        self,
        state: tuple[int, ...],
        actions: list[int],
        explore: bool = True,
    ) -> int:
        if not actions:
            raise ValueError("no available actions")

        if explore and self.random.random() < self.epsilon:
            return self.random.choice(actions)

        key = state_key(state)
        values = self.q[key]
        best_value = max(values.get(action, 0.0) for action in actions)
        best_actions = [
            action for action in actions if values.get(action, 0.0) == best_value
        ]
        return self.random.choice(best_actions)

    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...] | None,
        next_actions: list[int],
        done: bool,
    ) -> None:
        key = state_key(state)
        old = self.q[key].get(action, 0.0)
        if done or next_state is None or not next_actions:
            target = reward
        else:
            next_values = self.q[state_key(next_state)]
            opponent_value = max(next_values.get(next_action, 0.0) for next_action in next_actions)
            target = reward - self.gamma * opponent_value
        self.q[key][action] = old + self.alpha * (target - old)


def train(
    episodes: int,
    seed: int,
    epsilon: float,
    alpha: float,
    gamma: float,
    stochastic: bool,
) -> tuple[QLearner, dict[str, float]]:
    learner = QLearner(alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed)
    env = SuperTicTacToeEnv(seed=seed, stochastic=stochastic)
    episode_lengths: list[int] = []
    x_wins = 0
    o_wins = 0
    draws = 0
    history: list[dict[str, float | int | str]] = []
    recent_rewards: list[float] = []

    for episode in range(episodes):
        env.reset()
        turns = 0
        start_epsilon = learner.epsilon
        learner.epsilon = max(0.03, start_epsilon * (1.0 - episode / max(episodes, 1)))

        while not env.done:
            state = env.canonical_state()
            actions = env.available_actions()
            action = learner.choose_action(state, actions, explore=True)
            _, reward, done, _ = env.step(action)
            next_state = None if done else env.canonical_state()
            next_actions = [] if done else env.available_actions()
            learner.update(state, action, reward, next_state, next_actions, done)
            turns += 1

        learner.epsilon = start_epsilon
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

        recent_rewards.append(x_reward)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)
        history.append(
            {
                "episode": episode + 1,
                "turns": turns,
                "winner": player_name(env.winner),
                "x_reward": x_reward,
                "rolling_x_reward_50": sum(recent_rewards) / len(recent_rewards),
                "epsilon": learner.epsilon,
                "states": len(learner.q),
            }
        )

    metrics = {
        "episodes": float(episodes),
        "states": float(len(learner.q)),
        "avg_turns": mean(episode_lengths) if episode_lengths else 0.0,
        "x_win_rate": x_wins / episodes if episodes else 0.0,
        "o_win_rate": o_wins / episodes if episodes else 0.0,
        "draw_rate": draws / episodes if episodes else 0.0,
    }
    learner.training_history = history
    return learner, metrics


def evaluate(
    learner: QLearner,
    games: int,
    seed: int,
    stochastic: bool,
) -> dict[str, float]:
    env = SuperTicTacToeEnv(seed=seed, stochastic=stochastic)
    random_policy = random.Random(seed + 1)
    learner_wins = 0
    random_wins = 0
    draws = 0
    lengths: list[int] = []

    for game in range(games):
        env.reset()
        turns = 0
        learner_is_x = game % 2 == 0

        while not env.done:
            actions = env.available_actions()
            learner_turn = (env.current_player == 1 and learner_is_x) or (
                env.current_player == -1 and not learner_is_x
            )
            if learner_turn:
                action = learner.choose_action(env.canonical_state(), actions, explore=False)
            else:
                action = random_policy.choice(actions)
            env.step(action)
            turns += 1

        lengths.append(turns)
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
        "avg_turns": mean(lengths) if lengths else 0.0,
    }


def save_q_table(path: str, learner: QLearner, metadata: dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "q_table": {state: actions for state, actions in learner.q.items() if actions},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_training_history_csv(path: str, history: list[dict[str, float | int | str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "turns",
        "winner",
        "x_reward",
        "rolling_x_reward_50",
        "epsilon",
        "states",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_training_history_html(path: str, history: list[dict[str, float | int | str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    values = [float(row["rolling_x_reward_50"]) for row in history]
    chart = svg_line_chart(values, width=760, height=220)
    rows = "\n".join(
        "<tr>"
        f"<td>{row['episode']}</td>"
        f"<td>{row['winner']}</td>"
        f"<td>{row['turns']}</td>"
        f"<td>{float(row['x_reward']):.0f}</td>"
        f"<td>{float(row['rolling_x_reward_50']):.3f}</td>"
        f"<td>{float(row['epsilon']):.3f}</td>"
        f"<td>{row['states']}</td>"
        "</tr>"
        for row in history
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Q-Learning Reward History</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; color: #222; }}
    svg {{ width: 100%; max-width: 780px; height: auto; border: 1px solid #ddd; background: #fff; }}
    .axis {{ stroke: #bbb; stroke-width: 1; }}
    .line {{ fill: none; stroke: #2aa7df; stroke-width: 3; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 24px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>Q-Learning Reward History</h1>
  <p>Rolling reward is measured from X's perspective over the latest 50 episodes.</p>
  {chart}
  <table>
    <thead><tr><th>Episode</th><th>Winner</th><th>Turns</th><th>X Reward</th><th>Rolling X Reward</th><th>Epsilon</th><th>States</th></tr></thead>
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
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", default="results/checkpoints/q_table.json")
    parser.add_argument("--history-csv")
    parser.add_argument("--history-html")
    args = parser.parse_args()

    stochastic = not args.deterministic
    learner, train_metrics = train(
        episodes=args.episodes,
        seed=args.seed,
        epsilon=args.epsilon,
        alpha=args.alpha,
        gamma=args.gamma,
        stochastic=stochastic,
    )
    eval_metrics = evaluate(
        learner,
        games=args.eval_games,
        seed=args.seed + 10_000,
        stochastic=stochastic,
    )
    metadata = {
        "train": train_metrics,
        "eval": eval_metrics,
        "stochastic": stochastic,
        "note": "Tabular self-play baseline; suitable as a clear assignment starting point.",
    }
    save_q_table(args.output, learner, metadata)
    if args.history_csv:
        save_training_history_csv(args.history_csv, learner.training_history)
    if args.history_html:
        save_training_history_html(args.history_html, learner.training_history)

    print("Training metrics")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("Evaluation metrics")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"Saved Q-table to {args.output}")
    if args.history_csv:
        print(f"Saved reward history CSV to {args.history_csv}")
    if args.history_html:
        print(f"Saved reward history HTML to {args.history_html}")
    print(f"Last evaluated winner label format: {player_name(None)} means no winner")


if __name__ == "__main__":
    main()
