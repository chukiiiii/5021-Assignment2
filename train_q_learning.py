"""Train a simple tabular Q-learning baseline for super tic-tac-toe."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
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
        elif env.winner == -1:
            o_wins += 1
        else:
            draws += 1

    metrics = {
        "episodes": float(episodes),
        "states": float(len(learner.q)),
        "avg_turns": mean(episode_lengths) if episode_lengths else 0.0,
        "x_win_rate": x_wins / episodes if episodes else 0.0,
        "o_win_rate": o_wins / episodes if episodes else 0.0,
        "draw_rate": draws / episodes if episodes else 0.0,
    }
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
    payload = {
        "metadata": metadata,
        "q_table": {state: actions for state, actions in learner.q.items() if actions},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", default="q_table.json")
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

    print("Training metrics")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("Evaluation metrics")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"Saved Q-table to {args.output}")
    print(f"Last evaluated winner label format: {player_name(None)} means no winner")


if __name__ == "__main__":
    main()
