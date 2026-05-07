"""Reusable agents for evaluation and demos."""

from __future__ import annotations

import json
import random
from typing import Protocol

from super_tictactoe import CELL_TO_ACTION, EMPTY, SuperTicTacToeEnv, VALID_CELLS
from train_q_learning import state_key


class Agent(Protocol):
    name: str

    def reset(self) -> None:
        """Reset per-game state if the agent keeps any."""

    def select_action(self, env: SuperTicTacToeEnv) -> int:
        """Choose one legal action for the current environment."""


class RandomAgent:
    def __init__(self, seed: int | None = None, name: str = "random") -> None:
        self.name = name
        self.random = random.Random(seed)

    def reset(self) -> None:
        pass

    def select_action(self, env: SuperTicTacToeEnv) -> int:
        return self.random.choice(env.available_actions())


class QTableAgent:
    def __init__(
        self,
        path: str,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        self.path = path
        self.name = name or f"qtable:{path}"
        self.random = random.Random(seed)
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.q_table: dict[str, dict[int, float]] = {
            key: {int(action): float(value) for action, value in actions.items()}
            for key, actions in payload.get("q_table", {}).items()
        }

    def reset(self) -> None:
        pass

    def select_action(self, env: SuperTicTacToeEnv) -> int:
        actions = env.available_actions()
        values = self.q_table.get(state_key(env.canonical_state()), {})
        best_value = max(values.get(action, 0.0) for action in actions)
        best_actions = [
            action for action in actions if values.get(action, 0.0) == best_value
        ]
        return self.random.choice(best_actions)


class HumanConsoleAgent:
    def __init__(self, name: str = "human") -> None:
        self.name = name

    def reset(self) -> None:
        pass

    def select_action(self, env: SuperTicTacToeEnv) -> int:
        while True:
            raw = input("Enter move as 'row col' or action index: ").strip()
            try:
                parts = raw.replace(",", " ").split()
                if len(parts) == 1:
                    action = int(parts[0])
                    if action in env.available_actions():
                        return action
                elif len(parts) == 2:
                    row, col = int(parts[0]), int(parts[1])
                    action = CELL_TO_ACTION[(row, col)]
                    if env.board[row][col] == EMPTY:
                        return action
            except (KeyError, ValueError, IndexError):
                pass
            print("Invalid move. Use an empty valid cell, for example: 8 0")


def make_agent(spec: str, seed: int | None = None) -> Agent:
    """Build an agent from a CLI spec.

    Supported specs:
    - `random`
    - `qtable:path/to/q_table.json`
    - `human`
    """
    if spec == "random":
        return RandomAgent(seed=seed)
    if spec == "human":
        return HumanConsoleAgent()
    if spec.startswith("qtable:"):
        path = spec.split(":", 1)[1]
        return QTableAgent(path=path, seed=seed)
    raise ValueError(f"unknown agent spec: {spec}")


def action_label(action: int) -> str:
    row, col = VALID_CELLS[action]
    return f"{action} ({row},{col})"
