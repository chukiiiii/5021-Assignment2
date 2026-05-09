"""Reusable agents for evaluation and demos."""

from __future__ import annotations

import json
import random
from typing import Protocol

from super_tictactoe import (
    CELL_TO_ACTION,
    EMPTY,
    O,
    WINNING_LINES,
    X,
    SuperTicTacToeEnv,
    VALID_CELLS,
    adjacent_cells,
    is_valid_cell,
)
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


LINES_BY_CELL: dict[tuple[int, int], list[tuple[tuple[int, int], ...]]] = {
    cell: [] for cell in VALID_CELLS
}
for winning_line in WINNING_LINES:
    for line_cell in winning_line:
        LINES_BY_CELL[line_cell].append(winning_line)


def _placement_outcomes(
    env: SuperTicTacToeEnv,
    action: int,
) -> list[tuple[tuple[int, int] | None, float]]:
    requested = env.action_to_cell(action)
    if not env.stochastic:
        return [(requested, 1.0)]

    raw_targets = [requested] * 8 + adjacent_cells(*requested)
    counts: dict[tuple[int, int] | None, int] = {}
    for target in raw_targets:
        row, col = target
        if not is_valid_cell(row, col) or env.board[row][col] != EMPTY:
            target = None
        counts[target] = counts.get(target, 0) + 1
    return [(target, count / 16.0) for target, count in counts.items()]


def _line_score(player_count: int, empty_count: int, line_length: int) -> float:
    if player_count == line_length:
        return 100_000.0
    if empty_count == 0:
        return 0.0
    if player_count == line_length - 1:
        return 2_000.0
    if player_count == line_length - 2:
        return 120.0
    if player_count == line_length - 3:
        return 12.0
    return float(player_count)


def _board_score(env: SuperTicTacToeEnv, player: int) -> float:
    opponent = -player
    score = 0.0
    for line in WINNING_LINES:
        values = [env.board[row][col] for row, col in line]
        player_count = values.count(player)
        opponent_count = values.count(opponent)
        empty_count = values.count(EMPTY)
        line_length = len(line)
        if opponent_count == 0:
            score += _line_score(player_count, empty_count, line_length)
        if player_count == 0:
            score -= 1.15 * _line_score(opponent_count, empty_count, line_length)
    return score


def _cell_shape_score(env: SuperTicTacToeEnv, player: int, cell: tuple[int, int]) -> float:
    row, col = cell
    line_participation = len(LINES_BY_CELL[cell])
    neighbor_bonus = 0.0
    for near_row, near_col in adjacent_cells(row, col):
        if is_valid_cell(near_row, near_col):
            if env.board[near_row][near_col] == player:
                neighbor_bonus += 3.0
            elif env.board[near_row][near_col] == -player:
                neighbor_bonus += 0.8

    center_distance = abs(row - 7.0) + abs(col - 5.5)
    center_bonus = max(0.0, 12.0 - center_distance)
    return 0.8 * line_participation + neighbor_bonus + 0.2 * center_bonus


class HeuristicAgent:
    """Expected-value tactical baseline for the stochastic board.

    The agent evaluates each intended move by averaging over the actual
    placement outcomes. It heavily prefers immediate wins, blocks opponent
    threats, and otherwise scores line potential, local support, and board
    shape.
    """

    def __init__(self, seed: int | None = None, name: str = "heuristic") -> None:
        self.name = name
        self.random = random.Random(seed)

    def reset(self) -> None:
        pass

    def select_action(self, env: SuperTicTacToeEnv) -> int:
        actions = env.available_actions()
        scored = [(self.score_action(env, action), action) for action in actions]
        best_score = max(score for score, _ in scored)
        best_actions = [action for score, action in scored if score == best_score]
        return self.random.choice(best_actions)

    def score_action(self, env: SuperTicTacToeEnv, action: int) -> float:
        player = env.current_player
        requested = env.action_to_cell(action)
        total = 0.0

        for target, probability in _placement_outcomes(env, action):
            if target is None:
                total += probability * (_board_score(env, player) - 35.0)
                continue

            row, col = target
            env.board[row][col] = player
            winner = env.check_winner()
            if winner == player:
                outcome_score = 1_000_000.0
            else:
                outcome_score = _board_score(env, player)
                outcome_score += _cell_shape_score(env, player, target)
                if target == requested:
                    outcome_score += 2.0
            env.board[row][col] = EMPTY
            total += probability * outcome_score

        return total


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
    - `heuristic`
    - `qtable:path/to/q_table.json`
    - `human`
    """
    if spec == "random":
        return RandomAgent(seed=seed)
    if spec == "heuristic":
        return HeuristicAgent(seed=seed)
    if spec == "human":
        return HumanConsoleAgent()
    if spec.startswith("qtable:"):
        path = spec.split(":", 1)[1]
        return QTableAgent(path=path, seed=seed)
    raise ValueError(f"unknown agent spec: {spec}")


def action_label(action: int) -> str:
    row, col = VALID_CELLS[action]
    return f"{action} ({row},{col})"
