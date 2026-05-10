"""Super tic-tac-toe environment for Assignment 2.

The board is represented as a 12x12 grid with only a triangular subset valid:

    rows 0-3:   columns 4-7
    rows 4-7:   columns 2-9
    rows 8-11:  columns 0-11

That gives 6 square regions of size 4x4, for 96 playable cells.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable


EMPTY = 0
X = 1
O = -1
INVALID = None

BOARD_SIZE = 12
LEVEL_HEIGHT = 4
WIN_HORIZONTAL = 4
WIN_VERTICAL = 4
WIN_DIAGONAL = 5

Cell = tuple[int, int]


def level_for_row(row: int) -> int:
    """Return level number 1, 2, or 3 for a row."""
    if not 0 <= row < BOARD_SIZE:
        raise ValueError(f"row out of board: {row}")
    return row // LEVEL_HEIGHT + 1


def valid_columns_for_row(row: int) -> range:
    """Return the valid column range for a row in the triangular board."""
    level = level_for_row(row)
    if level == 1:
        return range(4, 8)
    if level == 2:
        return range(2, 10)
    return range(0, 12)


def is_valid_cell(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and col in valid_columns_for_row(row)


def build_valid_cells() -> list[Cell]:
    return [
        (row, col)
        for row in range(BOARD_SIZE)
        for col in valid_columns_for_row(row)
    ]


VALID_CELLS = build_valid_cells()
CELL_TO_ACTION = {cell: action for action, cell in enumerate(VALID_CELLS)}
NEIGHBOR_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def adjacent_cells(row: int, col: int) -> list[Cell]:
    """Return the eight adjacent coordinates, including coordinates off-board."""
    return [(row + dr, col + dc) for dr, dc in NEIGHBOR_OFFSETS]


def _all_valid(cells: Iterable[Cell]) -> bool:
    return all(is_valid_cell(row, col) for row, col in cells)


def build_winning_lines() -> list[tuple[Cell, ...]]:
    """Build all winning windows under the assignment rules.

    Horizontal wins need 4 contiguous marks. Vertical wins need 4 contiguous
    marks and must include at least two board levels. Diagonal wins need 5
    contiguous marks.
    """
    lines: list[tuple[Cell, ...]] = []

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            horizontal = tuple((row, col + offset) for offset in range(WIN_HORIZONTAL))
            if _all_valid(horizontal):
                lines.append(horizontal)

            vertical = tuple((row + offset, col) for offset in range(WIN_VERTICAL))
            if _all_valid(vertical):
                levels = {level_for_row(cell_row) for cell_row, _ in vertical}
                if len(levels) >= 2:
                    lines.append(vertical)

            diagonal_down = tuple(
                (row + offset, col + offset) for offset in range(WIN_DIAGONAL)
            )
            if _all_valid(diagonal_down):
                lines.append(diagonal_down)

            diagonal_up = tuple(
                (row + offset, col - offset) for offset in range(WIN_DIAGONAL)
            )
            if _all_valid(diagonal_up):
                lines.append(diagonal_up)

    return lines


WINNING_LINES = build_winning_lines()


@dataclass(frozen=True)
class MoveInfo:
    player: int
    requested: Cell
    placed: Cell | None
    accepted: bool
    forfeited: bool
    winner: int | None
    reason: str


class SuperTicTacToeEnv:
    """A small Gym-like environment without third-party dependencies."""

    action_count = len(VALID_CELLS)

    def __init__(self, seed: int | None = None, stochastic: bool = True) -> None:
        self.random = random.Random(seed)
        self.stochastic = stochastic
        self.board: list[list[int | None]] = []
        self.current_player = X
        self.done = False
        self.winner: int | None = None
        self.last_info: MoveInfo | None = None
        self.reset()

    def reset(self, starting_player: int = X) -> tuple[int, ...]:
        self.board = [
            [EMPTY if is_valid_cell(row, col) else INVALID for col in range(BOARD_SIZE)]
            for row in range(BOARD_SIZE)
        ]
        self.current_player = starting_player
        self.done = False
        self.winner = None
        self.last_info = None
        return self.state()

    def state(self) -> tuple[int, ...]:
        return tuple(self.board[row][col] for row, col in VALID_CELLS)  # type: ignore[misc]

    def canonical_state(self) -> tuple[int, ...]:
        """State from the current player's perspective."""
        return tuple(value * self.current_player for value in self.state())

    def available_actions(self) -> list[int]:
        return [
            action
            for action, (row, col) in enumerate(VALID_CELLS)
            if self.board[row][col] == EMPTY
        ]

    def action_to_cell(self, action: int) -> Cell:
        if not 0 <= action < len(VALID_CELLS):
            raise ValueError(f"action out of range: {action}")
        return VALID_CELLS[action]

    def cell_to_action(self, row: int, col: int) -> int:
        try:
            return CELL_TO_ACTION[(row, col)]
        except KeyError as exc:
            raise ValueError(f"invalid board cell: {(row, col)}") from exc

    def step(self, action: int) -> tuple[tuple[int, ...], float, bool, MoveInfo]:
        """Take one move for the current player.

        Returns ``state, reward, done, info`` where reward is from the acting
        player's perspective: 1 for a win, 0 otherwise. Training code can add
        its own shaping if desired.
        """
        if self.done:
            raise RuntimeError("cannot step after game is done; call reset()")

        requested = self.action_to_cell(action)
        player = self.current_player
        row, col = requested
        if self.board[row][col] != EMPTY:
            self.done = True
            self.winner = -player
            info = MoveInfo(
                player=player,
                requested=requested,
                placed=None,
                accepted=False,
                forfeited=True,
                winner=self.winner,
                reason="illegal occupied action",
            )
            self.last_info = info
            return self.state(), -1.0, self.done, info

        placed = self._resolve_placement(requested)
        accepted = placed == requested
        forfeited = placed is None
        reason = "accepted" if accepted else "redirected"

        if placed is None:
            reason = "forfeited"
        else:
            place_row, place_col = placed
            self.board[place_row][place_col] = player
            self.winner = self.check_winner()

        if self.winner is not None:
            self.done = True
        elif not self.available_actions():
            self.done = True
            reason = "draw"
        else:
            self.current_player = -self.current_player

        info = MoveInfo(
            player=player,
            requested=requested,
            placed=placed,
            accepted=accepted,
            forfeited=forfeited,
            winner=self.winner,
            reason=reason,
        )
        self.last_info = info
        reward = 1.0 if self.winner == player else 0.0
        return self.state(), reward, self.done, info

    def _resolve_placement(self, requested: Cell) -> Cell | None:
        if not self.stochastic:
            return requested

        roll = self.random.randrange(16)
        if roll < 8:
            return requested

        target = adjacent_cells(*requested)[roll - 8]
        row, col = target
        if not is_valid_cell(row, col):
            return None
        if self.board[row][col] != EMPTY:
            return None
        return target

    def check_winner(self) -> int | None:
        for line in WINNING_LINES:
            values = [self.board[row][col] for row, col in line]
            first = values[0]
            if first != EMPTY and all(value == first for value in values):
                return first  # type: ignore[return-value]
        return None

    def render(self) -> str:
        symbols = {X: "X", O: "O", EMPTY: ".", INVALID: " "}
        rows = []
        for row in range(BOARD_SIZE):
            rows.append(" ".join(symbols[self.board[row][col]] for col in range(BOARD_SIZE)))
        return "\n".join(rows)


def player_name(player: int | None) -> str:
    if player == X:
        return "X"
    if player == O:
        return "O"
    return "draw"
