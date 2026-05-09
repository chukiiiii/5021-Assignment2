"""Neural DQN model and tensor helpers for super tic-tac-toe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from super_tictactoe import BOARD_SIZE, EMPTY, SuperTicTacToeEnv, VALID_CELLS


def state_tensor(env: SuperTicTacToeEnv) -> torch.Tensor:
    """Encode the board from the current player's perspective.

    Channels:
    0. Current player's stones.
    1. Opponent stones.
    2. Empty legal cells.
    """
    tensor = torch.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    player = env.current_player
    for row, col in VALID_CELLS:
        value = env.board[row][col]
        if value == player:
            tensor[0, row, col] = 1.0
        elif value == -player:
            tensor[1, row, col] = 1.0
        elif value == EMPTY:
            tensor[2, row, col] = 1.0
    return tensor


def action_mask(env: SuperTicTacToeEnv) -> torch.Tensor:
    mask = torch.zeros(len(VALID_CELLS), dtype=torch.bool)
    for action in env.available_actions():
        mask[action] = True
    return mask


class DQN(nn.Module):
    def __init__(self, action_count: int = len(VALID_CELLS)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, action_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def masked_argmax(q_values: torch.Tensor, mask: torch.Tensor) -> int:
    masked = q_values.clone()
    masked[~mask] = -1e9
    return int(torch.argmax(masked).item())


def save_checkpoint(
    path: str,
    model: DQN,
    metadata: dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> tuple[DQN, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model = DQN()
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload.get("metadata", {})
