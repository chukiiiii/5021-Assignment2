"""PPO actor-critic model for super tic-tac-toe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from models.dqn_model import action_mask, state_tensor
from env.super_tictactoe import BOARD_SIZE, VALID_CELLS, SuperTicTacToeEnv


class PPOActorCritic(nn.Module):
    def __init__(self, action_count: int = len(VALID_CELLS)) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, action_count)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value


def masked_action_distribution(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    masked = logits.clone()
    masked[~mask] = -1e9
    return Categorical(logits=masked)


def select_greedy_action(model: PPOActorCritic, env: SuperTicTacToeEnv) -> int:
    with torch.no_grad():
        state = state_tensor(env).unsqueeze(0)
        mask = action_mask(env)
        logits, _ = model(state)
        masked = logits[0].clone()
        masked[~mask] = -1e9
        return int(torch.argmax(masked).item())


def save_checkpoint(
    path: str,
    model: PPOActorCritic,
    metadata: dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    map_location: str | torch.device = "cpu",
) -> tuple[PPOActorCritic, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model = PPOActorCritic()
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload.get("metadata", {})
