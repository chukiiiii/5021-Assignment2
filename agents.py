"""Reusable agents for evaluation and demos."""

from __future__ import annotations

import json
import math
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


class DQNAgent:
    def __init__(
        self,
        path: str,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        from dqn_model import action_mask, load_checkpoint, masked_argmax, state_tensor

        self.path = path
        self.name = name or f"dqn:{path}"
        self.random = random.Random(seed)
        self.model, self.metadata = load_checkpoint(path)
        self.action_mask = action_mask
        self.masked_argmax = masked_argmax
        self.state_tensor = state_tensor

    def reset(self) -> None:
        pass

    def select_action(self, env: SuperTicTacToeEnv) -> int:
        import torch

        with torch.no_grad():
            state = self.state_tensor(env).unsqueeze(0)
            mask = self.action_mask(env)
            q_values = self.model(state)[0]
            return self.masked_argmax(q_values, mask)


class PPOAgent:
    def __init__(
        self,
        path: str,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        from ppo_model import load_checkpoint, select_greedy_action

        self.path = path
        self.name = name or f"ppo:{path}"
        self.random = random.Random(seed)
        self.model, self.metadata = load_checkpoint(path)
        self.select_greedy_action = select_greedy_action

    def reset(self) -> None:
        pass

    def select_action(self, env: SuperTicTacToeEnv) -> int:
        return self.select_greedy_action(self.model, env)


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


def _immediate_winning_cells(
    env: SuperTicTacToeEnv,
    player: int,
) -> set[tuple[int, int]]:
    cells = set()
    for action in env.available_actions():
        row, col = env.action_to_cell(action)
        env.board[row][col] = player
        if env.check_winner() == player:
            cells.add((row, col))
        env.board[row][col] = EMPTY
    return cells


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
        player = env.current_player
        own_winning = _immediate_winning_cells(env, player)
        opponent_winning = _immediate_winning_cells(env, -player)
        scored = [
            (
                self._score_action(
                    env,
                    action,
                    own_winning=own_winning,
                    opponent_winning=opponent_winning,
                ),
                action,
            )
            for action in actions
        ]
        best_score = max(score for score, _ in scored)
        best_actions = [action for score, action in scored if score == best_score]
        return self.random.choice(best_actions)

    def score_action(self, env: SuperTicTacToeEnv, action: int) -> float:
        player = env.current_player
        return self._score_action(
            env,
            action,
            own_winning=_immediate_winning_cells(env, player),
            opponent_winning=_immediate_winning_cells(env, -player),
        )

    def _score_action(
        self,
        env: SuperTicTacToeEnv,
        action: int,
        own_winning: set[tuple[int, int]],
        opponent_winning: set[tuple[int, int]],
    ) -> float:
        player = env.current_player
        requested = env.action_to_cell(action)
        total = 0.0

        for target, probability in _placement_outcomes(env, action):
            if target is None:
                miss_penalty = 500_000.0 if opponent_winning else 0.0
                total += probability * (_board_score(env, player) - 35.0 - miss_penalty)
                continue

            row, col = target
            env.board[row][col] = player
            winner = env.check_winner()
            if winner == player:
                outcome_score = 10_000_000.0
            else:
                outcome_score = _board_score(env, player)
                outcome_score += _cell_shape_score(env, player, target)
                if target in own_winning:
                    outcome_score += 8_000_000.0
                if target in opponent_winning:
                    outcome_score += 4_000_000.0
                elif opponent_winning:
                    outcome_score -= 500_000.0
                if target == requested:
                    outcome_score += 2.0
            env.board[row][col] = EMPTY
            total += probability * outcome_score

        return total


class MCTSNode:
    def __init__(
        self,
        env: SuperTicTacToeEnv,
        action_scores: list[tuple[int, float]],
    ) -> None:
        self.state_id = (env.current_player, env.state())
        self.current_player = env.current_player
        self.untried_actions = [action for action, _ in action_scores]
        self.action_priors = {action: score for action, score in action_scores}
        self.children: dict[int, list[MCTSNode]] = {}
        self.visits = 0
        self.value = 0.0


def _clone_env(env: SuperTicTacToeEnv, seed: int) -> SuperTicTacToeEnv:
    clone = SuperTicTacToeEnv(seed=seed, stochastic=env.stochastic)
    clone.board = [row[:] for row in env.board]
    clone.current_player = env.current_player
    clone.done = env.done
    clone.winner = env.winner
    clone.last_info = env.last_info
    return clone


def _bounded_eval(env: SuperTicTacToeEnv, root_player: int) -> float:
    if env.winner == root_player:
        return 1.0
    if env.winner == -root_player:
        return -1.0
    if env.winner is None and env.done:
        return 0.0
    score = _board_score(env, root_player)
    return max(-0.5, min(0.5, score / 8_000.0))


class MCTSAgent:
    """Monte Carlo tree search agent with sampled stochastic transitions."""

    def __init__(
        self,
        simulations: int = 100,
        exploration: float = 1.4,
        rollout_depth: int = 10,
        rollout_randomness: float = 1.0,
        max_candidates: int = 12,
        prior_weight: float = 0.25,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        self.simulations = simulations
        self.exploration = exploration
        self.rollout_depth = rollout_depth
        self.rollout_randomness = rollout_randomness
        self.max_candidates = max_candidates
        self.prior_weight = prior_weight
        self.random = random.Random(seed)
        self.rollout_heuristic = HeuristicAgent(seed=seed)
        self.name = name or f"mcts:{simulations}"

    def reset(self) -> None:
        pass

    def select_action(self, env: SuperTicTacToeEnv) -> int:
        actions = env.available_actions()
        if len(actions) == 1:
            return actions[0]

        root = MCTSNode(env, self._candidate_action_scores(env))
        root_player = env.current_player

        for _ in range(self.simulations):
            sim_env = _clone_env(env, self.random.randrange(1_000_000_000))
            path = [root]
            node = root

            while not sim_env.done and not node.untried_actions:
                action = self._select_tree_action(node, root_player)
                sim_env.step(action)
                node = self._child_for_state(node, action, sim_env)
                path.append(node)

            if not sim_env.done and node.untried_actions:
                action = node.untried_actions.pop(0)
                sim_env.step(action)
                node = self._child_for_state(node, action, sim_env)
                path.append(node)

            value = self._rollout(sim_env, root_player)
            for visited in path:
                visited.visits += 1
                visited.value += value

        return self._best_root_action(root, actions)

    def _candidate_action_scores(self, env: SuperTicTacToeEnv) -> list[tuple[int, float]]:
        actions = env.available_actions()
        player = env.current_player
        own_winning = _immediate_winning_cells(env, player)
        opponent_winning = _immediate_winning_cells(env, -player)
        scored = [
            (
                self._fast_action_score(
                    env,
                    action,
                    player,
                    own_winning=own_winning,
                    opponent_winning=opponent_winning,
                ),
                self.random.random(),
                action,
            )
            for action in actions
        ]
        scored.sort(reverse=True)
        limit = min(self.max_candidates, len(scored))
        return [(action, score) for score, _, action in scored[:limit]]

    def _fast_action_score(
        self,
        env: SuperTicTacToeEnv,
        action: int,
        player: int,
        own_winning: set[tuple[int, int]],
        opponent_winning: set[tuple[int, int]],
    ) -> float:
        opponent = -player
        requested = env.action_to_cell(action)
        total = 0.0

        for target, probability in _placement_outcomes(env, action):
            if target is None:
                miss_penalty = 500_000.0 if opponent_winning else 25.0
                total -= miss_penalty * probability
                continue

            block_bonus = 0.0
            if target in own_winning:
                block_bonus += 8_000_000.0
            if target in opponent_winning:
                block_bonus += 4_000_000.0
            elif opponent_winning:
                block_bonus -= 500_000.0
            for line in LINES_BY_CELL[target]:
                values = [env.board[row][col] for row, col in line]
                if values.count(opponent) == len(line) - 1 and values.count(EMPTY) == 1:
                    block_bonus += 4_000.0

            row, col = target
            env.board[row][col] = player
            winner = env.check_winner()
            if winner == player:
                outcome_score = 10_000_000.0
            else:
                outcome_score = block_bonus + _cell_shape_score(env, player, target)
                for line in LINES_BY_CELL[target]:
                    values = [env.board[line_row][line_col] for line_row, line_col in line]
                    player_count = values.count(player)
                    opponent_count = values.count(opponent)
                    empty_count = values.count(EMPTY)
                    if opponent_count == 0:
                        outcome_score += _line_score(player_count, empty_count, len(line))
                    if player_count == 0:
                        outcome_score -= 1.1 * _line_score(opponent_count, empty_count, len(line))
                if target == requested:
                    outcome_score += 2.0
            env.board[row][col] = EMPTY
            total += probability * outcome_score

        return total

    def _select_tree_action(self, node: MCTSNode, root_player: int) -> int:
        log_parent = math.log(max(1, node.visits))
        best_score = -float("inf")
        best_actions: list[int] = []
        for action, children in node.children.items():
            visits = sum(child.visits for child in children)
            value = sum(child.value for child in children)
            if visits == 0:
                score = float("inf")
            else:
                exploit = value / visits
                if node.current_player != root_player:
                    exploit = -exploit
                explore = self.exploration * math.sqrt(log_parent / visits)
                prior = math.tanh(node.action_priors.get(action, 0.0) / 5_000.0)
                score = exploit + explore + self.prior_weight * prior
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
        return self.random.choice(best_actions)

    def _child_for_state(
        self,
        node: MCTSNode,
        action: int,
        env: SuperTicTacToeEnv,
    ) -> MCTSNode:
        state_id = (env.current_player, env.state())
        children = node.children.setdefault(action, [])
        for child in children:
            if child.state_id == state_id:
                return child
        child = MCTSNode(env, self._candidate_action_scores(env))
        children.append(child)
        return child

    def _rollout(self, env: SuperTicTacToeEnv, root_player: int) -> float:
        depth = 0
        while not env.done and depth < self.rollout_depth:
            actions = env.available_actions()
            if self.random.random() < self.rollout_randomness:
                action = self.random.choice(actions)
            else:
                action = self.rollout_heuristic.select_action(env)
            env.step(action)
            depth += 1
        return _bounded_eval(env, root_player)

    def _best_root_action(self, root: MCTSNode, fallback_actions: list[int]) -> int:
        visited_actions = [
            (
                sum(child.value for child in children)
                / max(1, sum(child.visits for child in children)),
                sum(child.visits for child in children),
                math.tanh(root.action_priors.get(action, 0.0) / 5_000.0),
                action,
            )
            for action, children in root.children.items()
        ]
        if not visited_actions:
            return self.random.choice(fallback_actions)
        best_value = max(
            value + self.prior_weight * prior
            for value, _, prior, _ in visited_actions
        )
        best_actions = [
            action
            for value, _, prior, action in visited_actions
            if value + self.prior_weight * prior == best_value
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
    - `heuristic`
    - `mcts`, `mcts:simulations`, or `mcts:simulations:max_candidates`
    - `qtable:path/to/q_table.json`
    - `dqn:path/to/checkpoint.pt`
    - `ppo:path/to/checkpoint.pt`
    - `human`
    """
    if spec == "random":
        return RandomAgent(seed=seed)
    if spec == "heuristic":
        return HeuristicAgent(seed=seed)
    if spec == "mcts":
        return MCTSAgent(seed=seed)
    if spec.startswith("mcts:"):
        parts = spec.split(":")
        simulations = int(parts[1])
        max_candidates = int(parts[2]) if len(parts) >= 3 else 12
        return MCTSAgent(
            simulations=simulations,
            max_candidates=max_candidates,
            seed=seed,
        )
    if spec == "human":
        return HumanConsoleAgent()
    if spec.startswith("qtable:"):
        path = spec.split(":", 1)[1]
        return QTableAgent(path=path, seed=seed)
    if spec.startswith("dqn:"):
        path = spec.split(":", 1)[1]
        return DQNAgent(path=path, seed=seed)
    if spec.startswith("ppo:"):
        path = spec.split(":", 1)[1]
        return PPOAgent(path=path, seed=seed)
    raise ValueError(f"unknown agent spec: {spec}")


def action_label(action: int) -> str:
    row, col = VALID_CELLS[action]
    return f"{action} ({row},{col})"
