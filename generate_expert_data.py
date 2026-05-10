"""Generate expert trajectories for behavior cloning."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import torch

from agents import Agent, make_agent
from dqn_model import action_mask, state_tensor
from super_tictactoe import SuperTicTacToeEnv, player_name


@dataclass(frozen=True)
class PartialSample:
    state: torch.Tensor
    mask: torch.Tensor
    action: int
    source_agent: str
    game_index: int
    turn: int
    player: int


def acting_outcome(winner: int | None, player: int) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner == player else -1.0


def play_expert_game(
    x_agent: Agent,
    o_agent: Agent,
    game_index: int,
    seed: int,
    stochastic: bool,
) -> tuple[list[PartialSample], int | None, int]:
    env = SuperTicTacToeEnv(seed=seed, stochastic=stochastic)
    x_agent.reset()
    o_agent.reset()
    samples: list[PartialSample] = []
    turn = 0

    while not env.done:
        agent = x_agent if env.current_player == 1 else o_agent
        sample = PartialSample(
            state=state_tensor(env),
            mask=action_mask(env),
            action=agent.select_action(env),
            source_agent=agent.name,
            game_index=game_index,
            turn=turn + 1,
            player=env.current_player,
        )
        env.step(sample.action)
        samples.append(sample)
        turn += 1

    return samples, env.winner, turn


def collect_dataset(
    heuristic_games: int,
    mcts_games: int,
    seed: int,
    stochastic: bool,
    mcts_spec: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    all_samples: list[PartialSample] = []
    all_outcomes: list[float] = []
    summary_rows: list[dict[str, object]] = []
    game_index = 0

    game_specs: list[tuple[str, str, str, int]] = [
        ("heuristic_self", "heuristic", "heuristic", heuristic_games),
        ("mcts_vs_heuristic", mcts_spec, "heuristic", mcts_games),
    ]
    for source, x_spec, o_spec, count in game_specs:
        turns: list[int] = []
        x_wins = 0
        o_wins = 0
        draws = 0
        start_samples = len(all_samples)
        for local_index in range(count):
            x_agent = make_agent(x_spec, seed=seed + game_index * 2)
            o_agent = make_agent(o_spec, seed=seed + game_index * 2 + 1)
            samples, winner, game_turns = play_expert_game(
                x_agent=x_agent,
                o_agent=o_agent,
                game_index=game_index,
                seed=seed + game_index,
                stochastic=stochastic,
            )
            all_samples.extend(samples)
            all_outcomes.extend(acting_outcome(winner, sample.player) for sample in samples)
            turns.append(game_turns)
            if winner == 1:
                x_wins += 1
            elif winner == -1:
                o_wins += 1
            else:
                draws += 1
            print(
                f"{source}: {local_index + 1}/{count}, winner={player_name(winner)}, turns={game_turns}",
                flush=True,
            )
            game_index += 1

        summary_rows.append(
            {
                "source": source,
                "x_spec": x_spec,
                "o_spec": o_spec,
                "games": count,
                "samples": len(all_samples) - start_samples,
                "x_win_rate": x_wins / count if count else 0.0,
                "o_win_rate": o_wins / count if count else 0.0,
                "draw_rate": draws / count if count else 0.0,
                "avg_turns": mean(turns) if turns else 0.0,
            }
        )

    records = [
        {
            "source_agent": sample.source_agent,
            "game_index": sample.game_index,
            "turn": sample.turn,
            "player": player_name(sample.player),
            "outcome": all_outcomes[index],
        }
        for index, sample in enumerate(all_samples)
    ]
    dataset = {
        "states": torch.stack([sample.state for sample in all_samples]) if all_samples else torch.empty(0, 3, 12, 12),
        "masks": torch.stack([sample.mask for sample in all_samples]) if all_samples else torch.empty(0, 96, dtype=torch.bool),
        "actions": torch.tensor([sample.action for sample in all_samples], dtype=torch.long),
        "outcomes": torch.tensor(all_outcomes, dtype=torch.float32),
        "records": records,
        "metadata": {
            "heuristic_games": heuristic_games,
            "mcts_games": mcts_games,
            "seed": seed,
            "stochastic": stochastic,
            "mcts_spec": mcts_spec,
            "note": "Mixed heuristic/MCTS expert data for behavior cloning.",
        },
    }
    return dataset, summary_rows


def write_summary(path: str, rows: list[dict[str, object]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heuristic-games", type=int, default=200)
    parser.add_argument("--mcts-games", type=int, default=30)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--mcts-spec", default="mcts:20:6")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", default="results/expert/mixed_expert.pt")
    parser.add_argument("--summary", default="results/expert/mixed_expert_summary.csv")
    args = parser.parse_args()

    dataset, summary_rows = collect_dataset(
        heuristic_games=args.heuristic_games,
        mcts_games=args.mcts_games,
        seed=args.seed,
        stochastic=not args.deterministic,
        mcts_spec=args.mcts_spec,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args.output)
    write_summary(args.summary, summary_rows)
    print(f"Saved expert data to {args.output}")
    print(f"Saved summary CSV to {args.summary}")
    print(f"samples: {int(dataset['actions'].shape[0])}")


if __name__ == "__main__":
    main()
