"""Reusable match runner and win-rate statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Callable

from agents.agents import Agent
from env.super_tictactoe import O, X, MoveInfo, SuperTicTacToeEnv, player_name


@dataclass(frozen=True)
class MoveRecord:
    turn: int
    player: str
    agent: str
    requested: tuple[int, int]
    placed: tuple[int, int] | None
    accepted: bool
    forfeited: bool
    reason: str


@dataclass(frozen=True)
class GameResult:
    game_index: int
    x_agent: str
    o_agent: str
    winner: str
    winner_agent: str | None
    turns: int
    forfeits: int
    redirects: int
    transcript: list[MoveRecord]
    final_board: str


@dataclass(frozen=True)
class MatchSummary:
    agent_a: str
    agent_b: str
    games: int
    agent_a_wins: int
    agent_b_wins: int
    draws: int
    agent_a_win_rate: float
    agent_b_win_rate: float
    draw_rate: float
    x_win_rate: float
    o_win_rate: float
    avg_turns: float
    avg_forfeits: float
    avg_redirects: float


def _agent_for_player(player: int, x_agent: Agent, o_agent: Agent) -> Agent:
    return x_agent if player == X else o_agent


def _winner_agent_name(winner: int | None, x_agent: Agent, o_agent: Agent) -> str | None:
    if winner == X:
        return x_agent.name
    if winner == O:
        return o_agent.name
    return None


def _record_move(turn: int, agent: Agent, info: MoveInfo) -> MoveRecord:
    return MoveRecord(
        turn=turn,
        player=player_name(info.player),
        agent=agent.name,
        requested=info.requested,
        placed=info.placed,
        accepted=info.accepted,
        forfeited=info.forfeited,
        reason=info.reason,
    )


def play_game(
    x_agent: Agent,
    o_agent: Agent,
    game_index: int = 0,
    seed: int | None = None,
    stochastic: bool = True,
    record: bool = False,
) -> GameResult:
    env = SuperTicTacToeEnv(seed=seed, stochastic=stochastic)
    x_agent.reset()
    o_agent.reset()

    transcript: list[MoveRecord] = []
    forfeits = 0
    redirects = 0
    turns = 0

    while not env.done:
        agent = _agent_for_player(env.current_player, x_agent, o_agent)
        action = agent.select_action(env)
        _, _, _, info = env.step(action)
        turns += 1
        if info.forfeited:
            forfeits += 1
        elif not info.accepted:
            redirects += 1
        if record:
            transcript.append(_record_move(turns, agent, info))

    return GameResult(
        game_index=game_index,
        x_agent=x_agent.name,
        o_agent=o_agent.name,
        winner=player_name(env.winner),
        winner_agent=_winner_agent_name(env.winner, x_agent, o_agent),
        turns=turns,
        forfeits=forfeits,
        redirects=redirects,
        transcript=transcript,
        final_board=env.render(),
    )


def evaluate_pair(
    agent_a: Agent,
    agent_b: Agent,
    games: int,
    seed: int = 7,
    stochastic: bool = True,
    alternate: bool = True,
    progress_callback: Callable[[int, GameResult], None] | None = None,
) -> tuple[MatchSummary, list[GameResult]]:
    results: list[GameResult] = []
    agent_a_wins = 0
    agent_b_wins = 0
    draws = 0
    x_wins = 0
    o_wins = 0

    for game_index in range(games):
        a_is_x = (game_index % 2 == 0) if alternate else True
        x_agent = agent_a if a_is_x else agent_b
        o_agent = agent_b if a_is_x else agent_a
        result = play_game(
            x_agent=x_agent,
            o_agent=o_agent,
            game_index=game_index,
            seed=seed + game_index,
            stochastic=stochastic,
            record=False,
        )
        results.append(result)
        if progress_callback is not None:
            progress_callback(game_index + 1, result)

        if result.winner == "draw":
            draws += 1
        elif result.winner == "X":
            x_wins += 1
            if a_is_x:
                agent_a_wins += 1
            else:
                agent_b_wins += 1
        elif result.winner == "O":
            o_wins += 1
            if a_is_x:
                agent_b_wins += 1
            else:
                agent_a_wins += 1

    summary = MatchSummary(
        agent_a=agent_a.name,
        agent_b=agent_b.name,
        games=games,
        agent_a_wins=agent_a_wins,
        agent_b_wins=agent_b_wins,
        draws=draws,
        agent_a_win_rate=agent_a_wins / games if games else 0.0,
        agent_b_win_rate=agent_b_wins / games if games else 0.0,
        draw_rate=draws / games if games else 0.0,
        x_win_rate=x_wins / games if games else 0.0,
        o_win_rate=o_wins / games if games else 0.0,
        avg_turns=mean(result.turns for result in results) if results else 0.0,
        avg_forfeits=mean(result.forfeits for result in results) if results else 0.0,
        avg_redirects=mean(result.redirects for result in results) if results else 0.0,
    )
    return summary, results


def summary_dict(summary: MatchSummary) -> dict[str, object]:
    return asdict(summary)


def results_dict(results: list[GameResult]) -> list[dict[str, object]]:
    rows = []
    for result in results:
        row = asdict(result)
        row.pop("transcript", None)
        row.pop("final_board", None)
        rows.append(row)
    return rows
