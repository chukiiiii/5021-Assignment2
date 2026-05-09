"""Interactive or scripted single-game display."""

from __future__ import annotations

import argparse

from agents import action_label, make_agent
from match import play_game
from super_tictactoe import SuperTicTacToeEnv, player_name


def print_action_map() -> None:
    env = SuperTicTacToeEnv(stochastic=False)
    print("Action map: action(row,col)")
    for row in range(12):
        labels = []
        for col in range(12):
            try:
                action = env.cell_to_action(row, col)
                labels.append(f"{action:02d}")
            except ValueError:
                labels.append("  ")
        print(" ".join(labels))


def play_verbose(x_spec: str, o_spec: str, seed: int, stochastic: bool) -> None:
    x_agent = make_agent(x_spec, seed=seed)
    o_agent = make_agent(o_spec, seed=seed + 1)
    env = SuperTicTacToeEnv(seed=seed, stochastic=stochastic)
    x_agent.reset()
    o_agent.reset()

    print_action_map()
    print()
    print(env.render())

    turn = 0
    while not env.done:
        turn += 1
        agent = x_agent if env.current_player == 1 else o_agent
        print()
        print(f"Turn {turn}: {agent.name} as {player_name(env.current_player)}")
        action = agent.select_action(env)
        print(f"Requested {action_label(action)}")
        _, _, _, info = env.step(action)
        if info.placed is None:
            print(f"Result: {info.reason}")
        else:
            print(f"Result: placed at {info.placed}, {info.reason}")
        print(env.render())

    print()
    if env.winner is None:
        print("Game over: draw")
    else:
        winner_agent = x_agent.name if env.winner == 1 else o_agent.name
        print(f"Game over: {winner_agent} wins as {player_name(env.winner)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", default="human", help="human | random | heuristic | qtable:path")
    parser.add_argument("--o", default="random", help="human | random | heuristic | qtable:path")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="print only final transcript summary")
    args = parser.parse_args()

    if args.quiet:
        result = play_game(
            x_agent=make_agent(args.x, seed=args.seed),
            o_agent=make_agent(args.o, seed=args.seed + 1),
            seed=args.seed,
            stochastic=not args.deterministic,
            record=True,
        )
        print(f"{result.x_agent} vs {result.o_agent}")
        print(f"winner: {result.winner_agent or result.winner}")
        print(f"turns: {result.turns}, forfeits: {result.forfeits}, redirects: {result.redirects}")
        print(result.final_board)
        return

    if args.x != "human" and args.o != "human":
        play_verbose(args.x, args.o, args.seed, stochastic=not args.deterministic)
        return

    play_verbose(args.x, args.o, args.seed, stochastic=not args.deterministic)


if __name__ == "__main__":
    main()
