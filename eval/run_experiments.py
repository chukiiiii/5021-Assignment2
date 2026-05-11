"""Run grouped multi-seed experiments and summarize results."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

from agents.agents import make_agent
from env.match import MatchSummary, evaluate_pair


@dataclass(frozen=True)
class Experiment:
    name: str
    agent_a: str
    agent_b: str
    stochastic: bool = True
    games: int | None = None


EXPERIMENTS = [
    Experiment("random_vs_random", "random", "random"),
    Experiment("qtable_vs_random", "qtable:results/checkpoints/q_table.json", "random"),
    Experiment("heuristic_vs_random", "heuristic", "random"),
    Experiment("mcts20c6_vs_random", "mcts:20:6", "random"),
    Experiment("mcts50c6_vs_heuristic", "mcts:50:6", "heuristic"),
    Experiment("dqn_stochastic_vs_random", "dqn:results/checkpoints/dqn_stochastic_smoke.pt", "random"),
    Experiment("dqn_stochastic_vs_heuristic", "dqn:results/checkpoints/dqn_stochastic_smoke.pt", "heuristic"),
    Experiment("ppo_stochastic_vs_random", "ppo:results/checkpoints/ppo_stochastic_smoke.pt", "random"),
    Experiment("ppo_stochastic_vs_heuristic", "ppo:results/checkpoints/ppo_stochastic_smoke.pt", "heuristic"),
    Experiment("bc_mixed_vs_random", "ppo:results/checkpoints/bc_mixed.pt", "random"),
    Experiment("bc_mixed_vs_heuristic", "ppo:results/checkpoints/bc_mixed.pt", "heuristic"),
    Experiment("sft_ppo_vs_random", "ppo:results/checkpoints/sft_ppo_seed801.pt", "random"),
    Experiment("sft_ppo_vs_heuristic", "ppo:results/checkpoints/sft_ppo_seed801.pt", "heuristic"),
    Experiment("sft_ppo_vs_mcts20c6", "ppo:results/checkpoints/sft_ppo_seed801.pt", "mcts:20:6"),
    Experiment("bc_large_vs_random", "ppo:results/checkpoints/bc_mixed_large.pt", "random"),
    Experiment("bc_large_vs_heuristic", "ppo:results/checkpoints/bc_mixed_large.pt", "heuristic"),
    Experiment("sft_ppo_conservative_vs_random", "ppo:results/checkpoints/sft_ppo_conservative_seed901.pt", "random"),
    Experiment("sft_ppo_conservative_vs_heuristic", "ppo:results/checkpoints/sft_ppo_conservative_seed901.pt", "heuristic"),
    Experiment("sft_ppo_conservative_vs_mcts20c6", "ppo:results/checkpoints/sft_ppo_conservative_seed901.pt", "mcts:20:6"),
]

SMOKE_EXPERIMENTS = [
    Experiment("random_vs_random", "random", "random"),
    Experiment("heuristic_vs_random", "heuristic", "random"),
    Experiment("mcts20c6_vs_random", "mcts:20:6", "random"),
    Experiment("dqn_stochastic_vs_random", "dqn:results/checkpoints/dqn_stochastic_smoke.pt", "random"),
    Experiment("ppo_stochastic_vs_random", "ppo:results/checkpoints/ppo_stochastic_smoke.pt", "random"),
]


def disambiguate(agent_a: object, agent_b: object) -> None:
    if getattr(agent_a, "name") == getattr(agent_b, "name"):
        agent_a.name = f"A:{agent_a.name}"  # type: ignore[attr-defined]
        agent_b.name = f"B:{agent_b.name}"  # type: ignore[attr-defined]


def reward(summary: MatchSummary) -> float:
    return summary.agent_a_win_rate - summary.agent_b_win_rate


def run_one(
    experiment: Experiment,
    seed: int,
    games: int,
    progress: bool,
) -> dict[str, float | int | str | bool]:
    agent_a = make_agent(experiment.agent_a, seed=seed)
    agent_b = make_agent(experiment.agent_b, seed=seed + 10_000)
    disambiguate(agent_a, agent_b)
    start = perf_counter()

    def progress_callback(done_games: int, result: object) -> None:
        if progress:
            print(
                f"{experiment.name} seed={seed}: {done_games}/{games}, "
                f"last={result.winner_agent or result.winner}, turns={result.turns}",
                flush=True,
            )

    summary, _ = evaluate_pair(
        agent_a=agent_a,
        agent_b=agent_b,
        games=games,
        seed=seed,
        stochastic=experiment.stochastic,
        alternate=True,
        progress_callback=progress_callback if progress else None,
    )
    elapsed = perf_counter() - start
    return {
        "experiment": experiment.name,
        "agent_a": summary.agent_a,
        "agent_b": summary.agent_b,
        "seed": seed,
        "games": games,
        "stochastic": experiment.stochastic,
        "agent_a_win_rate": summary.agent_a_win_rate,
        "agent_b_win_rate": summary.agent_b_win_rate,
        "draw_rate": summary.draw_rate,
        "agent_a_reward": reward(summary),
        "avg_turns": summary.avg_turns,
        "avg_forfeits": summary.avg_forfeits,
        "avg_redirects": summary.avg_redirects,
        "elapsed_seconds": elapsed,
    }


def summarize(rows: list[dict[str, float | int | str | bool]]) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[dict[str, float | int | str | bool]]] = {}
    for row in rows:
        grouped.setdefault(str(row["experiment"]), []).append(row)

    summary_rows = []
    for name, group in grouped.items():
        first = group[0]
        win_rates = [float(row["agent_a_win_rate"]) for row in group]
        rewards = [float(row["agent_a_reward"]) for row in group]
        turns = [float(row["avg_turns"]) for row in group]
        forfeits = [float(row["avg_forfeits"]) for row in group]
        redirects = [float(row["avg_redirects"]) for row in group]
        elapsed = [float(row["elapsed_seconds"]) for row in group]
        summary_rows.append(
            {
                "experiment": name,
                "agent_a": first["agent_a"],
                "agent_b": first["agent_b"],
                "seeds": len(group),
                "games_total": sum(int(row["games"]) for row in group),
                "mean_win_rate": mean(win_rates),
                "std_win_rate": stdev(win_rates) if len(win_rates) > 1 else 0.0,
                "mean_reward": mean(rewards),
                "std_reward": stdev(rewards) if len(rewards) > 1 else 0.0,
                "mean_turns": mean(turns),
                "mean_forfeits": mean(forfeits),
                "mean_redirects": mean(redirects),
                "total_elapsed_seconds": sum(elapsed),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, summary_rows: list[dict[str, object]], detail_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_body = "\n".join(
        "<tr>"
        f"<td>{row['experiment']}</td>"
        f"<td>{row['agent_a']}</td>"
        f"<td>{row['agent_b']}</td>"
        f"<td>{int(row['games_total'])}</td>"
        f"<td>{float(row['mean_win_rate']):.3f}</td>"
        f"<td>{float(row['std_win_rate']):.3f}</td>"
        f"<td>{float(row['mean_reward']):.3f}</td>"
        f"<td>{float(row['mean_turns']):.2f}</td>"
        f"<td>{float(row['mean_forfeits']):.2f}</td>"
        f"<td>{float(row['mean_redirects']):.2f}</td>"
        "</tr>"
        for row in summary_rows
    )
    detail_body = "\n".join(
        "<tr>"
        f"<td>{row['experiment']}</td>"
        f"<td>{row['seed']}</td>"
        f"<td>{row['games']}</td>"
        f"<td>{float(row['agent_a_win_rate']):.3f}</td>"
        f"<td>{float(row['agent_a_reward']):.3f}</td>"
        f"<td>{float(row['avg_turns']):.2f}</td>"
        f"<td>{float(row['elapsed_seconds']):.1f}</td>"
        "</tr>"
        for row in detail_rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Super Tic-Tac-Toe Experiment Summary</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0 32px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    h1, h2 {{ margin-bottom: 4px; }}
  </style>
</head>
<body>
  <h1>Experiment Summary</h1>
  <h2>Aggregate</h2>
  <table>
    <thead><tr><th>Experiment</th><th>Agent A</th><th>Agent B</th><th>Games</th><th>Mean Win</th><th>Std Win</th><th>Mean Reward</th><th>Turns</th><th>Forfeits</th><th>Redirects</th></tr></thead>
    <tbody>{summary_body}</tbody>
  </table>
  <h2>Per Seed</h2>
  <table>
    <thead><tr><th>Experiment</th><th>Seed</th><th>Games</th><th>Win Rate</th><th>Reward</th><th>Turns</th><th>Seconds</th></tr></thead>
    <tbody>{detail_body}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def choose_experiments(preset: str) -> list[Experiment]:
    if preset == "smoke":
        return SMOKE_EXPERIMENTS
    if preset == "full":
        return EXPERIMENTS
    raise ValueError(f"unknown preset: {preset}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--mcts-games", type=int, default=None)
    parser.add_argument("--output-dir", default="results/summary")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    experiments = choose_experiments(args.preset)
    detail_rows: list[dict[str, float | int | str | bool]] = []

    for experiment in experiments:
        games = experiment.games or args.games
        if args.mcts_games is not None and experiment.name.startswith("mcts"):
            games = args.mcts_games
        for seed in args.seeds:
            print(f"Running {experiment.name} seed={seed} games={games}", flush=True)
            detail_rows.append(run_one(experiment, seed, games, progress=args.progress))

    summary_rows = summarize(detail_rows)
    prefix = f"{args.preset}_s{len(args.seeds)}_g{args.games}"
    detail_path = output_dir / f"{prefix}_detail.csv"
    summary_path = output_dir / f"{prefix}_summary.csv"
    html_path = output_dir / f"{prefix}_summary.html"
    write_csv(detail_path, detail_rows)
    write_csv(summary_path, summary_rows)
    write_html(html_path, summary_rows, detail_rows)

    print(f"Saved detail CSV to {detail_path}")
    print(f"Saved summary CSV to {summary_path}")
    print(f"Saved summary HTML to {html_path}")
    print()
    for row in summary_rows:
        print(
            f"{row['experiment']}: mean_win={float(row['mean_win_rate']):.3f}, "
            f"mean_reward={float(row['mean_reward']):.3f}, "
            f"games={row['games_total']}"
        )


if __name__ == "__main__":
    main()
