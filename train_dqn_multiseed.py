"""Train DQN across several seeds and run diagnostic matchups."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

import torch

from agents import make_agent
from dqn_model import save_checkpoint
from match import MatchSummary, evaluate_pair
from train_dqn import save_history_csv, save_history_html, train


@dataclass(frozen=True)
class DiagnosticOpponent:
    name: str
    spec: str
    games: int


def reward(summary: MatchSummary) -> float:
    return summary.agent_a_win_rate - summary.agent_b_win_rate


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], group_key: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row[group_key]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for name, group in grouped.items():
        first = group[0]
        win_rates = [float(row["win_rate"]) for row in group]
        rewards = [float(row["avg_reward"]) for row in group]
        turns = [float(row["avg_turns"]) for row in group]
        summary_rows.append(
            {
                group_key: name,
                "seeds": len(group),
                "games_total": sum(int(row.get("games", 0)) for row in group),
                "mean_win_rate": mean(win_rates),
                "std_win_rate": stdev(win_rates) if len(win_rates) > 1 else 0.0,
                "mean_reward": mean(rewards),
                "std_reward": stdev(rewards) if len(rewards) > 1 else 0.0,
                "mean_turns": mean(turns),
                "agent_a": first.get("agent_a", ""),
                "agent_b": first.get("agent_b", ""),
            }
        )
    return summary_rows


def load_reward_points(history_path: Path) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    with history_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points.append((int(row["episode"]), float(row["rolling_x_reward_50"])))
    return points


def svg_multi_seed_chart(series: dict[str, list[tuple[int, float]]], width: int = 860, height: int = 260) -> str:
    if not series:
        return f"<svg viewBox='0 0 {width} {height}'></svg>"

    pad = 36
    max_episode = max(point[0] for points in series.values() for point in points)
    min_value = -1.0
    max_value = 1.0
    colors = ["#1565c0", "#2e7d32", "#c62828", "#6a1b9a", "#ef6c00", "#00838f"]

    def point(episode: int, value: float) -> tuple[float, float]:
        x = pad + (episode - 1) * (width - 2 * pad) / max(1, max_episode - 1)
        y = height - pad - (value - min_value) * (height - 2 * pad) / (max_value - min_value)
        return x, y

    zero_y = point(1, 0.0)[1]
    lines = [
        f"<line class='axis' x1='{pad}' y1='{zero_y:.1f}' x2='{width - pad}' y2='{zero_y:.1f}' />"
    ]
    legend = []
    for index, (label, points) in enumerate(series.items()):
        color = colors[index % len(colors)]
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(ep, value) for ep, value in points))
        lines.append(f"<polyline points='{coords}' fill='none' stroke='{color}' stroke-width='2.5' />")
        legend.append(
            f"<span class='legend-item'><span style='background:{color}'></span>{label}</span>"
        )

    return (
        f"<svg viewBox='0 0 {width} {height}' role='img'>"
        + "".join(lines)
        + f"<text x='{pad}' y='20'>rolling reward +1</text>"
        + f"<text x='{pad}' y='{height - 8}'>rolling reward -1</text>"
        + "</svg>"
        + "<div class='legend'>"
        + "".join(legend)
        + "</div>"
    )


def write_training_html(
    path: Path,
    training_rows: list[dict[str, object]],
    history_paths: dict[int, Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series = {f"seed {seed}": load_reward_points(history_path) for seed, history_path in history_paths.items()}
    chart = svg_multi_seed_chart(series)
    table_rows = "\n".join(
        "<tr>"
        f"<td>{row['seed']}</td>"
        f"<td>{row['episodes']}</td>"
        f"<td>{float(row['train_x_win_rate']):.3f}</td>"
        f"<td>{float(row['train_o_win_rate']):.3f}</td>"
        f"<td>{float(row['train_avg_turns']):.2f}</td>"
        f"<td>{float(row['final_rolling_x_reward_50']):.3f}</td>"
        f"<td>{float(row['elapsed_seconds']):.1f}</td>"
        "</tr>"
        for row in training_rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DQN Multi-Seed Reward Curves</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; color: #222; }}
    svg {{ width: 100%; max-width: 880px; height: auto; border: 1px solid #ddd; background: #fff; }}
    .axis {{ stroke: #bbb; stroke-width: 1; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 6px; font-size: 13px; }}
    .legend-item span {{ width: 22px; height: 3px; display: inline-block; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 24px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>DQN Multi-Seed Reward Curves</h1>
  <p>Rolling reward is measured from X's perspective over the latest 50 self-play episodes.</p>
  {chart}
  <table>
    <thead><tr><th>Seed</th><th>Episodes</th><th>X Win</th><th>O Win</th><th>Avg Turns</th><th>Final Rolling Reward</th><th>Seconds</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_diagnostics_html(
    path: Path,
    summary_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_body = "\n".join(
        "<tr>"
        f"<td>{row['opponent']}</td>"
        f"<td>{int(row['games_total'])}</td>"
        f"<td>{float(row['mean_win_rate']):.3f}</td>"
        f"<td>{float(row['std_win_rate']):.3f}</td>"
        f"<td>{float(row['mean_reward']):.3f}</td>"
        f"<td>{float(row['mean_turns']):.2f}</td>"
        "</tr>"
        for row in summary_rows
    )
    detail_body = "\n".join(
        "<tr>"
        f"<td>{row['seed']}</td>"
        f"<td>{row['opponent']}</td>"
        f"<td>{row['games']}</td>"
        f"<td>{float(row['win_rate']):.3f}</td>"
        f"<td>{float(row['avg_reward']):.3f}</td>"
        f"<td>{float(row['avg_turns']):.2f}</td>"
        "</tr>"
        for row in detail_rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DQN Diagnostic Matchups</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0 32px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>DQN Diagnostic Matchups</h1>
  <h2>Aggregate</h2>
  <table>
    <thead><tr><th>Opponent</th><th>Games</th><th>Mean Win</th><th>Std Win</th><th>Mean Reward</th><th>Mean Turns</th></tr></thead>
    <tbody>{summary_body}</tbody>
  </table>
  <h2>Per Seed</h2>
  <table>
    <thead><tr><th>Seed</th><th>Opponent</th><th>Games</th><th>Win Rate</th><th>Avg Reward</th><th>Avg Turns</th></tr></thead>
    <tbody>{detail_body}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def run_diagnostics(
    checkpoint_path: Path,
    seed: int,
    opponents: list[DiagnosticOpponent],
    stochastic: bool,
    progress: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for opponent in opponents:
        agent_a = make_agent(f"dqn:{checkpoint_path}", seed=seed)
        agent_b = make_agent(opponent.spec, seed=seed + 20_000)

        def progress_callback(done_games: int, result: object) -> None:
            if progress:
                print(
                    f"diagnostic seed={seed} vs {opponent.name}: {done_games}/{opponent.games}, "
                    f"last={result.winner_agent or result.winner}, turns={result.turns}",
                    flush=True,
                )

        summary, _ = evaluate_pair(
            agent_a=agent_a,
            agent_b=agent_b,
            games=opponent.games,
            seed=seed + 30_000,
            stochastic=stochastic,
            alternate=True,
            progress_callback=progress_callback if progress else None,
        )
        rows.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "opponent": opponent.name,
                "opponent_spec": opponent.spec,
                "games": opponent.games,
                "agent_a": summary.agent_a,
                "agent_b": summary.agent_b,
                "win_rate": summary.agent_a_win_rate,
                "avg_reward": reward(summary),
                "draw_rate": summary.draw_rate,
                "avg_turns": summary.avg_turns,
                "avg_forfeits": summary.avg_forfeits,
                "avg_redirects": summary.avg_redirects,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[201, 202, 203])
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--eval-games", type=int, default=12)
    parser.add_argument("--diagnostic-games", type=int, default=10)
    parser.add_argument("--mcts-games", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-size", type=int, default=20_000)
    parser.add_argument("--train-after", type=int, default=128)
    parser.add_argument("--target-update", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tag", default="dqn_long")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stochastic = not args.deterministic
    training_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    history_paths: dict[int, Path] = {}

    opponents = [
        DiagnosticOpponent("random", "random", args.diagnostic_games),
        DiagnosticOpponent("heuristic", "heuristic", args.diagnostic_games),
        DiagnosticOpponent("mcts20c6", "mcts:20:6", args.mcts_games),
    ]

    for seed in args.seeds:
        checkpoint_path = Path("results/checkpoints") / f"{args.tag}_seed{seed}.pt"
        history_csv = Path("results/history") / f"{args.tag}_seed{seed}_history.csv"
        history_html = Path("results/history") / f"{args.tag}_seed{seed}_history.html"
        print(f"Training DQN seed={seed} episodes={args.episodes}", flush=True)
        start = perf_counter()
        model, history, train_metrics = train(
            episodes=args.episodes,
            seed=seed,
            stochastic=stochastic,
            lr=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            replay_size=args.replay_size,
            train_after=args.train_after,
            target_update=args.target_update,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            device=device,
        )
        elapsed = perf_counter() - start
        metadata = {
            "train": train_metrics,
            "stochastic": stochastic,
            "seed": seed,
            "episodes": args.episodes,
            "note": "Multi-seed pure DQN baseline with no hand-written tactical rules.",
        }
        save_checkpoint(str(checkpoint_path), model.cpu(), metadata)
        save_history_csv(str(history_csv), history)
        save_history_html(str(history_html), history)
        history_paths[seed] = history_csv
        final_reward = float(history[-1]["rolling_x_reward_50"]) if history else 0.0
        training_rows.append(
            {
                "seed": seed,
                "episodes": args.episodes,
                "checkpoint": str(checkpoint_path),
                "history_csv": str(history_csv),
                "history_html": str(history_html),
                "train_x_win_rate": train_metrics["x_win_rate"],
                "train_o_win_rate": train_metrics["o_win_rate"],
                "train_draw_rate": train_metrics["draw_rate"],
                "train_avg_turns": train_metrics["avg_turns"],
                "train_avg_loss": train_metrics["avg_loss"],
                "final_rolling_x_reward_50": final_reward,
                "elapsed_seconds": elapsed,
            }
        )
        print(
            f"Saved {checkpoint_path}; final rolling reward={final_reward:.3f}, "
            f"train X win={train_metrics['x_win_rate']:.3f}",
            flush=True,
        )
        diagnostic_rows.extend(
            run_diagnostics(
                checkpoint_path=checkpoint_path,
                seed=seed,
                opponents=opponents,
                stochastic=stochastic,
                progress=args.progress,
            )
        )

    prefix = f"{args.tag}_s{len(args.seeds)}_e{args.episodes}"
    training_csv = Path("results/summary") / f"{prefix}_training.csv"
    training_html = Path("results/summary") / f"{prefix}_reward_curves.html"
    diagnostic_detail_csv = Path("results/evaluations") / f"{prefix}_diagnostics_detail.csv"
    diagnostic_summary_csv = Path("results/summary") / f"{prefix}_diagnostics_summary.csv"
    diagnostic_html = Path("results/summary") / f"{prefix}_diagnostics.html"
    diagnostic_summary = summarize(diagnostic_rows, group_key="opponent")

    write_csv(training_csv, training_rows)
    write_training_html(training_html, training_rows, history_paths)
    write_csv(diagnostic_detail_csv, diagnostic_rows)
    write_csv(diagnostic_summary_csv, diagnostic_summary)
    write_diagnostics_html(diagnostic_html, diagnostic_summary, diagnostic_rows)

    print(f"Saved training CSV to {training_csv}")
    print(f"Saved reward curves HTML to {training_html}")
    print(f"Saved diagnostic detail CSV to {diagnostic_detail_csv}")
    print(f"Saved diagnostic summary CSV to {diagnostic_summary_csv}")
    print(f"Saved diagnostic HTML to {diagnostic_html}")
    print()
    for row in diagnostic_summary:
        print(
            f"DQN vs {row['opponent']}: mean_win={float(row['mean_win_rate']):.3f}, "
            f"mean_reward={float(row['mean_reward']):.3f}, games={row['games_total']}"
        )


if __name__ == "__main__":
    main()
