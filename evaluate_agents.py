"""Run reusable win-rate tests between two agents."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from agents import make_agent
from match import MatchSummary, evaluate_pair, results_dict, summary_dict


def disambiguate_names(agent_a: object, agent_b: object) -> None:
    if getattr(agent_a, "name") == getattr(agent_b, "name"):
        agent_a.name = f"A:{agent_a.name}"  # type: ignore[attr-defined]
        agent_b.name = f"B:{agent_b.name}"  # type: ignore[attr-defined]


def bar(value: float, width: int = 28) -> str:
    filled = round(value * width)
    return "#" * filled + "-" * (width - filled)


def print_summary(summary: MatchSummary) -> None:
    agent_a_reward = summary.agent_a_win_rate - summary.agent_b_win_rate
    agent_b_reward = summary.agent_b_win_rate - summary.agent_a_win_rate
    print("Match summary")
    print(f"  {summary.agent_a} vs {summary.agent_b}")
    print(f"  games: {summary.games}")
    print(
        f"  {summary.agent_a} wins: {summary.agent_a_wins} "
        f"({summary.agent_a_win_rate:.2%}) [{bar(summary.agent_a_win_rate)}]"
    )
    print(
        f"  {summary.agent_b} wins: {summary.agent_b_wins} "
        f"({summary.agent_b_win_rate:.2%}) [{bar(summary.agent_b_win_rate)}]"
    )
    print(f"  draws: {summary.draws} ({summary.draw_rate:.2%}) [{bar(summary.draw_rate)}]")
    print(f"  X win rate: {summary.x_win_rate:.2%}")
    print(f"  O win rate: {summary.o_win_rate:.2%}")
    print(f"  {summary.agent_a} avg reward: {agent_a_reward:.3f}")
    print(f"  {summary.agent_b} avg reward: {agent_b_reward:.3f}")
    print(f"  avg turns: {summary.avg_turns:.2f}")
    print(f"  avg forfeits: {summary.avg_forfeits:.2f}")
    print(f"  avg redirects: {summary.avg_redirects:.2f}")


def add_reward_columns(
    rows: list[dict[str, object]],
    summary: MatchSummary,
    window: int = 20,
) -> list[dict[str, object]]:
    enriched = []
    cumulative_a = 0.0
    cumulative_b = 0.0
    recent_a: list[float] = []
    recent_b: list[float] = []

    for row in rows:
        winner_agent = row["winner_agent"]
        if winner_agent == summary.agent_a:
            reward_a = 1.0
            reward_b = -1.0
        elif winner_agent == summary.agent_b:
            reward_a = -1.0
            reward_b = 1.0
        else:
            reward_a = 0.0
            reward_b = 0.0

        cumulative_a += reward_a
        cumulative_b += reward_b
        recent_a.append(reward_a)
        recent_b.append(reward_b)
        if len(recent_a) > window:
            recent_a.pop(0)
            recent_b.pop(0)

        enriched_row = dict(row)
        enriched_row["agent_a_reward"] = reward_a
        enriched_row["agent_b_reward"] = reward_b
        enriched_row["agent_a_cumulative_reward"] = cumulative_a
        enriched_row["agent_b_cumulative_reward"] = cumulative_b
        enriched_row["agent_a_rolling_reward"] = sum(recent_a) / len(recent_a)
        enriched_row["agent_b_rolling_reward"] = sum(recent_b) / len(recent_b)
        enriched.append(enriched_row)

    return enriched


def save_json(path: str, summary: MatchSummary, results: list[dict[str, object]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary_dict(summary), "games": results}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_csv(path: str, rows: list[dict[str, object]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "game_index",
        "x_agent",
        "o_agent",
        "winner",
        "winner_agent",
        "turns",
        "forfeits",
        "redirects",
        "agent_a_reward",
        "agent_b_reward",
        "agent_a_cumulative_reward",
        "agent_b_cumulative_reward",
        "agent_a_rolling_reward",
        "agent_b_rolling_reward",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_html(path: str, summary: MatchSummary, rows: list[dict[str, object]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if rows and "agent_a_cumulative_reward" not in rows[0]:
        rows = add_reward_columns(rows, summary)
    reward_values = [float(row["agent_a_cumulative_reward"]) for row in rows]
    reward_chart = svg_line_chart(reward_values, width=760, height=220)
    game_rows = "\n".join(
        "<tr>"
        f"<td>{row['game_index']}</td>"
        f"<td>{row['x_agent']}</td>"
        f"<td>{row['o_agent']}</td>"
        f"<td>{row['winner']}</td>"
        f"<td>{row['winner_agent'] or ''}</td>"
        f"<td>{row['turns']}</td>"
        f"<td>{row['forfeits']}</td>"
        f"<td>{row['redirects']}</td>"
        f"<td>{float(row['agent_a_reward']):.0f}</td>"
        f"<td>{float(row['agent_a_cumulative_reward']):.0f}</td>"
        "</tr>"
        for row in rows
    )
    metrics = [
        (summary.agent_a, summary.agent_a_win_rate),
        (summary.agent_b, summary.agent_b_win_rate),
        ("draw", summary.draw_rate),
    ]
    metric_blocks = "\n".join(
        "<div class='metric'>"
        f"<div class='label'>{name}</div>"
        f"<div class='track'><div class='fill' style='width: {rate * 100:.2f}%'></div></div>"
        f"<div class='rate'>{rate:.2%}</div>"
        "</div>"
        for name, rate in metrics
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Super Tic-Tac-Toe Match Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; color: #222; }}
    h1 {{ margin-bottom: 4px; }}
    .subtle {{ color: #666; margin-top: 0; }}
    .metric {{ display: grid; grid-template-columns: 160px 1fr 80px; gap: 12px; align-items: center; margin: 10px 0; }}
    .track {{ height: 14px; background: #e8e8e8; border-radius: 7px; overflow: hidden; }}
    .fill {{ height: 100%; background: #2aa7df; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 24px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .chart {{ margin-top: 24px; max-width: 780px; }}
    svg {{ width: 100%; height: auto; border: 1px solid #ddd; background: #fff; }}
    .axis {{ stroke: #bbb; stroke-width: 1; }}
    .line {{ fill: none; stroke: #2aa7df; stroke-width: 3; }}
    .caption {{ color: #666; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>Super Tic-Tac-Toe Match Report</h1>
  <p class="subtle">{summary.agent_a} vs {summary.agent_b}, {summary.games} games</p>
  {metric_blocks}
  <p>Average turns: {summary.avg_turns:.2f}. Average forfeits: {summary.avg_forfeits:.2f}. Average redirects: {summary.avg_redirects:.2f}.</p>
  <div class="chart">
    <h2>{summary.agent_a} cumulative reward</h2>
    {reward_chart}
    <p class="caption">Each game gives +1 for a win, -1 for a loss, and 0 for a draw from {summary.agent_a}'s perspective.</p>
  </div>
  <table>
    <thead>
      <tr><th>Game</th><th>X Agent</th><th>O Agent</th><th>Winner</th><th>Winner Agent</th><th>Turns</th><th>Forfeits</th><th>Redirects</th><th>A Reward</th><th>A Cumulative</th></tr>
    </thead>
    <tbody>{game_rows}</tbody>
  </table>
</body>
</html>
"""
    Path(path).write_text(html, encoding="utf-8")


def svg_line_chart(values: list[float], width: int, height: int) -> str:
    if not values:
        return f"<svg viewBox='0 0 {width} {height}'></svg>"

    pad = 28
    min_value = min(0.0, min(values))
    max_value = max(0.0, max(values))
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0

    def point(index: int, value: float) -> tuple[float, float]:
        if len(values) == 1:
            x = width / 2
        else:
            x = pad + index * (width - 2 * pad) / (len(values) - 1)
        y = height - pad - (value - min_value) * (height - 2 * pad) / (max_value - min_value)
        return x, y

    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(i, v) for i, v in enumerate(values)))
    zero_y = point(0, 0.0)[1]
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img'>"
        f"<line class='axis' x1='{pad}' y1='{zero_y:.1f}' x2='{width - pad}' y2='{zero_y:.1f}' />"
        f"<polyline class='line' points='{points}' />"
        f"<text x='{pad}' y='18'>max {max_value:.0f}</text>"
        f"<text x='{pad}' y='{height - 8}'>min {min_value:.0f}</text>"
        "</svg>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-a", default="random", help="random | heuristic | mcts:N | qtable:path | dqn:path")
    parser.add_argument("--agent-b", default="random", help="random | heuristic | mcts:N | qtable:path | dqn:path")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-alternate", action="store_true")
    parser.add_argument("--json-output")
    parser.add_argument("--csv-output")
    parser.add_argument("--html-output")
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args()

    agent_a = make_agent(args.agent_a, seed=args.seed)
    agent_b = make_agent(args.agent_b, seed=args.seed + 1)
    disambiguate_names(agent_a, agent_b)
    def progress(done_games: int, result: object) -> None:
        if args.progress_every and done_games % args.progress_every == 0:
            print(
                f"progress: {done_games}/{args.games} games, "
                f"last winner={result.winner_agent or result.winner}, "
                f"turns={result.turns}",
                flush=True,
            )

    summary, results = evaluate_pair(
        agent_a=agent_a,
        agent_b=agent_b,
        games=args.games,
        seed=args.seed,
        stochastic=not args.deterministic,
        alternate=not args.no_alternate,
        progress_callback=progress if args.progress_every else None,
    )
    rows = add_reward_columns(results_dict(results), summary)
    print_summary(summary)

    if args.json_output:
        save_json(args.json_output, summary, rows)
        print(f"Saved JSON report to {args.json_output}")
    if args.csv_output:
        save_csv(args.csv_output, rows)
        print(f"Saved CSV report to {args.csv_output}")
    if args.html_output:
        save_html(args.html_output, summary, rows)
        print(f"Saved HTML report to {args.html_output}")


if __name__ == "__main__":
    main()
