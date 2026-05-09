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
    print(f"  avg turns: {summary.avg_turns:.2f}")
    print(f"  avg forfeits: {summary.avg_forfeits:.2f}")
    print(f"  avg redirects: {summary.avg_redirects:.2f}")


def save_json(path: str, summary: MatchSummary, results: list[dict[str, object]]) -> None:
    payload = {"summary": summary_dict(summary), "games": results}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_csv(path: str, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "game_index",
        "x_agent",
        "o_agent",
        "winner",
        "winner_agent",
        "turns",
        "forfeits",
        "redirects",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_html(path: str, summary: MatchSummary, rows: list[dict[str, object]]) -> None:
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
  </style>
</head>
<body>
  <h1>Super Tic-Tac-Toe Match Report</h1>
  <p class="subtle">{summary.agent_a} vs {summary.agent_b}, {summary.games} games</p>
  {metric_blocks}
  <p>Average turns: {summary.avg_turns:.2f}. Average forfeits: {summary.avg_forfeits:.2f}. Average redirects: {summary.avg_redirects:.2f}.</p>
  <table>
    <thead>
      <tr><th>Game</th><th>X Agent</th><th>O Agent</th><th>Winner</th><th>Winner Agent</th><th>Turns</th><th>Forfeits</th><th>Redirects</th></tr>
    </thead>
    <tbody>{game_rows}</tbody>
  </table>
</body>
</html>
"""
    Path(path).write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-a", default="random", help="random | heuristic | qtable:path")
    parser.add_argument("--agent-b", default="random", help="random | heuristic | qtable:path")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-alternate", action="store_true")
    parser.add_argument("--json-output")
    parser.add_argument("--csv-output")
    parser.add_argument("--html-output")
    args = parser.parse_args()

    agent_a = make_agent(args.agent_a, seed=args.seed)
    agent_b = make_agent(args.agent_b, seed=args.seed + 1)
    disambiguate_names(agent_a, agent_b)
    summary, results = evaluate_pair(
        agent_a=agent_a,
        agent_b=agent_b,
        games=args.games,
        seed=args.seed,
        stochastic=not args.deterministic,
        alternate=not args.no_alternate,
    )
    rows = results_dict(results)
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
