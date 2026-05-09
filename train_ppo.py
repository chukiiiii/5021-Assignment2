"""Train a PPO actor-critic baseline for super tic-tac-toe."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import random
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from dqn_model import action_mask, state_tensor
from match import evaluate_pair
from ppo_model import PPOActorCritic, masked_action_distribution, save_checkpoint
from agents import make_agent
from super_tictactoe import SuperTicTacToeEnv, player_name


@dataclass(frozen=True)
class PPOTransition:
    state: torch.Tensor
    mask: torch.Tensor
    action: int
    log_prob: float
    value: float
    player: int
    outcome: float


def select_action(
    model: PPOActorCritic,
    env: SuperTicTacToeEnv,
    device: torch.device,
) -> tuple[int, float, float]:
    with torch.no_grad():
        state = state_tensor(env).unsqueeze(0).to(device)
        mask = action_mask(env).to(device)
        logits, value = model(state)
        distribution = masked_action_distribution(logits[0], mask)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())


def collect_episode(
    model: PPOActorCritic,
    seed: int,
    stochastic: bool,
    device: torch.device,
    max_turns: int,
) -> tuple[list[PPOTransition], dict[str, float | int | str]]:
    env = SuperTicTacToeEnv(seed=seed, stochastic=stochastic)
    partial: list[dict[str, Any]] = []
    turns = 0

    while not env.done and turns < max_turns:
        state = state_tensor(env)
        mask = action_mask(env)
        player = env.current_player
        action, log_prob, value = select_action(model, env, device)
        env.step(action)
        partial.append(
            {
                "state": state,
                "mask": mask,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "player": player,
            }
        )
        turns += 1

    if not env.done:
        env.done = True
        env.winner = None

    transitions: list[PPOTransition] = []
    for item in partial:
        if env.winner is None:
            outcome = 0.0
        elif item["player"] == env.winner:
            outcome = 1.0
        else:
            outcome = -1.0
        transitions.append(
            PPOTransition(
                state=item["state"],
                mask=item["mask"],
                action=item["action"],
                log_prob=item["log_prob"],
                value=item["value"],
                player=item["player"],
                outcome=outcome,
            )
        )

    metrics = {
        "turns": turns,
        "winner": player_name(env.winner),
        "x_reward": 1.0 if env.winner == 1 else -1.0 if env.winner == -1 else 0.0,
    }
    return transitions, metrics


def ppo_update(
    model: PPOActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: list[PPOTransition],
    batch_size: int,
    update_epochs: int,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    device: torch.device,
) -> dict[str, float]:
    states = torch.stack([item.state for item in transitions]).to(device)
    masks = torch.stack([item.mask for item in transitions]).to(device)
    actions = torch.tensor([item.action for item in transitions], dtype=torch.long, device=device)
    old_log_probs = torch.tensor([item.log_prob for item in transitions], dtype=torch.float32, device=device)
    returns = torch.tensor([item.outcome for item in transitions], dtype=torch.float32, device=device)
    old_values = torch.tensor([item.value for item in transitions], dtype=torch.float32, device=device)
    advantages = returns - old_values
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []

    total = len(transitions)
    indices = torch.arange(total)
    for _ in range(update_epochs):
        permutation = indices[torch.randperm(total)]
        for start in range(0, total, batch_size):
            batch_idx = permutation[start : start + batch_size].to(device)
            logits, values = model(states[batch_idx])
            batch_masks = masks[batch_idx]
            masked_logits = logits.clone()
            masked_logits[~batch_masks] = -1e9
            distribution = torch.distributions.Categorical(logits=masked_logits)
            log_probs = distribution.log_prob(actions[batch_idx])
            entropy = distribution.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs[batch_idx])
            unclipped = ratio * advantages[batch_idx]
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[batch_idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns[batch_idx])
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))

    return {
        "loss": mean(losses) if losses else 0.0,
        "policy_loss": mean(policy_losses) if policy_losses else 0.0,
        "value_loss": mean(value_losses) if value_losses else 0.0,
        "entropy": mean(entropies) if entropies else 0.0,
    }


def train(
    episodes: int,
    seed: int,
    stochastic: bool,
    lr: float,
    batch_size: int,
    rollout_episodes: int,
    update_epochs: int,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    max_turns: int,
    device: torch.device,
) -> tuple[PPOActorCritic, list[dict[str, float | int | str]], dict[str, float]]:
    random.seed(seed)
    torch.manual_seed(seed)
    model = PPOActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float | int | str]] = []
    recent_x_rewards: list[float] = []
    all_turns: list[int] = []
    x_wins = 0
    o_wins = 0
    draws = 0
    episode = 0
    update_index = 0

    while episode < episodes:
        rollout: list[PPOTransition] = []
        metrics_batch: list[dict[str, float | int | str]] = []
        for _ in range(rollout_episodes):
            if episode >= episodes:
                break
            episode_seed = seed * 100_000 + episode
            transitions, metrics = collect_episode(
                model=model,
                seed=episode_seed,
                stochastic=stochastic,
                device=device,
                max_turns=max_turns,
            )
            rollout.extend(transitions)
            metrics_batch.append(metrics)
            episode += 1

        update_index += 1
        update_metrics = ppo_update(
            model=model,
            optimizer=optimizer,
            transitions=rollout,
            batch_size=batch_size,
            update_epochs=update_epochs,
            clip_ratio=clip_ratio,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            device=device,
        )

        for metrics in metrics_batch:
            turns = int(metrics["turns"])
            winner = str(metrics["winner"])
            x_reward = float(metrics["x_reward"])
            all_turns.append(turns)
            recent_x_rewards.append(x_reward)
            if len(recent_x_rewards) > 50:
                recent_x_rewards.pop(0)
            if winner == "X":
                x_wins += 1
            elif winner == "O":
                o_wins += 1
            else:
                draws += 1
            history.append(
                {
                    "episode": len(history) + 1,
                    "update": update_index,
                    "turns": turns,
                    "winner": winner,
                    "x_reward": x_reward,
                    "rolling_x_reward_50": sum(recent_x_rewards) / len(recent_x_rewards),
                    "loss": update_metrics["loss"],
                    "policy_loss": update_metrics["policy_loss"],
                    "value_loss": update_metrics["value_loss"],
                    "entropy": update_metrics["entropy"],
                }
            )

    metrics = {
        "episodes": float(episodes),
        "avg_turns": mean(all_turns) if all_turns else 0.0,
        "x_win_rate": x_wins / episodes if episodes else 0.0,
        "o_win_rate": o_wins / episodes if episodes else 0.0,
        "draw_rate": draws / episodes if episodes else 0.0,
        "final_rolling_x_reward_50": float(history[-1]["rolling_x_reward_50"]) if history else 0.0,
    }
    return model, history, metrics


def evaluate_checkpoint(
    checkpoint_path: str,
    games: int,
    seed: int,
    stochastic: bool,
) -> dict[str, float]:
    agent_a = make_agent(f"ppo:{checkpoint_path}", seed=seed)
    agent_b = make_agent("random", seed=seed + 1)
    summary, _ = evaluate_pair(
        agent_a=agent_a,
        agent_b=agent_b,
        games=games,
        seed=seed,
        stochastic=stochastic,
        alternate=True,
    )
    return {
        "games": float(games),
        "learner_win_rate": summary.agent_a_win_rate,
        "random_win_rate": summary.agent_b_win_rate,
        "draw_rate": summary.draw_rate,
        "avg_turns": summary.avg_turns,
    }


def save_history_csv(path: str, history: list[dict[str, float | int | str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "update",
        "turns",
        "winner",
        "x_reward",
        "rolling_x_reward_50",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_history_html(path: str, history: list[dict[str, float | int | str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    values = [float(row["rolling_x_reward_50"]) for row in history]
    points = svg_line_chart(values, width=760, height=220)
    rows = "\n".join(
        "<tr>"
        f"<td>{row['episode']}</td>"
        f"<td>{row['update']}</td>"
        f"<td>{row['winner']}</td>"
        f"<td>{row['turns']}</td>"
        f"<td>{float(row['x_reward']):.0f}</td>"
        f"<td>{float(row['rolling_x_reward_50']):.3f}</td>"
        f"<td>{float(row['loss']):.4f}</td>"
        f"<td>{float(row['entropy']):.3f}</td>"
        "</tr>"
        for row in history
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PPO Reward History</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; color: #222; }}
    svg {{ width: 100%; max-width: 780px; height: auto; border: 1px solid #ddd; background: #fff; }}
    .axis {{ stroke: #bbb; stroke-width: 1; }}
    .line {{ fill: none; stroke: #2e7d32; stroke-width: 3; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 24px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>PPO Reward History</h1>
  <p>Rolling reward is measured from X's perspective over the latest 50 self-play episodes.</p>
  {points}
  <table>
    <thead><tr><th>Episode</th><th>Update</th><th>Winner</th><th>Turns</th><th>X Reward</th><th>Rolling X Reward</th><th>Loss</th><th>Entropy</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
    Path(path).write_text(html, encoding="utf-8")


def svg_line_chart(values: list[float], width: int, height: int) -> str:
    if not values:
        return f"<svg viewBox='0 0 {width} {height}'></svg>"
    pad = 28
    min_value = -1.0
    max_value = 1.0

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
        f"<text x='{pad}' y='18'>+1</text>"
        f"<text x='{pad}' y='{height - 8}'>-1</text>"
        "</svg>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rollout-episodes", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-turns", type=int, default=240)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", default="results/checkpoints/ppo_checkpoint.pt")
    parser.add_argument("--history-csv")
    parser.add_argument("--history-html")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stochastic = not args.deterministic
    model, history, train_metrics = train(
        episodes=args.episodes,
        seed=args.seed,
        stochastic=stochastic,
        lr=args.lr,
        batch_size=args.batch_size,
        rollout_episodes=args.rollout_episodes,
        update_epochs=args.update_epochs,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_turns=args.max_turns,
        device=device,
    )
    metadata = {
        "train": train_metrics,
        "stochastic": stochastic,
        "seed": args.seed,
        "episodes": args.episodes,
        "note": "Pure PPO actor-critic baseline with no hand-written tactical rules.",
    }
    save_checkpoint(args.output, model.cpu(), metadata)
    if args.history_csv:
        save_history_csv(args.history_csv, history)
    if args.history_html:
        save_history_html(args.history_html, history)

    eval_metrics = evaluate_checkpoint(
        checkpoint_path=args.output,
        games=args.eval_games,
        seed=args.seed + 10_000,
        stochastic=stochastic,
    )
    print("Training metrics")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("Evaluation metrics")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"Saved PPO checkpoint to {args.output}")
    if args.history_csv:
        print(f"Saved reward history CSV to {args.history_csv}")
    if args.history_html:
        print(f"Saved reward history HTML to {args.history_html}")


if __name__ == "__main__":
    main()
