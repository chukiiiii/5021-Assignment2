"""Behavior cloning from mixed expert trajectories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
from statistics import mean

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ppo_model import PPOActorCritic, save_checkpoint


def load_dataset(path: str) -> dict[str, object]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = ["states", "masks", "actions"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"dataset missing keys: {missing}")
    return payload


def split_dataset(
    states: torch.Tensor,
    masks: torch.Tensor,
    actions: torch.Tensor,
    seed: int,
    train_fraction: float = 0.8,
) -> tuple[TensorDataset, TensorDataset]:
    total = actions.shape[0]
    if total < 2:
        raise ValueError("behavior cloning needs at least two samples")
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_size = max(1, min(total - 1, int(total * train_fraction)))
    train_idx = torch.tensor(indices[:train_size], dtype=torch.long)
    val_idx = torch.tensor(indices[train_size:], dtype=torch.long)
    return (
        TensorDataset(states[train_idx], masks[train_idx], actions[train_idx]),
        TensorDataset(states[val_idx], masks[val_idx], actions[val_idx]),
    )


def masked_cross_entropy(logits: torch.Tensor, masks: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.clone()
    masked_logits[~masks] = -1e9
    return F.cross_entropy(masked_logits, actions)


def evaluate(
    model: PPOActorCritic,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    legal_predictions = 0
    with torch.no_grad():
        for states, masks, actions in loader:
            states = states.to(device)
            masks = masks.to(device)
            actions = actions.to(device)
            logits, _ = model(states)
            loss = masked_cross_entropy(logits, masks, actions)
            masked_logits = logits.clone()
            masked_logits[~masks] = -1e9
            predictions = torch.argmax(masked_logits, dim=1)
            losses.append(float(loss.item()))
            correct += int((predictions == actions).sum().item())
            legal_predictions += int(masks.gather(1, predictions.unsqueeze(1)).sum().item())
            total += int(actions.shape[0])
    return {
        "loss": mean(losses) if losses else 0.0,
        "top1_accuracy": correct / total if total else 0.0,
        "legal_action_accuracy": legal_predictions / total if total else 0.0,
    }


def train_bc(
    data_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> tuple[PPOActorCritic, list[dict[str, float | int]], dict[str, float]]:
    torch.manual_seed(seed)
    dataset = load_dataset(data_path)
    states = dataset["states"].float()
    masks = dataset["masks"].bool()
    actions = dataset["actions"].long()
    train_data, val_data = split_dataset(states, masks, actions, seed=seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = PPOActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_states, batch_masks, batch_actions in train_loader:
            batch_states = batch_states.to(device)
            batch_masks = batch_masks.to(device)
            batch_actions = batch_actions.to(device)
            logits, _ = model(batch_states)
            loss = masked_cross_entropy(logits, batch_masks, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": mean(train_losses) if train_losses else 0.0,
            "train_top1_accuracy": train_metrics["top1_accuracy"],
            "train_legal_action_accuracy": train_metrics["legal_action_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_top1_accuracy": val_metrics["top1_accuracy"],
            "val_legal_action_accuracy": val_metrics["legal_action_accuracy"],
        }
        history.append(row)
        print(
            f"epoch {epoch}/{epochs}: val_acc={row['val_top1_accuracy']:.3f}, "
            f"val_loss={row['val_loss']:.4f}",
            flush=True,
        )

    final_metrics = {
        "samples": float(actions.shape[0]),
        "train_samples": float(len(train_data)),
        "val_samples": float(len(val_data)),
        "val_top1_accuracy": float(history[-1]["val_top1_accuracy"]) if history else 0.0,
        "val_legal_action_accuracy": float(history[-1]["val_legal_action_accuracy"]) if history else 0.0,
        "val_loss": float(history[-1]["val_loss"]) if history else 0.0,
    }
    return model, history, final_metrics


def save_history_csv(path: str, history: list[dict[str, float | int]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_history_html(path: str, history: list[dict[str, float | int]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        "<tr>"
        f"<td>{row['epoch']}</td>"
        f"<td>{float(row['train_loss']):.4f}</td>"
        f"<td>{float(row['train_top1_accuracy']):.3f}</td>"
        f"<td>{float(row['val_loss']):.4f}</td>"
        f"<td>{float(row['val_top1_accuracy']):.3f}</td>"
        f"<td>{float(row['val_legal_action_accuracy']):.3f}</td>"
        "</tr>"
        for row in history
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Behavior Cloning History</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 24px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>Behavior Cloning History</h1>
  <table>
    <thead><tr><th>Epoch</th><th>Train Loss</th><th>Train Top-1</th><th>Val Loss</th><th>Val Top-1</th><th>Val Legal</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
    Path(path).write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/expert/mixed_expert.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--output", default="results/checkpoints/bc_mixed.pt")
    parser.add_argument("--history-csv", default="results/history/bc_mixed_history.csv")
    parser.add_argument("--history-html", default="results/history/bc_mixed_history.html")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, history, metrics = train_bc(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=device,
    )
    metadata = {
        "bc": metrics,
        "data": args.data,
        "seed": args.seed,
        "epochs": args.epochs,
        "note": "Behavior cloning policy checkpoint compatible with PPOAgent.",
    }
    save_checkpoint(args.output, model.cpu(), metadata)
    save_history_csv(args.history_csv, history)
    save_history_html(args.history_html, history)
    print("Behavior cloning metrics")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"Saved BC checkpoint to {args.output}")
    print(f"Saved BC history CSV to {args.history_csv}")
    print(f"Saved BC history HTML to {args.history_html}")


if __name__ == "__main__":
    main()
