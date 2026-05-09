"""Local web UI for interactive super tic-tac-toe games."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any
from urllib.parse import urlparse

from agents import Agent, make_agent
from super_tictactoe import (
    CELL_TO_ACTION,
    EMPTY,
    O,
    WINNING_LINES,
    X,
    SuperTicTacToeEnv,
    VALID_CELLS,
    player_name,
)


PLAYER_LABELS = {X: "X", O: "O"}


class GameSession:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.seed = 7
        self.stochastic = True
        self.x_spec = "human"
        self.o_spec = "heuristic"
        self.env = SuperTicTacToeEnv(seed=self.seed, stochastic=self.stochastic)
        self.x_agent: Agent = make_agent(self.x_spec, seed=self.seed)
        self.o_agent: Agent = make_agent(self.o_spec, seed=self.seed + 1)
        self.turn = 0
        self.log: list[dict[str, Any]] = []

    def new_game(
        self,
        x_spec: str,
        o_spec: str,
        seed: int,
        stochastic: bool,
    ) -> dict[str, Any]:
        with self.lock:
            self.seed = seed
            self.stochastic = stochastic
            self.x_spec = x_spec
            self.o_spec = o_spec
            self.env = SuperTicTacToeEnv(seed=seed, stochastic=stochastic)
            self.x_agent = make_agent(x_spec, seed=seed)
            self.o_agent = make_agent(o_spec, seed=seed + 1)
            self.x_agent.reset()
            self.o_agent.reset()
            self.turn = 0
            self.log = []
            return self.serialize()

    def human_move(self, action: int) -> dict[str, Any]:
        with self.lock:
            if self.env.done:
                return self.serialize(error="Game is already finished.")
            if self.current_agent_spec() != "human":
                return self.serialize(error="Current player is controlled by an agent.")
            if action not in self.env.available_actions():
                return self.serialize(error="That cell is not available.")
            self._step(action, "human")
            return self.serialize()

    def agent_step(self) -> dict[str, Any]:
        with self.lock:
            if self.env.done:
                return self.serialize(error="Game is already finished.")
            agent = self.current_agent()
            if self.current_agent_spec() == "human":
                return self.serialize(error="Current player is human.")
            action = agent.select_action(self.env)
            self._step(action, agent.name)
            return self.serialize()

    def auto_play(self, max_steps: int = 200) -> dict[str, Any]:
        with self.lock:
            steps = 0
            while not self.env.done and self.current_agent_spec() != "human":
                agent = self.current_agent()
                action = agent.select_action(self.env)
                self._step(action, agent.name)
                steps += 1
                if steps >= max_steps:
                    break
            return self.serialize()

    def _step(self, action: int, agent_name: str) -> None:
        actor = player_name(self.env.current_player)
        _, _, _, info = self.env.step(action)
        self.turn += 1
        self.log.append(
            {
                "turn": self.turn,
                "player": actor,
                "agent": agent_name,
                "requested": info.requested,
                "placed": info.placed,
                "accepted": info.accepted,
                "forfeited": info.forfeited,
                "reason": info.reason,
            }
        )

    def current_agent_spec(self) -> str:
        return self.x_spec if self.env.current_player == X else self.o_spec

    def current_agent(self) -> Agent:
        return self.x_agent if self.env.current_player == X else self.o_agent

    def serialize(self, error: str | None = None) -> dict[str, Any]:
        legal = set(self.env.available_actions()) if not self.env.done else set()
        own_threats = set()
        opponent_threats = set()
        if not self.env.done:
            own_threats = immediate_winning_actions(self.env, self.env.current_player)
            opponent_threats = immediate_winning_actions(self.env, -self.env.current_player)
        winning_actions = winning_line_actions(self.env)
        cells = []
        for action, (row, col) in enumerate(VALID_CELLS):
            cells.append(
                {
                    "action": action,
                    "row": row,
                    "col": col,
                    "value": self.env.board[row][col],
                    "legal": action in legal,
                    "ownThreat": action in own_threats,
                    "opponentThreat": action in opponent_threats,
                    "winning": action in winning_actions,
                }
            )
        last = self.log[-1] if self.log else None
        winner_agent = None
        if self.env.winner == X:
            winner_agent = self.x_agent.name
        elif self.env.winner == O:
            winner_agent = self.o_agent.name

        return {
            "cells": cells,
            "currentPlayer": player_name(self.env.current_player),
            "currentAgent": self.current_agent_spec() if not self.env.done else None,
            "done": self.env.done,
            "winner": player_name(self.env.winner),
            "winnerAgent": winner_agent,
            "turn": self.turn,
            "stochastic": self.stochastic,
            "xSpec": self.x_spec,
            "oSpec": self.o_spec,
            "lastMove": last,
            "log": self.log[-80:],
            "error": error,
        }


def immediate_winning_actions(env: SuperTicTacToeEnv, player: int) -> set[int]:
    actions = set()
    for action in env.available_actions():
        row, col = env.action_to_cell(action)
        env.board[row][col] = player
        if env.check_winner() == player:
            actions.add(action)
        env.board[row][col] = EMPTY
    return actions


def winning_line_actions(env: SuperTicTacToeEnv) -> set[int]:
    if env.winner is None:
        return set()
    for line in WINNING_LINES:
        values = [env.board[row][col] for row, col in line]
        if values and values[0] == env.winner and all(value == env.winner for value in values):
            return {CELL_TO_ACTION[cell] for cell in line}
    return set()


SESSION = GameSession()


class GameUIHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/api/state":
            self._send_json(SESSION.serialize())
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/new":
                state = SESSION.new_game(
                    x_spec=str(payload.get("xSpec", "human")),
                    o_spec=str(payload.get("oSpec", "heuristic")),
                    seed=int(payload.get("seed", 7)),
                    stochastic=bool(payload.get("stochastic", True)),
                )
                self._send_json(state)
                return
            if parsed.path == "/api/move":
                self._send_json(SESSION.human_move(int(payload["action"])))
                return
            if parsed.path == "/api/agent-step":
                self._send_json(SESSION.agent_step())
                return
            if parsed.path == "/api/auto-play":
                self._send_json(SESSION.auto_play(max_steps=int(payload.get("maxSteps", 200))))
                return
        except Exception as exc:  # Keep the UI responsive with a JSON error.
            self._send_json(SESSION.serialize(error=str(exc)), status=400)
            return
        self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Super Tic-Tac-Toe</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #202124;
      --muted: #5f6368;
      --line: #d7dce2;
      --panel: #f7f8fa;
      --x: #1565c0;
      --o: #c62828;
      --accent: #16876b;
      --cell: #28a9df;
      --cell-hover: #46bde9;
      --placed: #f8fbff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }
    header {
      border-bottom: 1px solid var(--line);
      padding: 14px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1 { font-size: 20px; margin: 0; font-weight: 700; letter-spacing: 0; }
    main {
      display: grid;
      grid-template-columns: minmax(520px, 1fr) 380px;
      gap: 20px;
      padding: 20px;
      align-items: start;
    }
    .board-wrap {
      min-width: 0;
      display: flex;
      justify-content: center;
      align-items: flex-start;
      padding: 8px 0;
    }
    .board {
      display: grid;
      grid-template-columns: repeat(12, minmax(28px, 48px));
      grid-template-rows: repeat(12, minmax(28px, 48px));
      gap: 2px;
      width: min(100%, 620px);
      aspect-ratio: 1 / 1;
    }
    .cell {
      border: 1px solid #0f4157;
      background: var(--cell);
      color: var(--ink);
      font-size: 22px;
      font-weight: 800;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0;
      min-width: 0;
      min-height: 0;
      cursor: pointer;
      user-select: none;
    }
    .cell:hover { background: var(--cell-hover); }
    .cell:disabled { cursor: default; }
    .cell.invalid {
      visibility: hidden;
      pointer-events: none;
    }
    .cell.occupied {
      background: var(--placed);
    }
    .cell.x { color: var(--x); }
    .cell.o { color: var(--o); }
    .cell.last {
      outline: 3px solid var(--accent);
      outline-offset: -3px;
    }
    .cell.own-threat {
      box-shadow: inset 0 0 0 4px #f6c453;
    }
    .cell.opponent-threat {
      box-shadow: inset 0 0 0 4px #ef7d45;
    }
    .cell.winning {
      background: #fff3c4;
      box-shadow: inset 0 0 0 4px #d99a00;
    }
    aside {
      border-left: 1px solid var(--line);
      padding-left: 20px;
      min-height: calc(100vh - 82px);
    }
    .section {
      border-bottom: 1px solid var(--line);
      padding: 0 0 16px;
      margin-bottom: 16px;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    label {
      display: grid;
      gap: 5px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 600;
    }
    select, input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 9px;
      font: inherit;
      background: white;
      color: var(--ink);
    }
    .toggle {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--ink);
      font-size: 14px;
      font-weight: 600;
      margin-top: 10px;
    }
    .toggle input { width: auto; }
    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 12px;
    }
    button.control {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    button.control:disabled {
      opacity: 0.55;
      cursor: default;
    }
    .status {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 6px;
    }
    .metric {
      background: var(--panel);
      border-radius: 6px;
      padding: 9px 10px;
      min-height: 58px;
    }
    .metric b {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 3px;
    }
    .metric span {
      font-size: 16px;
      font-weight: 800;
    }
    .message {
      min-height: 24px;
      color: #9a3412;
      font-weight: 700;
      font-size: 13px;
      margin-top: 8px;
    }
    .log {
      max-height: 360px;
      overflow: auto;
      font-size: 13px;
      line-height: 1.4;
      padding-right: 6px;
    }
    .log-row {
      display: grid;
      grid-template-columns: 42px 1fr;
      gap: 8px;
      padding: 7px 0;
      border-bottom: 1px solid #edf0f3;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 24px;
      height: 24px;
      border-radius: 50%;
      font-weight: 900;
      background: #edf2f7;
    }
    .badge.x { color: var(--x); }
    .badge.o { color: var(--o); }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      aside { border-left: 0; padding-left: 0; min-height: auto; }
      .board { grid-template-columns: repeat(12, minmax(22px, 1fr)); grid-template-rows: repeat(12, minmax(22px, 1fr)); }
      .cell { font-size: 16px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Super Tic-Tac-Toe</h1>
    <div id="headline"></div>
  </header>
  <main>
    <section class="board-wrap">
      <div id="board" class="board"></div>
    </section>
    <aside>
      <section class="section">
        <div class="grid">
          <label>X Player
            <select id="xSpec">
              <option value="human">human</option>
              <option value="heuristic">heuristic</option>
              <option value="mcts:20:6">mcts:20:6</option>
              <option value="mcts:50:6">mcts:50:6</option>
              <option value="random">random</option>
              <option value="qtable:results/checkpoints/q_table.json">qtable:results/checkpoints/q_table.json</option>
              <option value="dqn:results/checkpoints/dqn_stochastic_smoke.pt">dqn:results/checkpoints/dqn_stochastic_smoke.pt</option>
              <option value="dqn:results/checkpoints/dqn_smoke.pt">dqn:results/checkpoints/dqn_smoke.pt</option>
            </select>
          </label>
          <label>O Player
            <select id="oSpec">
              <option value="heuristic">heuristic</option>
              <option value="human">human</option>
              <option value="mcts:20:6">mcts:20:6</option>
              <option value="mcts:50:6">mcts:50:6</option>
              <option value="random">random</option>
              <option value="qtable:results/checkpoints/q_table.json">qtable:results/checkpoints/q_table.json</option>
              <option value="dqn:results/checkpoints/dqn_stochastic_smoke.pt">dqn:results/checkpoints/dqn_stochastic_smoke.pt</option>
              <option value="dqn:results/checkpoints/dqn_smoke.pt">dqn:results/checkpoints/dqn_smoke.pt</option>
            </select>
          </label>
        </div>
        <div class="grid" style="margin-top: 10px;">
          <label>Seed
            <input id="seed" type="number" value="7">
          </label>
          <label>Custom Agent
            <input id="customSpec" type="text" placeholder="mcts:20:6">
          </label>
        </div>
        <label class="toggle"><input id="stochastic" type="checkbox" checked> Stochastic placement</label>
        <div class="actions">
          <button class="control primary" id="newGame">New Game</button>
          <button class="control" id="agentStep">Agent Step</button>
          <button class="control" id="autoPlay">Auto Play</button>
          <button class="control" id="useCustom">Use Custom For O</button>
        </div>
        <div id="message" class="message"></div>
      </section>
      <section class="section">
        <div class="status">
          <div class="metric"><b>Turn</b><span id="turnMetric">0</span></div>
          <div class="metric"><b>Current</b><span id="currentMetric">X</span></div>
          <div class="metric"><b>Winner</b><span id="winnerMetric">draw</span></div>
          <div class="metric"><b>Mode</b><span id="modeMetric">stochastic</span></div>
        </div>
      </section>
      <section>
        <div id="log" class="log"></div>
      </section>
    </aside>
  </main>
  <script>
    const boardEl = document.getElementById("board");
    const messageEl = document.getElementById("message");
    const headlineEl = document.getElementById("headline");
    let state = null;
    let busy = false;

    async function api(path, payload = null) {
      const options = payload ? {
        method: "POST",
        headers: {"content-type": "application/json"},
        body: JSON.stringify(payload)
      } : {};
      const response = await fetch(path, options);
      const data = await response.json();
      if (!response.ok && !data.error) {
        data.error = `Request failed: ${response.status}`;
      }
      return data;
    }

    function render(next) {
      state = next;
      messageEl.textContent = state.error || "";
      headlineEl.textContent = state.done
        ? (state.winnerAgent ? `${state.winnerAgent} wins` : "draw")
        : `${state.currentPlayer} / ${state.currentAgent}`;
      document.getElementById("turnMetric").textContent = state.turn;
      document.getElementById("currentMetric").textContent = state.done ? "-" : `${state.currentPlayer} (${state.currentAgent})`;
      document.getElementById("winnerMetric").textContent = state.done ? (state.winnerAgent || state.winner) : "-";
      document.getElementById("modeMetric").textContent = state.stochastic ? "stochastic" : "deterministic";
      document.getElementById("agentStep").disabled = busy || state.done || state.currentAgent === "human";
      document.getElementById("autoPlay").disabled = busy || state.done || state.currentAgent === "human";
      renderBoard();
      renderLog();
    }

    function renderBoard() {
      boardEl.innerHTML = "";
      const byCoord = new Map(state.cells.map(cell => [`${cell.row},${cell.col}`, cell]));
      const lastPlaced = state.lastMove && state.lastMove.placed ? `${state.lastMove.placed[0]},${state.lastMove.placed[1]}` : "";
      for (let row = 0; row < 12; row++) {
        for (let col = 0; col < 12; col++) {
          const cell = byCoord.get(`${row},${col}`);
          const button = document.createElement("button");
          button.className = "cell";
          if (!cell) {
            button.classList.add("invalid");
            button.disabled = true;
          } else {
            if (cell.value === 1) {
              button.textContent = "X";
              button.classList.add("occupied", "x");
            } else if (cell.value === -1) {
              button.textContent = "O";
              button.classList.add("occupied", "o");
            }
            if (`${row},${col}` === lastPlaced) button.classList.add("last");
            if (cell.ownThreat) button.classList.add("own-threat");
            if (cell.opponentThreat) button.classList.add("opponent-threat");
            if (cell.winning) button.classList.add("winning");
            button.disabled = busy || state.done || state.currentAgent !== "human" || !cell.legal;
            button.title = `${cell.action} (${row},${col})`;
            button.addEventListener("click", () => humanMove(cell.action));
          }
          boardEl.appendChild(button);
        }
      }
    }

    function renderLog() {
      const logEl = document.getElementById("log");
      if (!state.log.length) {
        logEl.innerHTML = "";
        return;
      }
      logEl.innerHTML = state.log.slice().reverse().map(row => {
        const placed = row.placed ? `to (${row.placed[0]},${row.placed[1]})` : "forfeited";
        const cls = row.player === "X" ? "x" : "o";
        return `<div class="log-row"><div><span class="badge ${cls}">${row.player}</span></div><div><b>${row.turn}. ${row.agent}</b><br>requested (${row.requested[0]},${row.requested[1]}) ${placed}<br>${row.reason}</div></div>`;
      }).join("");
    }

    async function newGame() {
      busy = true;
      const payload = {
        xSpec: document.getElementById("xSpec").value,
        oSpec: document.getElementById("oSpec").value,
        seed: Number(document.getElementById("seed").value || 7),
        stochastic: document.getElementById("stochastic").checked
      };
      render(await api("/api/new", payload));
      busy = false;
      render(state);
    }

    async function humanMove(action) {
      busy = true;
      render(state);
      render(await api("/api/move", {action}));
      busy = false;
      render(state);
    }

    async function agentStep() {
      busy = true;
      render(state);
      render(await api("/api/agent-step", {}));
      busy = false;
      render(state);
    }

    async function autoPlay() {
      busy = true;
      render(state);
      render(await api("/api/auto-play", {maxSteps: 200}));
      busy = false;
      render(state);
    }

    document.getElementById("newGame").addEventListener("click", newGame);
    document.getElementById("agentStep").addEventListener("click", agentStep);
    document.getElementById("autoPlay").addEventListener("click", autoPlay);
    document.getElementById("useCustom").addEventListener("click", () => {
      const value = document.getElementById("customSpec").value.trim();
      if (!value) return;
      const option = new Option(value, value, true, true);
      document.getElementById("oSpec").add(option);
    });

    api("/api/state").then(render);
  </script>
</body>
</html>
"""


def run_server(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), GameUIHandler)
    print(f"Game UI running at http://{host}:{port}", flush=True)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
