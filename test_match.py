import os
import tempfile
import unittest

from agents import HeuristicAgent, MCTSAgent, QTableAgent, RandomAgent
from agents import make_agent
from super_tictactoe import O, X, SuperTicTacToeEnv
from evaluate_agents import save_csv, save_html, save_json
from match import evaluate_pair, results_dict


class MatchTests(unittest.TestCase):
    def test_random_match_summary_counts_games(self) -> None:
        summary, results = evaluate_pair(
            RandomAgent(seed=1, name="random-a"),
            RandomAgent(seed=2, name="random-b"),
            games=6,
            seed=3,
            stochastic=False,
        )
        self.assertEqual(summary.games, 6)
        self.assertEqual(len(results), 6)
        self.assertEqual(summary.agent_a_wins + summary.agent_b_wins + summary.draws, 6)

    def test_qtable_agent_loads_existing_smoke_table(self) -> None:
        if not os.path.exists("results/checkpoints/q_table_smoke.json"):
            self.skipTest("results/checkpoints/q_table_smoke.json not available")
        agent = QTableAgent("results/checkpoints/q_table_smoke.json", seed=1)
        self.assertGreater(len(agent.q_table), 0)

    def test_heuristic_takes_immediate_deterministic_win(self) -> None:
        env = SuperTicTacToeEnv(stochastic=False)
        env.board[8][0] = X
        env.board[8][1] = X
        env.board[8][2] = X
        agent = HeuristicAgent(seed=1)
        action = agent.select_action(env)
        self.assertEqual(env.action_to_cell(action), (8, 3))

    def test_heuristic_blocks_immediate_opponent_win(self) -> None:
        env = SuperTicTacToeEnv(stochastic=False)
        env.board[4][2] = O
        env.board[4][3] = O
        env.board[4][4] = O
        agent = HeuristicAgent(seed=1)
        action = agent.select_action(env)
        self.assertEqual(env.action_to_cell(action), (4, 5))

    def test_make_agent_supports_heuristic(self) -> None:
        agent = make_agent("heuristic", seed=1)
        self.assertEqual(agent.name, "heuristic")

    def test_make_agent_supports_mcts_candidates(self) -> None:
        agent = make_agent("mcts:2:4", seed=1)
        self.assertEqual(agent.name, "mcts:2")
        self.assertEqual(agent.max_candidates, 4)

    def test_mcts_returns_legal_action(self) -> None:
        env = SuperTicTacToeEnv(stochastic=False)
        agent = MCTSAgent(simulations=3, rollout_depth=2, seed=1)
        action = agent.select_action(env)
        self.assertIn(action, env.available_actions())

    def test_mcts_candidates_include_immediate_block(self) -> None:
        env = SuperTicTacToeEnv(stochastic=False)
        env.board[4][2] = O
        env.board[4][3] = O
        env.board[4][4] = O
        agent = MCTSAgent(simulations=3, max_candidates=4, seed=1)
        candidate_actions = [action for action, _ in agent._candidate_action_scores(env)]
        self.assertIn(env.cell_to_action(4, 5), candidate_actions)

    def test_reports_are_written(self) -> None:
        summary, results = evaluate_pair(
            RandomAgent(seed=1, name="random-a"),
            RandomAgent(seed=2, name="random-b"),
            games=2,
            seed=3,
            stochastic=False,
        )
        rows = results_dict(results)
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "report.json")
            csv_path = os.path.join(tmpdir, "report.csv")
            html_path = os.path.join(tmpdir, "report.html")
            save_json(json_path, summary, rows)
            save_csv(csv_path, rows)
            save_html(html_path, summary, rows)
            self.assertTrue(os.path.getsize(json_path) > 0)
            self.assertTrue(os.path.getsize(csv_path) > 0)
            self.assertTrue(os.path.getsize(html_path) > 0)


if __name__ == "__main__":
    unittest.main()
