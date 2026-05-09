import os
import tempfile
import unittest

from agents import HeuristicAgent, QTableAgent, RandomAgent
from agents import make_agent
from super_tictactoe import X, SuperTicTacToeEnv
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
        if not os.path.exists("q_table_smoke.json"):
            self.skipTest("q_table_smoke.json not available")
        agent = QTableAgent("q_table_smoke.json", seed=1)
        self.assertGreater(len(agent.q_table), 0)

    def test_heuristic_takes_immediate_deterministic_win(self) -> None:
        env = SuperTicTacToeEnv(stochastic=False)
        env.board[8][0] = X
        env.board[8][1] = X
        env.board[8][2] = X
        agent = HeuristicAgent(seed=1)
        action = agent.select_action(env)
        self.assertEqual(env.action_to_cell(action), (8, 3))

    def test_make_agent_supports_heuristic(self) -> None:
        agent = make_agent("heuristic", seed=1)
        self.assertEqual(agent.name, "heuristic")

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
