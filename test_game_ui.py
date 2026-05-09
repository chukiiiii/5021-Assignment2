import unittest

from game_ui import GameSession


class GameUITests(unittest.TestCase):
    def test_new_game_serializes_cells(self) -> None:
        session = GameSession()
        state = session.new_game("human", "random", seed=3, stochastic=False)
        self.assertEqual(len(state["cells"]), 96)
        self.assertEqual(state["currentAgent"], "human")
        self.assertFalse(state["done"])

    def test_human_move_updates_log(self) -> None:
        session = GameSession()
        session.new_game("human", "random", seed=3, stochastic=False)
        state = session.human_move(0)
        self.assertEqual(state["turn"], 1)
        self.assertEqual(len(state["log"]), 1)
        self.assertEqual(state["currentAgent"], "random")

    def test_agent_step_rejects_human_turn(self) -> None:
        session = GameSession()
        session.new_game("human", "random", seed=3, stochastic=False)
        state = session.agent_step()
        self.assertIn("human", state["error"])

    def test_auto_play_stops_before_human_turn(self) -> None:
        session = GameSession()
        session.new_game("random", "human", seed=3, stochastic=False)
        state = session.auto_play()
        self.assertEqual(state["turn"], 1)
        self.assertEqual(state["currentAgent"], "human")


if __name__ == "__main__":
    unittest.main()
