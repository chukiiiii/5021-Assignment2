import unittest

from game_ui import GameSession
from super_tictactoe import O, X


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

    def test_state_marks_opponent_threats(self) -> None:
        session = GameSession()
        session.new_game("human", "random", seed=3, stochastic=False)
        session.env.board[4][2] = O
        session.env.board[4][3] = O
        session.env.board[4][4] = O
        state = session.serialize()
        threat_cells = [cell for cell in state["cells"] if cell["opponentThreat"]]
        self.assertEqual(len(threat_cells), 1)
        self.assertEqual((threat_cells[0]["row"], threat_cells[0]["col"]), (4, 5))

    def test_state_marks_winning_line(self) -> None:
        session = GameSession()
        session.new_game("human", "random", seed=3, stochastic=False)
        for col in range(4):
            session.env.board[8][col] = X
        session.env.winner = X
        session.env.done = True
        state = session.serialize()
        winning_cells = [cell for cell in state["cells"] if cell["winning"]]
        self.assertEqual(len(winning_cells), 4)


if __name__ == "__main__":
    unittest.main()
