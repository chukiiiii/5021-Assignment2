import unittest

from super_tictactoe import (
    O,
    VALID_CELLS,
    WINNING_LINES,
    X,
    SuperTicTacToeEnv,
    adjacent_cells,
    is_valid_cell,
)


class SuperTicTacToeTests(unittest.TestCase):
    def test_board_has_six_4_by_4_regions(self) -> None:
        self.assertEqual(len(VALID_CELLS), 96)
        self.assertTrue(is_valid_cell(0, 4))
        self.assertTrue(is_valid_cell(11, 11))
        self.assertFalse(is_valid_cell(0, 3))
        self.assertFalse(is_valid_cell(4, 1))

    def test_corner_has_five_outside_adjacent_targets(self) -> None:
        outside = [
            cell for cell in adjacent_cells(8, 0) if not is_valid_cell(cell[0], cell[1])
        ]
        self.assertEqual(len(outside), 5)

    def test_horizontal_four_wins(self) -> None:
        env = SuperTicTacToeEnv(stochastic=False)
        for col in range(4):
            env.board[8][col] = X
        self.assertEqual(env.check_winner(), X)

    def test_vertical_four_must_cross_levels(self) -> None:
        env = SuperTicTacToeEnv(stochastic=False)
        for row in range(8, 12):
            env.board[row][0] = O
        self.assertIsNone(env.check_winner())

        env = SuperTicTacToeEnv(stochastic=False)
        for row in range(5, 9):
            env.board[row][2] = O
        self.assertEqual(env.check_winner(), O)

    def test_diagonal_five_wins(self) -> None:
        env = SuperTicTacToeEnv(stochastic=False)
        for row, col in [(4, 2), (5, 3), (6, 4), (7, 5), (8, 6)]:
            env.board[row][col] = X
        self.assertEqual(env.check_winner(), X)

    def test_winning_lines_are_precomputed(self) -> None:
        self.assertGreater(len(WINNING_LINES), 0)

    def test_step_switches_player_after_forfeit(self) -> None:
        env = SuperTicTacToeEnv(seed=0, stochastic=True)
        action = env.cell_to_action(8, 0)
        for _ in range(50):
            env.reset()
            _, _, _, info = env.step(action)
            if info.forfeited:
                self.assertEqual(env.current_player, O)
                return
        self.fail("expected at least one forfeited corner move with fixed seed sequence")


if __name__ == "__main__":
    unittest.main()
