import os
import tempfile
import unittest

from agents import make_agent
from dqn_model import DQN, action_mask, save_checkpoint, state_tensor
from super_tictactoe import SuperTicTacToeEnv


class DQNTests(unittest.TestCase):
    def test_state_tensor_and_mask_shapes(self) -> None:
        env = SuperTicTacToeEnv(stochastic=False)
        self.assertEqual(tuple(state_tensor(env).shape), (3, 12, 12))
        self.assertEqual(tuple(action_mask(env).shape), (96,))
        self.assertEqual(int(action_mask(env).sum().item()), 96)

    def test_dqn_forward_shape(self) -> None:
        env = SuperTicTacToeEnv(stochastic=False)
        model = DQN()
        output = model(state_tensor(env).unsqueeze(0))
        self.assertEqual(tuple(output.shape), (1, 96))

    def test_dqn_agent_loads_checkpoint_and_returns_legal_action(self) -> None:
        env = SuperTicTacToeEnv(stochastic=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            save_checkpoint(path, DQN(), {"test": True})
            agent = make_agent(f"dqn:{path}", seed=1)
            action = agent.select_action(env)
            self.assertIn(action, env.available_actions())


if __name__ == "__main__":
    unittest.main()
