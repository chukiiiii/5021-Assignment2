import os
import tempfile
import unittest

from agents.agents import make_agent
from models.dqn_model import action_mask, state_tensor
from models.ppo_model import PPOActorCritic, save_checkpoint
from env.super_tictactoe import SuperTicTacToeEnv


class PPOTests(unittest.TestCase):
    def test_ppo_forward_shapes(self) -> None:
        env = SuperTicTacToeEnv(stochastic=False)
        model = PPOActorCritic()
        logits, value = model(state_tensor(env).unsqueeze(0))
        self.assertEqual(tuple(logits.shape), (1, 96))
        self.assertEqual(tuple(value.shape), (1,))
        self.assertEqual(tuple(action_mask(env).shape), (96,))

    def test_ppo_agent_loads_checkpoint_and_returns_legal_action(self) -> None:
        env = SuperTicTacToeEnv(stochastic=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ppo.pt")
            save_checkpoint(path, PPOActorCritic(), {"test": True})
            agent = make_agent(f"ppo:{path}", seed=1)
            action = agent.select_action(env)
            self.assertIn(action, env.available_actions())


if __name__ == "__main__":
    unittest.main()
