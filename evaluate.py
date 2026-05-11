"""
评估模型胜率
"""

import numpy as np
import sys
sys.path.append('.')
from env import SuperTicTacToeEnv, MultiAgentEnv, RuleBasedOpponent
from stable_baselines3 import PPO


def evaluate_vs_rulebased(n_episodes=100):
    """vs Rule-based对手评估"""
    env = MultiAgentEnv(opponent_type='rule-based', opponent_prob=1.0)
    model = PPO.load('models/best_model/best_model.zip')

    wins = 0
    losses = 0
    draws = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result

        # 检查reward判断胜负
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1}: Win={wins}, Loss={losses}, Draw={draws}")

    print(f"\n=== vs Rule-based ({n_episodes}局) ===")
    print(f"Win:  {wins}/{n_episodes} ({100*wins/n_episodes:.1f}%)")
    print(f"Loss: {losses}/{n_episodes} ({100*losses/n_episodes:.1f}%)")
    print(f"Draw: {draws}/{n_episodes} ({100*draws/n_episodes:.1f}%)")


def evaluate_random(n_episodes=100):
    """vs Random对手评估"""
    env = MultiAgentEnv(opponent_type='random', opponent_prob=1.0)
    model = PPO.load('models/best_model/best_model.zip')

    wins = 0
    losses = 0
    draws = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    print(f"\n=== vs Random ({n_episodes}局) ===")
    print(f"Win:  {wins}/{n_episodes} ({100*wins/n_episodes:.1f}%)")
    print(f"Loss: {losses}/{n_episodes} ({100*losses/n_episodes:.1f}%)")
    print(f"Draw: {draws}/{n_episodes} ({100*draws/n_episodes:.1f}%)")


if __name__ == '__main__':
    print("加载最佳模型: models/best_model/best_model.zip")
    evaluate_vs_rulebased(100)
    evaluate_random(100)