"""
Super Tic-Tac-Toe PPO Training
"""

import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env import MultiAgentEnv


def make_env(opponent_type='rule-based', opponent_prob=0.5):
    """创建训练环境"""
    def _init():
        env = MultiAgentEnv(opponent_type=opponent_type, opponent_prob=opponent_prob)
        env = Monitor(env)
        return env
    return _init


def train():
    """PPO训练"""

    # 创建环境
    env = DummyVecEnv([make_env(opponent_type='rule-based', opponent_prob=0.5)])
    eval_env = DummyVecEnv([make_env(opponent_type='rule-based', opponent_prob=0.5)])

    # 创建模型
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log='./logs/',
    )

    # 回调
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='ppo_super_tictactoe')
    eval_callback = EvalCallback(eval_env, best_model_save_path='./models/best_model/', n_eval_episodes=100, eval_freq=5000)

    # 训练
    print("开始训练...")
    model.learn(
        total_timesteps=200_000,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # 保存最终模型
    model.save('./models/ppo_final')
    print("训练完成！")

    env.close()
    eval_env.close()


def evaluate(n_episodes=100):
    """评估模型"""
    from env import RuleBasedOpponent

    env = MultiAgentEnv(opponent_type='rule-based', opponent_prob=1.0)

    # 加载模型
    model = PPO.load('./models/best_model')

    wins = 0
    losses = 0
    draws = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1}: Win={wins}, Loss={losses}, Draw={draws}")

    print(f"\n=== 最终结果 (vs Rule-based) ===")
    print(f"Win: {wins}/{n_episodes} ({100*wins/n_episodes:.1f}%)")
    print(f"Loss: {losses}/{n_episodes} ({100*losses/n_episodes:.1f}%)")
    print(f"Draw: {draws}/{n_episodes} ({100*draws/n_episodes:.1f}%)")


if __name__ == '__main__':
    train()
    # evaluate(100)