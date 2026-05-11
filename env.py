"""
Super Tic-Tac-Toe RL Environment with Opponent
"""

import numpy as np
from enum import Enum
import gymnasium as gym
from gymnasium import spaces


class Player(Enum):
    PLAYER_1 = 1   # 当前智能体
    PLAYER_2 = -1  # 对手


class SuperTicTacToeEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super().__init__()

        # 棋盘结构: 6个4x4方块，3层（1+2+3=6个方块）
        self.board_shape = (6, 4, 4)
        self.num_squares = 6 * 4 * 4  # 96

        # 每个位置属于哪层 (用于Level 2限制检查)
        # Level 0: index 0 (1个方块)
        # Level 1: index 1-2 (2个方块)
        # Level 2: index 3-5 (3个方块)
        self.level = np.array([0] * 1 + [1] * 2 + [2] * 3, dtype=np.int32)

        # action: 选择96个位置之一
        self.action_space = spaces.Discrete(self.num_squares)
        # state: 棋盘 + 当前玩家
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.num_squares + 1,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(self.board_shape, dtype=np.int8)
        self.current_player = Player.PLAYER_1.value
        self.winner = None
        self.terminated = False
        self.truncated = False
        return self._get_obs(), {}

    def _get_obs(self):
        """返回观测: 展平棋盘 + 当前玩家"""
        board_flat = self.board.flatten().astype(np.float32)
        player = np.array([self.current_player], dtype=np.float32)
        return np.concatenate([board_flat, player])

    def _get_pos_coords(self, pos_idx):
        """将位置索引转换为(层, row, col)坐标"""
        layer = pos_idx // 16
        remainder = pos_idx % 16
        row = remainder // 4
        col = remainder % 4
        return layer, row, col

    def _pos_mask(self):
        """返回有效位置的mask"""
        return (self.board == 0).flatten()

    def _get_neighbors(self, layer, row, col):
        """获取2D的8个邻居位置（同层内）"""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < 4 and 0 <= nc < 4:
                    neighbors.append((layer, nr, nc))
        return neighbors

    def _apply_action(self, action):
        """应用动作，返回实际落子位置索引，无效返回-1"""
        layer, row, col = self._get_pos_coords(action)

        # 随机机制: 50%正常, 50%随机邻居
        if np.random.random() < 0.5:
            target_layer, target_row, target_col = layer, row, col
        else:
            neighbors = self._get_neighbors(layer, row, col)
            if neighbors:
                target_layer, target_row, target_col = neighbors[np.random.randint(len(neighbors))]
            else:
                target_layer, target_row, target_col = layer, row, col

        # 越界检查
        if not (0 <= target_layer < 6 and 0 <= target_row < 4 and 0 <= target_col < 4):
            return -1

        # 占用检查
        if self.board[target_layer, target_row, target_col] != 0:
            return -1

        # 落子成功
        self.board[target_layer, target_row, target_col] = self.current_player
        return target_layer * 16 + target_row * 4 + target_col

    def step(self, action):
        if self.terminated:
            return self._get_obs(), 0, True, False, {}

        # 检查动作是否合法
        valid_mask = self._pos_mask()
        if not valid_mask[action]:
            return self._get_obs(), -0.05, True, False, {'invalid': True}

        # 应用动作
        result = self._apply_action(action)

        if result == -1:
            return self._get_obs(), -0.1, True, False, {'invalid': True}

        # 计算reward
        reward = -0.005  # 步长惩罚

        # 检查胜利
        winner = self._check_winner()
        if winner is not None:
            self.winner = winner
            self.terminated = True
            if winner == self.current_player:
                reward += 10  # 赢: +10
            else:
                reward -= 10  # 输: -10
            return self._get_obs(), reward, True, False, {}

        # 检查平局
        if np.all(self.board != 0):
            self.terminated = True
            return self._get_obs(), 0, True, False, {}

        # 连子奖励
        reward += self._count_connect_reward()

        # 防守奖励：检查是否阻止对方连子
        reward += self._count_block_reward()

        # 切换玩家
        self.current_player = -self.current_player

        return self._get_obs(), reward, False, False, {}

    def _count_connect_reward(self):
        """计算连子奖励"""
        reward = 0
        player = self.current_player

        directions = [
            (1, 0, 0),   # 纵向(层内)
            (0, 1, 0),   # 行方向
            (0, 0, 1),   # 列方向
            (1, 1, 0),   # 对角线(层+行)
            (1, 0, 1),   # 对角线(层+列)
            (0, 1, 1),   # 对角线(行+列)
            (1, 1, 1),  # 立体对角线
        ]

        for layer in range(6):
            for row in range(4):
                for col in range(4):
                    if self.board[layer, row, col] != player:
                        continue

                    for dl, dr, dc in directions:
                        length = 1
                        while True:
                            nl, nr, nc = layer + dl * length, row + dr * length, col + dc * length
                            if 0 <= nl < 6 and 0 <= nr < 4 and 0 <= nc < 4:
                                if self.board[nl, nr, nc] == player:
                                    length += 1
                                else:
                                    break
                            else:
                                break

                        if length >= 2:
                            if length == 2:
                                reward += 0.02
                            elif length == 3:
                                reward += 0.08
                            elif length >= 4:
                                reward += 0.2
                            break

        return reward

    def _count_block_reward(self):
        """计算防守奖励：检查是否阻止对方形成3连"""
        reward = 0
        player = -self.current_player  # 对手

        # 简单检查：对手最近形成的连子长度
        # 如果对手有3连，阻止+0.1
        count_3 = self._check_threat_length(player, 3)
        if count_3 > 0:
            reward += 0.1 * count_3

        return reward

    def _check_threat_length(self, player, min_length):
        """检查玩家是否有min_length的连子"""
        directions = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
        ]
        count = 0

        for layer in range(6):
            for row in range(4):
                for col in range(4):
                    if self.board[layer, row, col] != player:
                        continue

                    for dl, dr, dc in directions:
                        length = 1
                        while True:
                            nl, nr, nc = layer + dl * length, row + dr * length, col + dc * length
                            if 0 <= nl < 6 and 0 <= nr < 4 and 0 <= nc < 4:
                                if self.board[nl, nr, nc] == player:
                                    length += 1
                                else:
                                    break
                            else:
                                break

                        if length == min_length:
                            count += 1
                            break
        return count

    def _check_winner(self):
        """检查胜利者"""
        winner = self._check_line_winner()
        if winner is not None:
            return winner
        return self._check_diagonal_winner()

    def _check_line_winner(self):
        """检查横竖胜利(4连)"""
        player = self.current_player

        directions = [(1, 0), (0, 1)]  # (行方向, 列方向)

        for layer in range(6):
            for row in range(4):
                for col in range(4):
                    if self.board[layer, row, col] != player:
                        continue

                    for dr, dc in directions:
                        length = 1
                        has_level2 = (self.level[layer] == 1)

                        while True:
                            nr, nc = row + dr * length, col + dc * length
                            if 0 <= nr < 4 and 0 <= nc < 4:
                                if self.board[layer, nr, nc] == player:
                                    if self.level[layer] == 1:
                                        has_level2 = True
                                    length += 1
                                else:
                                    break
                            else:
                                break

                        if length >= 4 and has_level2:
                            return player

        return None

    def _check_diagonal_winner(self):
        """检查对角线胜利(5连)"""
        player = self.current_player

        directions = [
            (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
        ]

        for layer in range(6):
            for row in range(4):
                for col in range(4):
                    if self.board[layer, row, col] != player:
                        continue

                    for dl, dr, dc in directions:
                        length = 1

                        while True:
                            nl, nr, nc = layer + dl * length, row + dr * length, col + dc * length
                            if 0 <= nl < 6 and 0 <= nr < 4 and 0 <= nc < 4:
                                if self.board[nl, nr, nc] == player:
                                    length += 1
                                else:
                                    break
                            else:
                                break

                        if length >= 5:
                            return player

        return None

    def render(self):
        for layer in range(6):
            print(f"Layer {layer}:")
            print(self.board[layer])
            print()


# ================== Rule-based 对手 ==================

class RuleBasedOpponent:
    """规则-based对手"""

    def __init__(self, player_id=-1):
        self.player_id = player_id

    def get_action(self, env):
        """获取对手动作"""
        board = env.board.copy()

        # 1. 能赢则赢
        win_action = self._find_winning_move(env, board)
        if win_action is not None:
            return win_action

        # 2. 能防则防
        block_action = self._find_blocking_move(env, board)
        if block_action is not None:
            return block_action

        # 3. 随机选择
        valid_actions = np.where(env._pos_mask())[0]
        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
        else:
            action = env.action_space.sample()

        return action

    def _find_winning_move(self, env, board):
        """找到能赢的动作"""
        player = self.player_id
        valid_actions = np.where(env._pos_mask())[0]

        for action in valid_actions:
            test_board = board.copy()
            layer, row, col = env._get_pos_coords(action)
            test_board[layer, row, col] = player

            # 临时检查胜利
            original = env.current_player
            env.current_player = player
            env.board = test_board
            winner = env._check_winner()
            env.board = board
            env.current_player = original

            if winner == player:
                return action

        return None

    def _find_blocking_move(self, env, board):
        """找到能阻止对方赢的动作"""
        player = self.player_id
        opponent = -player
        valid_actions = np.where(env._pos_mask())[0]

        for action in valid_actions:
            test_board = board.copy()
            layer, row, col = env._get_pos_coords(action)
            test_board[layer, row, col] = opponent

            original = env.current_player
            env.current_player = opponent
            env.board = test_board
            winner = env._check_winner()
            env.board = board
            env.current_player = original

            if winner == opponent:
                return action

        return None


# ================== Multi-agent 对战环境 ==================

class MultiAgentEnv(gym.Env):
    """支持智能体vs对手对战的环境"""

    def __init__(self, opponent_type='rule-based', opponent_prob=0.5):
        super().__init__()
        self.env = SuperTicTacToeEnv()
        self.opponent_type = opponent_type
        self.opponent_prob = opponent_prob
        self.opponent = RuleBasedOpponent()

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        # 玩家1（智能体）行动
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        if done:
            return obs, reward, done, False, info

        # 对手行动
        if self.opponent_type == 'rule-based' and np.random.random() < self.opponent_prob:
            opp_action = self.opponent.get_action(self.env)
        else:
            valid_actions = np.where(self.env._pos_mask())[0]
            opp_action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0

        result2 = self.env.step(opp_action)
        if len(result2) == 5:
            obs2, reward2, terminated2, truncated2, info2 = result2
            done2 = terminated2 or truncated2
        else:
            obs2, reward2, done2, info2 = result2

        # 智能体的reward是对手输赢的相反
        reward = -reward2
        done = done2
        info['opponent_action'] = opp_action

        return obs, reward, done, False, info

    def _get_obs(self):
        return self.env._get_obs()

    @property
    def board(self):
        return self.env.board

    @property
    def current_player(self):
        return self.env.current_player


if __name__ == '__main__':
    # 测试环境
    print("=== 测试单机环境 ===")
    env = SuperTicTacToeEnv()
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")

    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(f"Step {i}: action={action}, reward={reward:.3f}, done={done}")

        if done:
            env.render()
            print(f"Winner: {env.winner}")
            break

    print("\n=== 测试对战环境 ===")
    env2 = MultiAgentEnv()
    obs, _ = env2.reset()

    for i in range(20):
        action = env2.action_space.sample()
        obs, reward, done, info = env2.step(action)
        print(f"Step {i}: action={action}, reward={reward:.3f}, done={done}")

        if done:
            print(f"Winner: {env2.env.winner}")
            break