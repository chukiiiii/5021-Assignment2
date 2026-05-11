# 项目结构快速参考

## 📂 目录说明

| 目录 | 用途 | 主要文件 |
|------|------|----------|
| `src/env/` | 游戏环境 | 游戏规则、匹配系统、UI界面 |
| `src/agents/` | Agent实现 | 各种Agent算法、人机对战 |
| `src/models/` | 深度学习模型 | DQN、PPO网络架构 |
| `train/` | 训练脚本 | Q-learning、DQN、PPO、BC训练 |
| `eval/` | 评估实验 | Agent对战、实验运行 |
| `tests/` | 单元测试 | 各模块测试用例 |
| `docs/` | 文档 | README、实验计划、报告 |
| `results/` | 实验结果 | 检查点、评估报告、训练历史 |
| `utils/` | 工具脚本 | （预留） |

## 🚀 常用命令速查

### 运行测试
```bash
python3 -m pytest tests/ -v
```

### 训练模型
```bash
# Q-Learning
python3 train/train_q_learning.py

# DQN
python3 train/train_dqn.py --episodes 100

# PPO
python3 train/train_ppo.py --episodes 100

# 多种子训练
python3 train/train_dqn_multiseed.py --seeds 201 202 203 --episodes 120
```

### 行为克隆
```bash
# 生成专家数据
python3 train/generate_expert_data.py

# 训练BC模型
python3 train/train_bc.py

# SFT微调
python3 train/train_ppo_multiseed.py --init-checkpoint results/checkpoints/bc_mixed.pt
```

### 评估Agent
```bash
# 单场评估
python3 eval/evaluate_agents.py --agent-a dqn:ckpt.pt --agent-b random --games 20

# 批量实验
python3 eval/run_experiments.py --preset full --seeds 1 2 3
```

### 游戏体验
```bash
# 终端模式
python3 src/agents/play_game.py

# 浏览器UI
python3 src/env/game_ui.py --host 127.0.0.1 --port 8765
```

## 📊 结果输出位置

- **模型检查点**: `results/checkpoints/`
- **评估报告**: `results/evaluations/`
- **训练历史**: `results/history/`
- **汇总报告**: `results/summary/`

## 🔧 依赖安装

```bash
pip install -r requirements.txt
```

## 📝 文档

- [主文档](docs/README.md) - 完整的项目说明
- [实验计划](docs/EXPERIMENT_PLAN.md) - 实验路线图
- [迁移指南](docs/MIGRATION_GUIDE.md) - 目录重构说明
- [交接文档](docs/HANDOFF.md) - 详细的工作日志
