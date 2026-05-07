# Super Tic-Tac-Toe 实验路线文档

## 目标

在当前已完成的环境和 tabular Q-learning baseline 基础上，设计一条可逐步推进的实验路线，用于训练并评估更强的 super tic-tac-toe 智能体。

核心目标分三层：

1. 交付一个能稳定运行、可解释、可复现实验结果的作业版本。
2. 在 baseline 之外实现至少一种更强策略，例如 heuristic、MCTS 或 PPO。
3. 如果时间允许，探索 SFT 到 RL 的路线，用启发式或搜索策略生成数据，再用 RL 微调。

## 当前基础

已有文件：

- `super_tictactoe.py`: 游戏环境、棋盘几何、随机落子、胜负判断。
- `train_q_learning.py`: tabular Q-learning self-play baseline。
- `test_super_tictactoe.py`: 规则测试。
- `HANDOFF.md`: 当前实现交接记录。

当前 baseline 的定位：

- 优点：依赖少、容易跑通、能证明环境和训练闭环正确。
- 缺点：96 个可行动作、超大状态空间、随机落子机制让 tabular Q-learning 很难学到强策略。
- 后续实验应把它当作最低学习型 baseline，而不是最终主力 agent。

## 统一实验设定

为了方便公平比较，所有后续方法尽量共享以下设定。

### 状态表示

推荐神经网络方法使用 `3 x 12 x 12` 张量：

- 通道 1：当前玩家棋子。
- 通道 2：对手棋子。
- 通道 3：有效格/可落子 mask。

也可以增加扩展通道：

- 已占用格。
- 当前格子的 forfeiture risk。
- 每个格子参与潜在 winning lines 的数量。
- 当前玩家和对手的 immediate threat map。

### 动作空间

- 固定 96 个动作，对应 `VALID_CELLS`。
- 已占用格必须 action mask。
- 无效的 12x12 空白区域不进入动作空间。

### 奖励设计

第一版保持稀疏奖励：

- 赢：`+1`
- 输：`-1`
- 平：`0`
- 非终局：`0`

如果 PPO 训练不稳定，再考虑 shaping：

- 当前动作创造 immediate winning threat：小正奖励。
- 当前动作阻止对方 immediate win：小正奖励。
- 当前动作导致 forfeited：小负奖励，但不宜过大，因为随机偏移部分不可控。
- 选择边角或高风险格：只在没有战术收益时轻微惩罚。

### 评估指标

每种 agent 至少记录：

- vs random 的胜率。
- vs heuristic 的胜率。
- vs tabular Q-learning 的胜率。
- self-play 平均局长。
- forfeited move 比例。
- immediate win missed rate：存在一步获胜机会但没选的比例。
- immediate block missed rate：对手下一步能赢但没阻止的比例。
- 训练时间和推理时间。

建议固定多个 seed：

- 开发 smoke test：`seed = 7`
- 正式评估：`seed in [1, 2, 3, 4, 5]`

## 实验阶段 0：确认规则与基础环境

目标：

- 确认当前环境解释与课程要求一致。
- 尤其确认 column win 的 level 规则。

要做：

1. 复核题目或询问老师：竖向 4 连是否必须跨至少两个 level。
2. 如果规则不同，只改 `super_tictactoe.py` 的 `build_winning_lines()`。
3. 跑 `python3 -m unittest -v`。

成功标准：

- 规则解释明确。
- 所有测试通过。
- `HANDOFF.md` 中的歧义被更新或确认。

风险：

- 规则解释若错，后续训练结果都不可靠。

## 实验阶段 1：Heuristic Baseline

目标：

- 建立一个强于 random、推理速度快、解释性强的非学习 baseline。
- 为 MCTS rollout 和 SFT 数据生成提供基础策略。

策略优先级：

1. 如果某个动作在期望上能立即形成胜利，优先选。
2. 如果对手下一步有 immediate win，优先阻止。
3. 选择能形成多条 threat 的动作。
4. 选择靠近已有己方棋子的动作。
5. 避免高 forfeiture risk 的边界格，除非有直接战术收益。
6. 平局面使用中心性和 winning-line participation 评分。

注意 stochastic placement：

- 一个 intended action 不一定落在目标格。
- 评分时应计算 16 个 placement outcomes 的期望收益。
- 如果 redirect 到非法格或 occupied 格，要把 forfeiture 计入期望。

评估：

- heuristic vs random。
- heuristic vs tabular Q。
- deterministic 环境和 stochastic 环境分别评估。

成功标准：

- stochastic 环境下 vs random 胜率明显高于 50%。
- immediate win missed rate 接近 0。
- immediate block missed rate 显著低于 random。

风险：

- 如果 heuristic 只按 deterministic 棋局评分，会在随机落子环境中高估边界动作。

## 实验阶段 2：MCTS

目标：

- 用搜索增强棋类决策能力。
- 在不需要大规模训练的情况下得到一个强策略。

核心建模：

- 玩家节点：选择 intended action。
- Chance 节点：根据 16 个随机结果处理 placement。
- 终局节点：返回胜负结果。
- Rollout policy：不要纯随机，优先使用 heuristic。

推荐实现路线：

1. 先做 deterministic MCTS，验证树搜索逻辑。
2. 再加入 stochastic placement：
   - 简化版：每次 simulation 对 placement 采样。
   - 完整版：显式 chance node，按概率展开。
3. rollout policy 使用 heuristic-random 混合：
   - 70% heuristic。
   - 30% random exploration。
4. 使用 UCB/PUCT 选择动作。
5. 限制每步模拟次数，例如 100、500、1000，比较强度与耗时。

评估：

- MCTS-100 vs heuristic。
- MCTS-500 vs heuristic。
- MCTS-1000 vs heuristic。
- MCTS vs PPO，如果 PPO 已完成。

成功标准：

- 随模拟次数增加，胜率应稳定提升。
- MCTS-500 应明显强于纯 heuristic 或至少不弱。

风险：

- 分支因子很大，完整 chance node 会慢。
- 如果 rollout policy 太弱，MCTS 收敛慢。
- 搜索过程中必须小心复制环境状态，避免污染原局面。

## 实验阶段 3：PPO

目标：

- 用神经网络 policy/value 泛化局面，解决 tabular 方法状态空间太大的问题。

推荐模型：

- 输入：`3 x 12 x 12` 或更多通道。
- Backbone：小型 CNN。
- Policy head：输出 96 个动作 logits。
- Value head：输出当前玩家视角的 state value。
- Action mask：已占用格 logits 置为很小的负数。

训练方式：

- Self-play。
- 每一步都使用 canonical state，即从当前玩家视角看棋盘。
- trajectory 中保存：
  - state
  - action
  - log_prob
  - reward
  - done
  - value
  - action_mask
- 使用 GAE 计算 advantage。

推荐超参起点：

- learning rate: `3e-4`
- gamma: `0.99`
- lambda: `0.95`
- clip range: `0.2`
- entropy coefficient: `0.01`
- value coefficient: `0.5`
- rollout steps: `2048` 或 `4096`
- minibatch size: `256`
- PPO epochs: `4`

训练阶段：

1. deterministic 环境 smoke test。
2. stochastic 环境短训练。
3. 加入 reward shaping 或 curriculum。
4. 长训练并保存 checkpoint。

可选 curriculum：

- 先 deterministic placement。
- 再 25% stochastic。
- 最后完整 stochastic。

成功标准：

- PPO vs random 胜率持续上升。
- PPO vs heuristic 接近或超过 50%。
- 训练曲线没有 value loss 爆炸或 entropy 过早归零。

风险：

- 稀疏奖励导致学习慢。
- self-play 容易策略坍缩，需要保留旧版本 opponent pool。
- action mask 若处理错误，policy 会学到非法动作。

## 实验阶段 4：SFT 到 RL

目标：

- 先通过模仿学习得到一个不太离谱的初始 policy，再用 PPO fine-tune。
- 减少 PPO 冷启动成本。

数据来源：

- heuristic self-play。
- MCTS self-play。
- MCTS vs heuristic。
- human-designed tactical positions。

数据格式：

- state tensor。
- action label。
- action mask。
- optional: expert action probabilities。
- optional: game outcome。

SFT 训练：

- policy-only supervised learning。
- loss 使用 masked cross entropy。
- 如果使用 MCTS visit counts，可训练到 soft target distribution。
- 验证集看 top-1 action accuracy、legal action accuracy、immediate tactic accuracy。

SFT 后 RL：

1. 用 SFT policy 初始化 PPO policy head 和 backbone。
2. value head 可以从零初始化，也可以用 outcome regression 预训练。
3. PPO fine-tune 时保持较小 learning rate。
4. 评估 SFT-only、PPO-from-scratch、SFT-PPO 三者差异。

成功标准：

- SFT-only 能强于 random。
- SFT-PPO 前期学习速度快于 PPO-from-scratch。
- 最终胜率不低于 PPO-from-scratch。

风险：

- 如果 expert 数据质量低，SFT 会固化弱策略。
- 数据分布偏向 expert，RL 后期仍需要足够 exploration。

## 推荐执行顺序

优先级从稳到强：

1. 确认规则。
2. 实现 heuristic baseline。
3. 实现评估框架和指标记录。
4. 实现 MCTS。
5. 实现 PPO。
6. 用 heuristic/MCTS 生成 SFT 数据。
7. 训练 SFT policy。
8. 用 SFT 初始化 PPO。
9. 写报告和对比表。

如果时间紧：

- 最小强版本：heuristic + MCTS + 当前 tabular Q。
- 最小 RL 版本：当前 tabular Q + PPO。
- 最完整版本：heuristic + MCTS + PPO + SFT-PPO。

## 报告结构建议

1. Problem definition
   - 棋盘结构。
   - 随机落子规则。
   - 胜利条件。
2. Environment design
   - 状态空间。
   - 动作空间。
   - 奖励。
   - stochastic transition。
3. Baselines
   - random。
   - tabular Q-learning。
   - heuristic。
4. Advanced agents
   - MCTS。
   - PPO。
   - optional SFT-PPO。
5. Experiments
   - seeds。
   - evaluation opponents。
   - metrics。
6. Results
   - 胜率表。
   - 训练曲线。
   - 搜索次数 vs 胜率。
7. Discussion
   - 哪些方法有效。
   - 随机落子带来的困难。
   - 当前限制。
8. Future work
   - 更强的 neural network。
   - opponent pool。
   - TorchRL/TF-Agents/RLlib bonus。

## 建议结果表模板

| Agent | Train env | Eval env | Opponent | Win rate | Draw rate | Avg turns | Forfeit rate |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Random | none | stochastic | Random | TBD | TBD | TBD | TBD |
| Tabular Q | stochastic | stochastic | Random | 0.44 | 0.00 | 60.26 | TBD |
| Heuristic | none | stochastic | Random | TBD | TBD | TBD | TBD |
| MCTS-500 | none | stochastic | Heuristic | TBD | TBD | TBD | TBD |
| PPO | stochastic | stochastic | Random | TBD | TBD | TBD | TBD |
| SFT-PPO | stochastic | stochastic | Random | TBD | TBD | TBD | TBD |

## 下一步建议

下一步最适合先做 heuristic baseline 和统一 evaluator。

原因：

- heuristic 是 MCTS rollout 的基础。
- evaluator 是后续所有实验的共同地基。
- 做完后就能立刻给报告增加一个可靠对照组。
- PPO 和 SFT 都会受益于更好的评估指标和专家数据生成器。
