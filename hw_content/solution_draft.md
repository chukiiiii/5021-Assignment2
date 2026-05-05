# Super Tic-Tac-Toe RL Solution Design

## 一、问题建模

该问题属于：

* 双人对抗
* 随机转移（stochastic transition）
* 离散动作空间

可建模为：

> Markov Decision Process (MDP) + 对抗环境

简化方式：

* 单智能体 + 固定对手（rule-based）
* 或 self-play

---

## 二、环境设计（Env）

### 1. State（状态）

建议表示为：

* 一维或三维棋盘张量：

  * 0 = 空
  * 1 = 当前玩家
  * -1 = 对手

包含信息：

* 棋盘状态
* 当前玩家

---

### 2. Action（动作）

* 离散空间：

  * action ∈ [0, N-1]
* 表示选择一个格子

需要：

* action mask（避免选非法格子）

---

### 3. Transition（转移）

核心逻辑：

1. 玩家选择 action
2. 以 0.5 概率：

   * 正常落子
3. 以 0.5 概率：

   * 随机选择 8 邻居
4. 检查合法性：

   * 越界 / 已占 → 无效

---

### 4. Done（终止）

* 达成胜利条件
* 或棋盘填满

---

## 三、Reward 设计

### 1. 总体结构

reward =

* 终局奖励
* 进攻奖励
* 防守奖励
* 无效惩罚
* 步长惩罚

---

### 2. 终局奖励（核心）

* 赢：+1
* 输：-1
* 平：0

---

### 3. 进攻奖励

基于连子长度：

* 2连：+0.02
* 3连：+0.08
* 4连：+0.2

采用：

> reward = new_score - old_score

避免重复计算

---

### 4. 防守奖励

当成功阻止对方连子：

* 阻止3连：+0.08

---

### 5. 无效动作惩罚

* 落子失败：-0.05

---

### 6. 步长惩罚

* 每步：-0.005

作用：

* 防止拖延

---

## 四、对手设计

### 1. Rule-based

策略：

* 能赢 → 赢
* 能防 → 防
* 否则随机

---

### 2. Self-play

* agent vs agent
* 提高上限

---

### 3. 混合策略（推荐）

* 50% rule-based
* 50% self-play

---

## 五、算法选择

推荐：

### PPO（首选）

优点：

* 稳定
* 适合 stochastic 环境
* 支持 action mask

---

备选：

* DQN（不太稳定）
* GRPO（复杂度高）

---

## 六、实现流程

### Step 1：实现环境

* reset
* step
* reward
* done

---

### Step 2：测试环境

* random agent

---

### Step 3：加入对手

* rule-based

---

### Step 4：接入 RL

* PPO / RLlib / TorchRL

---

### Step 5：训练

* 观察 reward 曲线
* 调整 reward scaling

---

## 七、关键难点

### 1. 随机扰动

问题：

* action ≠ 执行结果

影响：

* variance 高
* 学习变慢

---

### 2. reward 设计

* 太 sparse → 学不动
* 太 dense → 学歪

---

### 3. 胜利检测

* 横/竖/对角
* 中层限制

---

### 4. credit assignment

* 结果延迟
* PPO + GAE 可缓解

---

## 八、优化方向

### 1. Potential-based reward

R = γΦ(s') - Φ(s)

---

### 2. Curriculum Learning

* 先无随机
* 再加噪声

---

### 3. Self-play curriculum

* 逐步提升对手强度

---

## 九、总结

该问题核心在于：

* 随机环境建模
* reward 设计
* 稳定训练

推荐策略：

* PPO + 轻量 reward shaping + rule-based opponent

即可达到较好效果。
