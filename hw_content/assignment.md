# Super Tic-Tac-Toe Assignment

## 📌 Assignment Description

**Assignment 2 (40%) – Super Tic-Tac-Toe**

The game is similar to traditional tic-tac-toe, but with modified winning conditions:

* A player wins if they get:

  * **4 in a row horizontally or vertically**, OR
  * **5 in a row diagonally**
* Additionally:

  * For horizontal or vertical wins, **at least one move must be in Level 2 (middle layer)**

---

## 📌 棋盘结构（Board Structure）

The board is shaped like a **triangular structure** composed of multiple layers:

* The board consists of **6 squares**
* Each square is of size **4 × 4**
* The structure is divided into **three levels**:

  * Level 1 (top)
  * Level 2 (middle)
  * Level 3 (bottom)

---

## 📌 游戏规则（Game Rules）

* Two players take turns
* Each turn:

  * A player selects an empty square
  * Places either:

    * **nought (O)** or
    * **cross (X)**

---

## 📌 随机机制（Stochastic Mechanism）

After a player selects a square:

* With probability **1/2**:

  * The mark is placed in the chosen square

* With probability **1/2**:

  * The move is replaced by a **random neighboring square**

Details:

* One of the **8 neighboring squares** is selected
* Each neighbor has probability **1/16**
* Boundaries are ignored when sampling

Then:

* If the selected square is:

  * Outside the board → ignored
  * Already occupied → ignored
  * Otherwise → mark is placed

Example:

* If the chosen square is a **corner**
* Then **5/16 probability** leads to outside-board selection

---

## 📌 终止条件（Game Termination）

The game ends when:

* A player satisfies the winning condition
* OR
* The board is completely filled (draw)

---

## 📌 任务要求（Task Requirement）

Train a **Reinforcement Learning (RL) agent** to play this game.

---

## 📌 Bonus（加分项）

If you implement using:

* TensorFlow Agent
* TorchRL
* RLlib

Then:

* Your score is multiplied by **1.5**
* Bonus is **capped at 50%**

---

# ================= 中文版本 =================

## 📌 作业说明

**作业 2（占 40%）：超级井字棋**

该游戏类似传统井字棋，但规则有所变化：

* 获胜条件：

  * 横向或纵向 **4 连**
  * 对角线 **5 连**
* 额外限制：

  * 横/竖取胜时，**至少一个子必须在中间层（Level 2）**

---

## 📌 棋盘结构

* 棋盘为**三层三角形结构**
* 共由 **6 个 4×4 的方块组成**
* 分为三层：

  * 第一层（顶部）
  * 第二层（中间）
  * 第三层（底部）

---

## 📌 游戏规则

* 两名玩家轮流行动
* 每回合：

  * 选择一个空格
  * 放置：

    * 圈（O）或叉（X）

---

## 📌 随机机制

玩家选择一个位置后：

* **50% 概率**：

  * 落子成功

* **50% 概率**：

  * 落子会被随机扰动

具体：

* 从 **8 个邻居格子**中随机选一个
* 每个概率为 **1/16**
* 忽略边界限制

然后：

* 如果目标位置：

  * 越界 → 忽略
  * 已占用 → 忽略
  * 否则 → 落子

示例：

* 如果选角落：

  * 有 **5/16 概率落到棋盘外**

---

## 📌 终止条件

* 任一玩家满足胜利条件
* 或棋盘填满（平局）

---

## 📌 任务

使用**强化学习方法**训练一个智能体完成该游戏。

---

## 📌 加分项

使用以下框架：

* TensorFlow Agent
* TorchRL
* RLlib

可获得：

* 分数 ×1.5
* 上限为 50%
