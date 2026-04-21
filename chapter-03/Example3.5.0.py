# ============================================================
# GridWorld（基于 Sutton & Barto Example 3.5 的简化实现）
#
# 这个程序实现了一个经典的强化学习网格世界（GridWorld）环境，
# 用来练习：
#   1. MDP（马尔可夫决策过程）的建模
#   2. 状态价值 / 动作价值的更新
#   3. 策略学习（如 value iteration、policy iteration、Q-learning 等）
#
# ------------------------------------------------------------
# 一、环境目标
#
# 在一个二维网格中，agent 每一步都处在某个格子（state）上，
# 并从四个动作中选择一个：
#   - north  : 向上
#   - south  : 向下
#   - east   : 向右
#   - west   : 向左
#
# agent 的目标是通过不断试错，学会在这个环境中采取更优动作，
# 使长期累积奖励最大。
#
# 如果当前实验重点放在特殊状态 B 上，
# 则可以重点观察 agent 是否会逐渐学会主动靠近 B，
# 并利用 “从 B 出发会跳转到 B' 且获得奖励” 这一机制。
#
# ------------------------------------------------------------
# 二、状态设计
#
# 网格中的每一个格子都对应一个状态 state。
# 通常可以用：
#   - (row, col)
# 或
#   - 一个整数编号
# 来表示状态。
#
# 此环境中包含普通状态，以及两个特殊状态：
#   - A
#   - B
#
# 同时还定义两个特殊跳转目标：
#   - A'
#   - B'
#
# ------------------------------------------------------------
# 三、动作设计
#
# 在每个状态下，agent 可选动作集合固定为：
#   ACTIONS = [up, down, left, right]
#
# 动作是确定性的（deterministic）：
#   - 选择向上，就尝试向上移动一格
#   - 选择向下，就尝试向下移动一格
#   - 选择向左，就尝试向左移动一格
#   - 选择向右，就尝试向右移动一格
#
# ------------------------------------------------------------
# 四、奖励机制
#
# 1. 普通移动
#   如果 agent 从普通格子移动到网格内另一个合法格子，
#   则奖励为：
#       reward = 0
#
# 2. 越界移动
#   如果动作会使 agent 走出网格边界，
#   则：
#       - agent 保持在原地不动
#       - reward = -1
#
# 3. 特殊状态 A
#   如果 agent 当前位于 A，
#   那么无论采取哪个动作，都会：
#       - 获得 reward = +10
#       - 下一状态强制跳转到 A'
#
# 4. 特殊状态 B
#   如果 agent 当前位于 B，
#   那么无论采取哪个动作，都会：
#       - 获得 reward = +5
#       - 下一状态强制跳转到 B'
#
# 也就是说，A 和 B 不遵循普通网格移动规则，
# 它们是环境手动定义的特殊跃迁状态。
#
# ------------------------------------------------------------
# 五、当前采用的策略（Q 版本贪心控制）
#
# 本实验采用动作价值函数 Q(s, a) 来做控制，
# 使用 Bellman 最优性更新：
#
#   Q_{k+1}(s, a) = sum_{s', r} p(s', r | s, a)
#                   [ r + gamma * max_{a'} Q_k(s', a') ]
#
# 对于本 GridWorld（确定性转移）可简化为：
#
#   Q_{k+1}(s, a) = r(s, a) + gamma * max_{a'} Q_k(s', a')
#
# 执行动作时使用贪心选择：
#
#   a = argmax_a Q(s, a)
#
# 若多个动作 Q 值相同，可在并列最大值中随机选一个动作。
# ============================================================
import numpy as np
import random
import matplotlib.pyplot as plt
class Agent():
    def __init__(self, epsilon = 0.1, alpha = 0.1, gamma = 0.9):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.Q = np.zeros((5,5), dtype = float)
        self.N = np.zeros((5,5), dtype = int)
        self.directions = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }
    def get_location(self, a, b):
        self.a = a
        self.b = b
    def Locate(self):
        rows, cols = self.Q.shape
        
        neighbors = {}
        for name, (da, db) in self.directions.items():
            na, nb = self.a + da, self.b + db
            if 0 <= na < rows and 0 <= nb < cols:
                neighbors[name] = self.Q[na, nb]
            else:
                neighbors[name] = None
        up = neighbors["up"]
        down = neighbors["down"]
        left = neighbors["left"]
        right = neighbors["right"]
        return up, down, left, right
    def Action(self):
        up, down, left, right = self.Locate()
        candidates = {
        "up": up,
        "down": down,
        "left": left,
        "right": right
        }
        valid = {k: v for k, v in candidates.items() if v is not None}
        best_action = max(valid, key=valid.get)
        if random.random() > self.epsilon:
            action = best_action
        else:
            action = random.choice(list(valid.keys()))
        return self.directions[action]
    def Update(self, r_next, a_next, b_next):
        self.N[self.a, self.b] += 1
        self.Q[self.a, self.b] += self.alpha * (r_next + self.gamma * self.Q[a_next, b_next] - self.Q[self.a, self.b])
class Env():
    def __init__(self):
        pass 
    def Critic(self, a, b, action):
        c21 = (a == 1 and b == 0)
        c22 = (a == 3 and b == 0)
        if c21 or c22:
            c1 = True
        else:
            na = a + action[0]
            nb = b + action[1]
            c1 = (0 <= na < 5 and 0 <= nb < 5)
        return c1, c21, c22
    def update_locate(self, a, b, action):
        c1, c21, c22 = self.Critic(a, b, action)
        if c1 == True:       
            if c21 == True:
                a = 1
                b = 4
            elif c22 == True:
                a = 3
                b = 2
            else:
                a += action[0]
                b += action[1]
        return a,b
    def reward(self, a, b, action):
        c1, c21, c22 = self.Critic(a, b, action)
        if c1 == True:       
            if c21 == True:
                r = 10
            elif c22 == True:
                r = 5
            else:
                r = 0
        else:
            r = -1
        return r

agent = Agent()
env = Env()
a = random.randrange(5)
b = random.randrange(5)
r = 0
reward_sum = 0.0
plot_steps = []
plot_avg_rewards = []
for i in range (100000):
    agent.get_location(a,b)
    move = agent.Action()
    r = env.reward(a, b, move)
    a_next, b_next = env.update_locate(a, b, move)
    agent.Update(r, a_next, b_next)
    a, b = a_next, b_next
    reward_sum += r
    if (i + 1) % 500 == 0:
        avg_reward = reward_sum / 500
        print(f"step {i + 1}: avg_reward={avg_reward:.4f}")
        plot_steps.append(i + 1)
        plot_avg_rewards.append(avg_reward)
        reward_sum = 0.0

plt.figure(figsize=(8, 4))
plt.plot(plot_steps, plot_avg_rewards, marker='o')
plt.xlabel('Step')
plt.ylabel('Average Reward (per 500 steps)')
plt.title('Training Average Reward')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()