# ============================================================
# GridWorld v3 (Bellman Optimal State-Value)
#
# 目标：
#   按题意求解 Example 3.5 的最优状态价值函数 v*(s)，
#   并由 v*(s) 导出最优策略（一个状态可对应多个最优动作）。
#
# 环境设定（5x5）：
#   - 普通合法移动：reward = 0
#   - 越界移动：原地不动，reward = -1
#   - A: reward = +10，下一状态强制到 A' = (1, 4)
#   - B: reward = +5，下一状态强制到 B' = (3, 2)
#
# Bellman 最优状态方程：
#   v*(s) = max_a [ r(s, a) + gamma * v*(s') ]
#
# 代码更新逻辑：
#   1. 对每个状态 s 枚举 4 个动作
#   2. 计算每个动作的一步回报 q(s,a)=r+gamma*v(s')
#   3. 用 max_a q(s,a) 更新该状态价值
#   4. 反复全表扫描直到收敛（delta < theta）
#
# 与采样式 TD 的区别：
#   - 不依赖 epsilon/softmax 采样轨迹
#   - 不使用 alpha 增量学习率
#   - 每轮对所有状态做 Bellman 最优备份
#
# ============================================================
import numpy as np
import random

class Agent():
    def __init__(self, gamma=0.9, theta=1e-4):
        self.Q = np.zeros((5, 5), dtype=float)
        self.gamma = gamma
        self.theta = theta
    def Update(self, i, j, maxm):
        self.Q[i, j] = maxm


def print_value_table(table, title):
    rows, cols = table.shape
    print(title)
    border = "+" + "+".join(["-" * 10] * (cols + 1)) + "+"
    header = "|{:^10}".format("row\\col") + "".join(
        ["|{:^10}".format(c) for c in range(cols)]
    ) + "|"

    print(border)
    print(header)
    print(border)
    for r in range(rows):
        line = "|{:^10}".format(r) + "".join(
            ["|{:>10.3f}".format(table[r, c]) for c in range(cols)]
        ) + "|"
        print(line)
        print(border)


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

dic = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

iterations = 0
while True:
    iterations += 1
    delta = 0.0
    for i in range(5):
        for j in range(5):
            v_all_dir = []
            for _, action in dic.items():
                c1, c21, c22 = env.Critic(i, j, action)
                if c21:
                    a, b = 1, 4
                elif c22:
                    a, b = 3, 2
                elif c1:
                    a = i + action[0]
                    b = j + action[1]
                else:
                    a, b = i, j

                r = env.reward(i, j, action)
                v = r + agent.gamma * agent.Q[a, b]
                v_all_dir.append(v)

            m = max(v_all_dir)
            old_v = agent.Q[i, j]
            agent.Update(i, j, m)
            delta = max(delta, abs(old_v - agent.Q[i, j]))

    if delta < agent.theta:
        break

print(f"Converged after {iterations} iterations, delta={delta:.6f}")
print_value_table(agent.Q, "Final Q Table (state values)")