# ============================================================
# GridWorld v2 (Bellman Action-Value Control)
#
# 本文件是 Example3.5.0 的第二版，核心目标是：
#   使用 Bellman 行动价值思路，直接维护 Q(s, a)，
#   并通过 softmax 概率策略在探索与利用之间平衡，
#   让 agent 在 5x5 GridWorld 中学到更优行为策略。
#
# ------------------------------------------------------------
# 一、版本目标
#
# 1. 继续使用 5x5 网格世界（坐标范围 0..4）
# 2. 保留特殊状态 A / B 及其强制跳转和奖励
# 3. 采用基于 Q 的在线更新（TD 风格）
# 4. 通过训练曲线观察学习趋势（每 N 步平均奖励）
#
# ------------------------------------------------------------
# 二、环境与奖励约定
#
# 状态：
#   - 每个格子是一个状态 s = (a, b)
#   - a, b 均在 [0, 4]
#
# 动作：
#   - up / down / left / right
#   - 建议使用 direction 字典统一维护动作位移
#
# 奖励：
#   - 普通合法移动：reward = 0
#   - 越界移动：保持原地，reward = -1
#   - 在 A 状态：reward = +10，下一状态强制到 A'
#   - 在 B 状态：reward = +5，下一状态强制到 B'
#
# ------------------------------------------------------------
# 三、学习思想（Bellman 行动价值）
#
# 理论目标（最优行动价值）：
#
#   Q*(s, a) = sum_{s', r} p(s', r | s, a)
#              [ r + gamma * max_{a'} Q*(s', a') ]
#
# 在确定性 GridWorld 中可简化为：
#
#   Q(s, a) <- r + gamma * max_{a'} Q(s', a')
#
# 实际实现中通常使用增量更新（TD 形式）：
#
#   Q(s, a) <- Q(s, a)
#              + alpha * [ r + gamma * max_{a'} Q(s', a') - Q(s, a) ]
#
# 注：
#   - 如果当前实现继续维护状态表 V(a, b)，则对应的是状态价值 TD。
#   - 如果想严格实现本节的“行动价值版本”，应让 Q 具有动作维度，
#     例如 Q[a, b, action_id]。
#
# ------------------------------------------------------------
# 四、行为策略（softmax）
#
# 每一步：
#   - 根据当前状态下四个动作的 Q 值计算 softmax 概率
#   - 按概率分布采样动作，实现有偏探索
#
# 设计建议：
#   - 温度参数 tau 越大，策略越随机；越小，策略越接近贪心
#   - 为防数值溢出，可在指数前做稳定化处理
#
# ------------------------------------------------------------
# 五、单步交互流程
#
# 1. 读取当前状态 s_t = (a, b)
# 2. 根据 softmax 概率采样动作 a_t
# 3. 环境返回 (r_{t+1}, s_{t+1})
# 4. 执行 TD 更新
# 5. 状态推进到 s_{t+1}
# 6. 重复上述过程直到达到训练步数
#
# ------------------------------------------------------------
# 六、监控与输出
#
# 1. 每 500 步统计一次平均奖励：
#      avg_reward = window_reward_sum / 500
# 2. 将 (step, avg_reward) 记录到列表
# 3. 训练后用 matplotlib 绘图
#
# 观察重点：
#   - 曲线整体是否上升
#   - 后期波动是否减小
#   - 学习是否在 A/B 附近形成稳定偏好
#
# ------------------------------------------------------------
# 七、关键超参数
#
#   alpha   = 0.1
#   gamma   = 0.9
#   tau     = 0.1
#
# ------------------------------------------------------------
# 八、实现提示
#
# 1. 尽量让 Env 只负责“状态转移与奖励”，不负责学习。
# 2. Agent 只负责“选动作与更新价值”。
# 3. 保持坐标和方向定义全程一致（特别是 up/down 对应哪一维）。
# 4. 边界检查统一在一个函数中实现，避免逻辑分叉。
#
# ============================================================
import numpy as np
import random
import math
import matplotlib.pyplot as plt
class Agent():
    def __init__(self, tau = 0.1, gamma = 0.9, alpha = 0.1):
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.Q = np.zeros((5,5,4), dtype = float)
        self.N = np.zeros((5,5,4), dtype = int)
        self.directions = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }
    def get_location(self, a, b):
        self.a = a
        self.b = b
    def Softmax(self):
        exp_up = math.exp(self.value_up / self.tau ) 
        exp_down = math.exp(self.value_down / self.tau ) 
        exp_left = math.exp(self.value_left / self.tau ) 
        exp_right = math.exp(self.value_right / self.tau ) 
        sum_exp = exp_down+exp_left+exp_right+exp_up
        pu = exp_up/sum_exp
        pd = exp_down/sum_exp
        pl = exp_left/sum_exp
        pr = exp_right/sum_exp
        return pu,pd,pl,pr
    def get_value(self):
        self.value_up = self.Q[self.a, self.b, 0]
        self.value_down = self.Q[self.a, self.b, 1]
        self.value_left = self.Q[self.a, self.b, 2]
        self.value_right = self.Q[self.a, self.b, 3]
    def Action(self):
        self.get_value()
        pu,pd,pl,pr= self.Softmax()
        p_list = [pu, pd, pl, pr]
        t = 0
        x = random.random()
        action_to_id = {"up": 0, "down": 1, "left": 2, "right": 3}
        id_to_action = {v: k for k, v in action_to_id.items()}
        for i in range (4):
            t += p_list[i]
            if x <= t:
                self.best_action =  id_to_action[i]
                break
        return self.directions[self.best_action]
    def action_to_id(self):
        atid ={
            "up": 0,
            "down": 1,
            "left": 2,
            "right": 3,
        }
        return  atid[self.best_action]
    def Update(self, r_next, a_next, b_next):
        action_id = self.action_to_id()
        target = r_next + self.gamma * np.max(self.Q[a_next, b_next, :])
        self.Q[self.a, self.b, action_id] += self.alpha * (target - self.Q[self.a, self.b, action_id])
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
for i in range (20000):
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