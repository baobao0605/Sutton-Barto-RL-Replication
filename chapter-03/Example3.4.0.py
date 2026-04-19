# ============================================
# Mini Balance：极简平衡控制任务
#
# 目标：
# 让一个“简化杆”尽量长时间保持直立，不要倒下。
#
# 状态：
#   theta : 杆子的倾斜程度，取值范围为 -9 ~ +9
#           可近似理解为 -90° ~ +90°
#           theta = 0 表示完全竖直
#           theta < 0 表示向左倾斜
#           theta > 0 表示向右倾斜
#
#   omega : 杆子的角速度
#           omega < 0 表示继续向左倒
#           omega > 0 表示继续向右倒
#
# 动作：
#   action ∈ {-1, 0, +1}
#   -1 : 向左施加控制
#    0 : 不施加控制
#   +1 : 向右施加控制
#
# 状态更新：
#   每一步根据当前 theta、omega 和 action 更新下一时刻状态。
#   这里不使用真实物理引擎，只用简化规则模拟“会倒、可纠正”的动态过程。
#
# 终止条件：
#   如果 theta 超出允许范围（例如 < -9 或 > +9），
#   则认为平衡失败，当前回合结束。
#
# 奖励：
#   每成功存活一步，reward = +1
#   若失败，回合结束
#
# 训练目标：
#   学习一个策略，使总奖励最大，
#   也就是让杆子保持平衡尽可能久。
# ============================================
import numpy as np
import random
import matplotlib.pyplot as plt

class Agent():
    def __init__(self, theta_max = 9, epsilon = 0.1):
        self.theta_max = theta_max
        self.epsilon = epsilon
        self.a = [-1, 0, 1]
        self.a_idx = 0
        self.b = 0
        self.bin_edges = np.array([-9.0, -7.875, -6.75, -4.5, 0.0, 4.5, 6.75, 7.875, 9.0])
        self.Q = np.zeros((8,3), dtype=float)
        self.N = np.zeros((8,3), dtype=int)

    def state_to_bin(self, theta):
        theta = max(-9.0, min(9.0, theta))
        if theta == 9.0:
            return 7
        return int(np.searchsorted(self.bin_edges, theta, side="right") - 1)

    def act(self, theta):
        b = self.state_to_bin(theta)
        if random.random() < self.epsilon:
           a_idx = random.choice([0, 1, 2])
        else:
            a_idx = int(np.argmax(self.Q[b]))
        action = self.a[a_idx]
        return a_idx,action

    def award(self, theta):
        if abs(theta) < self.theta_max:
            reward = 1
        else:
            reward = -1
        return reward
    def update(self, theta, r, a_idx):
        b = self.state_to_bin(theta)
        self.N[b, a_idx] += 1
        self.Q[b, a_idx] += (r - self.Q[b, a_idx]) / self.N[b, a_idx]

class Env():
    def __init__(self, omega=None, theta=None):
        if omega is None:
            omega = random.uniform(-0.5, 0.5)
        if theta is None:
            theta = random.uniform(-2, 2)
        self.omega = omega
        self.theta = theta
               
    def reset(self):
        self.omega = random.uniform(-0.5, 0.5)
        self.theta = random.uniform(-2, 2)
    
    def update(self, action):
        noise = random.uniform(-0.02, 0.02)
        self.theta += self.omega
        self.omega = 0.95 * self.omega + 0.15 * self.theta + 0.5 * action + noise

env = Env()
agent = Agent()
gamma = 0.1
bin_best_action_history = []
for episode in range(1000):
    steps = 0
    discounted_return = 0
    env.reset()
    aidxlist = []
    rewards = []
    theta = []
    while True:
        steps += 1
        theta.append(env.theta)
        aidx,a = agent.act(env.theta)
        aidxlist.append(aidx)
        env.update(a)
        rewards.append(agent.award(env.theta))
        if abs(env.theta) >= 9 or steps >= 10000:
            break
    for j in range(steps-1, -1, -1):
        discounted_return = rewards[j] + gamma * discounted_return
        agent.update(theta[j], discounted_return, aidxlist[j])
    best_a_idx_per_bin = np.argmax(agent.Q, axis=1)
    best_action_per_bin = np.array(agent.a)[best_a_idx_per_bin]
    bin_best_action_history.append(best_action_per_bin.copy())

bin_best_action_history = np.array(bin_best_action_history)
fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
axes = axes.ravel()

for b in range(agent.Q.shape[0]):
    left_edge = agent.bin_edges[b]
    right_edge = agent.bin_edges[b + 1]
    axes[b].plot(bin_best_action_history[:, b], linewidth=1.2)
    axes[b].set_title(f"bin {b} [{left_edge:.3f}, {right_edge:.3f})")
    axes[b].set_ylim(-1.2, 1.2)
    axes[b].set_yticks([-1, 0, 1])
    axes[b].grid(True, alpha=0.3)

fig.suptitle("Best Action Per Bin Across Episodes")
fig.supxlabel("Episode")
fig.supylabel("Best Action")
plt.tight_layout()
plt.show()


