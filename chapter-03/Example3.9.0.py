# 程序完成目的简介：
# 1) 按题目给定的 MDP 环境（状态、动作、转移概率、奖励、折扣因子）建立模型；
# 2) 使用 Bellman 最优方程进行值迭代，在未收敛阶段持续更新状态价值；
# 3) 通过收敛阈值判断迭代结束，得到最优状态价值函数 v*(h), v*(l)；
# 4) 基于收敛后的价值函数提取最优策略 pi*；
# 5) 输出迭代过程关键信息（如每轮最大变化量 delta）与最终结果，便于校验。

# 已知回收机器人有两个状态：高电量 (h) 和低电量 (l)。可选动作分别为：搜索 (s)、等待 (w)、充电 (re)。其中，在状态 (h) 下可选动作是 (s,w)；在状态 (l) 下可选动作是 (s,w,re)。折扣因子为 (\gamma=0.9)。

# 状态转移与即时奖励如下：

# * 在 (h) 状态执行 (s)：以概率 (0.6) 转移到 (h)，奖励为 (5)；
#                       以概率 (0.4) 转移到 (l)，奖励为 (5)。

# * 在 (h) 状态执行 (w)：以概率 (1) 保持在 (h)，奖励为 (2)。

# * 在 (l) 状态执行 (s)：以概率 (0.7) 保持在 (l)，奖励为 (5)；
#                       以概率 (0.3) 转移到 (h)，奖励为 (-3)。

# * 在 (l) 状态执行 (w)：以概率 (1) 保持在 (l)，奖励为 (2)。
# * 在 (l) 状态执行 (re)：以概率 (1) 转移到 (h)，奖励为 (0)。

# 请根据上述 MDP 模型，写出该问题的 **Bellman 最优策略方程**，并求出最优状态价值函数 (v^*(h), v^*(l)) 以及最优策略 (\pi^*)。
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    gamma: float = 0.9
    theta: float = 1e-6
    max_iter: int = 10000
    p_hh: float = 0.6
    p_hl: float = 0.4
    p_ll: float = 0.7
    p_lh: float = 0.3


CFG = Config()

state = ['h', 'l']
action = ['s', 'w', 're']
action_set = {
    'h': ['s', 'w'],
    'l': ['s', 'w', 're']
}


def Bellman_Q(s, a, V):
    h_idx = 0
    l_idx = 1
    if s == 'h' and a == 's':
        q = CFG.p_hh * (5 + CFG.gamma * V[h_idx]) + CFG.p_hl * (5 + CFG.gamma * V[l_idx])
    elif s == 'h' and a == 'w':
        q = 2 + CFG.gamma * V[h_idx]
    elif s == 'l' and a == 's':
        q = CFG.p_ll * (5 + CFG.gamma * V[l_idx]) + CFG.p_lh * (-3 + CFG.gamma * V[h_idx])
    elif s == 'l' and a == 'w':
        q = 2 + CFG.gamma * V[l_idx]
    elif s == 'l' and a == 're':
        q = 0 + CFG.gamma * V[h_idx]
    else:
        q = -1e18
    return q


def Value_Iteration():
    V = np.array([0.0, 0.0], dtype=float)
    n_iter = 0
    final_delta = 0.0

    for i in range(CFG.max_iter):
        V_new = V.copy()
        delta = 0.0
        for s in state:
            s_idx = 0 if s == 'h' else 1
            q_list = []
            for a in action_set[s]:
                q = Bellman_Q(s, a, V)
                q_list.append(q)
            V_new[s_idx] = max(q_list)
            delta = max(delta, abs(V_new[s_idx] - V[s_idx]))

        V = V_new
        n_iter = i + 1
        final_delta = delta

        if delta < CFG.theta:
            break

    return V, n_iter, final_delta


def Get_Optimal_Policy(V):
    pi = {}
    for s in state:
        best_a = None
        best_q = -1e18
        for a in action_set[s]:
            q = Bellman_Q(s, a, V)
            if q > best_q:
                best_q = q
                best_a = a
        pi[s] = best_a
    return pi


if __name__ == '__main__':
    V_star, n_iter, final_delta = Value_Iteration()
    pi_star = Get_Optimal_Policy(V_star)

    print('Bellman 最优方程:')
    print('v*(h) = max{ 0.6*(5 + gamma*v*(h)) + 0.4*(5 + gamma*v*(l)), 2 + gamma*v*(h) }')
    print('v*(l) = max{ 0.7*(5 + gamma*v*(l)) + 0.3*(-3 + gamma*v*(h)), 2 + gamma*v*(l), gamma*v*(h) }')
    print('-' * 60)

    print('迭代轮数 =', n_iter)
    print('最终 delta =', final_delta)
    print('v*(h) =', round(float(V_star[0]), 6))
    print('v*(l) =', round(float(V_star[1]), 6))
    print('pi*(h) =', pi_star['h'])
    print('pi*(l) =', pi_star['l'])