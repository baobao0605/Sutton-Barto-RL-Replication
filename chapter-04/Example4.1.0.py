"""
程序简介

本程序任务是实现“迭代策略评估（Iterative Policy Evaluation）”，用于计算 4×4 网格世界中各状态的状态价值函数。

已知条件如下：
1. 环境为 4×4 网格世界。
2. 左上角与右下角为终止状态。
3. 其余 14 个格子为非终止状态。
4. 每一步即时奖励为 -1。
5. 折扣因子 γ = 1。
6. 在每个非终止状态下，智能体以等概率随机选择上、下、左、右四个动作。
7. 每个动作概率均为 0.25。

核心方法：
程序反复使用 Bellman 期望方程对所有非终止状态价值进行同步更新。
由于本题状态转移是确定性的，更新可写为：
v_{k+1}(s) = Σ_a π(a|s)[-1 + γ v_k(s')]
其中 s' 为在状态 s 执行动作 a 后到达的下一状态。

收敛判据：
程序持续迭代，直到所有状态在一次迭代中的价值变化都小于阈值 θ，
即认为价值函数已收敛并输出最终状态价值。
"""
import numpy as np
ACTIONS = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1),
}

TERMINAL_STATES = {(0, 0), (3, 3)}


def next_state(state, action, size=4):
    ni = state[0] + action[0]
    nj = state[1] + action[1]
    if 0 <= ni < size and 0 <= nj < size:
        return ni, nj
    return state


def iterative_policy_evaluation(theta=1e-4, gamma=1.0, size=4):
    v = np.zeros((size, size), dtype=float)
    iterations = 0

    while True:
        delta = 0.0
        v_old = v.copy()

        for i in range(size):
            for j in range(size):
                state = (i, j)
                if state in TERMINAL_STATES:
                    continue

                new_v = 0.0
                for action in ACTIONS.values():
                    s_next = next_state(state, action, size=size)
                    reward = -1.0
                    new_v += 0.25 * (reward + gamma * v_old[s_next[0], s_next[1]])

                v[i, j] = new_v
                delta = max(delta, abs(v[i, j] - v_old[i, j]))

        iterations += 1
        if delta < theta:
            break

    return v, iterations


def print_value_table(v):
    table = np.array2string(v, formatter={'float_kind': lambda x: f"{x:6.1f}"})
    print(table)


if __name__ == '__main__':
    values, iters = iterative_policy_evaluation(theta=1e-4, gamma=1.0, size=4)
    print(f"Converged in {iters} iterations")
    print("Final state-value table:")
    print_value_table(values)

                       

                        
                    
                    