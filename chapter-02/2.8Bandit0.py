import random
import matplotlib.pyplot as plt
import math


class Bandit:
    def __init__(self, k=4, alpha=0.1, baseline=0.0):
        self.k = k
        self.alpha = alpha
        self.R_baseline = baseline
        self.H_t = [0.0 for _ in range(k)]
        self.softmax = [1.0 / k for _ in range(k)]

    def H_update(self, action, R_t):
        self.H_t[action] += self.alpha * (R_t - self.R_baseline) * (1 - self.softmax[action])
        for i in range(self.k):
            if i != action:
                self.H_t[i] -= self.alpha * (R_t - self.R_baseline) * self.softmax[i]

    def Pi_update(self):
        # Numerically stable softmax.
        max_h = max(self.H_t)
        exp_vals = [math.exp(h - max_h) for h in self.H_t]
        exp_sum = sum(exp_vals)
        self.softmax = [x / exp_sum for x in exp_vals]

    def baseline_update(self, R_t, t):
        self.R_baseline += (R_t - self.R_baseline) / (t + 1)
        return self.R_baseline

    def sample_from_probs(self, probs):
        r = random.random()  # [0.0, 1.0)
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if r < c:
                return i
        return len(probs) - 1  # 处理浮点误差

    def action(self):
        action = self.sample_from_probs(self.softmax)
        return action


k=4
true_means = [random.randrange(2) for _ in range(k)]
true_sigma = 1


def get_reward(action):
    mu = true_means[action]
    reward = random.gauss(mu, true_sigma)
    return reward

bandit = Bandit(k=4, alpha=0.1)

rewards = []
avg_rewards = []
total = 0.0
steps = 1000

for t in range(steps):
    a = bandit.action()
    r = get_reward(a)

    bandit.H_update(a, r)
    bandit.baseline_update(r, t)
    bandit.Pi_update()

    rewards.append(r)
    total += r
    avg_rewards.append(total / (t + 1))

plt.plot(avg_rewards)
plt.xlabel("Step")
plt.ylabel("Average Reward")
plt.title("softmax")
plt.show()