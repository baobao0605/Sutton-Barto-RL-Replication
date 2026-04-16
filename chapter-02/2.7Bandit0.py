import random
import matplotlib.pyplot as plt
from math import sqrt, log, inf

class Bandit:
    def __init__(self, k=4, c=2):
        self.k = k
        self.c = c
        self.Q = [0.0 for _ in range(k)]
        self.N = [0 for _ in range(k)]
        self.UCB = [0.0 for _ in range(k)]

    def act(self, t):
        action = self.UCB.index(max(self.UCB))
        return action
    
    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
    
    def UCB_calcu(self, t):
        for j in range (self.k):
            if self.N[j] ==0:
                self.UCB[j] = inf
            else:
                self.UCB[j] = self.Q[j] + self.c * sqrt(log(t + 1)) / sqrt(self.N[j])
        

true_means = [random.randrange(2), random.randrange(2), random.randrange(2), random.randrange(2)]
true_sigma = 1
def get_reward(action):
    mu = true_means[action]
    reward = random.gauss(mu, true_sigma)
    return reward


bandit = Bandit(k=4)

rewards = []
avg_rewards = []
total = 0.0
steps = 1000

for t in range(steps):
    bandit.UCB_calcu(t)
    a = bandit.act(t)
    r = get_reward(a)
    bandit.update(a, r)


    rewards.append(r)
    total += r
    avg_rewards.append(total / (t + 1))

plt.plot(avg_rewards)
plt.xlabel("Step")
plt.ylabel("Average Reward")
plt.title("Epsilon-Greedy Bandit (epsilon=0.1)")
plt.show()