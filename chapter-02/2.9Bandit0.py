import random
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self,k=4, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon 
        self.Q = [0.0 for _ in range(k)]
        self.N = [0 for _ in range(k)]

    def act(self):
        if random.random() < self.epsilon:
            action = random.randrange(self.k)
        else:
            action = self.Q.index(max(self.Q))
        return action 
    
    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

k=4
true_means = [random.randrange(2) for _ in range(k)]
true_sigma = 1
def get_reward(action):
    mu = true_means[action]
    reward = random.gauss(mu, true_sigma)
    return reward

bandit = Bandit(k=4, epsilon=0.1)

rewards = []
avg_rewards = []
total = 0.0
steps = 1000

for t in range(steps):
    a = bandit.act()
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