import random
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k = 4, epsilon = 0.1):
        self.k = k
        self.epsilon = epsilon
        self.Q = [0.0 for _ in range(self.k)]
        self.N = [0 for _ in range(self.k)]
    
    def act(self):
        if random.random <= self.epsilon:
            action = random.randrange(self.k)
        else:
            action = self.Q.index(max(self.Q))
        return action
    
    def update(self, rewards, action):
        self.N[action] += 1
        self.Q[action] += (rewards - self.Q[action]) / self.N[action]

true_means = [random.randrange(2), random.randrange(2), random.randrange(2), random.randrange(2)]
true_sigma = 1

def get_rewards(action):
    return random.gauss(true_means, true_sigma)

bandit = Bandit(k = 4, epsilon = 0.1)

for i in range (1000):
    action = bandit.act()
    rewards = get_rewards(action)
    bandit.update(a, r)
