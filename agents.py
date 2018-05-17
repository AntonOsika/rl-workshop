import random
from collections import defaultdict
import numpy as np

class Random:
    def __init__(self, actions):
        self.actions = actions

    def act(self, state):
        return random.randint(0, self.actions - 1)

    def feedback(self, reward):
        pass

    def new_episode(self):
        pass


class MC:
    def __init__(self, actions, eps=0.0):
        self.actions = actions
        self.reward = defaultdict(lambda: np.zeros(actions))
        self.count = defaultdict(lambda: 1e-9*np.ones(actions))

        self.state_action_buffer = []
        self.eps = eps
        self.learn = True

    def act(self, state):
        if random.random() < self.eps:
            a = random.randint(0, self.actions-1)
        else:
            a = np.argmax(self.reward[state]/self.count[state])
        self.state_action_buffer.append([state, a])
        return a

    def feedback(self, r):
        if self.learn:
            for state, a in self.state_action_buffer:
                self.reward[state][a] += r
                self.count[state][a] += 1

    def get_coeff(self):
        res = {}
        for k in self.reward:
            res[k] = self.reward[k]/self.count[k]
        return sorted(res.items())
    
    def new_episode(self):
        self.state_action_buffer = []


class SARSA:
    def __init__(self, actions):
        self.actions = actions
        self.Q = defaultdict(lambda: np.zeros(actions))

        self.state_action_buffer = []
        self.eps = 0.0
        self.alpha = 0.01
        self.gamma = 1.0

    def act(self, state):
        if random.random() < self.eps:
            a = random.randint(0, self.actions-1)
        else:
            a = np.argmax(self.Q[state])
        self.state_action_buffer.append([state, a])
        return a

    def feedback(self, r):

        for i in range(len(self.state_action_buffer)-1):
            s, a = self.state_action_buffer[i]
            sp, ap = self.state_action_buffer[i+1]
            delta = r + self.gamma*self.Q[sp][ap] - self.Q[s][a]
            self.Q[s][a] += self.alpha*delta

        # Final step last:
        s, a = self.state_action_buffer[-1]
        delta = r - self.Q[s][a]
        self.Q[s][a] += self.alpha*delta

    def new_episode(self):
        self.state_action_buffer = []


class SARSAReversed(SARSA):
    def feedback(self, r):

        # Final step first:
        s, a = self.state_action_buffer[-1]
        delta = r - self.Q[s][a]
        self.Q[s][a] += self.alpha*delta

        for i in range(len(self.state_action_buffer)-1)[::-1]:
            s, a = self.state_action_buffer[i]
            sp, ap = self.state_action_buffer[i+1]
            delta = r + self.gamma*self.Q[sp][ap] - self.Q[s][a]
            self.Q[s][a] += self.alpha*delta

    def new_episode(self):
        self.state_action_buffer = []


class Q:
    def __init__(self, actions):
        self.actions = actions
        self.Q = defaultdict(lambda: np.zeros(actions))

        self.state_action_buffer = []
        self.eps = 0.0
        self.alpha = 0.01
        self.gamma = 1.0

    def act(self, state):
        if random.random() < self.eps:
            a = random.randint(0, self.actions-1)
        else:
            a = np.argmax(self.Q[state])
        self.state_action_buffer.append([state, a])
        return a

    def feedback(self, r):

        for i in range(len(self.state_action_buffer)-1):
            s, a = self.state_action_buffer[i]
            sp, ap = self.state_action_buffer[i+1]
            delta = r + self.gamma*max(self.Q[sp]) - self.Q[s][a]
            self.Q[s][a] += self.alpha*delta

        # Final step last:
        s, a = self.state_action_buffer[-1]
        delta = r - self.Q[s][a]
        self.Q[s][a] += self.alpha*delta

    def new_episode(self):
        self.state_action_buffer = []


class QReversed:
    def __init__(self, actions):
        self.actions = actions
        self.Q = defaultdict(lambda: np.zeros(actions))

        self.state_action_buffer = []
        self.eps = 0.0
        self.alpha = 0.01
        self.gamma = 1.0

    def act(self, state):
        if random.random() < self.eps:
            a = random.randint(0, self.actions-1)
        else:
            a = np.argmax(self.Q[state])
        self.state_action_buffer.append([state, a])
        return a

    def feedback(self, r):

        # Final step first:
        s, a = self.state_action_buffer[-1]
        delta = r - self.Q[s][a]
        self.Q[s][a] += self.alpha*delta

        for i in range(len(self.state_action_buffer)-1):
            s, a = self.state_action_buffer[i]
            sp, ap = self.state_action_buffer[i+1]
            delta = r + self.gamma*max(self.Q[sp]) - self.Q[s][a]
            self.Q[s][a] += self.alpha*delta

    def new_episode(self):
        self.state_action_buffer = []


class Human:
    def act(self, state):
        print("State: ", state)
        return int(input("")) - 1

    def feedback(self, r):
        pass

    def new_episode(self):
        pass


class Perfect:
    def act(self, state):
        if (state - 1) % 4 == 0:
            return 0
        else:
            action = state - (4*((state-1)//4) + 1)
            assert 0 < action <= 3
            return action - 1

    def feedback(self, r):
        pass

    def new_episode(self):
        pass
