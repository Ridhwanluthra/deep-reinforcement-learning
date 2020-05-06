import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, decay_rate=0.9999, alpha=0.05, gamma=0.9, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.e = 1.0
        self.alpha = alpha
        self.gamma = gamma
        self.decay_rate = decay_rate
    
    def epsilon_greedy_probs(self, Q_s, i_episode, eps=None):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        epsilon = 1.0 / i_episode
        if eps is not None:
            epsilon = eps
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s
    
    def e_greedy(self, state):
        if np.random.random() >= self.e:
            return np.argmax(self.Q[state])
        else:
            return np.random.randint(self.nA)

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
#         policy_s = self.epsilon_greedy_probs(self.Q[state], 1, 0.05)
#         return np.random.choice(np.arange(self.nA), p=policy_s)
        return self.e_greedy(state)
    
    def expected_sarsa(self, state, action, reward, next_state, done):
        if done:
            self.Q[state][action] += (reward + (self.gamma * 0) - self.Q[state][action]) * self.alpha
            return

        policy_s = self.epsilon_greedy_probs(self.Q[next_state], 1, 0.05)
        er = np.dot(self.Q[next_state], policy_s)
        self.Q[state][action] += (reward + (self.gamma * er) - self.Q[state][action]) * self.alpha
    
    def q_learning(self, state, action, reward, next_state, done):
        if done:
            self.Q[state][action] += (reward + (self.gamma * 0) - self.Q[state][action]) * self.alpha
            return

        self.Q[state][action] += (reward + (self.gamma * np.max(self.Q[next_state])) - self.Q[state][action]) * self.alpha

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
#         self.Q[state][action] += 1
#         self.expected_sarsa(state, action, reward, next_state, done)
        self.q_learning(state, action, reward, next_state, done)
        self.e *= self.decay_rate
