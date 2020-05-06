from agent import Agent
from monitor import interact
import gym
import numpy as np
from bayes_opt import BayesianOptimization

env = gym.make('Taxi-v2')

def interact_wrapper(decay_rate, alpha, gamma):
    agent = Agent(decay_rate, alpha, gamma)
    avg_rewards, best_avg_reward = interact(env, agent, 15000)
    return best_avg_reward

pbounds = {'decay_rate': (0.9999, 0.9990), 'alpha': (0.01, 0.5), 'gamma': (0.5, 1.0)}

optimizer = BayesianOptimization(
    f=interact_wrapper,
    pbounds=pbounds,
    random_state=47
)

optimizer.probe(
    params={'decay_rate': 0.9999, 'alpha': 0.1, 'gamma': 0.9},
    lazy=True,
)

optimizer.maximize(
    init_points=3,
    n_iter=25
)
