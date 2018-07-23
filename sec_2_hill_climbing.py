import tensorflow as tf
import numpy as np
import gym
import logging

logger = logging.getLogger('r1')
logger.setLevel(logging.DEBUG)


class Harness:

    def run_episode(self, env, agent):
        observation = env.reset()
        total_reward = 0

        for _ in range(1000):
            action = agent.next_action(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                break
        return total_reward


class LinearAgent:

    def __init__(self):
        self.parameters = np.random.rand(4) * 2 - 1  # given random 4 parameters corresponding to 4 number of each state

    def next_action(self, observation):
        return 0 if np.matmul(self.parameters, observation) < 0 else 1
        # multiply the parameters with observations, if the result is less than 0 we go left(0), else go right(1)


def random_search():
    env = gym.make('CartPole-v0')
    best_params = None
    best_reward = 0
    agent = LinearAgent()
    harness = Harness()

    for step in range(1000):
        agent.parameters = np.random.rand(4) * 2 - 1
        reward = harness.run_episode(env, agent)

        if reward > best_reward:
            best_params = agent.parameters

            if best_reward == 200:
                print(best_params)
                break

        if step % 100 == 0:
            print(reward)

    print(best_params)


def hill_climbing():

    env = gym.make('CartPole-v0')
    noise_scaling = 0.1
    best_reward = 0
    best_params = None
    agent = LinearAgent()
    harness = Harness()

    for step in range(10000):
        old_params = agent.parameters
        agent.parameters += noise_scaling * (np.random.rand(4) * 2 - 1)
        reward = harness.run_episode(env, agent)
        if reward > best_reward:
            best_reward = reward
            best_params = agent.parameters
        else:
            agent.parameters = old_params

        if best_reward == 200:
            break
        if step % 200 == 0:
            print(best_reward)
            print(best_params)


hill_climbing()
