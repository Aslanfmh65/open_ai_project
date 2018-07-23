import gym
import time

env = gym.make('CartPole-v0')
env.reset()

for step in range(1000):
    env.render()  # rendering the environment at each step
    env.step(env.action_space.sample()) # feed the env with random actions that exist in all possible actions
    time.sleep(0.1)
