import gym
import lib
import numpy as np

GAMMA = 1.0
THETA = 1e-5


def value_iteration(env):
    num_states = env.nS
    num_actions = env.nA
    transitions = env.P

    # initialize array V(value) arbitrarily
    v = np.zeros(num_states)
    # repeat
    while True:
        # initialize delta to zero
        delta = 0
        # for each s in S
        for state in range(num_states):
            # v = V(s)
            old_value = v[state]
            new_action_values = np.zeros(num_actions)
            # update rule is: v(s) = max_a(sum(p(s',r|s, a) * [r + gamma * v(s')]))
            for action in range(num_actions):
                for probability, next_state, reward, _ in transitions[state][action]:
                    new_action_values[action] += probability * (reward + GAMMA * v[next_state])
            v[state] = np.max(new_action_values)
            # calculate delta = max(delta, |v - V(s)|)
            delta = max(delta, np.abs(v[state] - old_value))
        # until delta is smaller than theta
        if delta < THETA:
            break

    # output a deterministic policy (the optimal one)
    optimal_policy = np.zeros([num_states, num_actions])
    for state in range(num_states):
        action_values = np.zeros(num_actions)
        for action in range(num_actions):
            for probability, next_state, reward, _ in transitions[state][action]:
                action_values[action] += probability * (reward + GAMMA * v[next_state])
        optimal_policy[state] = np.eye(num_actions)[np.argmax(action_values)]

    return optimal_policy, v


env = gym.make('GridWorld-v0').unwrapped
optimal_policy, optimal_value = value_iteration(env)

print('Policy Probability Distribution:')
print(optimal_policy)

print('Value Function:')
print(optimal_value.reshape(env.shape))
