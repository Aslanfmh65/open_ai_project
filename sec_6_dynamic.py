import gym
import lib
import numpy as np

GAMMA = 1.0
THETA = 1e-5


def policy_evaluation(policy, env):
    num_states = env.nS  # nS = number of states
    num_actions = env.nA  # nA = number of actions
    transitions = env.P  # transition

    #  initialize an array V(s) = 0, for all states in state set
    v = np.array(np.zeros(num_states))

    while True:
        delta = 0
        # for each s(state) in S(state set):
        for state in range(num_states):
            # v = V(s)
            new_value = 0
            # update rule is: V(s) = sum(pi(a|s) * sum(p(s, a) * [r + gamma * V(s')]))
            # sum over action(a)
            for action, p_action in enumerate(policy[state]):
                # sum over s', r
                for probability, next_state, reward, _ in transitions[state][action]:
                    new_value += p_action * probability * (reward + GAMMA * v[next_state])
            delta = max(delta, np.abs(new_value - v[state]))
            v[state] = new_value
        if delta < THETA:
            break
    return v


env = gym.make('GridWorld-v0').unwrapped
random_policy = np.ones([env.nS, env.nA]) * 0.25
v_k = policy_evaluation(random_policy, env)
print(v_k.reshape(env.shape))

