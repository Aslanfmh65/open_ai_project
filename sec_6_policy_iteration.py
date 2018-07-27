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


def policy_iteration(policy, env):
    num_states = env.nS
    num_actions = env.nA
    transitions = env.P

    policy_stable = True

    while policy_stable:
        # step 2: policy evaluation
        v = policy_evaluation(policy, env)
        # for each s in S:
        for state in range(num_states):
            # old_action = pi_s
            old_action = np.argmax(policy[state])
            # update rule is: pi_s argmax_a(sum(s', r|s, a) * [r + gamma * V(s")])
            new_action_values = np.zeros(num_actions)
            for action in range(num_actions):
                for probability, next_state, reward, _ in transitions[state][action]:
                    new_action_values[action] += probability * (reward + GAMMA * v[next_state])
            new_action = np.argmax(new_action_values)
            policy_stable = (new_action == old_action)
            policy[state] = np.eye(num_actions)[new_action]

        # if the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, v


env = gym.make('GridWorld-v0').unwrapped
random_policy = np.ones([env.nS, env.nA]) * 0.25
v_k = policy_evaluation(random_policy, env)
print(v_k.reshape(env.shape))

optimal_policy, optimal_value = policy_iteration(random_policy, env)
print(optimal_policy)





env = gym.make('GridWorld-v0').unwrapped
random_policy = np.ones([env.nS, env.nA]) * 0.25
v_k = policy_evaluation(random_policy, env)
print(v_k.reshape(env.shape))
