import numpy as np
import gymnasium as gym
from GymTraffic import GymTrafficEnv


def get_action(Q, epsilon, nu=1e-6):
    if np.random.rand() < epsilon:
        q_norm =  Q / (np.sum(Q) + nu)
        probs = np.exp(q_norm) / np.sum(np.exp(q_norm))
        return np.random.choice([0, 1], p=probs)
    else:
        return np.argmax(Q)
    
def q_val(q1, q2, l, c, V, max_queue, beta, arr_prob1, arr_prob2, dep_prob):
    exp_r = -(q1 + q2)
    nxt_r = np.zeros((2,))

    for a in range(2):
        for arrive1 in range(2):
            for arrive2 in range(2):
                for depart1 in range(2):
                    for depart2 in range(2):

                        prob = 1.0

                        nq1, nq2, nl, nc = q1, q2, l, c

                        if nq1 + 1 <= max_queue:
                            if arrive1:
                                prob *= arr_prob1
                                nq1 += 1
                            else:
                                prob *= (1 - arr_prob1)

                        if nq2 + 1 <= max_queue:
                            if arrive2:
                                prob *= arr_prob2
                                nq2 += 1
                            else:
                                prob *= (1 - arr_prob2)

                        if c == 0:
                            nc = 0 if a == 0 else 1
                        elif 0 < c < 10:
                            nc = c + 1 if a == 0 else 0
                        elif c == 10:
                            nc = 0
                            if a == 0:
                                nl = 1 if l == 0 else 0

                        if l == 0 and nq1 > 0:
                            prob *= dep_prob if depart1 else (1 - dep_prob)
                            if depart1:
                                nq1 -= 1

                        elif l == 1 and nq2 > 0:
                            prob *= dep_prob if depart2 else (1 - dep_prob)
                            if depart2:
                                nq2 -= 1
                        nxt_r[a] += prob * V[nq1, nq2, nl, nc]
                        
    return exp_r + beta * nxt_r

def SARSA(env, beta, Nepisodes, alpha):
    max_queue = 20
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    Q = np.zeros((max_queue+1, max_queue+1, 2, 11, 2))

    for ep in range(Nepisodes):
        state, _ = env.reset()
        q1, q2, l, c = state
        q1, q2 = min(q1, max_queue), min(q2, max_queue)
        action = get_action(Q[q1, q2, l, c], epsilon)

        truncated = False
        while not truncated:
            next_state, reward, terminated, truncated, _ = env.step(action)
            nq1, nq2, nl, nc = next_state
            nq1, nq2 = min(nq1, max_queue), min(nq2, max_queue)
            next_action = get_action(Q[nq1, nq2, nl, nc], epsilon)

            Q[q1, q2, l, c, action] += alpha * (reward + beta * Q[nq1, nq2, nl, nc, next_action] - Q[q1, q2, l, c, action])

            q1, q2, l, c, action = nq1, nq2, nl, nc, next_action

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    policy = np.argmax(Q, axis=-1)
    return policy


def ExpectedSARSA(env, beta, Nepisodes, alpha):
    max_queue = 20
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    nu = 1e-6
    Q = np.zeros((max_queue+1, max_queue+1, 2, 11, 2))

    for ep in range(Nepisodes):
        state, _ = env.reset()
        truncated = False
        while not truncated:
            q1, q2, l, c = state
            q1, q2 = min(q1, max_queue), min(q2, max_queue)
            action = get_action(Q[q1, q2, l, c], epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)

            nq1, nq2, nl, nc = next_state
            nq1, nq2 = min(nq1, max_queue), min(nq2, max_queue)

            q_values = Q[nq1, nq2, nl, nc]
            q_norm =  q_values / (np.sum(q_values) + nu)
            probs = np.exp(q_norm) / np.sum(np.exp(q_norm))

            final_probs = np.zeros((2,))

            for a in range(2):
                if a == np.argmax(q_values):
                    final_probs[a] = (1 - epsilon) + epsilon * probs[a]
                else:
                    final_probs[a] = epsilon * probs[a]

            expected_q = np.dot(final_probs, q_values)

            Q[q1, q2, l, c, action] += alpha * (reward + beta * expected_q - Q[q1, q2, l, c, action])

            state = next_state

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    policy = np.argmax(Q, axis=-1)
    return policy


def ValueFunctionSARSA(env, beta, Nepisodes, alpha):
    max_queue = 20
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    nu = 1e-6
    V = np.zeros((max_queue + 1, max_queue + 1, 2, 11))

    for ep in range(Nepisodes):
        state, _ = env.reset()
        truncated = False

        while not truncated:
            q1, q2, l, c = state
            q1, q2 = min(q1, max_queue), min(q2, max_queue)

            if c == 0:
                dep_prob = env.departure_max_prob
            else:
                dep_prob = env._red_departure_prob(c - 1)

            q_values = q_val(q1, q2, l, c, V, max_queue, beta, env.arrival_prob[0], env.arrival_prob[1], dep_prob)

            action = get_action(q_values, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)

            nq1, nq2, nl, nc = next_state
            nq1, nq2 = min(nq1, max_queue), min(nq2, max_queue)

            V[q1, q2, l, c] += alpha * (reward + beta * V[nq1, nq2, nl, nc] - V[q1, q2, l, c])

            state = next_state

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    policy = np.zeros((max_queue+1, max_queue+1, 2, 11), dtype=int)
    for q1 in range(max_queue+1):
        for q2 in range(max_queue+1):
            for l in range(2):
                for c in range(11):
                    if c == 0:
                        dep_prob = env.departure_max_prob
                    else:
                        dep_prob = env._red_departure_prob(c - 1)

                    q_values = q_val(q1, q2, l, c, V, max_queue, beta, env.arrival_prob[0], env.arrival_prob[1], dep_prob)
                    
                    policy[q1, q2, l, c] = np.argmax(q_values)                  

    return policy


# Main runner
env = GymTrafficEnv() # Create and instance of the traffic controller environment.

Nepisodes = 2000   # Number of episodes to train
alpha = 0.1         # Learning rate
beta = 0.997        # Discount factor

# Learn the optimal policies using two different TD learning approaches
policy1 = SARSA(env, beta, Nepisodes, alpha)
policy2 = ExpectedSARSA(env, beta, Nepisodes, alpha)
policy3 = ValueFunctionSARSA(env, beta, Nepisodes, alpha)

# Save the policies
np.save("policy1.npy", policy1)
np.save("policy2.npy", policy2)
np.save("policy3.npy", policy3)

env.close() # Close the environment