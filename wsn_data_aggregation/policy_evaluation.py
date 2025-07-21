import numpy as np
from Utils import prob_vector_generator, markov_matrix_generator
from itertools import product

def greedy_policy(Swind, lmbda, eta, tau, phi, z, b, c):
    action = -1
    action_transmit = 0
    if 0 < c <= tau and b >= eta:
        action_space_transmit = np.copy(Swind)
        action_transmit = np.argmin(lmbda * (phi - action_space_transmit) ** 2 + (1 - lmbda) * (phi - z) ** 2)
        action_idx = action_transmit if  (
            lmbda * ((phi - action_space_transmit[action_transmit]) ** 2) +
            (1 - lmbda) * ((phi - z) ** 2)
        ) < ((phi - z) ** 2) else -1
        if action_idx != -1:
            action = action_space_transmit[action_idx].item()
    return action

def policy_evaluation(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin):
    # This function must return the value function of the greedy policy.
    n_states = len(Swind)
    Swind = np.round(Swind, 2)
    V = np.zeros((n_states, n_states, B + 1, tau + 1))
    V_new = np.zeros_like(V)
    difference = float('inf')
    i = 1
    del_dims = np.arange(Delta + 1)
    policy = np.zeros((n_states, n_states, B + 1, tau + 1))
    for phi_idx, z_idx, b, c in product(range(n_states), range(n_states),range(B + 1), range(tau + 1)):
            phi, z = Swind[phi_idx], Swind[z_idx]
            policy[phi_idx, z_idx, b, c] = greedy_policy(Swind,lmbda, eta, tau, phi, z, b, c)

    while difference > theta or i <= Kmin:
        for phi_idx, z_idx, b, c in product(range(n_states), range(n_states),range(B + 1), range(tau + 1)):
            phi, z = Swind[phi_idx], Swind[z_idx]
            x = (phi_idx, z_idx, b, c)
            action = policy[phi_idx, z_idx, b, c]
            if action != -1:
                action_idx = np.where(Swind == action)[0][0]

            if c == 0:
                V_new[x] = (
                            (phi - z) ** 2 + 
                            beta * np.sum(
                                        (1 - gamma)  * P[x[0]].reshape(-1, 1) * alpha * V[:, x[1], np.minimum(b + del_dims, B), 0] + 
                                        gamma * P[x[0]].reshape(-1, 1) * alpha * V[:, x[1], np.minimum(b + del_dims, B), 1]
                                )
                    )
            elif 0 < c < tau and action == -1:
                V_new[x] = (
                    (phi - z) ** 2 + 
                    beta * np.sum(
                        P[x[0]].reshape(-1,1) * alpha * V[:, x[1], np.minimum(b + del_dims, B), c + 1]
                        )
                )
            elif 0 < c < tau and action != -1: 
                V_new[x] = (
                    lmbda * (
                        (phi - action) ** 2) + (1 - lmbda) * ((phi - z) ** 2) + 
                    beta * np.sum(
                        P[x[0]].reshape(-1, 1) * alpha * (
                            lmbda * V[:, action_idx, np.minimum(b + del_dims - eta, B), c + 1] + 
                            (1 - lmbda) * V[:, x[1], np.minimum(b + del_dims - eta, B), c + 1]
                            )
                        )
                )
            elif c == tau and action == -1:
                V_new[x] = (
                    (phi - z) ** 2 + 
                    beta * np.sum(
                        P[x[0]].reshape(-1, 1) * alpha * V[:, x[1], np.minimum(b + del_dims, B), 0]
                        )
                )
            elif c == tau and action != -1:
                V_new[x] = (
                    lmbda * ((phi - action) ** 2) + (1 - lmbda) * ((phi - z) ** 2) + 
                    beta * np.sum(
                        P[x[0]].reshape(-1, 1) * alpha * (
                            lmbda * V[:, action_idx, np.minimum(b + del_dims - eta, B), 0] + 
                            (1 - lmbda) * V[:, x[1], np.minimum(b + del_dims - eta, B), 0]
                            )
                        )
                )

        difference = np.max(np.abs(V_new - V))
        V = np.copy(V_new)

        print(f"Iteration {i}, Max Delta: {difference}")
        i += 1
    return V

    pass




# System parameters (set to default values)
Swind = np.linspace(0, 1, 21)                      # The set of all possible normalized wind speed.
mu_wind = 0.3                                      # Mean wind speed. You can vary this between 0.2 to 0.8.
z_wind = 0.5                                       # Z-factor of the wind speed. You can vary this between 0.25 to 0.75.
                                                   # Z-factor = Standard deviation divided by mean.
                                                   # Higher the Z-factor, the more is the fluctuation in wind speed.
stddev_wind = z_wind*np.sqrt(mu_wind*(1-mu_wind))  # Standard deviation of the wind speed.
retention_prob = 0.9                               # Retention probability is the probability that the wind speed in the current and the next time slot is the same.
                                                   # You can vary the retention probability between 0.05 to 0.95.
                                                   # Higher retention probability implies lower fluctuation in wind speed.
P = markov_matrix_generator(Swind, mu_wind, stddev_wind, retention_prob)  # Markovian probability matrix governing wind speed.


lmbda = 0.7  # Probability of successful transmission.


B = 10         # Maximum battery capacity.
eta = 2        # Battery power required for one transmission.
Delta = 3      # Maximum solar power in one time slot.
mu_delta = 2   # Mean of the solar power in one time slot.
z_delta = 0.5  # Z-factor of the slower power in one time slot. You can vary this between 0.25 to 0.75.                  
stddev_delta = z_delta*np.sqrt(Delta*(Delta-mu_delta))  # Standard deviation of the solar power in one time slot.
alpha = prob_vector_generator(np.arange(Delta+1), mu_delta, stddev_delta)  # Probability distribution of solar power in one time slot.


tau = 4       # Number of time slots in active phase.
gamma = 1/15  # Probability of getting chance to transmit. It can vary between 0.01 to 0.99.


beta = 0.95   # Discount factor.
theta = 0.01  # Convergence criteria: Maximum allowable change in value function to allow convergence.
Kmin = 10     # Convergence criteria: Minimum number of iterations to allow convergence.




# Call policy evaluation function.
V = policy_evaluation(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin)

