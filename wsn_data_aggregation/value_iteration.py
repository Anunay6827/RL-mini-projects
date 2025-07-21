from Utils import prob_vector_generator, markov_matrix_generator
import numpy as np
from itertools import product

# Q(x,a)
def qfunc(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, V, phi_idx, z_idx, phi, z, b, c):    
    
    q = 0.0
    q1 = 0.0
    q2 = 0.0
    a = -1
    
    if c == 0:
        q = (
            (phi - z) ** 2 +
            beta * np.sum(
                (1 - gamma)  * P[phi_idx].reshape(-1, 1) * alpha * V[:, z_idx, np.minimum(b + np.arange(Delta + 1), B), 0] +
                gamma * P[phi_idx].reshape(-1, 1) * alpha * V[:, z_idx, np.minimum(b + np.arange(Delta + 1), B), 1]
            )
        )
    
    elif 0 < c < tau:
        q1 = (
            (phi - z) ** 2 +
            beta * np.sum(
                P[phi_idx].reshape(-1,1) * alpha * V[:, z_idx, np.minimum(b + np.arange(Delta + 1), B), c + 1]
            )
        )

        if b >= eta:
            q2 = (
                lmbda * ((phi - Swind) ** 2) + (1 - lmbda) * ((phi - z) ** 2) +
                beta * np.sum(
                    P[phi_idx].reshape(-1, 1) * alpha * (
                        lmbda * V.swapaxes(0, 1)[:, :, np.minimum(b + np.arange(Delta + 1) - eta, B), c + 1] +
                        (1 - lmbda) * V.swapaxes(0, 1)[:,z_idx, np.minimum(b + np.arange(Delta + 1) - eta, B), c + 1]
                        ),
                    axis=(1, 2)
                )
            )
            
            q2_min = np.min(q2)
            a = np.argmin(q2)
            q = min(q1, q2_min)
            if q == q1:
                a = -1
        else:
            q = q1
            a = -1
        
    elif c == tau:
        q1 = (
            (phi - z) ** 2 +
            beta * np.sum(P[phi_idx].reshape(-1, 1) * alpha * V[:, z_idx, np.minimum(b + np.arange(Delta + 1), B), 0])
        )
        if b >= eta:
            q2 = (
                lmbda * ((phi - Swind) ** 2) + (1 - lmbda) * ((phi - z) ** 2) +
                beta * np.sum(
                    P[phi_idx].reshape(-1, 1) * alpha * (
                        lmbda * V.swapaxes(0, 1)[:, :, np.minimum(b + np.arange(Delta + 1) - eta, B), 0] +
                        (1 - lmbda) * V.swapaxes(0, 1)[:, z_idx, np.minimum(b + np.arange(Delta + 1) - eta, B), 0]
                    ),
                    axis=(1, 2)
                )
            )
            q2_min = np.min(q2)
            a = np.argmin(q2)
            q = min(q1, q2_min)
            if q == q1:
                a = -1

        else:
            q = q1
            a = -1
            
    if a == -1:
        return q, a
    else:
        return q, Swind[a]

def value_iteration(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin):
    # This function must return the optimal value function and the optimal policy.
    n_states = len(Swind)
    V = np.zeros((n_states, n_states, B + 1, tau + 1))
    V_new = np.zeros_like(V)

    threshold_delta = float('inf')
    iteration = 1

    while threshold_delta > theta or iteration <= Kmin:
  
        for phi_idx, z_idx, b, c in product(range(n_states), range(n_states), range(B + 1), range(tau + 1)):
            phi, z = Swind[phi_idx], Swind[z_idx]
            q, action = qfunc(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, V, phi_idx, z_idx, phi, z, b, c)
            V_new[phi_idx, z_idx, b, c] = q
            
        threshold_delta = np.max(np.abs(V_new - V))
        V = np.copy(V_new)
        
        print(f"Iteration {iteration}, Max Delta: {threshold_delta}")
        iteration+=1
    
    policy = np.full((len(Swind),len(Swind),B+1,tau+1), -1)

    for phi_idx, z_idx, b, c in product(range(n_states), range(n_states), range(B + 1), range(tau + 1)):
        phi, z = Swind[phi_idx], Swind[z_idx]
        q, action = qfunc(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, V,phi_idx, z_idx, phi, z, b, c)
        policy[phi_idx, z_idx, b, c] = action
        
    return V, policy


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
alpha = prob_vector_generator(np.arange(Delta+1), mu_delta, stddev_delta)  # Probability distribution of solar power in one time slot.﻿
tau = 4       # Number of time slots in active phase.
gamma = 1/15  # Probability of getting chance to transmit. It can vary between 0.01 to 0.99.

beta = 0.95   # Discount factor.
theta = 0.01  # Convergence criteria: Maximum allowable change in value function to allow convergence.
Kmin = 10     # Convergence criteria: Minimum number of iterations to allow convergence.

# Call value iteration function.
V_optimal_vi, policy_optimal_vi = value_iteration(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin)
