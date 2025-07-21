from Utils import prob_vector_generator, markov_matrix_generator
import numpy as np

from itertools import product

def policy_evaluation(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin, policy, V, n_states):
    max_delta = float('inf')
    iteration_ipe = 0
    del_dims = np.arange(Delta + 1)
    while max_delta > theta or iteration_ipe <= Kmin:
        iteration_ipe += 1
        V_new = np.copy(V)

        for phi_idx, z_idx, b, c in product(range(n_states), range(n_states), range(B + 1), range(tau + 1)):
            phi, z = Swind[phi_idx], Swind[z_idx]
            action = policy[phi_idx, z_idx, b, c]
            if c == 0:
                V_new[phi_idx, z_idx, b, c] = (
                    (phi - z) ** 2 +
                    beta * np.sum(
                        (1 - gamma)  * P[phi_idx].reshape(-1, 1) * alpha * V[:, z_idx, np.minimum(b + del_dims, B), 0] +
                        gamma * P[phi_idx].reshape(-1, 1) * alpha * V[:, z_idx, np.minimum(b + del_dims, B), 1]
                    )
                )

            elif 0 < c < tau and action == -1:
                V_new[phi_idx, z_idx, b, c] = (
                    (phi - z)**2 +
                    beta * np.sum(
                        P[phi_idx].reshape(-1,1) * alpha * V[:, z_idx, np.minimum(b + del_dims, B), c+1]
                    )
                )

            elif 0 < c < tau and action != -1:
                a_idx = np.where(Swind == action)[0][0]
                V_new[phi_idx, z_idx, b, c] = (
                    lmbda * ((phi - action) ** 2) + (1 - lmbda) * ((phi - z) ** 2) +
                    beta * np.sum(
                        P[phi_idx].reshape(-1, 1) * alpha * (
                            lmbda * V[:, a_idx, np.minimum(b + del_dims - eta, B), c + 1] +
                            (1 - lmbda) * V[:, z_idx, np.minimum(b + del_dims - eta, B), c + 1]
                        )
                    )
                )

            elif c == tau and action == -1:
                V_new[phi_idx, z_idx, b, c] = (
                    (phi - z)**2 +
                    beta * np.sum(
                        P[phi_idx].reshape(-1,1) * alpha * V[:, z_idx, np.minimum(b + del_dims, B), 0]
                    )
                )

            elif c == tau and action != -1:
                a_idx = np.where(Swind == action)[0][0]
                V_new[phi_idx, z_idx, b, c] = (
                    lmbda * ((phi - action) ** 2) + (1 - lmbda) * ((phi - z) ** 2) +
                    beta * np.sum(
                        P[phi_idx].reshape(-1, 1) * alpha * (
                            lmbda * V[:, a_idx, np.minimum(b + del_dims - eta, B), 0] +
                            (1 - lmbda) * V[:, z_idx, np.minimum(b + del_dims - eta, B), 0]
                        )
                    )
                )


        max_delta = np.max(np.abs(V_new - V))
        # print(f"Iteration of PI: {iteration_pi}, Iteration of IPE: {iteration_ipe}, Max Delta: {max_delta}")
        V = np.copy(V_new)
        # print(f"Value Function: {np.min(V)}")

    return V, max_delta

def policy_iteration(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin):
    n_states = len(Swind)
    V = np.zeros((n_states, n_states, B+1, tau+1), dtype = np.float64)
    V_pi = np.zeros_like(V)
    policy = np.full((n_states, n_states, B+1, tau+1), -1, dtype = np.float64)

    max_delta = 0
    iteration_pi = 0
    del_dims = np.arange(Delta + 1)
    converged = False

    while not converged:
        iteration_pi += 1
        V, max_delta = policy_evaluation(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin, policy, V, n_states)
        # print(policy)
        V_pi = np.copy(V)
        policy_new = np.full((n_states, n_states, B + 1, tau + 1), -1, dtype = np.float64)
        for phi_idx, z_idx, b, c in product(range(n_states), range(n_states), range(B + 1), range(tau + 1)):
            phi, z = Swind[phi_idx], Swind[z_idx]
            q1 = 0
            q2 = 0
            q2_min = 0
            q_min = 0
            a = -1

            if 0 < c < tau:
                q1 = (
                    (phi - z) ** 2 +
                    beta * np.sum(
                        P[phi_idx].reshape(-1,1) * alpha * V_pi[:, z_idx, np.minimum(b + del_dims, B), c + 1]
                    )
                )
                if b >= eta:
                    q2 = (
                        lmbda * ((phi - Swind) ** 2) + (1 - lmbda) * ((phi - z) ** 2) +
                        beta * np.sum(
                            P[phi_idx].reshape(-1, 1) * alpha * (
                                lmbda * V_pi.swapaxes(0,1)[:, :, np.minimum(b + del_dims - eta, B), c + 1] +
                                (1 - lmbda) * V_pi.swapaxes(0, 1)[:,z_idx, np.minimum(b + del_dims - eta, B), c + 1]
                                ),
                            axis=(1, 2)
                        )
                    )

                    q2_min = np.min(q2)
                    a = np.argmin(q2)
                    q_min = min(q1, q2_min)

                    if q_min == q1:
                        a = -1
                else:
                    q_min = q1
                    a = -1

            elif c == tau:
                q1 = (
                    (phi - z) ** 2 +
                    beta * np.sum(P[phi_idx].reshape(-1, 1) * alpha * V_pi[:, z_idx, np.minimum(b + del_dims, B), 0])
                )
                if b >= eta:
                    q2 = (
                        lmbda * ((phi - Swind) ** 2) + (1 - lmbda) * ((phi - z) ** 2) +
                        beta * np.sum(
                            P[phi_idx].reshape(-1, 1) * alpha * (
                                lmbda * V_pi.swapaxes(0, 1)[:, :, np.minimum(b + del_dims - eta, B), 0] +
                                (1 - lmbda) * V_pi.swapaxes(0, 1)[:, z_idx, np.minimum(b + del_dims - eta, B), 0]
                            ),
                            axis=(1, 2)
                        )
                    )
                    q2_min = np.min(q2)
                    a = np.argmin(q2)
                    q_min = min(q1, q2_min)
                    if q_min == q1:
                        a = -1
                else:
                    q_min = q1
                    a = -1
            if q_min < V_pi[phi_idx, z_idx, b, c]:
                policy_new[phi_idx, z_idx, b, c] = Swind[a] if a != -1 else -1
            else:
                policy_new[phi_idx, z_idx, b, c] = np.copy(policy[phi_idx, z_idx, b, c])

        converged = True
        for phi_idx, z_idx, b, c in product(range(n_states), range(n_states), range(B + 1), range(tau + 1)):
            if policy_new[phi_idx, z_idx, b, c] != policy[phi_idx, z_idx, b, c]:
                converged = False
        policy = np.copy(policy_new)
        V = np.copy(V_pi)
        print(f"Iteration of PI: {iteration_pi}, Max Delta: {max_delta}, Value: {np.min(V)}")

    return V, policy


Swind = np.linspace(0, 1, 21)
mu_wind = 0.3
z_wind = 0.5

stddev_wind = z_wind * np.sqrt(mu_wind * (1 - mu_wind))
retention_prob = 0.9

P = markov_matrix_generator(Swind, mu_wind, stddev_wind, retention_prob)

lmbda = 0.7
B = 10
eta = 2
Delta = 3
mu_delta = 2
z_delta = 0.5
stddev_delta = z_delta * np.sqrt(Delta * (Delta - mu_delta))
alpha = prob_vector_generator(np.arange(Delta + 1), mu_delta, stddev_delta)

tau = 4
gamma = 1/15

beta = 0.95
theta = 0.01
Kmin = 10

V_optimal, policy_optimal = policy_iteration(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin)