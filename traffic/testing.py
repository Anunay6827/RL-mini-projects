import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from GymTraffic import GymTrafficEnv

def TestPolicy(env, policy):
    state, _ = env.reset()
    q1_list = np.zeros((env.max_time + 1))
    q2_list = np.zeros((env.max_time + 1))
    actions_taken = np.zeros((env.max_time), dtype = int)
    avg_queue_sum = 0
    steps = 0

    while True:
        q1, q2, l, c = state
        q1, q2 = min(q1, 20), min(q2, 20)

        action = policy[q1, q2, l, c]
        actions_taken[steps] = action

        state, reward, terminated, truncated, _ = env.step(action)

        q1_list[steps] = q1
        q2_list[steps] = q2
        steps += 1
        avg_queue_sum += ((q1 + q2) - avg_queue_sum) / steps

        if truncated:
            break
    
    q1, q2, l, c = state
    q1, q2 = min(q1, 20), min(q2, 20)
    q1_list[steps] = q1
    q2_list[steps] = q2
    steps += 1
    avg_queue_sum += ((q1 + q2) - avg_queue_sum) / steps

    plt.figure(figsize=(12, 5))
    plt.subplot(2, 1, 1)
    plt.plot(q1_list, label="Queue Road 1")
    plt.plot(q2_list, label="Queue Road 2")
    plt.xlabel("Time Step")
    plt.ylabel("Queue Length")
    plt.title("Queue Lengths Over Time")
    plt.legend()
    plt.grid(True)

    print(f"Average Sum of Queue Lengths per Time Step: {avg_queue_sum:.2f}")
    print(f"Actions taken : {actions_taken}")



env = GymTrafficEnv()

    # Load a trained policy
policy1 = np.load('policy1.npy')  # SARSA policy
policy2 = np.load('policy2.npy')  # Expected SARSA policy
policy3 = np.load('policy3.npy')  # Value Function SARSA policy

TestPolicy(env, policy1)
TestPolicy(env, policy2)
TestPolicy(env, policy3)
env.close()