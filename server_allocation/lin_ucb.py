import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import gymnasium
from CloudComputing import ServerAllocationEnv

def processContext(context):
    numJobs = len(context)
    
    context = np.array(context, dtype = object)
    
    jobPriorities = context[:, 0].astype(np.float64)
    jobTypes = context[:, 1].astype(str)
    networkUsages = context[:, 2].astype(np.float64)
    estimatedTimes = context[:, 3].astype(np.float64)
    
    jobTypesCounts = np.array([np.sum(jobTypes == 'A'), np.sum(jobTypes == 'B'), np.sum(jobTypes == 'C')], dtype=np.float64)
            
    prioritiesMean = np.mean(jobPriorities) 
    networkMean = np.mean(networkUsages)
    timesMean = np.mean(estimatedTimes)

    return np.array([prioritiesMean, *jobTypesCounts, networkMean, timesMean, 1], dtype = np.float64).reshape(-1, 1)

def linearUCB(env):
    numJobTypes = env.NTypes
    numJobFeatures = len(env.observation_space.sample()[0])
    numActions = env.MaxServers
    numJobs = env.MaxJobs
    numFeatures = 6

    epsilon = 1.0
    decayRate = 0.001
    minEpsilon = 0.05
    
    theta = np.zeros((numActions, numFeatures + 1))
    A = np.array([np.eye(numFeatures + 1) * 0.01 for _ in range(numActions)])  
    B = np.zeros((numActions, numFeatures + 1))
    
    numEpisodes = 500
    rewardArr = []
    totalReward = 0
    
    for episode in range(numEpisodes):
        observation, _ = env.reset()
        z = processContext(observation)
        truncated = False
        
        while not truncated:
            action = 1
            A_inv = np.linalg.inv(A[action - 1])
            maxActionSelection = theta[action - 1] @ z + epsilon * np.sqrt(z.T @ A_inv @ z)
            for j in range(2, numActions + 1):
                A_inv = np.linalg.inv(A[j - 1])
                currentActionSelection = theta[j - 1] @ z + epsilon * np.sqrt(z.T @ A_inv @ z)
                if currentActionSelection > maxActionSelection:
                    maxActionSelection = currentActionSelection
                    action = j
                    
            observation_next, reward, _, truncated, _ = env.step(action)
            
            rewardArr.append(reward)
            totalReward += reward
            
            if not truncated:
                A[action - 1] += z @ z.T
                B[action - 1] += reward * (z.reshape(numFeatures + 1,))
                theta[action - 1] = np.linalg.inv(A[action - 1]) @ B[action - 1]

                observation = observation_next
                z = processContext(observation)
        epsilon = minEpsilon + (epsilon - minEpsilon) * np.exp(-decayRate * episode)

        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}, Total Reward: {totalReward}, Epsilon {epsilon}")

    return np.array(rewardArr), theta

env = ServerAllocationEnv()
rewardArr, theta = linearUCB(env)

# Apply smoothing
window_size = 500
rolling_avg = np.convolve(rewardArr, np.ones(window_size)/window_size, mode="valid")

# Plot reward trend
plt.figure(figsize=(20, 5))
plt.plot(rolling_avg, label="Moving Average", color='red')
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.xlabel("Time steps")
plt.ylabel("Window Averaged reward")
plt.title("Linear UCB Contextual Bandit Performance")
plt.legend()
plt.grid(True)
plt.show()