import numpy as np
import gymnasium as gym
from scipy.stats import truncnorm, poisson, truncexpon
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from CloudComputing import ServerAllocationEnv

max_actions = 8
num_features_used = 7

# Policy Gradient model with variable hidden layers
class PolicyGradient(nn.Module):
    def __init__(self, input_dim, output_dim = max_actions, hidden_layers = [64,64], activation = nn.ReLU(), use_softmax = True):
        super(PolicyGradient, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for each_layer in hidden_layers:
            layers.append(nn.Linear(current_dim, each_layer))
            layers.append(activation)
            current_dim = each_layer
        
        layers.append(nn.Linear(current_dim, output_dim))
        if(use_softmax):
            layers.append(nn.Softmax(dim=-1))

        self.pg = nn.Sequential(*layers)

    def forward(self, state):
        return self.pg(state)
    
#Incremental Normalizer for one feature vector at a time
class Normalization:
    def __init__(self, feature_dimensions):
        self.num_features = feature_dimensions
        self.mean = np.zeros(feature_dimensions, dtype=np.float64)
        self.var = np.zeros(feature_dimensions, dtype=np.float64)
        self.count = 0

    def update(self, x):
        if(x.shape[0] != self.num_features):
            raise Exception("The features are not matching")
        
        self.count += 1
        
        if(self.count == 1):
            self.mean[:] = x
            self.var = np.zeros(self.num_features, dtype=np.float64)
        else:
            prev_mean = self.mean.copy()
            self.mean += (x - prev_mean) / self.count
            self.var += (x - prev_mean) * (x - self.mean)

    def normalize(self, x, epsilon = 1e-8):
        if self.count < 2:
            return x
        std = np.sqrt(self.var / (self.count - 1) + 1e-8) # Added epsilon to fix zero division issue that comes in next line
        return (x - self.mean) / std
        
    def full_reset(self):
        self.buffer.clear()
        self.state.clear()
        self.reset()
        
# Extracting features from the context
# 7 features from priority, job type, times and network
# Mean of priorities, Mean of estimated times, Mean of network usage, counts for job A,B,C and bias
def feature_extraction(current_context):
    num_jobs = len(current_context)
    if(num_jobs == 0):
        return np.zeros(num_features_used, dtype=np.float32)
        
    priority_list  = []
    estimated_times_list = []
    network_usage_list = []
    type_counts = {'A': 0, 'B': 0, 'C': 0}
    
    for each_job in current_context:
        priority = each_job[0]
        job_type = each_job[1]
        network_usage = each_job[2]
        estimated_time = each_job[3]
        
        priority_list.append(priority)
        estimated_times_list.append(estimated_time)
        network_usage_list.append(network_usage)
        type_counts[job_type] += 1

    final_features = np.array([
        np.mean(priority_list),
        np.mean(estimated_times_list),
        np.mean(network_usage_list),
        type_counts['A'] / num_jobs, 
        type_counts['B'] / num_jobs,
        type_counts['C'] / num_jobs,
    ], dtype=np.float32)

    features = np.append(final_features, 1)
    
    return features

# Sampling and selecting an action from the space
def action_selection(pg, all_features):
    features_tensored = torch.tensor(all_features, dtype=torch.float32)
    action_prob = pg(features_tensored)
    
    sample_action = torch.multinomial(action_prob, num_samples = 1).item() + 1 # adding 1 fixes the indexing issue
    log_prob = torch.log(action_prob[sample_action - 1]) # 1 is subtracted to align with the previous indexing
    
    return sample_action, log_prob

# Setting hyperparameters for training and the model
learning_rate = 0.01
num_batches = 100
batch_size = 10
hidden_layers = [64,64]
activation_pg = nn.ReLU()
optimizer = 1
epsilon = 1e-8
input_dimensions = num_features_used

# Training
env = ServerAllocationEnv()
pg_model = PolicyGradient(input_dimensions, output_dim = max_actions, hidden_layers = hidden_layers, activation = activation_pg)

if(optimizer == 1):
    optimizer = optim.Adam(pg_model.parameters(), lr = learning_rate) # Adam optimizer
    
normalizer = Normalization(input_dimensions)

track_timesteps_rewards = [] # for plotting

for each_batch in range(1, num_batches + 1):
    batch_training_data = []
    
    for each_episode in range(batch_size):
        episode_rewards_list = []
        log_prob_list = []

        resultant = env.reset()
        context = resultant[0] # retreiving the current context
        
        for each_timestep in range(env.Horizon):
            # Extract and normalize features for each timestep
            initial_features = feature_extraction(context)
            normalizer.update(initial_features)
            normalized_features = normalizer.normalize(initial_features)

            # Sampling and getting action with its log probability
            sampled = action_selection(pg_model, normalized_features)
            sampled_action = sampled[0]
            log_prob = sampled[1]

            # Getting data for that action from the computing file
            retreived_data = env.step(sampled_action)
            observation = retreived_data[0]
            reward = retreived_data[1]
            terminated = retreived_data[2]
            truncated = retreived_data[3]
            
            track_timesteps_rewards.append(reward)
            episode_rewards_list.append(reward)
            log_prob_list.append(log_prob)

            # Updating context based on received data
            context = observation
            if(truncated):
                break
                
        batch_training_data.append((episode_rewards_list, log_prob_list))

    
    optimizer.zero_grad()
    batch_losses = []
    total_batch_reward = 0
    
    for episode_rewards_list, log_prob_list in batch_training_data:
        tensored_rewards = torch.tensor(episode_rewards_list, dtype=torch.float32)
        total_batch_reward += sum(episode_rewards_list)

        entire_episode_loss = 0
        for i, log_prob in enumerate(log_prob_list):
            entire_episode_loss -= log_prob * tensored_rewards[i]

        batch_losses.append(entire_episode_loss)

    tensored_batch_loss = torch.stack(batch_losses)
    mean_loss = tensored_batch_loss.mean()
    mean_loss.backward()
    optimizer.step()
             
    batch_average_reward = total_batch_reward / (batch_size * env.Horizon)
    print(f"Batch: {each_batch}, Average Reward: {batch_average_reward:.2f}")
    
window_size = 500
window_rewards = []
track_timesteps_rewards = np.array(track_timesteps_rewards)

for i in range(len(track_timesteps_rewards) - window_size + 1):
    window_area = track_timesteps_rewards[i : i + window_size]
    window_averaged = np.mean(window_area)
    window_rewards.append(window_averaged)

plt.figure(figsize=(12, 6))
plt.plot(range(window_size, len(track_timesteps_rewards) + 1), window_rewards)
plt.xlabel('Time Steps')
plt.ylabel(f'Window-Averaged Reward (Size=500)')
plt.title('Policy Gradient - Time Steps vs Receding Window-Averaged Reward')
plt.grid(True)
plt.show()