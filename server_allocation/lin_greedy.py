import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from CloudComputing import ServerAllocationEnv

class EpsilonGreedyLinearBandit:
    def __init__(self, env, epsilon=0.1, alpha=0.01):
        self.env = env
        self.initial_epsilon = epsilon
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate
        self.num_actions = env.MaxServers  # Actions: Number of servers (1-8)
        self.weights = np.random.randn(4, self.num_actions) * 0.01  # Small random weights
        self.rewards = []  # Store rewards for plotting
        self.reward_mean = 0
        self.reward_std = 1
    
    def featurize(self, context):
        # Convert variable-length context into a fixed-length feature vector
        features = np.zeros(4)
        num_jobs = len(context)
        
        if num_jobs > 0:
            priority_avg = np.mean([job[0] for job in context]) / self.env.MinPriority
            type_avg = np.mean([ord(job[1]) - ord('A') for job in context]) / (self.env.NTypes - 1)
            network_avg = np.mean([job[2] for job in context])
            processing_avg = np.mean([job[3] for job in context]) / self.env.MaxProcessingTime
            
            features = np.array([priority_avg, type_avg, network_avg, processing_avg])
        
        return features
    
    def select_action(self, features):
        if np.random.rand() < self.epsilon:
            return np.random.randint(1, self.num_actions + 1)  # Random action (exploration)
        
        q_values = features @ self.weights  # Linear model prediction
        return np.argmax(q_values) + 1  # Best action (exploitation)
    
    def update_weights(self, features, action, reward):
        action_idx = action - 1
        q_values = features @ self.weights
        
        # Normalize reward dynamically
        self.reward_mean = 0.99 * self.reward_mean + 0.01 * reward
        self.reward_std = 0.99 * self.reward_std + 0.01 * (reward - self.reward_mean) ** 2
        target = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        
        error = target - q_values[action_idx]
        self.weights[:, action_idx] = 0.9 * self.weights[:, action_idx] + 0.1 * (features * error)  # Smooth update
    
    def train(self, episodes=1000, window_size=500):
        env = self.env
        rewards = []
        
        for episode in range(episodes):
            context, _ = env.reset()
            total_reward = 0
            
            for t in range(env.Horizon):
                features = self.featurize(context)
                action = self.select_action(features)
                next_context, reward, _, truncated, _ = env.step(action)
                self.update_weights(features, action, reward)
                total_reward += reward
                
                if truncated:
                    break
                context = next_context
            
            rewards.append(total_reward)
            
            # Compute receding window time-averaged reward
            if episode >= window_size:
                avg_reward = np.mean(rewards[-window_size:])
            else:
                avg_reward = np.mean(rewards)
            self.rewards.append(avg_reward)
            
            # Decay epsilon over time for better exploration-exploitation tradeoff
            self.epsilon = max(0.01, self.initial_epsilon * np.exp(-0.001 * episode))
            
            if episode % 1 == 0:
                print(f"Episode {episode}/{episodes}, Avg Reward: {avg_reward:.2f}")
        
    def plot_rewards(self):
        plt.plot(self.rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Receding Window Avg Reward")
        plt.title("Epsilon-Greedy Linear Bandit Performance")
        plt.show()

# Initialize environment and agent
env = ServerAllocationEnv()
agent = EpsilonGreedyLinearBandit(env, epsilon=0.3, alpha=0.01)

# Train and plot results
agent.train(episodes=2000)
agent.plot_rewards()