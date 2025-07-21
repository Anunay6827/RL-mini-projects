import numpy as np
import pandas as pd
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def build_NN(Nactions, Nobservations):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(Nobservations + Nactions,)))
    model.add(Dense(1))
    return model

def choose_action(state, model, Nactions, epsilon, nu =1e-6):
    state = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
    state_batch = tf.repeat(state, repeats=Nactions, axis=0)
    
    actions = tf.eye(Nactions, dtype=tf.float32)

    input_batch = tf.concat([state_batch, actions], axis=1)
    Q = model.predict(input_batch, verbose = False)
    Q = tf.squeeze(Q, axis=1).numpy()
    if np.random.uniform()<epsilon:
        q_norm =  Q / (np.sum(np.abs(Q)) + nu)
        probs = np.exp(q_norm) / np.sum(np.exp(q_norm))

        return np.random.choice(np.arange(Nactions), p=probs)
    else:
        action = np.argmax(Q)
    return action

def add_to_buffer(x, a, r, x_dash, terminated, state_buffer, action_buffer, reward_buffer, next_state_buffer, terminated_buffer, buffer_size, buffer_counter, buffer_ix):        
    state_buffer[buffer_ix] = x
    action_buffer[buffer_ix] = a
    reward_buffer[buffer_ix] = r
    
    if terminated:
        next_state_buffer[buffer_ix] = np.zeros_like(x)  # If terminated=True, then the next_state is None. So, we fill the next state with dummy zero value.
    else:
        next_state_buffer[buffer_ix] = x_dash
        
    terminated_buffer[buffer_ix] = int(terminated) # If the episode terminated, then we save 1 or else 0.
    
    # buffer_counter is only updated till replay buffer is full. After that
    # the number of sample in replay buffer is equal to buffer_size. Hence,
    # no update required.
    if buffer_counter<buffer_size:
        buffer_counter+=1
    
    # Cyclic update of buffer_ix as mentioned in DQN_training() function.
    buffer_ix+=1
    if buffer_ix==buffer_size:
        buffer_ix = 0
        
    return buffer_counter, buffer_ix


def generate_training_data(state_buffer, action_buffer, reward_buffer, next_state_buffer, terminated_buffer, buffer_counter, model_target, Nb, beta, Nactions, epsilon, nu=1e-6):
    
    # Sample a random batch from replay buffer. Line 12 of psuedocode.
    ix = np.arange(buffer_counter)
    np.random.shuffle(ix)
    state_batch = state_buffer[ix[:Nb]]
    action_batch = action_buffer[ix[:Nb]]
    reward_batch = reward_buffer[ix[:Nb]]
    next_state_batch = next_state_buffer[ix[:Nb]]
    terminated_batch = terminated_buffer[ix[:Nb]]
    
    # The remaining line in this function generates the training batch using the
    # samples obtained from replay buffer. Line 13 of psuedocode.
    
    # Generate input, X, and target, y, of the training data
      
    # This line creates dummy target values using predict DQN (refer to notes.pdf in this folder)
    
    # These two lines create target values for the relevant actions.
    # It is quite complex to explain it in writting. Essentially, the line
    # y[np.arange(Nb), action_batch] is indexing of Numpy array, y, to assign
    # the target value.
    next_state_batch = np.repeat(next_state_batch, repeats=Nactions, axis=0)
    next_action_batch = np.tile(np.eye(Nactions), (Nb, 1))
    next_state_batch = np.concatenate([next_state_batch, next_action_batch], axis = 1)
    Q_next = model_target.predict(next_state_batch, verbose=0).reshape(Nb, Nactions) #Using target DQN to generate the target.
    Q_next_norm = Q_next / (np.sum(np.abs(Q_next), axis=1, keepdims=True) + nu)
    probs = np.exp(Q_next_norm) / np.sum(np.exp(Q_next_norm), axis=1, keepdims=True)
    
    greedy_actions = np.argmax(Q_next, axis=1)

    pi = epsilon * probs

    pi[np.arange(Nb), greedy_actions] += (1 - epsilon)

    Q_expected = np.sum(pi * Q_next, axis=1)


    action_one_hot = np.eye(Nactions)[action_batch]
    X = np.concatenate([state_batch, action_one_hot], axis = 1)
    y = reward_batch + beta * Q_expected * (1-terminated_batch) #(1-terminated) = 0 if terminated=1, implying no future reward.
    
    return X, y



def load_offline_data(path, min_score):    
    state_data = []
    action_data = []
    reward_data = []
    next_state_data = []
    terminated_data = []
    
    dataset = pd.read_csv(path)
    dataset_group = dataset.groupby('Play #')
    for play_no, df in dataset_group:
        # Skip first row if it contains a dictionary format
        start_idx = 0
        if isinstance(df.iloc[0, 1], str) and '{}' in df.iloc[0, 1]:
            start_idx = 1
        
        df = df[start_idx:]
        
        # Parse state - handle both string representation and array format
        state = []
        for s in df.iloc[:, 1]:
            if isinstance(s, str):
                # Handle string format like "[-0.5944583, 0.0]"
                s = s.replace('[', '').replace(']', '').split()
                state.append([float(val.strip(',')) for val in s])
            else:
                # It's already an array format
                state.append(s)
        state = np.array(state)
        
        action = np.array(df.iloc[:, 2]).astype(int)
        reward = np.array(df.iloc[:, 3]).astype(np.float32)
        
        # Parse next_state with the same approach
        next_state = []
        for s in df.iloc[:, 4]:
            if isinstance(s, str):
                s = s.replace('[', '').replace(']', '').split()
                next_state.append([float(val.strip(',')) for val in s])
            else:
                next_state.append(s)
        next_state = np.array(next_state)
        
        terminated = np.array(df.iloc[:, 5]).astype(int)
        
        total_reward = np.sum(reward)
        if total_reward >= min_score:
            state_data.append(state)
            action_data.append(action)
            reward_data.append(reward)
            next_state_data.append(next_state)
            terminated_data.append(terminated)
    
    if not state_data:  # Check if any data was collected
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
            
    state_data = np.concatenate(state_data)
    action_data = np.concatenate(action_data)
    reward_data = np.concatenate(reward_data)
    next_state_data = np.concatenate(next_state_data)
    terminated_data = np.concatenate(terminated_data)
    
    return state_data, action_data, reward_data, next_state_data, terminated_data




def plot_reward(total_reward_per_episode, window_length):
    # This function should display:
    # (i)  total reward per episode.
    # (ii) moving average of the total reward. The window for moving average
    #      should slide by one episode every time.

    plt.figure(figsize=(10, 5))
    plt.plot(total_reward_per_episode, label="Total Reward", color='blue')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Total Reward of Deep Expected SARSA")
    plt.legend()
    plt.grid(True)
    plt.show()

    movingAvg = np.convolve(total_reward_per_episode, np.ones(window_length)/window_length, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(movingAvg, label="Moving Average", color='red')
    plt.xlabel("Episodes")
    plt.ylabel("Moving Average of Total Reward")
    plt.title("Moving Average of Total Reward of Deep Expected SARSA")
    plt.legend()
    plt.grid(True)
    plt.show()




def DQN_training(env, offline_data, use_offline_data):
    # The function should return the final trained predict DQN model and
    # total reward of every episode.
    Nu = 1               # Predict DQN training interval.
    Nb = 100             # Training batch size.
    Nt = 10              # Target DQN update interval.
    beta = 0.99          # Discount factor.
    alpha = 0.001        # Learning rate. We are using a fixed learning rate.
                         # Even though theory demand decaying learning rate,
                         # in practice, fixed learning rate is used.
    Nsave = 50           # How often to save the model.
    buffer_size = 50000  # Replay buffer size.
    Nactions = 3         # There are only two actions in action space in cartpole environment.
    Nobservations = 2    # Dimension of the observation space of the cartpole environment.
    


    
    # Build predict and target DQNs. Similar to line 1 of the psuedocode.
    model_predict = build_NN(Nactions, Nobservations)
    model_target = build_NN(Nactions, Nobservations)
    model_target.set_weights(model_predict.get_weights())  # Copying predict DQN's weight to target DQN
    
    optimizer = Adam(learning_rate = alpha)
    model_predict.compile(loss='mse', optimizer=optimizer)

    # NOTE: We don't need to compile the target DQN because we are NOT training
    #       it. We are simply copying the weights from predict DQN to target DQN.
    
    
    # Initializing the replay buffer. Similar to line 2 of the psuedocode.
    state_buffer = np.zeros((buffer_size, Nobservations))
    action_buffer = np.zeros((buffer_size), dtype=np.uint8)     # Action indices and not the actual action
    reward_buffer = np.zeros((buffer_size))
    next_state_buffer = np.zeros((buffer_size, Nobservations))
    terminated_buffer = np.zeros((buffer_size), dtype=np.uint8)     # This is required to detect end of episode

    buffer_counter = 0        # Buffer counter represents how many samples are ther in the buffer.
                              # Once the buffer is full buffer_counter will be always equal to
                              # buffer_size.
    
    buffer_ix = 0             # Indicates the index in replay buffer where
                              # new samples can be inserted. buffer_ix gets updated
                              # in a cyclic fashion:
                              # 0, 1,....,buffer_size-1,0,1,...
                              #                         |
                              #                        Again reset to zero to immitate a FIFO queue.
    
    # Initlializing counters for various periodic operations like saving predict DQN,
    # update target DQN, and training predict DQN. Unlike psuedocode, we are using
    # seperate counters for each operation. This is better because in the psuedocode
    # the counter variable will always keep increasing and may gwt outside of the
    # range if int. Here, we use counter variable in a cyclical fashion.
    if (use_offline_data):
        num_offline = len(offline_data[0])

        if num_offline < buffer_size:
            state_buffer[:num_offline] = offline_data[0]
            action_buffer[:num_offline] = offline_data[1]
            reward_buffer[:num_offline] = offline_data[2]
            next_state_buffer[:num_offline] = offline_data[3]
            terminated_buffer[:num_offline] = offline_data[4]

            buffer_counter = num_offline
            buffer_ix = num_offline
        else:
            buffer_counter = buffer_size
            state_buffer[:buffer_size] = offline_data[0][num_offline - buffer_size : num_offline]
            action_buffer[:buffer_size] = offline_data[1][num_offline - buffer_size : num_offline]
            reward_buffer[:buffer_size] = offline_data[2][num_offline - buffer_size : num_offline]
            next_state_buffer[:buffer_size] = offline_data[3][num_offline - buffer_size : num_offline]
            terminated_buffer[:buffer_size] = offline_data[4][num_offline - buffer_size : num_offline]

            buffer_ix = num_offline % buffer_size

            
    counter_save = 0
    counter_target = 0
    counter_predict = 0
    
    total_reward_per_episode = [] # This list will contain the total reward of every episode.

    avg_total_reward = 0
    threshold = -120
    epsilon = 1.0
    decay_rate = 0.001
    min_epsilon = 0.05
    E = 100

    # This is required to detect end of episode
    episode = 0
    while (avg_total_reward < threshold or episode < 50) and not(episode >=3000):
        x, _ = env.reset()

        total_reward = 0
        end_episode = False
        while not(end_episode):

            a = choose_action(x, model_predict, Nactions, epsilon)

            x_dash, r, terminated, truncated, _ = env.step(a)
            total_reward+=r
            if not(use_offline_data) or ((episode + 1) >= E):
                buffer_counter, buffer_ix = add_to_buffer(x, a, r, x_dash, terminated, state_buffer, action_buffer, reward_buffer, next_state_buffer, terminated_buffer, buffer_size, buffer_counter, buffer_ix)

            counter_predict+=1
            if counter_predict==Nu:
                if buffer_counter>=Nb:
                    X, y = generate_training_data(state_buffer, action_buffer, reward_buffer, next_state_buffer, terminated_buffer, buffer_counter, model_target, Nb, beta, Nactions, epsilon)
                    model_predict.train_on_batch(X, y)
                
                counter_predict = 0
                
                counter_target+=1
                if counter_target==Nt:
                    model_target.set_weights(model_predict.get_weights())
                    counter_target=0
                
                counter_save+=1
                if counter_save==Nsave:
                    model_predict.save('Mountain_Car_model.h5')
                    counter_save = 0
            
            x = np.copy(x_dash)

            if terminated or truncated:
                end_episode=True

        total_reward_per_episode.append(total_reward)

        if episode < 50:
            avg_total_reward += (total_reward - avg_total_reward)/(episode + 1)
        else:
            avg_total_reward = np.mean(total_reward_per_episode[-50:])

        print('Episode = {}, Total reward = {}, Epsilon = {}'.format(episode+1, np.round(total_reward, 2),  epsilon))

        episode+=1

        epsilon = min_epsilon + (epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    return model_predict, total_reward_per_episode




# Initiate the mountain car environment.
# NO RENDERING. It will slow the training process.
env = gym.make('MountainCar-v0')


# Load the offline data collected in step 3. Also, process the dataset.
path = 'car_dataset.csv' # This should contain the path to the collected dataset.
min_score = -np.inf # The minimum total reward of an episode that should be used for training.
offline_data = load_offline_data(path, min_score)

# Train DQN model of Architecture type 1
use_offline_data = True # If True then the offline data will be used. Else, offline data will not be used.
final_model, total_reward_per_episode = DQN_training(env, offline_data, use_offline_data)


# Plot reward per episode and moving average reward
window_length = 50    # Window length for moving average reward.
plot_reward(total_reward_per_episode, window_length)


# Save the final model
final_model.save('DQN_offline_true.h5') # This line is for Keras. Replace this appropriate code.

use_offline_data = False # If True then the offline data will be used. Else, offline data will not be used.
final_model, total_reward_per_episode = DQN_training(env, offline_data, use_offline_data)

# # Plot reward per episode and moving average reward
window_length = 50    # Window length for moving average reward.
plot_reward(total_reward_per_episode, window_length)

final_model.save('DQN_offline_false.h5')

env.close()