import numpy as np
import gymnasium as gym
import pygame
import tensorflow as tf
from tensorflow.keras.models import load_model

def choose_action(state, model, Nactions):
    # This function should choose action based on the DQN model and the current state.
    # While choosing action here, exploration is not required.
    # You have to set the argumnents of the function and write the required code.

    state = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
    state_batch = tf.repeat(state, repeats=Nactions, axis=0)
    actions = tf.eye(Nactions, dtype=tf.float32)
    input_batch = tf.concat([state_batch, actions], axis=1)
    Q = model.predict(input_batch, verbose = False)
    Q = tf.squeeze(Q, axis=1).numpy()
    action = np.argmax(Q)

    return action


# The following line load the DQN model. Write (commented) paths for both models, with and without offline data.
# HERE GOES THE LINE.
# loaded_model = load_model('DQN_offline_true.h5', compile=False)
# loaded_model = load_model('DQN_offline_false.h5', compile=False)


# The following line initializes the Mountain Car environment with render_mode
# set to 'human'.
# HERE GOES THE LINE.
env = gym.make('MountainCar-v0', render_mode='human')


# The following line resets the environment,
# HERE GOES THE LINE.
x, _ = env.reset()


end_episode = False
total_reward = 0
while not(end_episode):

    # The following line picks an action using choose_action() function.
    # HERE GOES THE LINE.
    action = choose_action(x, loaded_model,3)


    # The following line takes that the picked action. After taking the action,
    # it gets next state,reward, terminated, truncated, and info.
    # HERE GOES THE LINE.
    x_dash, r, terminated, truncated, _ = env.step(action)


    # The following line update the total reward
    # HERE GOES THE LINE.
    total_reward += r


    # The following line decides the state for the next time slot.
    # HERE GOES THE LINE.
    x = x_dash


    env.render()
    pygame.time.wait(20)


    # The following line decides end_episode for the next time slot.
    # HERE GOES THE LINE.
    if truncated or terminated:
        end_episode = True


# The following line prints the total reward.
# HERE GOES THE LINE.
print(total_reward)


# The following line closes the environment.
# HERE GOES THE LINE.
env.close()

pygame.display.quit()