import numpy as np
import gymnasium as gym
import pygame
from RobotNavigation import RobotNavigationEnv
from stable_baselines3.common.callbacks import BaseCallback
# Imported the libraries which I think are useful. You can import the rest.
from stable_baselines3 import DQN


class LoggingAndSavingCallback(BaseCallback):
    def init(self, test_period, test_count, verbose=0):
        super().init(verbose)
        # test_period is the number of time steps (env.step()) after which we
        # want to test the model. You also have to save the latest model every
        # test_period.
        
        # test_count is the number of episodes for which we want to test the model.
        
        self.test_period = test_period
        self.test_count = test_count
        self.training_rewards = []
        self.testing_rewards = []
        self.best_mean_reward =  -np.inf
        self.current_episode_reward = 0
        
        # You can declare other variables here that are required to do the
        # tasks mentioned in _on_step() function.
        

    def _on_step(self) -> bool:
        
        # This function should do the following:
        #  1. Calculate the sum of reward of every episode and save it in a
        #     .npy file named training_log.npy during training. This MUST be
        #     done in the end of every episode. To do this, you can access the
        #     reward of the current time step using self.locals['rewards'][0]
        #     and you can check if the current episode is terminated/truncated
        #     using self.locals['dones'][0].
        self.current_episode_reward += self.locals['rewards'][0]
        if self.locals['dones'][0]:
            self.training_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            np.save('training_log.npy', np.array(self.training_rewards))
        #
        #  2. Every self.test_period time steps:
        #      2.a. Save the latest model as LATEST_MODEL. You can access the
        #           latest model using self.model. 
        #
        #      2.b. Test the latest model by calculating the average sum reward
        #           of the latest model over self.test_count episodes. After
        #           calculating average sum of reward, append this latest test
        #           result to a .npy file named testing_log.npy. Also, if the
        #           average sum of reward is highest till now, save the latest
        #           model as BEST_MODEL.
        #
        #           VERY IMPORTANT: While testing, you need to initiate a LOCAL
        #           robot navigation environment here. You MUST NOT use the
        #           environment that you initiated for training for testing
        #           purposes.
        if self.num_timesteps % self.test_period == 0:
            self.model.save('LATEST_MODEL')
            test_env = RobotNavigationEnv()
            mean_reward = 0
            for _ in range(self.test_count):
                x, _ = test_env.reset()
                dones = False
                while not dones:
                    action = self.model.predict(x, deterministic=True)
                    # Ensure action is scalar, not a tuple
                    if isinstance(action, (tuple, list, np.ndarray)):
                        action = action[0]
                        x, reward, terminated, truncated, _ = test_env.step(action)

                    dones = terminated or truncated
                    mean_reward += reward
            test_env.close()
            mean_reward /= self.test_count
            self.testing_rewards.append(mean_reward)
            np.save('testing_log.npy', np.array(self.testing_rewards))
            if self.best_mean_reward < mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save('BEST_MODEL')


        
            
        return True # This MUST return True unless you want the training to stop.


# Initiate the robot navigation environment.
env = RobotNavigationEnv()


# Initiate an instance of the LoggingAndSavingCallback. Desription of test_period
# and test_count are there in init_ function of LoggingAndSavingCallback.
test_period = 20000   # Default value. You can change it.
test_count = 10       # Default value. You can change it.
# callback = LoggingAndSavingCallback(test_period, test_count)


# The code that you use to train the RL agent for the robot navigation environment
# goes below this line. The total number of lines is unlikely to be more than 10.
model = DQN('MlpPolicy', env, verbose=1,learning_starts = 10000, batch_size=128, policy_kwargs={"net_arch": [256, 256, 256]}, buffer_size=100000)
model.learn(total_timesteps=2000000, log_interval=1)

# Close the robot navigation environment.
env.close()


# Write just ONE line of code below to save the model that you have trained.
# YOU HAVE TO SUBMIT THIS MODEL. THE NAME OF THE MODEL MUST BE MODEL3.
model.save('MODEL3')