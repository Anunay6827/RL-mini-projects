import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GymTrafficEnv(gym.Env):
    def __init__(self):
        self.arrival_prob = np.array([0.28, 0.40])
        self.departure_max_prob = 0.9
        self.max_queue = 1810
        self.max_time = 1800

        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.max_queue + 1),
            spaces.Discrete(self.max_queue + 1),
            spaces.Discrete(2),
            spaces.Discrete(11),
        ))
        
        self.action_space = spaces.Discrete(2)
        
        self.q1 = None
        self.q2 = None
        self.signal = None
        self.counter = None
        self.time = None
        
    def step(self, action):
        self.time += 1
        reward = -(self.q1 + self.q2)

        if np.random.rand() < self.arrival_prob[0]:
            self.q1 = min(self.q1 + 1, self.max_queue)
            
        if np.random.rand() < self.arrival_prob[1]:
            self.q2 = min(self.q2 + 1, self.max_queue)


        if self.signal == 0 and self.counter == 0 and self.q1 > 0 and np.random.rand() < self.departure_max_prob:  # east-west green
                self.q1 -= 1
            
        elif self.signal == 1 and self.q2 > 0 and self.counter == 0 and np.random.rand() < self.departure_max_prob:  # north-south green 
                self.q2 -= 1

        if 0 < self.counter <= 10:
            p = self._red_departure_prob(self.counter-1)
            if self.signal == 0 and self.q1 > 0:
                if np.random.rand() < p:
                    self.q1 -= 1
            elif self.signal == 1 and self.q2 > 0:
                if np.random.rand() < p:
                    self.q2 -= 1
            
        if self.counter == 0 and action == 1:
            self.counter = 1
                
        elif 0 < self.counter < 10 and action == 0:
            self.counter += 1
        elif 0 < self.counter <= 10 and action == 1:
            self.counter = 0
        elif self.counter == 10 and action == 0:
            if self.signal == 0:
                self.signal = 1
            else:
                self.signal = 0
            self.counter = 0

        truncated = self.time >= self.max_time
        terminated = False
        next_state = (self.q1,self.q2,self.signal,self.counter)

        return next_state,reward,terminated,truncated,{}

    def _red_departure_prob(self, delta):
        t = self.time
        prob = self.departure_max_prob * (1 - ((delta ** 2) / 100))
        
        return prob 
        
    def reset(self):
        self.q1 = np.random.randint(0,11)
        self.q2 = np.random.randint(0,11)
        self.signal = np.random.choice([0,1])
        self.counter = 0
        self.time = 0
        return (self.q1,self.q2,self.signal,self.counter),{}
    
    def render(self):
        # Used to graphics. NOT NEEDED FOR THIS ASSIGNMENT.
        pass