import numpy as np
import gymnasium as gym
from typing import Optional
import pygame
import pygame.gfxdraw

class RobotNavigationEnv(gym.Env):
    def __init__(self, render_mode: Optional[str] = None):
        self.GridSize = 500        
        self.delta = 1.0/self.GridSize
        
        self.goal = np.array([20,20]).astype(float)
        self.goal[0] = self.delta*(0.5 + (self.goal[0] - 1))
        self.goal[1] = self.delta*(0.5 + (self.goal[1] - 1))
        self.goal_radius = 0.02
        
        self.MaxObstacles = 7
        self.radius = np.array([0.1, 0.4])/2
        self.obstacles = None
        
        # 0: Up, 1: Down, 2: Left, 3: Right
        self.action_dict = {0:np.array([0,1]), 1:np.array([0,-1]), 2:np.array([-1,0]), 3:np.array([1,0])}
        
        self.observation_space = gym.spaces.Box(np.zeros(36+2), np.sqrt(2)*np.ones(36+2), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
            
        self.robot_position = None
        self.observation = None
        
        self.Horizon = 5*(2*self.GridSize) # Maximum length of an episode.
        self.t = None
        
        # These variables are required for rendering/animation/graphics.
        self.render_mode = render_mode
        self.trail = None
        self.screen_width = 500
        self.screen_height = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        
    
    def check_collision(self):
        # Check collision with boundary
        delta_error = 0.0001
        if self.robot_position[0]<self.delta/2-delta_error or self.robot_position[0]>1-self.delta/2+delta_error:
            return True
        
        if self.robot_position[1]<self.delta/2-delta_error or self.robot_position[1]>1-self.delta/2+delta_error:
            return True
            
        # Check collision with circular obstacles
        for x, y, r in self.obstacles:
            if (x - self.robot_position[0])**2 + (y - self.robot_position[1])**2<=r**2:
                return True
            
        return False
    
    
    def step(self, action):
        assert self.robot_position is not None, "Call reset before using step method."
                
        reward = -np.sqrt(np.sum((self.robot_position - self.goal)**2))
        
        self.robot_position = self.robot_position + self.action_dict[action+0]*self.delta
        
        self.observation = np.concatenate((self.get_lidar_reading(),self.robot_position))
        self.trail.append(self.robot_position) # This is the path trail of the robot. Required for rendering/graphics.
        
        # Check for collision
        terminated = self.check_collision()
        if terminated:
            reward = -10000
        
        # Check if goal is reached
        if not(terminated):
            if (self.goal[0] - self.robot_position[0])**2 + (self.goal[1] - self.robot_position[1])**2<=self.goal_radius**2:
                terminated = True
                reward = 1000
                
        self.t += 1
        truncated = False
        if self.t>self.Horizon:
            truncated = True
            
        if self.render_mode=="human":
            self.render()
        
        return self.observation, reward, terminated, truncated, {}
    

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Randomly create obstacles
        self.create_obstacles()
            
        check = False
        while not(check):            
            # Randomly generate robot position
            self.robot_position = np.zeros(2, dtype=float)
            segment = np.random.choice(['top right', 'top left', 'bottom right', 'bottom left'], p=[0.5, 0.2, 0.2, 0.1])
            if segment=='top right':
                self.robot_position[0] = np.random.randint(int(0.5*self.GridSize), self.GridSize+1)
                self.robot_position[1] = np.random.randint(int(0.5*self.GridSize), self.GridSize+1)
            elif segment=='top left':
                self.robot_position[0] = np.random.randint(1 , int(0.5*self.GridSize)+1)
                self.robot_position[1] = np.random.randint(int(0.5*self.GridSize), self.GridSize+1)
            elif segment=='bottom right':
                self.robot_position[0] = np.random.randint(int(0.5*self.GridSize), self.GridSize+1)
                self.robot_position[1] = np.random.randint(1 , int(0.5*self.GridSize)+1)
            else:
                self.robot_position[0] = np.random.randint(1 , int(0.5*self.GridSize)+1)
                self.robot_position[1] = np.random.randint(1 , int(0.5*self.GridSize)+1)
            
            self.robot_position[0] = self.delta*(0.5 + (self.robot_position[0] - 1))
            self.robot_position[1] = self.delta*(0.5 + (self.robot_position[1] - 1))
            
            
            # Verify that the robot position is NOT inside one of the obstacles
            check = True
            for x, y, r in self.obstacles:
                if (x - self.robot_position[0])**2 + (y - self.robot_position[1])**2<=r**2:
                    check = False
                    break
                    
        
        self.observation = np.concatenate((self.get_lidar_reading(),self.robot_position))
        self.trail = [self.robot_position]
        self.t = 0
        
        if self.render_mode=="human":
            self.render()
        
        return self.observation, {}
    
    
    def intersection_circle(self, D):
        # This function helps in generating lidar readings. It calculates the
        # distance of the robot from the circular obstacles in a specified direction.
        
        min_distance = np.inf

        for circle in self.obstacles:
            x_c, y_c, r_c = circle
        
            # Vector from circle center to robot
            f = (self.robot_position[0]-x_c, self.robot_position[1]-y_c)

            b = 2 * (D[0]*f[0] + D[1]*f[1])
            c = f[0]**2 + f[1]**2 - r_c**2

            discriminant = b*b-4*c

            if discriminant < 0: # No intersection
                continue

            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b - sqrt_disc)/2
            t2 = (-b + sqrt_disc)/2

            if t1 > 0 and t1 < min_distance:
                min_distance = t1
            
            if t2 > 0 and t2 < min_distance:
                min_distance = t2

        return min_distance
    
    
    def intersection_line(self, D, line):
        # This function helps in generating lidar readings. It calculates the
        # distance of the robot from the grid boundaries in a specified direction.
        
        Dx, Dy = D
        
        if line[0]=='v': # Vertical line
            X_target = line[1]

            if np.abs(Dx)<=0.0001:
                return np.inf

            t = (X_target-self.robot_position[0])/Dx
            if t<0:
                return np.inf

            return t

        else:  # Horizontal line
            Y_target = line[1]
            
            if np.abs(Dy)<=0.0001:
                return np.inf

            t = (Y_target-self.robot_position[1])/Dy
            if t<0:
                return np.inf

            return t
    
    
    def get_lidar_reading(self):
        distance = []
        for angle in range(0, 360, 10):
            D = np.array([np.cos(angle*np.pi/180.0), np.sin(angle*np.pi/180.0)])
            distance.append(np.sqrt(2))
            
            # Calculate the distance of the robot from the grid boundaries in direction specified by vector D.
            for boundary in [['v', 0], ['v', 1], ['h', 0], ['h', 1]]:
                distance[-1] = min(distance[-1], self.intersection_line(D, boundary))
            
            # Calculate the distance of the robot from the obstacles in direction specified by vector D.
            distance[-1] = min(distance[-1], self.intersection_circle(D))
            
        return np.array(distance)
    
            
    def check_obstacles(self, X, Y, R):
        # This function checks if the circular obstacles that are generated
        # (i) are overlapping, (ii) if the circular obstacles goes beyond the
        # grid boundaries, and (iii) the circular obstacles are such that the
        # goal lies inside the obstacle. If any of these three happens, it implies
        # that the obstacles are not valid and hence needs to be re-generated.
        
        N = len(X)
        check = True
        for n in range(N):
            x, y, r = X[n], Y[n], R[n]
            dist = np.sqrt((x - self.goal[0])**2 + (y - self.goal[1])**2)
            if dist<=r+self.goal_radius:
                check = False
                break
            
            x_lb = x - r
            x_ub = x + r
            y_lb = y - r
            y_ub = y + r
            if x_lb<0 or x_ub>1 or y_lb<0 or y_ub>1:
                check = False
                break
                
            for m in range(N):
                if m!=n:
                    dist = np.sqrt((x - X[m])**2 + (y - Y[m])**2)
                    if dist<=r+R[m]:
                        check = False
                        break
                    
            if not(check):
                break
            
        return check
        
            
    def create_obstacles(self):
        # This function randomly generate the circular obstaicles.
        Nobstacles = np.random.randint(3, self.MaxObstacles+1) # The number of circular obstacles.
        correct_generation = False
        while not(correct_generation):
            # Randomly generate teh center (X,Y) and the radius R of the circular obstacles.
            X = np.random.uniform(0, 1, size=(Nobstacles))
            Y = np.random.uniform(0, 1, size=(Nobstacles))
            R = np.random.uniform(self.radius[0], self.radius[1], size=(Nobstacles))
            
            # Check of the randomly generated obstacles are valid or not.
            correct_generation = self.check_obstacles(X, Y, R)
        
        self.obstacles = [(x, y, r) for x, y, r in zip(X, Y, R)]
    
    
    def render(self):
        # This function is used to generate graphics/animation
        
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        world_width = 1.0
        scale = self.screen_width/world_width
                
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((0, 128, 0))
        
        for x, y, r in self.obstacles:
            x_graphics = int(x*scale)
            y_graphics = int(y*scale)
            r_graphics = int(r*scale)
            
            pygame.gfxdraw.filled_circle(
                self.surf,
                x_graphics,
                y_graphics,
                r_graphics,
                (150, 75, 0),
            )
            
        x_graphics = int(self.goal[0]*scale)
        y_graphics = int(self.goal[1]*scale)
        r_graphics = int(self.goal_radius*scale)
        
        pygame.gfxdraw.filled_circle(
            self.surf,
            x_graphics,
            y_graphics,
            r_graphics,
            (0, 0, 0),
        )
        
        if len(self.trail)>1:
            Ntrail = len(self.trail)
            for i in range(Ntrail-1):
                x1_graphics = int(self.trail[i][0]*scale)
                y1_graphics = int(self.trail[i][1]*scale)
                x2_graphics = int(self.trail[i+1][0]*scale)
                y2_graphics = int(self.trail[i+1][1]*scale)
                
                pygame.gfxdraw.line(
                    self.surf,
                    x1_graphics,
                    y1_graphics,
                    x2_graphics,
                    y2_graphics,
                    (0, 0, 255),
                )
        
        
        x_graphics = int(self.robot_position[0]*scale)
        y_graphics = int(self.robot_position[1]*scale)
        r_graphics = int(0.01*scale)
        
        pygame.gfxdraw.filled_circle(
            self.surf,
            x_graphics,
            y_graphics,
            r_graphics,
            (255, 0, 0),
        )        
        
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        
        fps = 50
        pygame.event.pump()
        self.clock.tick(fps)
        pygame.display.flip()
   
            
    def close(self):
        if self.screen is not None:
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        
            pygame.display.quit()
            pygame.quit()
            self.isopen = False