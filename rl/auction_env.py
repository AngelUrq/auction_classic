import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AuctionEnv(gym.Env):
    """Custom Environment that follows gymnasium interface"""

    def __init__(self):
        super(AuctionEnv, self).__init__()
        # Define action and observation space
        # They must be gymnasium.spaces objects
        # Example for using discrete actions:
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using box actions:
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(N_ACTIONS,), dtype=np.float32)

        # Example for using discrete observation space:
        # self.observation_space = spaces.Discrete(N_DISCRETE_OBSERVATIONS)
        # Example for using box observation space:
        # self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
        
    def step(self, action):
        # Execute one time step within the environment
        # return observation, reward, done, info
        # observation: the observation of the environment after taking the action
        # reward: the reward obtained from taking the action
        # done: whether the episode has ended, returning the agent to a new state
        # info: additional information about the environment
        pass

    def reset(self):
        # Reset the state of the environment to an initial state
        # return the initial observation
        pass

    def close(self):
        # Clean up the environment's resources
        pass
