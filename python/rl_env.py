import gymnasium as gym
from gymnasium import spaces

from env_helpers import *

# documentation: https://gymnasium.farama.org/api/env/
class KhatrisEnv(gym.Env):
    def __init__(self):
        # observation space contains multiple variables
        # hold: the current hold piece
        # queue: list of upcoming pieces
        # combo: how many lines have been cleared consecutively
        # b2b: back-to-back status (last clear was a T-spin or Tetris) 
        # board: all possible Tetris boards (40x10) as 2d list of bits
        spaces = {
            'hold': spaces.Discrete(7),
            'queue': spaces.Discrete(6),
            'combo': spaces.Discrete(1000),
            'b2b': spaces.Discrete(2),
            'board': spaces.MultiBinary([40, 10])
        }
        self.observation_space = gym.spaces.Dict(spaces)

        # TODO: initialize this with the find_moves function
        move_list = []
        self.action_space = spaces.Discrete(len(move_list))

        # map the encoded/abstract actions to actual ones
        self._action_to_move = dict((x, y) for x, y in enumerate(move_list))

    # translates the environment’s state
    def _get_obs(self):
        pass 

        # return {
        #     'hold': spaces.Discrete(7),
        #     'queue': spaces.Discrete(6),
        #     'combo': spaces.Discrete(1000),
        #     'b2b': spaces.Discrete(2),
        #     'board': spaces.MultiBinary([40, 10])
        # }

    def reset(self, seed=None, options=None):
        pass 

        # this is to seed self.np_random
        #super().reset(seed=seed)

        # TODO: implementation
        
        #return observation, info

    def step(self, action):
        pass
    
        #return observation, reward, terminated (bool), False (bool), info (dict: auxiliary diagnostic info)
