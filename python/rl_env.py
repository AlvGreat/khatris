import gymnasium as gym
from gymnasium import spaces
from pylibtetris.pylibtetris import *

from env_helpers import *

# documentation: https://gymnasium.farama.org/api/env/
class KhatrisEnv(gym.Env):
    def __init__(self):
        # observation space contains multiple variables
        # board: all possible Tetris boards (40x10) as 2d list of bits
        # hold: the current hold piece
        # queue: list of upcoming pieces
        # combo: how many lines have been cleared consecutively
        # b2b: back-to-back status (last clear was a T-spin or Tetris) 
        spaces = {
            'hold': spaces.Discrete(7),
            'queue': spaces.MultiDiscrete([7 for _ in range(6)]),
            'combo': spaces.Discrete(1000),
            'b2b': spaces.Discrete(2),
            'board': spaces.MultiBinary([40, 10])
        }
        self.observation_space = gym.spaces.Dict(spaces)

        # TODO: initialize this with the find_moves function
        move_list = []
        self.action_space = spaces.Discrete(len(move_list))
        b = new_board_with_queue()
        self.board = b.board
        self.hold = b.hold
        self.queue = b.queue
        self.b2b = b.b2b
        self.combo = b.combo
        
        # map the encoded/abstract actions to actual ones
        self._action_to_move = dict((x, y) for x, y in enumerate(move_list))

    # translates the environment’s state
    def _get_obs(self):

        return {
            'hold': self.hold,
            'queue': self.queue,
            'combo': self.combo,
            'b2b': self.b2b,
            'board': self.board,
        }

    def reset(self, seed=None, options=None):
        b = new_board_with_queue()
        self.board = b.board
        self.hold = b.hold
        self.queue = b.queue
        self.b2b = b.b2b
        self.combo = b.combo
        pass 

        # this is to seed self.np_random
        #super().reset(seed=seed)

        # TODO: implementation
        
        #return observation, info

    def step(self, action):
        #take move based on move suggested by rl agent
        
        pass
    
        #return observation, reward, terminated (bool), False (bool), info (dict: auxiliary diagnostic info)
