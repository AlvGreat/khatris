import gymnasium as gym
from gymnasium import spaces
from pylibtetris.pylibtetris import *
import random

from env_helpers import *

PIECES = ['I', 'O', 'T', 'L', 'J', 'S', 'Z']
ROTATION_DICT = {
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3,
}
TSPIN_DICT = {
    'None': 0,
    'Mini': 1,
    'Full': 2,
}

# documentation: https://gymnasium.farama.org/api/env/
class KhatrisEnv(gym.Env):
    def __init__(self):
        # observation space contains multiple variables
        # board: all possible Tetris boards (40x10) as 2d list of bits
        # hold: the current hold piece
        # queue: list of upcoming pieces
        # combo: how many lines have been cleared consecutively
        # b2b: back-to-back status (last clear was a T-spin or Tetris) 
        s = {
            'hold': spaces.Discrete(7),
            'queue': spaces.MultiDiscrete([7 for _ in range(6)]),
            'combo': spaces.Discrete(1000),
            'b2b': spaces.Discrete(2),
            'board': spaces.MultiBinary([40, 10]),
            'piece': spaces.Discrete(7),
        }
        self.observation_space = gym.spaces.Dict(s)
        self.action_space = spaces.Discrete(190)

        b = new_board_with_queue()
        self.pyboard = b
        self.board = b.field
        self.hold = b.hold
        self.queue = b.next_pieces
        self.b2b = b.b2b
        self.combo = b.combo
        self.piece = self.queue[0]

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
        super().reset(seed=seed)
        b = new_board_with_queue()
        self.pyboard = b
        self.board = b.field
        self.hold = b.hold
        self.queue = b.next_pieces
        self.b2b = b.b2b
        self.combo = b.combo
        self.piece = self.queue[0]

        observation = {
            'hold': self.hold,
            'queue': self.queue,
            'combo': self.combo,
            'b2b': self.b2b,
            'board': self.board,
            'piece': self.piece,
        } 
        info = {}
        return observation, info
        #return observation, info

    def check_valid_action(self, action):
        action_list = find_moves_py(self.board, self.piece, 0, 5, 19, 0, 0) 
        if len(action_list) == 0:
            return True
        if action > len(action_list) - 1:
            return False
        else:
            return True
         
    def step(self, action):
        #take move based on move suggested by rl agent
        action_list = find_moves_py(self.board, self.piece, 0, 5, 19, 0, 0)
        if action_list[0].x == 5 and action_list[0].y == 19:
            observation = {
                'hold': self.hold,
                'queue': self.queue,
                'combo': self.combo,
                'b2b': self.b2b,
                'board': self.board,
                'piece': self.piece,
            }
            reward = -1000
            terminated = True
            info = {}
            return observation, reward, terminated, info
        else:
            action_taken = action_list[action]
            new_board, lock_res = get_placement_res(self.pyboard, self.piece, ROTATION_DICT[action_taken.rotation_state], action_taken.x, action_taken.y, TSPIN_DICT[action_taken.tspin])
            self.pyboard = new_board
            self.board = new_board.field
            self.hold = new_board.hold
            self.queue = new_board.next_pieces
            self.combo = new_board.combo
            self.b2b = new_board.b2b
            self.piece = new_board.next_pieces[0]
            
            isb2b = 1 if lock_res.b2b else 0
            isPerfectClear = 10 if lock_res.perfect_clear else 0
            reward = lock_res.garbage_sent + lock_res.combo + isb2b + isPerfectClear

            observation = {
                'hold': self.hold,
                'queue': self.queue,
                'combo': self.combo,
                'b2b': self.b2b,
                'board': self.board,
                'piece': self.piece,
            }

            terminated = False
            info = {}
            return observation, reward, terminated, info
    
        #return observation, reward, terminated (bool), False (bool), info (dict: auxiliary diagnostic info)

    def render(self):
        print(self.pyboard)