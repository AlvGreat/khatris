import numpy as np
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
            'board': spaces.MultiBinary([40, 10]),
            'hold': spaces.Discrete(7),
            'combo': spaces.Discrete(1000),
            'b2b': spaces.Discrete(2),
            'queue': spaces.MultiDiscrete([7 for _ in range(6)]),
        }
        self.total_reward = 0
        self.observation_space = gym.spaces.Dict(s)
        self.action_space = spaces.Discrete(400)

        b = new_board_with_queue()
        self.pyboard = b
        self.spawn_point = (5, 19)  # fixed: where the tetris piece spawns


    # translates the environment’s state
    def _get_obs(self):
        flat_board = np.array(self.pyboard.field).reshape(-1)
        hold = self.pyboard.get_hold_int()
        combo = self.pyboard.combo
        b2b = self.pyboard.b2b
        queue = np.array(self.pyboard.get_next_pieces_int())
        return np.concatenate([flat_board, hold, combo, b2b, queue], axis=None)
    
    def test_obs(self):
        return self._get_obs()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.total_reward = 0
        
        # create a new board
        b = new_board_with_queue()
        self.pyboard = b

        observation = self._get_obs()
        info = {}
        return observation, info

    def get_action_list(self, separated=False):
        # the hold should never be empty
        if self.pyboard.hold == ' ':
            print("Critical Error: Hold is empty")

        # get all possible placements from both current piece and hold piece
        cur_piece_actions = find_moves_py(self.pyboard.field, self.pyboard.next_pieces[0], 0, self.spawn_point[0], self.spawn_point[1], 0, 0) 
        hold_piece_actions = find_moves_py(self.pyboard.field, self.pyboard.hold, 0, self.spawn_point[0], self.spawn_point[1], 0, 0) 

        if separated: 
            return cur_piece_actions, hold_piece_actions
        return cur_piece_actions + hold_piece_actions

    def check_valid_action(self, action):
        action_list = self.get_action_list()

        if len(action_list) == 0:
            return True
        
        # if action is out of bounds
        if action > len(action_list) - 1:
            return False
        else:
            return True
         
    def step(self, action):
        cur_piece_actions, hold_piece_actions = self.get_action_list(separated=True)

        # if there is only 1 action left at the spawn, we've topped out
        if len(cur_piece_actions) == 1 and cur_piece_actions[0].x == self.spawn_point[0] and cur_piece_actions[0].y == self.spawn_point[1] and \
                len(hold_piece_actions) == 1 and hold_piece_actions[0].x == self.spawn_point[0] and hold_piece_actions[0].y == self.spawn_point[1]:
            observation = self._get_obs()
            reward = self.total_reward
            terminated = True
            truncated = False
            info = {}
            return observation, reward, terminated, truncated, info
        else:
            is_current_piece = True
            if action < len(cur_piece_actions):
                action_taken = cur_piece_actions[action]
            else:
                is_current_piece = False
                action_taken = hold_piece_actions[action - len(cur_piece_actions)]
            
            new_board, lock_res = get_placement_res(self.pyboard, is_current_piece, ROTATION_DICT[action_taken.rotation_state], action_taken.x, action_taken.y, TSPIN_DICT[action_taken.tspin])
            self.pyboard = new_board
            
            # isb2b = 1 if lock_res.b2b else 0
            # isPerfectClear = 10 if lock_res.perfect_clear else 0
            # reward = lock_res.garbage_sent + lock_res.combo + isb2b + isPerfectClear
            reward = lock_res.garbage_sent
            self.total_reward += reward

            observation = self._get_obs()
            terminated = False
            truncated = False
            info = {}
            return observation, reward, terminated, truncated, info
    
        #return observation, reward, terminated (bool), False (bool), info (dict: auxiliary diagnostic info)

    def render(self):
        print(self.pyboard)
