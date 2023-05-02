from pylibtetris.pylibtetris import *
from rl_env import KhatrisEnv
from DQN import DQNAgent
from gymnasium.wrappers import FlattenObservation
import numpy as np
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
    
# import os
# os.environ["RUST_BACKTRACE"] = "1"

#print(pylibtetris.__)
# help(pylibtetris)
#print(pylibtetris.__dir__())

# y = PyLockResult(1, True, True, 0, 1)
# x = PiecePlacement(["CCW"], 1, "T", "North", 1, 1, "T")

# print(x)
# print(y)


# blank_board = [[False for _ in range(10)] for _ in range(40)]

# test_board = [4*[True] + 6*[False]]
# test_board += [[False for _ in range(10)] for _ in range(39)]

# test_pyboard = PyBoard(test_board, "T", True, 5, ["S", "Z", "L", "J", "I", "O"], ["S", "Z", "L", "J", "I", "O", "T"])
# print(test_pyboard)

# # fn find_moves_py(board: [[bool; 10]; 40], piece: char, rotation_state: u8, x: i32, y: i32, t_spin_status:i8, mode: u8)

# piece_placements = find_moves_py(test_board, 'S', 0, 5, 19, 0, 0)
# print(piece_placements[0])

# # fn get_placement_res(board_arr: [[bool; 10]; 40], hold: char, b2b: bool, combo: u32, next_pieces: [char; 6], bag: [char; 7],
# #                      piece: char, rotation_state: u8, x: i32, y: i32, t_spin_status:i8) -> PyResult<(PyBoard, PyLockResult)> {

# new_board, lock_res = get_placement_res(test_pyboard, 'S', 1, 8, 1, 0)
# print(new_board, lock_res)

env = KhatrisEnv()
env = FlattenObservation(env)
# observation, info = env.reset(seed=42)
# for _ in range(100):
#     found_action = False
#     while(not found_action):
#         action = env.action_space.sample()  # this is where you would insert your policy
#         if (env.check_valid_action(action)):
#             found_action = True
#             observation, reward, terminated, info = env.step(action)
#             env.render()
#     if terminated:
#         observation, info = env.reset()
# env.close()
# Set the random seed for reproducibility
seed = 123
env.reset(seed=seed)
np.random.seed(seed)

# Define the state and action spaces
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Instantiate the DQN agent
agent = DQNAgent(state_size, action_size)
#agent.load("tetris-dqn.h5")

# Train the agent
n_episodes = 500
batch_size = 25

for e in range(n_episodes):
    state, info = env.reset(seed=seed)
    state = np.reshape(state, [1, state_size])
    done = False
    score = 0
    while not done:
        all_actions = env.get_action_list()
        # length of action array represents the total number of actions possible
        max_action = len(all_actions)
        # we pass in the maximum number of actions for action masking
        action = agent.act(state, max_action)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            print("episode: {}/{}, score: {}".format(e, n_episodes, score))
            if e % 200 == 0:
                agent.save(f'tetris-dqp-{e}.h5')
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

# Save the trained model
agent.save("tetris-dqn.h5")

# Test the trained agent
n_test_episodes = 10

for e in range(n_test_episodes):
    state = env.reset()
