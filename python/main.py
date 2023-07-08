import numpy as np
import tensorflow as tf
from gymnasium.wrappers import FlattenObservation

from pylibtetris.pylibtetris import *

from rl_env_flat import KhatrisEnv
from models.dqn import DQNAgent
from models.convoluted_dqn import ConvolutedDQNAgent

# Enabling GPU usage
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Parameters for training the agent
N_EPISODES = 500
BATCH_SIZE = 25
BOARD_SIZE = 40*10

# Initialize environment and flatten the observations from it
env = KhatrisEnv()

# Set the random seed for reproducibility
seed = 123
env.reset(seed=seed)
np.random.seed(seed)

# Define the state and action spaces
state_size = env.get_obs().shape[0]
action_size = env.action_space.n

# Instantiate the DQN agent
# agent = DQNAgent(state_size, action_size)
agent = ConvolutedDQNAgent(state_size, action_size)


def reshape_for_cdqn(state):
    board = np.reshape(state[:BOARD_SIZE], (1, 40, 10, 1))
    other_stats = np.reshape(state[BOARD_SIZE:], (1, 9))
    return [board, other_stats]

for e in range(N_EPISODES):
    state, info = env.reset(seed=seed)
    state = reshape_for_cdqn(state)
    done = False
    score = 0
    while not done:
        all_actions = env.get_action_list()
        
        # Length of action array represents the total number of actions possible
        max_action = len(all_actions)
        
        # Pass in the maximum number of actions for action masking
        action = agent.act(state, max_action)
        next_state, reward, done, truncated, info = env.step(action)
        
        # fix this later pls 
        next_state = reshape_for_cdqn(next_state)

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            print("episode: {}/{}, score: {}".format(e, N_EPISODES, score))
            break
    if len(agent.memory) > BATCH_SIZE:
        agent.replay(BATCH_SIZE)

# Save the trained model
agent.save("tetris-dqn.h5")

# Test the trained agent
n_test_episodes = 10

for e in range(n_test_episodes):
    state = env.reset()
