import numpy as np
import tensorflow as tf
from gymnasium.wrappers import FlattenObservation

from pylibtetris.pylibtetris import *

from rl_env import KhatrisEnv
from models.dqn import DQNAgent
from models.convoluted_dqn import ConvolutedDQNAgent

# Enabling GPU usage
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Parameters for training the agent
N_EPISODES = 500
BATCH_SIZE = 25

# Initialize environment and flatten the observations from it
env = KhatrisEnv()
env = FlattenObservation(env)

# Set the random seed for reproducibility
seed = 123
env.reset(seed=seed)
np.random.seed(seed)

# Define the state and action spaces
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Instantiate the DQN agent
agent = DQNAgent(state_size, action_size)
# agent = ConvolutedDQNAgent(state_size, action_size)


for e in range(N_EPISODES):
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
