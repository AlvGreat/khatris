from pylibtetris.pylibtetris import *
from rl_env_flat import KhatrisEnv
from multi_head_DQN import MultiHeadAgent
from gymnasium.wrappers import FlattenObservation
import numpy as np
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
    
env = KhatrisEnv()

seed = 123
env.reset(seed=seed)
np.random.seed(seed)

# Define the state and action spaces
state_size = env.get_obs().shape[0]

# Instantiate the DQN agent
agent = MultiHeadAgent(state_size)
#agent.load("tetris-dqn.h5")

# Train the agent
n_episodes = 2000
batch_size = 25

for e in range(n_episodes):
    state, info = env.reset(seed=seed)
    state = np.reshape(state, [1, state_size])
    done = False
    score = 0
    while not done:
        all_actions = env.get_action_list()
        # we pass in the list of all possible actions
        action = agent.act(state, all_actions)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, all_actions, reward, next_state, done)
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
# agent.save("tetris-dqn.h5")

# Test the trained agent
n_test_episodes = 10

for e in range(n_test_episodes):
    state = env.reset()
