import os
import argparse

import numpy as np
import tensorflow as tf
from gymnasium.wrappers import FlattenObservation

from pylibtetris.pylibtetris import *

from rl_env_flat import KhatrisEnv
from models.dqn import DQNAgent
from models.convoluted_dqn import ConvolutedDQNAgent

# Enabling GPU usage
# gpu_devices = tf.config.experimental.list_physical_devices("GPU")
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)

# Parameters for saving output files
RESULTS_FOLDER = 'train_results' 

# Parameters for training the agent
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

def write_number_to_file(number, filename):
    with open(filename, 'w') as file:
        file.write(str(number))

def read_number_from_file(filename):
    with open(filename, 'r') as file:
        number = int(file.read())
        return number

def reshape_for_cdqn(state):
    board = np.reshape(state[:BOARD_SIZE], (1, 40, 10, 1))
    other_stats = np.reshape(state[BOARD_SIZE:], (1, 9))
    return [board, other_stats]

def train(episodes):
    """
    Trains an agent and saves its results to files inside `train_results`
    The file `train_iter` stores the number of times the training process
    has been run. File names of saved trained models are based on the number.

    Note: If a trained model exists, it will load weights from the 
    latest saved model and continue training from there.
    """

    iter = read_number_from_file(f'{RESULTS_FOLDER}/train_iter.txt')
    saved_model_name = f'{RESULTS_FOLDER}/tetris-cdqn-{iter}.h5'
    if os.path.exists(saved_model_name):
        agent.load(saved_model_name)

    score_history = []
    for e in range(episodes):
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
            next_state = reshape_for_cdqn(next_state)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                score_history.append(score)
                print("episode: {}/{}, score: {}".format(e+1, episodes, score))
                break
        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)
        
    # Save the trained model
    new_save_name = f'{RESULTS_FOLDER}/tetris-cdqn-{iter+1}.h5'
    agent.save(new_save_name)

    # Also save its training history so that we can view it later
    score_history = np.array(score_history)
    np.savetxt(f'{RESULTS_FOLDER}/score-history-{iter+1}.csv', score_history, delimiter=',')

    # Update the number
    write_number_to_file(iter+1, f'{RESULTS_FOLDER}/train_iter.txt')


def main():
    """
    Run with: "python main.py <episodes>"
    """
    parser = argparse.ArgumentParser(description='Train the Khatris RL agent')
    parser.add_argument('episodes', type=int,
                        help='Number of episodes to train for')
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("Episode count must be > 0")

    train(args.episodes)
            

if __name__ == '__main__':
    main()
