import random
import numpy as np
import tensorflow as tf

from keras.layers import (
    Input,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    AveragePooling2D,
    Concatenate,
)
from keras.models import Model
from keras.optimizers import Adam

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


class ConvolutedDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Create and compile model
        model = init_model(self.action_size)
        model.compile(
            loss="mse", optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, max_action):
        if np.random.rand() <= self.epsilon:
            return random.randrange(max_action)
        act_values = self.model.predict(state, verbose=0)

        best_actions = np.argsort(act_values[0])[::-1]
        for action in best_actions:
            if action < max_action:
                return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state, verbose=0)[0]
                )
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        # Load saved weights before compiling model
        model = init_model(self.action_size)
        model.load_weights(name)
        model.compile(
            loss="mse", optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate)
        )

        # Update model
        self.model = model

    def save(self, name):
        self.model.save_weights(name)


def init_model(action_size):
    BOARD_SIZE = (40, 10)
    # hold, combo, b2b, 6 pieces in queue
    OTHER_VAR_SIZE = 9

    # Create input layers for the board and other variables separately
    input1 = Input(shape=BOARD_SIZE + (1,))
    input2 = Input(shape=(OTHER_VAR_SIZE,))

    # CNN layers for the board
    x1 = Conv2D(32, (3, 3), activation="relu", padding="same")(input1)
    x1 = AveragePooling2D()(x1)
    x1 = Dropout(0.2)(x1)

    x1 = Conv2D(32, (3, 3), activation="relu", padding="same")(x1)
    x1 = AveragePooling2D()(x1)
    x1 = Dropout(0.2)(x1)

    x1 = Conv2D(64, (3, 3), activation="relu", padding="same")(x1)
    x1 = AveragePooling2D()(x1)
    x1 = Dropout(0.2)(x1)

    x1 = Conv2D(64, (3, 3), activation="relu", padding="same")(x1)
    x1 = Dropout(0.2)(x1)

    # Convert convolution layer back into Dense
    x1 = Flatten()(x1)
    x1 = Dense(128, activation="relu")(x1)

    # Combine board after CNN with rest of observation data
    merged = Concatenate(axis=1)([x1, input2])

    # Now, we use all observation data in fully-connected layers
    merged = Dense(96, activation="relu")(merged)
    merged = Dense(48, activation="relu")(merged)
    merged = Dense(action_size, activation="softmax")(merged)

    # Create the final model
    model = Model(inputs=[input1, input2], outputs=merged)
    return model
