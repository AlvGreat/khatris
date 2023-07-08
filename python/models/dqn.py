import random
import numpy as np
import tensorflow as tf


class DQNAgent:
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
        # Basic neural network
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Dense(1024, input_dim=self.state_size, activation="relu")
        )
        model.add(tf.keras.layers.Dense(1024, activation="relu"))
        model.add(tf.keras.layers.Dense(512, activation="relu"))
        model.add(tf.keras.layers.Dense(self.action_size, activation="softmax"))
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
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Dense(1024, input_dim=self.state_size, activation="relu")
        )
        model.add(tf.keras.layers.Dense(1024, activation="relu"))
        model.add(tf.keras.layers.Dense(512, activation="relu"))
        model.add(tf.keras.layers.Dense(self.action_size, activation="softmax"))

        model.load_weights(name)

        model.compile(
            loss="mse", optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate)
        )
        self.model = model

    def save(self, name):
        self.model.save_weights(name)
