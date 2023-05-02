import random
import numpy as np
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
LAYER_SIZES=((512, 256, 128, 64, 10), 
             (512, 256, 128, 64, 20), 
             (512, 256, 128, 64, 4), 
             (512, 256, 128, 64, 2))

ROTATION_DICT = {
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3,
}
    
class MultiHeadAgent:
    def __init__(self, state_size, layer_sizes=LAYER_SIZES):
        self.state_size = state_size
        self.layer_sizes = layer_sizes
        self.memory = []
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.xCoordHead, self.yCoordHead, self.rotationHead, self.blockHead = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # Create networks to predict x, y, rotation, piece (10, 20, 4, 2)
        heads = []
        for head_layers in self.layer_sizes:
            head = tf.keras.models.Sequential()
            for idx, layer in enumerate(head_layers):
                if idx == 0:
                    head.add(tf.keras.layers.Dense(layer, input_dim=self.state_size, activation='relu'))
                elif idx == len(head_layers) - 1:
                    head.add(tf.keras.layers.Dense(layer, activation='softmax'))
                else:
                    head.add(tf.keras.layers.Denes(layer, activation='relu'))
            head.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
            heads.append(head)
        return heads
                             
    def remember(self, state, action, valid_actions, reward, next_state, done):
        self.memory.append((state, action, valid_actions, reward, next_state, done))
    
    def _get_action_masks(self, valid_actions):
        x_values = []
        y_values = []
        r_values = []
        p_values = []
        # keep track of where the pieces are located (starting index of each piece)
        new_piece_idx = []
        for idx, action in enumerate(valid_actions):
            if action.x not in x_values: x_values.append(action.x)
            if action.y not in y_values: y_values.append(action.y)
            if action.rotation_state not in r_values: r_values.append(action.rotation_state)
            if action.piece_type not in p_values:
                p_values.append(action.piece_type)
                new_piece_idx.append(idx)
        
        x_mask = np.zeros(10, dtype=int)
        x_mask[x_values] = 1
        x_mask[x_mask == 0] = -np.inf
                    
        y_mask = np.zeros(20, dtype=int)
        y_mask[y_values] = 1
        y_mask[y_mask == 0] = -np.inf

        for idx, value in enumerate(r_values):
            r_values[idx] = ROTATION_DICT[value]
            
        r_mask = np.zeros(4, dtype=int)
        r_mask[r_values] = 1
        r_mask[r_mask==0] = -np.inf

        p_mask = np.ones(2, dtype=int)
        if (new_piece_idx[1] - new_piece_idx[0] == 1):
            p_mask[0] = -np.inf
        if (len(valid_actions) - 1 == new_piece_idx[1]):
            p_mask[1] = -np.inf

        return (x_mask, x_values), (y_mask, y_values), (r_mask, r_values), (p_mask, p_values)
        

        
        
        

    def act(self, state, valid_actions):
        x_mask, y_mask, r_mask, p_mask = self._get_action_masks(valid_actions)
                
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
                target = (reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # names should be in x, y, rotation, piece order
    def load(self, names):
        heads = []
        for idx, head_layers in enumerate(self.layer_sizes):
            head = tf.keras.models.Sequential()
            for idx, layer in enumerate(head_layers):
                if idx == 0:
                    head.add(tf.keras.layers.Dense(layer, input_dim=self.state_size, activation='relu'))
                elif idx == len(head_layers) - 1:
                    head.add(tf.keras.layers.Dense(layer, activation='softmax')
                else:
                    head.add(tf.keras.layers.Denes(layer, activation='relu')
            head.load_weights(names[idx])
            head.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
            heads.append(head)
        self.xCoordHead, self.yCoordHead, self.rotationHead, self.blockHead = heads 

    def save(self, name):
        self.model.save_weights(name)
