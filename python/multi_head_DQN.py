import random
import numpy as np
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
LAYER_SIZES=((512, 256, 128, 64, 10), 
             (512, 256, 128, 64, 25), 
             (512, 256, 128, 64, 4), 
             (512, 256, 128, 64, 2))

ROTATION_DICT = {
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3,
}

ROTATION_INDICES = {
    0: 'North',
    1: 'South',
    2: 'East',
    3: 'West',
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
        # Create networks to predict x, y, rotation, piece (10, 25, 4, 2)
        heads = []
        for head_layers in self.layer_sizes:
            head = tf.keras.models.Sequential()
            for idx, layer in enumerate(head_layers):
                if idx == 0:
                    head.add(tf.keras.layers.Dense(layer, input_dim=self.state_size, activation='relu'))
                elif idx == len(head_layers) - 1:
                    head.add(tf.keras.layers.Dense(layer, activation='softmax'))
                else:
                    head.add(tf.keras.layers.Dense(layer, activation='relu'))
            head.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
            heads.append(head)
        return heads
                             
    def remember(self, state, action, valid_actions, reward, next_state, done):
        self.memory.append((state, action, valid_actions, reward, next_state, done))
    
    def _get_action_masks(self, valid_actions, piece_chosen):
        
        # x coordinates and indices of actions
        x_values = []
        # y coordinates and indices of actions
        y_values = []
        # rotation values Ex: 'North'
        r_values = []
        # rotation action indices
        r_indices = []
        # piece values Ex: 'O'
        p_values = []
        # piece action indices
        p_indices = []
        # keep track of where the pieces are located (starting index of each piece in p_values)
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
        x_mask[x_mask == 0] = np.iinfo(np.int64).min
                    
        y_mask = np.zeros(25, dtype=int)
        y_mask[y_values] = 1
        y_mask[y_mask == 0] = np.iinfo(np.int64).min

        r_indices = [i for i in r_values]
        for idx, value in enumerate(r_values):
            r_indices[idx] = ROTATION_DICT[value]
            
        r_mask = np.zeros(4, dtype=int)
        r_mask[r_indices] = 1
        r_mask[r_mask==0] = np.iinfo(np.int64).min

        p_mask = np.ones(2, dtype=int)
        if piece_chosen:
            p_mask = None
        else:
            if (len(new_piece_idx) == 2) :
                if (new_piece_idx[1] - new_piece_idx[0] == 1):
                    p_mask[0] = np.iinfo(np.int64).min
                else:
                    p_indices.append(0)
                if (len(valid_actions) - 1 == new_piece_idx[1]):
                    p_mask[1] = np.iinfo(np.int64).min
                else:
                    p_indices.append(1)

        return (x_mask, x_values), (y_mask, y_values), (r_mask, r_values, r_indices), (p_mask, p_values, p_indices)
    
    
    def _get_best_valid_action(self, x_act_values, y_act_values, r_act_values, p_act_values, p_values, valid_actions):
        
        best_x = None
        best_y = None
        best_p = None
        best_r = None

        best_p = np.argmax(p_act_values)
        for idx, action in enumerate(valid_actions):
            if action.piece_type != p_values[best_p]:
                del valid_actions[idx]
        
        x_mask, _, _, _ = self._get_action_masks(valid_actions, True)
        x_act_values = x_act_values*x_mask[0]
        best_x = np.argmax(x_act_values)
        for idx, action in enumerate(valid_actions):
            if action.x != best_x:
                del valid_actions[idx]
        
        _, y_mask, _, _ = self._get_action_masks(valid_actions, True)
        y_act_values = y_act_values*y_mask[0]
        best_y = np.argmax(y_act_values)
        for idx, action in enumerate(valid_actions):
            if action.y != best_y:
                del valid_actions[idx]
        
        _, _, r_mask, _ = self._get_action_masks(valid_actions, True)
        r_act_values = r_act_values*r_mask[0]
        best_r = np.argmax(r_act_values)

        for idx, action in enumerate(valid_actions):
            if action.x == best_x and action.y == best_y and action.rotation_state == ROTATION_INDICES[best_r] and action.piece_type == p_values[best_p]:
                return idx
        
    

    def act(self, state, valid_actions):
        x_mask, y_mask, r_mask, p_mask = self._get_action_masks(valid_actions, False)
        if np.random.rand() <= self.epsilon:
            return random.randint(0, len(valid_actions) - 1)
        x_act_values = self.xCoordHead.predict(state, verbose=0)[0] * x_mask[0]
        y_act_values = self.yCoordHead.predict(state, verbose=0)[0] * y_mask[0]
        r_act_values = self.rotationHead.predict(state, verbose=0)[0] * r_mask[0]
        p_act_values = self.blockHead.predict(state, verbose=0)[0] * p_mask[0]
        return self._get_best_valid_action(x_act_values, y_act_values, r_act_values, p_act_values, p_mask[1], valid_actions)
        

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, valid_actions, reward, next_state, done in minibatch:
            x_mask, y_mask, r_mask, p_mask = self._get_action_masks(valid_actions, False)
            chosen_action = valid_actions[action]
            best_x, best_y, best_r, best_p = chosen_action.x , chosen_action.y, ROTATION_DICT[chosen_action.rotation_state], p_mask[1].index(chosen_action.piece_type) 
            for (model, mask, best_action) in [(self.xCoordHead, x_mask, best_x), (self.yCoordHead, y_mask, best_y), (self.rotationHead, r_mask, best_r), (self.blockHead, p_mask, best_p)]:
                target = reward
                if not done:
                    target = (reward + self.gamma * np.amax(model.predict(next_state, verbose=0)[0] * mask[0]))
                target_f = model.predict(state, verbose=0)
                target_f[0][best_action] = target
                model.fit(state, target_f, epochs=1, verbose=0)
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
                    head.add(tf.keras.layers.Dense(layer, activation='softmax'))
                else:
                    head.add(tf.keras.layers.Dense(layer, activation='relu'))
            head.load_weights(names[idx])
            head.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
            heads.append(head)
        self.xCoordHead, self.yCoordHead, self.rotationHead, self.blockHead = heads 

    def save(self, names):
        self.xCoordHead.save_weights(names[0])
        self.yCoordHead.save_weights(names[1])
        self.rotationHead.save_weights(names[2])
        self.blockHead.save_weights(names[3])
