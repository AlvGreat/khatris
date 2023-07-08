import sys
sys.path.append('../')

from python.models.convoluted_dqn import init_model
import tensorflow as tf

from keras.utils import plot_model
import numpy as np

def test_cdqn():
    # This is the real size used as well
    TEST_ACTION_SIZE = 400

    model = init_model(TEST_ACTION_SIZE)
    model.compile(
        loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )

    # These are the correct input shapes!!
    test_input1 = np.random.rand(1, 40, 10, 1)
    test_input2 = np.random.rand(1, 9)

    model.predict([test_input1, test_input2])

    # model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)

test_cdqn()
