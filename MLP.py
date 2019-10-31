# This imports the sequential model, the layers,
# the SGD optimizer, the regularizers from keras.
# This comes from Reference 1 in Referenes.
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import regularizers
import numpy as np

def build_model(drop_rate, l2_factor, hidden_act, out_act,
                n_hidden, in_dense, hid_dense, x):
    
    # This defines the model as a sequential model.
    # This comes from References 1 & 2 in References.
    model = Sequential()

    # This is the input layer.
    # This comes from References 1 & 3 in References.
    model.add(Dense(in_dense, activation = hidden_act,
        kernel_regularizer = regularizers.l2(l2_factor),
        input_dim = np.size(x,1)))
    model.add(Dropout(drop_rate))

    # This creates the hidden layers.
    # This comes from Reference 2 in References.
    for i in range(0,100):
        model.add(Dense(hid_dense,
                activation=hidden_act,
                kernel_regularizer = regularizers.l2(l2_factor)))
        model.add(Dropout(drop_rate))

    # This creates the output layer.
    # This comes from Reference 1 in References.
    model.add(Dense(1, activation=out_act))
    # This returns the model.
    return model
    

# References
# 1. https://keras.io/getting-started/sequential-model-guide/
# 2. https://datascience.stackexchange.com/questions/19407/keras-built-in-multi-layer-shortcut
# 3. https://docs.scipy.org/doc/numpy/reference/generated/numpy.ma.size.html