# This imports the sequential model, the layers,
# the SGD optimizer, the regularizers from keras.
# This comes from Reference 1 in Referenes.
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import regularizers
import numpy as np

def build_model(dropout, l2_factor, hidden_act, out_act,
                n_hidden, x, y):
    hidden_layer_sizes = list(range(1,101,1))
    hidden_layer_sizes = hidden_layer_sizes.sort(
                        reverse=True)
    # This defines the model as a sequential model.
    # This comes from References 1 & 2 in References.
    model = Sequential()

    # This is the input layer.
    # This comes from References 1 & 3 in References.
    model.add(Dense(hidden_layer_sizes[0], activation = hidden_act,
        kernel_regularizer = regularizers.l2(l2_factor),
        input_dim = np.size(x,1)))
    model.add(Dropout(dropout))





# References
# 1. https://keras.io/getting-started/sequential-model-guide/
# 2. https://datascience.stackexchange.com/questions/19407/keras-built-in-multi-layer-shortcut
# 3. https://docs.scipy.org/doc/numpy/reference/generated/numpy.ma.size.html