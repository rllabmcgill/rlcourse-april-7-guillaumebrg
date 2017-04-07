__author__ = 'Guillaume'

from keras.models import Model
from keras.layers import Input, Convolution2D, Activation, Flatten, Dense, \
    merge, RepeatVector, TimeDistributed, LSTM
from keras.optimizers import SGD, Adam
import keras.backend as K


def DQN_7x7(grid_shape, lr=0.1, adam=False):
    # Input node
    grid = Input(grid_shape)
    # Hidden layer
    x = Convolution2D(16, 3, 3, subsample=(2,2))(grid)
    x = Activation("relu")(x)
    x = Convolution2D(16, 3, 3)(x)
    x = Activation("relu")(x)
    # Output layer
    q = Convolution2D(4, 1, 1)(x)
    q = Flatten()(q)
    # Keras model
    neuralnet = Model(grid, q)
    # Compile
    #neuralnet.compile(Adam(lr=lr), loss="mse")
    if adam:
        neuralnet.compile(Adam(lr=lr), loss="mse")
    else:
        neuralnet.compile(SGD(lr=lr), loss="mse")
    return neuralnet


def logloss(y_true, y_pred, eps=10e-6):
    return K.sum(-y_true*K.log(y_pred+eps), axis=1)


def DPN_3x3(grid_shape, lr=0.001, adam=False):
    # Input node
    grid = Input(grid_shape)
    # Hidden layer
    #x1 = Convolution2D(16, 3, 3, subsample=(2,2))(grid)
    #x1 = Activation("relu")(x1)
    x1 = Convolution2D(16, 3, 3)(grid)
    x1 = Activation("relu")(x1)
    x1 = Flatten()(x1)
    pi = Dense(4, activation="softmax")(x1)
    #
    #x2 = Convolution2D(16, 3, 3, subsample=(2,2))(grid)
    #x2 = Activation("relu")(x2)
    x2 = Convolution2D(16, 3, 3)(grid)
    x2 = Activation("relu")(x2)
    x2 = Flatten()(x2)
    v = Dense(1)(x2)
    # Keras model
    neuralnet = Model(grid, [pi, v])
    # Compile
    #neuralnet.compile(Adam(lr=lr), loss="mse")
    if adam:
        neuralnet.compile(Adam(lr=lr), loss = [logloss, "mse"])
    else:
        neuralnet.compile(SGD(lr=lr), loss = [logloss, "mse"])
    return neuralnet


def DPN_7x7(grid_shape, lr=0.001, adam=False):
    # Input node
    grid = Input(grid_shape)
    # Hidden layer
    x1 = Convolution2D(16, 3, 3, subsample=(2,2))(grid)
    x1 = Activation("relu")(x1)
    x1 = Convolution2D(16, 3, 3)(x1)
    x1 = Activation("relu")(x1)
    x1 = Flatten()(x1)
    pi = Dense(4, activation="softmax")(x1)
    #
    x2 = Convolution2D(16, 3, 3, subsample=(2,2))(grid)
    x2 = Activation("relu")(x2)
    x2 = Convolution2D(16, 3, 3)(x2)
    x2 = Activation("relu")(x2)
    x2 = Flatten()(x2)
    v = Dense(1)(x2)
    # Keras model
    neuralnet = Model(grid, [pi, v])
    # Compile
    #neuralnet.compile(Adam(lr=lr), loss="mse")
    if adam:
        neuralnet.compile(Adam(lr=lr), loss = [logloss, "mse"])
    else:
        neuralnet.compile(SGD(lr=lr), loss = [logloss, "mse"])
    return neuralnet


def DPN_7x7_shared(grid_shape, lr=0.001, adam=False):
    # Input node
    grid = Input(grid_shape)
    # Hidden layer
    x1 = Convolution2D(16, 3, 3, subsample=(2,2))(grid)
    x1 = Activation("relu")(x1)
    x1 = Convolution2D(16, 3, 3)(x1)
    x1 = Activation("relu")(x1)
    x1 = Flatten()(x1)
    pi = Dense(4, activation="softmax")(x1)
    #
    v = Dense(1)(x1)
    # Keras model
    neuralnet = Model(grid, [pi, v])
    # Compile
    #neuralnet.compile(Adam(lr=lr), loss="mse")
    if adam:
        neuralnet.compile(Adam(lr=lr), loss = [logloss, "mse"])
    else:
        neuralnet.compile(SGD(lr=lr), loss = [logloss, "mse"])
    return neuralnet