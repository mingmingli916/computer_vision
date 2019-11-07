from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras import backend as K


def conv_module(x, K, kx, ky, stride, chan_dim, padding='same'):
    # define a CONV => RELU => BN pattern
    # Functional API
    # Each layer instance in a Model is callable on a tensor and also returns a tensor.
    # Therefore, we can supply the inputs to a given layer by calling it as a function
    # once the object is instantiated
    x = Conv2D(K, (kx, ky), strides=stride, padding=padding)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=chan_dim)(x)

    return x


def inception_module(x, num_K_1x1, num_K_3x3, chan_dim):
    conv_1x1 = conv_module(x, num_K_1x1, 1, 1, (1, 1), chan_dim)
    conv_3x3 = conv_module(x, num_K_3x3, 3, 3, (1, 1), chan_dim)
    x = concatenate([conv_1x1, conv_3x3], axis=chan_dim)

    return x


def downample_module(x, K, chan_dim):
    conv_3x3 = conv_module(x, K, 3, 3, (2, 2), chan_dim, padding='valid')
    pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate([conv_3x3, pool], axis=chan_dim)

    return x


def MiniGoogLeNet(width, height, depth, classes):
    input_shape = height, width, depth
    chan_dim = -1

    if K.image_data_format() == 'channels_first':
        input_shape = depth, height, width
        chan_dim = 1

    # define the model input and first CONV module
    inputs = Input(shape=input_shape)
    x = conv_module(inputs, 96, 3, 3, (1, 1), chan_dim)

    # two inception modules followed by a downsample module
    x = inception_module(x, 32, 32, chan_dim)
    x = inception_module(x, 32, 48, chan_dim)
    x = downample_module(x, 80, chan_dim)

    # for inception modules followed by a downsample module
    x = inception_module(x, 112, 48, chan_dim)
    x = inception_module(x, 96, 64, chan_dim)
    x = inception_module(x, 80, 80, chan_dim)
    x = inception_module(x, 48, 96, chan_dim)
    x = downample_module(x, 96, chan_dim)

    # two inception modules followed by global POOL and dropout
    x = inception_module(x, 176, 160, chan_dim)
    x = inception_module(x, 176, 160, chan_dim)
    x = AveragePooling2D((7, 7))(x)
    x = Dropout(.5)(x)

    # softmax classifier
    x = Flatten()(x)
    x = Dense(classes)(x)
    x = Activation('softmax')(x)

    # create model
    model = Model(inputs, x, name='mini googlenet')

    return model
