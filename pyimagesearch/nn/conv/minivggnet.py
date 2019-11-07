from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras import backend as K


class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # determine the input shape and channels dimension
        input_shape = (height, width, depth)
        chan_dim = -1
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            chan_dim = 1

        # initialize the model
        model = Sequential()

        # first CONV -> RELU -> CONV -> RELU -> POOL layer set
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # a  node from the POOL layer will randomly disconnect
        # from the next layer with a probability of 25% during training.
        model.add(Dropout(.25))

        # second CONV -> RELU -> CONV -> RELU -> POOL layer set
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.25))

        # first (and only) set of FC -> RELU layer
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Dropout(.25))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
