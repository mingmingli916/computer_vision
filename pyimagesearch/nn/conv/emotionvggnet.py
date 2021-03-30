from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


def EmotionVGGNet(width, height, depth, classes):
    model = Sequential()
    input_shape = (height, width, depth)
    chan_dim = -1  # channel dimension

    # if 'channel first', update the input shape and channels dimension
    if K.image_data_format() == 'channels_first':
        input_shape = (depth, height, width)
        chan_dim = 1

    # see emotion_vgg_net.png
    # block1
    print()
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     padding='same',
                     # He et al. initialization tends to work better for the VGG family of networks
                     kernel_initializer='he_normal',
                     input_shape=input_shape))
    model.add(ELU())
    model.add(BatchNormalization(axis=chan_dim))
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     padding='same',
                     kernel_initializer='he_normal'))
    model.add(ELU())
    model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    # block2
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     kernel_initializer='he_normal',
                     padding='same'))
    model.add(ELU())
    model.add(BatchNormalization(axis=chan_dim))
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     kernel_initializer='he_normal',
                     padding='same'))
    model.add(ELU())
    model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=.25))

    # block3
    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     kernel_initializer='he_normal',
                     padding='same'))
    model.add(ELU())
    model.add(BatchNormalization(axis=chan_dim))
    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     kernel_initializer='he_normal',
                     padding='same'))
    model.add(ELU())
    model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    # block4, FC layers
    model.add(Flatten())
    model.add(Dense(units=64,
                    kernel_initializer='he_normal'))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))

    model.add(Dense(units=64,
                    kernel_initializer='he_normal'))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))

    model.add(Dense(units=classes,
                    kernel_initializer='he_normal'))
    model.add(Activation(activation='softmax'))

    return model
