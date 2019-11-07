from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten, Dense
from keras import backend as K


class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be "channels last"
        model = Sequential()  # do not forget the parentheses
        input_shape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        # define teh first (and only) CONV => RELU layer
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
