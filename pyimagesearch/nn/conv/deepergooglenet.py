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
from keras.regularizers import l2


def conv_module(x, filters, k_size, stride, chan_dim, padding='same', reg=.0005, name=None):
    # initialize the CONV, BN, and RELU layer names
    conv_name, bn_name, act_name = None, None, None

    # if a layer name was supplied, prepend it
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
        act_name = name + '_act'

    # define a CONV => BN => RELU pattern
    x = Conv2D(filters, k_size, strides=stride, padding=padding, kernel_regularizer=l2(reg), name=conv_name)(x)
    x = BatchNormalization(axis=chan_dim, name=bn_name)(x)
    x = Activation('relu', name=act_name)(x)

    return x


def inception_model(x,
                    num_1x1, num_3x3_reduce, num_3x3,
                    num_5x5_reduce, num_5x5,
                    num_1x1_proj,
                    chan_dim, stage, reg=.0005):
    # define the first branch of the inception module
    # which consists of 1x1 convolutions
    first = conv_module(x, num_1x1, (1, 1), (1, 1), chan_dim, reg=reg, name=stage + '_first')

    # define the second branch of the inception module
    # which consists of 1x1 and 3x3 convolutions
    second = conv_module(x, num_3x3_reduce, (1, 1), (1, 1), chan_dim, reg=reg, name=stage + '_second1')
    second = conv_module(second, num_3x3, (3, 3), (1, 1), chan_dim, reg=reg, name=stage + '_second2')

    # define the third branch of the inception module
    # which are 1x1 and 5x5 convolutions
    third = conv_module(x, num_5x5_reduce, (1, 1), (1, 1), chan_dim, reg=reg, name=stage + '_third1')
    third = conv_module(third, num_5x5, (5, 5), (1, 1), chan_dim, reg=reg, name=stage + '_third2')

    # define the fourth branch of the inception module
    # which is the POOL projection
    fourth = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name=stage + '_pool')(x)
    fourth = conv_module(fourth, num_1x1_proj, (1, 1), (1, 1), chan_dim, reg=reg, name=stage + '_fourth')

    # concatenate across the channel dimension
    x = concatenate([first, second, third, fourth], axis=chan_dim, name=stage + '_mixed')

    return x


def DeeperGoogLeNet(width, height, depth, classes, reg=.0005):
    input_shape = height, width, depth
    chan_dim = -1

    if K.image_data_format() == 'channels_first':
        input_shape = depth, height, width
        chan_dim = 1

    # define the model input, followed be a sequence of
    # CONV => POOL => (CONV * 2) => POOL layers
    inputs = Input(shape=input_shape)
    x = conv_module(inputs, 64, (5, 5), (1, 1), chan_dim, reg=reg, name='block1')
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1')(x)
    x = conv_module(x, 64, (1, 1), (1, 1), chan_dim, reg=reg, name='block2')
    x = conv_module(x, 192, (3, 3), (1, 1), chan_dim, reg=reg, name='block3')
    x = MaxPooling2D((3, 3), padding='same', name='pool2')(x)

    # apply two inception modules followed by a POOL
    x = inception_model(x, 64, 96, 128, 16, 32, 32, chan_dim, stage='3a', reg=reg)
    x = inception_model(x, 128, 128, 192, 32, 96, 64, chan_dim, stage='3b', reg=reg)
    x = MaxPooling2D((3, 3), (2, 2), padding='same', name='pool3')(x)

    # apply five inception modules followed by POOL
    x = inception_model(x, 192, 96, 208, 16, 48, 64, chan_dim, stage='4a', reg=reg)
    x = inception_model(x, 160, 112, 224, 24, 64, 64, chan_dim, stage='4b', reg=reg)
    x = inception_model(x, 128, 128, 256, 24, 64, 64, chan_dim, stage='4c', reg=reg)
    x = inception_model(x, 112, 144, 288, 32, 64, 64, chan_dim, stage='4d', reg=reg)
    x = inception_model(x, 256, 160, 320, 32, 128, 128, chan_dim, stage='4e', reg=reg)
    x = MaxPooling2D((3, 3), (2, 2), padding='same', name='pool4')(x)

    # apply a POOL layer (average) followed by dropout
    x = AveragePooling2D((4, 4), (1, 1), padding='same', name='pool5')(x)
    x = Dropout(.4, name='dropout')(x)

    # softmax classifier
    x = Flatten(name='flatten')(x)
    x = Dense(classes, kernel_regularizer=l2(reg), name='labels')(x)
    x = Activation('softmax', name='softmax')(x)

    # create the model
    model = Model(inputs, x, name='googlenet')

    return model
