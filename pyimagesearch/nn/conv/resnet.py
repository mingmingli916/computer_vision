from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense
from keras.layers import Flatten, Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K


def residual_module(x, filters, stride, chan_dim, reduce=False, reg=.0001, bn_eps=2e-5, bn_mon=.9):
    """
    
    :param x: the input to the residual module 
    :param filters: defines the number of filters that will be learned by the final CONV in the bottleneck
    :param stride: stride of the convolution
    :param chan_dim: defines the axis which will perform batch normalization
    :param reduce: controls whether we are reducing spatial dimensions or not
    :param reg: regularization strength
    :param bn_eps: controls the epsilon responsible for avoiding 'division by zero' error
    :param bn_mon: controls the momentum for the moving average
    :return: 
    """
    # the shortcut branch of the ResNet module should be
    # initialize as the input (identity) data
    shortcut = x

    # the first block of the ResNet module are the 1x1 CONVs
    bn1 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mon)(x)
    act1 = Activation('relu')(bn1)
    # the biases are in the BN layers that immediately follow the convolutions, so there is no need
    # to introduce a second bias term.
    # default strides is (1,1) and (1,1) CONV uses it.
    conv1 = Conv2D(int(filters * .25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

    # the second block of the ResNet module are the 3x3 CONVs
    bn2 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mon)(conv1)
    act2 = Activation('relu')(bn2)
    conv2 = Conv2D(int(filters * .25), (3, 3), strides=stride, padding='same', use_bias=False)(act2)

    # the third block of the ResNet module is another set of 1x1 CONVs
    bn3 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mon)(conv2)
    act3 = Activation('relu')(bn3)
    conv3 = Conv2D(filters, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

    # if we are to reduce the spatial size, apply a CONV layer to the shortcut
    if reduce:
        shortcut = Conv2D(filters, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

    # add together the shortcut and the final CONV
    x = add([conv3, shortcut])

    return x


def ResNet(width, height, depth, classes, stages, filters, reg=.0001, bn_eps=2e-5, bn_mon=.9, dataset='cifar'):
    input_shape = (height, width, depth)
    chan_dim = -1

    if K.image_data_format() == 'channels_first':
        input_shape = (depth, height, width)
        chan_dim = 1

    # set the input and apply BN
    inputs = Input(shape=input_shape)
    x = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mon)(inputs)

    # check if we are using the CIFAR dataset
    if dataset == 'cifar':
        # apply a single CONV layer
        x = Conv2D(filters[0], (3, 3), use_bias=False, padding='same', kernel_regularizer=l2(reg))(x)
    elif dataset == 'tiny_imagenet':
        # apply CONV => BN => ACT => POOL to reduce spatial size
        x = Conv2D(filters[0], (5, 5), use_bias=False, padding='same', kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mon)(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        # the output volume is 32x32, ensuring we can easily reuse the rest of
        # the ResNet implementation without having to make any additional changes.
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # loop over the number of stages
    for i in range(0, len(stages)):
        # initialize the stride, then apply a residual module
        # used to reduce the spatial size of the input volume
        # To reduce volume size without pooling layers, we must set the stride of the convolution。
        stride = (1, 1) if i == 0 else (2, 2)
        # The reason we use i + 1 as the index into filters is
        # because the first filter value was used on if statement above.
        x = residual_module(x, filters[i + 1], stride, chan_dim, reduce=True, bn_eps=bn_eps, bn_mon=bn_mon)

        # loop over the number of layers in the stage
        for j in range(0, stages[i] - 1):
            x = residual_module(x, filters[i + 1], (1, 1), chan_dim, bn_eps=bn_eps, bn_mon=bn_mon)

    # In order to avoid using dense fully-connected layers, we’ll instead apply average pooling to
    # reduce the volume size to 1 x 1 x classes
    # apply BN => ACT => POOL
    x = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mon)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((8, 8))(x)

    # softmax classifier
    x = Flatten()(x)
    x = Dense(classes, kernel_regularizer=l2(reg))(x)
    x = Activation('softmax')(x)

    # create the model
    model = Model(inputs=inputs, outputs=x, name='resnet')

    return model
