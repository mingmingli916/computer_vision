import mxnet as mx


# paper: http://chyson.net/papers/Age%20and%20Gender%20Classification%20using%20Convolutional%20Neural%20Networks.pdf

def MxAgeGenderNet(classes):
    # data input
    data = mx.sym.Variable('data')

    # Block #1: CONV => RELU => POOL
    conv1_1 = mx.sym.Convolution(
        data=data,
        num_filter=96,
        kernel=(7, 7),
        stride=(4, 4)
    )
    act1_1 = mx.sym.Activation(data=conv1_1, act_type='relu')
    bn1_1 = mx.sym.BatchNorm(data=act1_1)
    pool1 = mx.sym.Pooling(
        data=bn1_1,
        pool_type='max',
        kernel=(3, 3),
        stride=(2, 2)
    )
    do1 = mx.sym.Dropout(data=pool1, p=0.25)

    # Block #2: CONV => RELU => POOL
    conv2_1 = mx.sym.Convolution(
        data=do1,
        num_filter=256,
        kernel=(5, 5),
        pad=(2, 2)
    )
    act2_1 = mx.sym.Activation(data=conv2_1, act_type='relu')
    bn2_1 = mx.sym.BatchNorm(data=act2_1)
    pool2 = mx.sym.Pooling(
        data=bn2_1,
        pool_type='max',
        kernel=(3, 3),
        stride=(2, 2)
    )
    do2 = mx.sym.Dropout(data=pool2, p=0.25)

    # Block #3: CONV => RELU => POOL
    conv3_1 = mx.sym.Convolution(
        data=do2,
        kernel=(3, 3),
        pad=(1, 1),
        num_filter=384
    )
    act3_1 = mx.sym.Activation(data=conv3_1, act_type='relu')
    bn3_1 = mx.sym.BatchNorm(data=act3_1)
    pool3 = mx.sym.Pooling(
        data=bn3_1,
        pool_type='max',
        kernel=(3, 3),
        stride=(2, 2)
    )
    do3 = mx.sym.Dropout(pool3, p=0.25)

    # Block #4: FC
    flatten = mx.sym.Flatten(do3)
    fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=512)
    act4_1 = mx.sym.Activation(data=fc1, act_type='relu')
    bn4_1 = mx.sym.BatchNorm(data=act4_1)
    do4 = mx.sym.Dropout(data=bn4_1, p=0.5)

    # Block #5: FC
    fc2 = mx.sym.FullyConnected(data=do4, num_hidden=512)
    act5_1 = mx.sym.Activation(data=fc2, act_type='relu')
    bn5_1 = mx.sym.BatchNorm(data=act5_1)
    do5 = mx.sym.Dropout(data=bn5_1, p=0.5)

    # softmax classifier
    fc3 = mx.sym.FullyConnected(data=do5, num_hidden=classes)
    model = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

    return model
