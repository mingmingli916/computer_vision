import os
from pyimagesearch.messages import info
from pyimagesearch.utils import cifar10utils
import matplotlib
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.resnet import ResNet
from pyimagesearch.callbacks.epochcheckpoint import EpochCheckpoint
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import load_model
import keras.backend as K
import numpy as np
import argparse
import sys
from sklearn.model_selection import train_test_split

matplotlib.use('Agg')

# Set the maximum depth of the Python interpreter stack to n.  This
# limit prevents infinite recursion from causing an overflow of the C
# stack and crashing Python.
sys.setrecursionlimit(5000)  # for Theano
BATCH_SIZE = 32

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
                help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
args = vars(ap.parse_args())

# load the training and testing data, converting the images from integers to floats
trainx, trainy, testx, testy = cifar10utils.get_cifar10()

# # for debug
# trainx = trainx[:500]  # 'ProgbarLogger' object has no attribute 'log_values'
# trainy = trainy[:500]  # samples should be greater than batch size
# testx = testx[:500]
# testy = testy[:500]

# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=.1,
                         height_shift_range=.1,
                         horizontal_flip=True,
                         fill_mode='nearest')

# if there is no specific model checkpoint supplied,
# then initialize the network and compile the model
if args['model'] is None:
    print(info.compiling_model)
    opt = SGD(1e-1)
    model = ResNet(width=32,
                   height=32,
                   depth=3,
                   classes=10,
                   stages=(9, 9, 9),
                   filters=(64, 64, 128, 256),
                   reg=.0005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
else:
    print('[INFO] loading {}...'.format(args['model']))
    model = load_model(args['model'])

    # update the learning rate
    print('[INFO] old learning rate: {}'.format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-5)
    print('[INFO] new learning rate: {}'.format(K.get_value(model.optimizer.lr)))

if not os.path.exists('output'):
    os.mkdir('output')

# callbacks
callbacks = [
    EpochCheckpoint(args['checkpoints'], every=5, start_at=args['start_epoch']),
    TrainingMonitor('output/resnet56_cifar10.png', 'output/resnet56_cifar10.json', start_at=args['start_epoch'])
]

# train the network
print(info.training_model)
model.fit_generator(aug.flow(trainx, trainy, batch_size=BATCH_SIZE),
                    steps_per_epoch=len(trainx) // BATCH_SIZE,
                    validation_data=(testx, testy),
                    epochs=10,
                    callbacks=callbacks)

# ValueError: Input arrays should have the same number of samples as target arrays.
# Found 50000 input samples and 10000 target samples.
# bug: testx: 50000,  testy: 10000,
# testx = trainx.astype('float') # bug
