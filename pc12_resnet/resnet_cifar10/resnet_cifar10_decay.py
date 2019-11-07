import matplotlib

from pyimagesearch.messages import info
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.resnet import ResNet
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from pyimagesearch.utils import cifar10utils
import argparse
import sys
import os

matplotlib.use('Agg')
sys.setrecursionlimit(5000)  # for Theano

NUM_EPOCHS = 100
INIT_LR = 1e-1
BATCH_SIZE = 32


def poly_decay(epoch):
    max_epochs = NUM_EPOCHS
    base_lr = INIT_LR
    power = 1.0  # linear rate decay

    # compute the new learning rate based on polynomial decay
    alpha = base_lr * (1 - (epoch / float(max_epochs))) ** power

    return alpha


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory (logs, plots, etc.)")
args = vars(ap.parse_args())

# load cifar10 data
trainx, trainy, testx, testy = cifar10utils.get_cifar10()

# data augmentation
aug = ImageDataGenerator(width_shift_range=.1,
                         height_shift_range=.1,
                         horizontal_flip=True)

# callbacks
fig_path = os.path.sep.join([args['output'], '{}.png'.format(os.getpid())])
json_path = os.path.join(args['output'], '{}.json'.format(os.getpid()))
callbacks = [
    TrainingMonitor(fig_path, json_path),
    LearningRateScheduler(poly_decay)
]

# compile
print(info.compiling_model)
opt = SGD(lr=INIT_LR, momentum=.9)
model = ResNet(width=32,
               height=32,
               depth=3,
               classes=10,
               stages=(9, 9, 9),
               filters=(64, 64, 128, 256),
               reg=.0005)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train
print(info.training_model)
model.fit_generator(aug.flow(trainx, trainy, batch_size=BATCH_SIZE),
                    steps_per_epoch=len(trainx) // BATCH_SIZE,
                    validation_data=(testx, testy),
                    epochs=100,
                    callbacks=callbacks)

# save the model
print(info.saving_model)
model.save(args['model'])
