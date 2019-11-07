from pyimagesearch.messages import info
import matplotlib
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.minigooglenet import MiniGoogLeNet
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import os

matplotlib.use("Agg")

NUM_EPOCHS = 70
INIT_LR = 1e-2
BATCH_SIZE = 32


def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    max_epochs = NUM_EPOCHS
    base_lr = INIT_LR
    power = 1.0

    # compute the new learning rate based on polynomial decay
    alpha = base_lr * (1 - (epoch / float(max_epochs))) ** power

    return alpha


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory (logs, plots, etc.)")
args = vars(ap.parse_args())

# load the training and testing data, converting the images
# from integers to float
print('[INFO] loading CIFAR-10 data...')
(trainx, trainy), (testx, testy) = cifar10.load_data()
trainx = trainx.astype('float')
testx = testx.astype('float')

# apply mean subtraction to the data
mean = np.mean(trainx, axis=0)
trainx -= mean
testx -= mean

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainy = lb.fit_transform(trainy)
testy = lb.transform(testy)

# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=.1,
                         height_shift_range=.1,
                         horizontal_flip=True,
                         fill_mode='nearest')

# construct the set of callbacks
fig_path = os.path.join(args['output'], '{}.png'.format(os.getpid()))
json_path = os.path.join(args['output'], '{}.json'.format(os.getpid()))
callbacks = [TrainingMonitor(fig_path, json_path), LearningRateScheduler(poly_decay)]

# initialize the optimizer and model
print(info.compiling_model)
opt = SGD(lr=INIT_LR, momentum=.9)
model = MiniGoogLeNet(width=32, height=32, depth=3, classes=10)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
print(info.training_model)
model.fit_generator(aug.flow(trainx, trainy, batch_size=BATCH_SIZE),
                    steps_per_epoch=len(trainx) // BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    callbacks=callbacks,
                    validation_data=(testx, testy))

# save the network to disk
print(info.saving_model)
model.save(args['model'])
