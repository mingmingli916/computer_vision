import matplotlib
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os

matplotlib.use("Agg")

# construct the argument parse and parse the augments
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to the output directory')
args = vars(ap.parse_args())

# show information on the process ID
print('[INFO] process ID: {}'.format(os.getpid()))

# load the training and testing data, then scale it into the range [0, 1]
print('[INFO] loading CIFAR-10 data...')
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype('float') / 255.0
testX = testX.astype('float') / 255.0

# convert num labels to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# initialize the SGD optimizer, but without any learning rate decay
print('[INFO] compiling model...')
opt = SGD(lr=.01, momentum=.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# construct the set of callbacks
fig_path = os.path.sep.join([args['output'], '{}.png'.format(os.getpid())])
json_path = os.path.sep.join([args['output'], '{}.json'.format(os.getpid())])
callbacks = [TrainingMonitor(fig_path, json_path=json_path)]

# train the network
print('[INFO] training network...')
model.fit(trainX, trainY, batch_size=64, epochs=100, callbacks=callbacks, validation_data=(testX, testY), verbose=1)
