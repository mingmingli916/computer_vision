# set the matplotlib backend so figures can be saved in the background
import matplotlib

from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse

matplotlib.use('Agg')


def step_decay(epoch):
    # initialize the base initial learning rate, drop factor, and epochs to drop every
    init_alpha = .01
    factor = .5
    drop_every = 5

    # compute learning rate for the current epoch
    alpha = init_alpha * (factor ** np.floor((1 + epoch) / drop_every))

    return alpha


# commandline arguments
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to the output loss/accuracy plot')
ap.add_argument('-m', '--model', required=True, help='path to model')
args = vars(ap.parse_args())

# data
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

# define the set of callback to be passed to the model during training
callbacks = [LearningRateScheduler(step_decay)]

# initialize the optimizer and model
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
opt = SGD(lr=.01, momentum=.9, nesterov=True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
H = model.fit(trainX, trainY, batch_size=64, epochs=40, verbose=1, callbacks=callbacks, validation_data=(testX, testY))

# # save model
model.save(args['model'])

# evaluate the network
print('[INFO] evaluating network...')
preds = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=label_names))

# plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 40), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 40), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 40), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, 40), H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy on CIFAR-10')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig(args['output'])

# factor: 0.25
#               precision    recall  f1-score   support
#
#     airplane       0.85      0.82      0.84      1000
#   automobile       0.92      0.89      0.90      1000
#         bird       0.77      0.70      0.73      1000
#          cat       0.64      0.63      0.64      1000
#         deer       0.78      0.81      0.80      1000
#          dog       0.71      0.73      0.72      1000
#         frog       0.81      0.88      0.84      1000
#        horse       0.85      0.84      0.85      1000
#         ship       0.90      0.89      0.89      1000
#        truck       0.87      0.88      0.87      1000
#
#     accuracy                           0.81     10000
#    macro avg       0.81      0.81      0.81     10000
# weighted avg       0.81      0.81      0.81     10000

# factor： 0.5
#               precision    recall  f1-score   support
#
#     airplane       0.85      0.84      0.84      1000
#   automobile       0.93      0.90      0.91      1000
#         bird       0.77      0.72      0.75      1000
#          cat       0.67      0.65      0.66      1000
#         deer       0.77      0.81      0.79      1000
#          dog       0.73      0.74      0.74      1000
#         frog       0.84      0.88      0.86      1000
#        horse       0.85      0.86      0.86      1000
#         ship       0.90      0.92      0.91      1000
#        truck       0.90      0.89      0.89      1000
#
#     accuracy                           0.82     10000
#    macro avg       0.82      0.82      0.82     10000
# weighted avg       0.82      0.82      0.82     10000