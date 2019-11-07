import matplotlib
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Depending on what your default matplotlib backend is and whether
# you are accessing your deep learning machine remotely (via SSH, for instance), X11 session may
# timeout. If that happens, matplotlib will error out when it tries to display your figure. Instead,
# we can simply set the background to Agg and write the plot to disk when we are done training our
# network.
# set the matplotlib backend so figures can be saved in the background
matplotlib.use('Agg')

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

# initialize the optimizer and model
print('[INFO] compiling model...')
# decay: slowly reduce the learning rate over time
# A common setting for decay is to divide the initial learning rate by the total number of epochs
opt = SGD(lr=.01, decay=.01 / 40, momentum=.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# train the network
print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=40, verbose=1)

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

# with batch normalization
#               precision    recall  f1-score   support
#
#     airplane       0.86      0.82      0.84      1000
#   automobile       0.93      0.90      0.91      1000
#         bird       0.76      0.71      0.73      1000
#          cat       0.67      0.64      0.65      1000
#         deer       0.78      0.81      0.80      1000
#          dog       0.73      0.75      0.74      1000
#         frog       0.81      0.91      0.85      1000
#        horse       0.89      0.86      0.87      1000
#         ship       0.90      0.91      0.90      1000
#        truck       0.87      0.91      0.89      1000
#
#     accuracy                           0.82     10000
#    macro avg       0.82      0.82      0.82     10000
# weighted avg       0.82      0.82      0.82     10000