from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
import numpy as np


def get_cifar10():
    # load the training and testing data, converting the images from integers to floats
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
    testy = lb.fit_transform(testy)

    return trainx, trainy, testx, testy
