import matplotlib
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os
from pyimagesearch.utils import pltutils

matplotlib.use('Agg')

# commandline argument parse
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to output directory')
ap.add_argument('-m', '--models', required=True, help='path to output models directory')
ap.add_argument('-n', '--num-models', type=int, default=5, help='# of models to train')
args = vars(ap.parse_args())

# load data
(trainx, trainy), (testx, testy) = cifar10.load_data()
trainx = trainx.astype('float') / 255
testx = testx.astype('float') / 255

# convert labels into vectors
lb = LabelBinarizer()
lb.fit(trainy)
trainy = lb.transform(trainy)
testy = lb.transform(testy)

# label names
label_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# data augmentation
aug = ImageDataGenerator(rotation_range=10,
                         width_shift_range=.1,
                         height_shift_range=.1,
                         horizontal_flip=True,
                         fill_mode='nearest')

# loop over the number of models to train
for i in range(args['num_models']):
    print('[INFO] training model {}/{}'.format(i, args['num_models']))
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=len(label_names))

    opt = SGD(lr=.01, decay=.01 / 40, momentum=.9, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # train the network
    H = model.fit_generator(aug.flow(trainx, trainy, batch_size=64),
                            epochs=40,
                            steps_per_epoch=len(trainx) / 40,
                            verbose=2,
                            validation_data=(testx, testy))

    # save the model
    p = [args['models'], 'model_{}.model'.format(i)]
    model.save(os.path.sep.join(p))

    # evaluate the network
    preds = model.predict(testx, batch_size=64)
    report = classification_report(testy.argmax(axis=1), preds.argmax(axis=1), target_names=label_names)

    # save the classification report to file
    p = [args['output'], 'model_{}.txt'.format(i)]
    with open(os.path.sep.join(p), 'w') as f:
        f.write(report)

    # plot the training loss and accuracy
    p = [args['output'], 'model_{}.png'.format(i)]
    pltutils.save_loss_acc(H, os.path.sep.join(p))
