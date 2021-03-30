from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from pyimagesearch.datasets.simpledetasetloader import SimpleDatasetLoader
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from imutils import paths
import numpy as np
import argparse
import os
from pyimagesearch.messages import info
from pyimagesearch.utils import pltutils
from keras.preprocessing.image import ImageDataGenerator

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
args = vars(ap.parse_args())

# grab the list of images that we'll be describing, then extract
# the class label names from the image paths
print(info.loading_image)
image_paths = list(paths.list_images(args['dataset']))
# flowers17/{species}/{image}
class_names = [pt.split(os.path.sep)[-2] for pt in image_paths]
class_names = [str(x) for x in np.unique(class_names)]

# initialize the image proprocessors
aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
data, labels = sdl.load(image_paths, verbose=500)
data = data.astype('float') / 255

trainx, testx, trainy, testy = train_test_split(data, labels, test_size=.25, random_state=42)

# convert the labels from integers to vectors
lb = LabelBinarizer()
lb.fit(trainy)
trainy = lb.transform(trainy)
testy = lb.transform(testy)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=.1,
                         height_shift_range=.1,
                         shear_range=.2,
                         zoom_range=.2,
                         horizontal_flip=True,
                         fill_mode='nearest')

# initialize the optimizer and model
print(info.compiling_model)
opt = SGD(.05)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(class_names))
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
print(info.training_model)
# H = model.fit(trainx, trainy, batch_size=32, epochs=100, verbose=1, validation_data=(testx, testy))
H = model.fit_generator(aug.flow(trainx, trainy, batch_size=32), steps_per_epoch=len(trainx) // 32, epochs=100,
                        verbose=1, validation_data=(testx, testy))

# evaluate the network
print(info.evaluating_model)
preds = model.predict(testx, batch_size=32)
print(classification_report(testy.argmax(axis=1), preds.argmax(axis=1), target_names=lb.classes_))

# plot loss and accuracy
pltutils.plot_loss_acc(H)

#               precision    recall  f1-score   support
#
#     bluebell       0.81      0.87      0.84        15
#    buttercup       0.89      0.84      0.86        19
#   colts_foot       0.83      0.62      0.71        16
#      cowslip       0.42      0.75      0.54        20
#       crocus       0.60      0.75      0.67        20
#     daffodil       0.56      0.65      0.60        23
#        daisy       0.95      0.95      0.95        20
#    dandelion       0.77      0.87      0.82        23
#   fritillary       0.78      0.74      0.76        19
#         iris       0.88      0.78      0.82        18
#  lily_valley       0.74      0.78      0.76        18
#        pansy       0.95      0.67      0.78        27
#     snowdrop       0.70      0.64      0.67        22
#    sunflower       0.88      0.88      0.88        16
#    tigerlily       0.93      0.67      0.78        21
#        tulip       0.44      0.35      0.39        23
#   windflower       0.89      0.85      0.87        20
#
#     accuracy                           0.74       340
#    macro avg       0.77      0.74      0.75       340
# weighted avg       0.76      0.74      0.74       340
