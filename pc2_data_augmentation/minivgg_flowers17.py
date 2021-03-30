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

# initialize the optimizer and model
print(info.compiling_model)
opt = SGD(.05)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(class_names))
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
print(info.training_model)
H = model.fit(trainx, trainy, batch_size=32, epochs=100, verbose=1, validation_data=(testx, testy))

# evaluate the network
print(info.evaluating_model)
preds = model.predict(testx, batch_size=32)
print(classification_report(testy.argmax(axis=1), preds.argmax(axis=1), target_names=lb.classes_))

pltutils.plot_loss_acc(H)

#               precision    recall  f1-score   support
#
#     bluebell       0.59      0.67      0.62        15
#    buttercup       0.50      0.47      0.49        19
#   colts_foot       0.56      0.50      0.53        20
#      cowslip       0.31      0.44      0.36        18
#       crocus       0.52      0.60      0.56        20
#     daffodil       0.50      0.38      0.43        21
#        daisy       0.65      0.65      0.65        20
#    dandelion       0.46      0.81      0.59        16
#   fritillary       0.82      0.78      0.80        23
#         iris       0.67      0.88      0.76        16
#  lily_valley       0.72      0.59      0.65        22
#        pansy       0.87      0.68      0.76        19
#     snowdrop       0.55      0.55      0.55        20
#    sunflower       1.00      0.81      0.90        27
#    tigerlily       0.78      0.78      0.78        18
#        tulip       0.39      0.30      0.34        23
#   windflower       0.60      0.52      0.56        23
#
#     accuracy                           0.61       340
#    macro avg       0.62      0.61      0.61       340
# weighted avg       0.63      0.61      0.61       340
