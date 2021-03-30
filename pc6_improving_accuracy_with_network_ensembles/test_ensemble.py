from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import argparse
import glob
import os
from pyimagesearch.messages import info

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True,
                help="path to models directory")
args = vars(ap.parse_args())

# load data
(testx, testy) = cifar10.load_data()[1]
testx = testx.astype('float') / 255

# convert labels into vectors
lb = LabelBinarizer()
testy = lb.fit_transform(testy)

# label names
label_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# construct the path used to collect the models then
# initialize the models list
model_paths = os.path.sep.join([args['models'], '*.model'])
model_paths = list(glob.glob(model_paths))
models = []

# load models
for i, model_path in enumerate(model_paths):
    print('[INFO] loading model {}/{}'.format(i, len(model_paths)))
    models.append(load_model(model_path))

print('[INFO] evaluating ensemble...')
preds = []
for model in models:
    preds.append(model.predict(testx, batch_size=64))

# averaging the probabilities across all model predictions,
# then show a classification report
preds = np.average(preds, axis=0)
print(classification_report(testy.argmax(axis=1), preds.argmax(axis=1), target_names=label_names))

#               precision    recall  f1-score   support
#
#     airplane       0.90      0.87      0.88      1000
#   automobile       0.93      0.95      0.94      1000
#         bird       0.84      0.80      0.82      1000
#          cat       0.80      0.69      0.74      1000
#         deer       0.84      0.89      0.86      1000
#          dog       0.78      0.83      0.80      1000
#         frog       0.89      0.92      0.91      1000
#        horse       0.91      0.92      0.91      1000
#         ship       0.93      0.93      0.93      1000
#        truck       0.90      0.95      0.92      1000
#
#     accuracy                           0.87     10000
#    macro avg       0.87      0.87      0.87     10000
# weighted avg       0.87      0.87      0.87     10000
