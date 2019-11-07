from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from pyimagesearch.nn.conv.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset of faces")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

# initialize the list of data and labels
data = []
labels = []

# prepare the data and labels
for image_path in sorted(list(paths.list_images(args['dataset']))):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    # SMILEs/positives/positives7/10007.jpg
    label = image_path.split(os.path.sep)[-3]
    label = 'smiling' if label == 'positives' else 'not_smiling'
    labels.append(label)

# pre-process
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

# convert the labels
le = LabelEncoder()
le.fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# account the skew in the labeled data
class_totals = labels.sum(axis=0)  # [9475,3690]
class_weight = class_totals.max() / class_totals  # [1. , 2.56775068]

#     stratify : array-like or None (default=None)
#         If not None, data is split in a stratified fashion, using this as
#         the class labels.
train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=.2, stratify=labels, random_state=42)

# initialize the model
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the network
H = model.fit(train_x, train_y, batch_size=32, epochs=15, verbose=1, validation_data=(test_x, test_y),
              class_weight=class_weight)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
X = np.arange(0, 15)
plt.plot(X, H.history["loss"], label="train_loss")
plt.plot(X, H.history["val_loss"], label="val_loss")
plt.plot(X, H.history["acc"], label="acc")
plt.plot(X, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

#               precision    recall  f1-score   support
#
#  not_smiling       0.94      0.93      0.93      1895
#      smiling       0.82      0.84      0.83       738
#
#     accuracy                           0.90      2633
#    macro avg       0.88      0.88      0.88      2633
# weighted avg       0.90      0.90      0.90      2633
