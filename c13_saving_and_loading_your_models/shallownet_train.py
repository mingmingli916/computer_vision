from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.datasets.simpledetasetloader import SimpleDatasetLoader
from pyimagesearch.nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument('-m', '--model', required=True, help='path to output model')
args = vars(ap.parse_args())

# grab the list of images that we’ll be describing
print("[INFO] loading images...")
image_paths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
data, labels = sdl.load(image_paths, verbose=500)
data = data.astype('float') / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=2)

# save the network to disk
print("[INFO] serializing network...")
model.save(args['model'])

# evaluate the network
preds = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=['cat', 'dog', 'pandas']))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

#               precision    recall  f1-score   support
#
#          cat       0.65      0.59      0.62       239
#          dog       0.60      0.71      0.65       262
#       pandas       0.87      0.78      0.83       249
#
#     accuracy                           0.70       750
#    macro avg       0.71      0.69      0.70       750
# weighted avg       0.71      0.70      0.70       750
