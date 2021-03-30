from pyimagesearch.nn.conv.lenet import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

# grab the MNIST dataset (if this is your first time using this
# dataset then the 55MB download may take a minute)
print("[INFO] accessing MNIST...")
dataset = datasets.fetch_mldata("MNIST Original")
# dataset = datasets.fetch_openml('MNIST Original')
data = dataset.data
data = data / 255.0
targets = dataset.target.astype('int')

# if we are using "channels first" ordering, then reshape the
# design matrix such that the matrix is:
# num_samples x depth x rows x columns
# scale the input data to the range [0, 1] and perform a train/test split
trainX, testX, trainY, testY = train_test_split(data, targets, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=128, epochs=20, verbose=2)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in le.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

#               precision    recall  f1-score   support
#
#            0       0.98      0.99      0.99      1677
#            1       0.98      0.99      0.99      1935
#            2       0.98      0.99      0.98      1767
#            3       0.98      0.98      0.98      1766
#            4       0.92      1.00      0.96      1691
#            5       0.99      0.98      0.99      1653
#            6       1.00      0.98      0.99      1754
#            7       0.98      0.99      0.98      1846
#            8       0.97      0.96      0.97      1702
#            9       0.99      0.90      0.95      1709
#
#     accuracy                           0.98     17500
#    macro avg       0.98      0.98      0.98     17500
# weighted avg       0.98      0.98      0.98     17500
