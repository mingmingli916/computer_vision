from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from pyimagesearch.nn.conv.lenet import LeNet
from pyimagesearch.utils.captchahelper import preprocess
from imutils.paths import list_images
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# commandline argument parse
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-m', '--model', required=True, help='path to output model')
args = vars(ap.parse_args())

# initialize the data and labels
data = []
labels = []

for image_path in list_images(args['dataset']):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess(image, 28, 28)
    image = img_to_array(image)
    data.append(image)

    label = image_path.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

# split into train and test
train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=.25, random_state=42)

# convert the labels from integers to vectors
lb = LabelBinarizer()
lb.fit(train_y)
train_y = lb.transform(train_y)
test_y = lb.transform(test_y)

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape, sep='\n')

# initialize the mode
print('[INFO] compiling model...')
model = LeNet.build(width=28, height=28, depth=1, classes=9)
opt = SGD(lr=.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
# ValueError: Error when checking target: expected activation_4 to have shape (10,) but got array with shape (9,)
# in lenet.py
#         # softmax classifier
#         model.add(Dense(10))
#         model.add(Activation('softmax'))
# should be model.add(Dense(classes)
H = model.fit(train_x, train_y, batch_size=32, epochs=15, verbose=1, validation_data=(test_x, test_y))

# evaluate the network
print('[INFO] evaluating network...')
preds = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1), preds.argmax(axis=1), target_names=lb.classes_))

# save the model to disk
print('[INFO] serializing network...')
model.save(args['model'])

# plot the training + testing loss and accuracy
plt.style.use('ggplot')
plt.figure()
X = np.arange(0, 15)
plt.plot(X, H.history['loss'], label='train_loss')
plt.plot(X, H.history['val_loss'], label='val_loss')
plt.plot(X, H.history['acc'], label='acc')
plt.plot(X, H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()

#               precision    recall  f1-score   support
#
#            1       1.00      1.00      1.00        48
#            2       1.00      1.00      1.00        53
#            3       1.00      1.00      1.00        41
#            4       1.00      1.00      1.00        44
#            5       1.00      0.98      0.99        64
#            6       0.99      1.00      0.99        75
#            7       1.00      1.00      1.00        54
#            8       1.00      1.00      1.00        67
#            9       1.00      1.00      1.00        44
#
#     accuracy                           1.00       490
#    macro avg       1.00      1.00      1.00       490
# weighted avg       1.00      1.00      1.00       490
