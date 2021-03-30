from pyimagesearch.messages import info
from pyimagesearch.utils import labelhelper
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from pyimagesearch.datasets.simpledetasetloader import SimpleDatasetLoader
from pyimagesearch.nn.conv.fcheadnet import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from imutils import paths
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
args = vars(ap.parse_args())

# data augmentation
aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=.1,
                         height_shift_range=.1,
                         shear_range=.2,
                         zoom_range=.2,
                         horizontal_flip=True,
                         fill_mode='nearest')

# image paths and labels
print(info.loading_image)
image_paths = list(paths.list_images(args['dataset']))
class_names = labelhelper.unique_labels(image_paths)

# initialize the image proprocessors
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
data, labels = sdl.load(image_paths, verbose=500)
data = data.astype('float') / 255

# partitions the data into training and testing
trainx, testx, trainy, testy = train_test_split(data, labels, test_size=.25, random_state=42)

# convert labels to vectors
lb = LabelBinarizer()
lb.fit(trainy)
trainy = lb.transform(trainy)
testy = lb.transform(testy)

# load the VGG16 network, ensuring the head FC layer sets are left off
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# initialize the new head of the network
head_model = FCHeadNet.build(base_model, len(class_names), 256)

# place the head FC model on the top of the base model
# this will become the actual model we will train
model = Model(inputs=base_model.input, outputs=head_model)

# loop over all layers in the base model and freeze them
# so they will not be updated during the training process
for layer in base_model.layers:
    layer.trainable = False

# compile the model
print(info.compiling_model)
opt = RMSprop(lr=.001)  # small learning rate to warm up FC head
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the head of the network for a few epochs
# this will allow the new FC layers to start to
# become initialized with actual "learned" values
# versus pure random
print(info.training_model)
model.fit_generator(aug.flow(trainx, trainy, batch_size=32),
                    validation_data=(testx, testy),
                    epochs=25,  # usually 10-30 for warm up
                    steps_per_epoch=len(trainx) // 32,
                    verbose=2)

# evaluate the network after initialization
print(info.evaluating_model)
preds = model.predict(testx, batch_size=32)
print(classification_report(testy.argmax(axis=1), preds.argmax(axis=1), target_names=lb.classes_))

# now that the head FC layers have been trainable/initialized,
# lets unfreeze the final set of CONV layers and make them trainable
for layer in base_model.layers[15:]:
    # For deeper architectures with many parameters such as VGG, I suggest only
    # unfreezing the top CONV layers and then continuing training. If classification accuracy continues to
    # improve (without overfitting), you may want to consider unfreezing more layers in the body.
    layer.trainable = True

# for the changes to the mdel to take effect we need to recompile
# the model, this time using SGD with a very small learning rate
print('[INFO] re-compiling model...')
opt = SGD(.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the model again, this time fine-tuning both the
# final set of CONV layers along with our set of FC layers
print('[INFO] fine-tuning model...')
model.fit_generator(aug.flow(trainx, trainy, batch_size=32),
                    validation_data=(testx, testy),
                    epochs=100,
                    steps_per_epoch=len(trainx) // 32,
                    verbose=2)

# evaluate the network on the fine-tuned model
print('[INFO] evaluating after fine-tuning...')
preds = model.predict(testx, batch_size=32)
print(classification_report(testy.argmax(axis=1), preds.argmax(axis=1), target_names=lb.classes_))

# save the model
print(info.saving_model)
model.save(args['model'])

#               precision    recall  f1-score   support
#
#     bluebell       1.00      1.00      1.00        15
#    buttercup       0.90      1.00      0.95        19
#   colts_foot       0.93      0.88      0.90        16
#      cowslip       0.84      0.80      0.82        20
#       crocus       0.86      0.95      0.90        20
#     daffodil       0.95      0.87      0.91        23
#        daisy       1.00      1.00      1.00        20
#    dandelion       1.00      0.96      0.98        23
#   fritillary       0.95      0.95      0.95        19
#         iris       1.00      0.94      0.97        18
#  lily_valley       1.00      0.94      0.97        18
#        pansy       0.96      0.89      0.92        27
#     snowdrop       0.85      1.00      0.92        22
#    sunflower       1.00      1.00      1.00        16
#    tigerlily       1.00      0.95      0.98        21
#        tulip       0.70      0.83      0.76        23
#   windflower       1.00      0.90      0.95        20
#
#     accuracy                           0.93       340
#    macro avg       0.94      0.93      0.93       340
# weighted avg       0.93      0.93      0.93       340
