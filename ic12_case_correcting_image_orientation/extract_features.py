import argparse
import random

import numpy as np
import progressbar
from imutils import paths
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder

from pyimagesearch.io.hdf5datasewriter import HDF5DatasetWriter
from pyimagesearch.messages import info
from pyimagesearch.utils import labelhelper

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-o', '--output', required=True, help='path to output HDF5 file')
ap.add_argument('-b', '--batch-size', type=int, default=32, help='batch size of image to be passed through network')
ap.add_argument('-s', '--buffer-size', type=int, default=100, help='size of feature extraction buffer')
args = vars(ap.parse_args())

# store the batch size in a convenience variable
bs = args['batch_size']  # notice, in the dict, batch-size is saved as batch_size

# grab the list of images then randomly shuffle them to allow for easy
# training and testing splits via array slicing during training time
print(info.loading_image)
image_paths = list(paths.list_images(args['dataset']))
# since we’ll be working with datasets too large to fit into memory, we won’t be
# able to perform this shuffle in memory – therefore, we shuffle the image paths before
# we extract the features.
random.shuffle(image_paths)

# extract the class labels from the image paths then encode the labels
labels = labelhelper.get_labels(image_paths)
le = LabelEncoder()
labels = le.fit_transform(labels)

# load the VGG16 network
print(info.loading_model)
model = VGG16(weights='imagenet', include_top=False)

# initialize the HDF5 dataset writer, then store the class label names in the dataset
dataset = HDF5DatasetWriter((len(image_paths), 512 * 7 * 7), args['output'], data_key='features',
                            buf_size=args['buffer_size'])
dataset.store_class_labels(le.classes_)

# initialize the progress bar
widgets = ['Extracting Features: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
           progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(image_paths), widgets=widgets).start()

for i in np.arange(0, len(image_paths), bs):  # similar to range(0, len(image_paths), bs)
    # extract the batch of images and labels, then initialize the list of actual images that
    # will be passed through the network for feature extraction
    batch_paths = image_paths[i:i + bs]
    batch_labels = labels[i:i + bs]
    batch_images = []

    # loop over the images and labels in the current batch
    for j, image_path in enumerate(batch_paths):
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)

        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)  # subtract mean
        batch_images.append(image)

    # pass the images through the network and use the output as our actual features
    batch_images = np.vstack(batch_images)
    features = model.predict(batch_images, batch_size=bs)

    # reshape the features so that each image is represented by
    # a flattened feature vector of the 'MaxPooling2D' output
    features = features.reshape(features.shape[0], 512 * 7 * 7)

    # add the features and labels to our HDF5 dataset
    #     raise TypeError("Can't broadcast %s -> %s" % (target_shape, self.mshape))
    # TypeError: Can't broadcast (96000,) -> (1024,)
    # I made the above bug because of I mistype batch_labels into labels
    dataset.add(features, batch_labels)
    pbar.update(i)

dataset.close()
pbar.finish()
