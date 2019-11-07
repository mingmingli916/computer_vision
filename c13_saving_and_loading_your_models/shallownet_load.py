from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.datasets.simpledetasetloader import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path  to input dataset')
ap.add_argument('-m', '--model', required=True, help='path to pre-trained model')
args = vars(ap.parse_args())

# initialize the class labels
class_labels = ['cat', 'dog', 'panda']

# grab the list of images in the dataset then randomly
# sample indexes into the image paths list
print('[INFO] sampling images')
image_paths = np.array(list(paths.list_images(args['dataset'])))
idxs = np.random.randint(0, len(image_paths), size=(10,))  # random sample
image_paths = image_paths[idxs]

# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk the scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
# testing images are preprocessed in the same way as your training images
data, labels = sdl.load(image_paths)
data = data.astype('float') / 255.0

# load the pre-trained network
print('[INFO] loading pre-trained network...')
model = load_model(args['model'])

# make predictions on the images
print('[INFO] predicting...')
preds = model.predict(data, batch_size=32).argmax(axis=1)

# loop over the sample images
for i, image_path in enumerate(image_paths):
    # load the example image, draw the prediction, and display it to our screen
    image = cv2.imread(image_path)
    cv2.putText(image, 'Label: {}'.format(class_labels[preds[i]]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
