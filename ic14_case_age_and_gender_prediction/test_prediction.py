import cv2
from ic14_case_age_and_gender_prediction.config import age_gender_deploy as deploy
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyimagesearch.preprocessing.croppreprocessor import CropPreprocessor  # to boost accuracy
from pyimagesearch.utils.agegenderhelper import AgeGenderHelper
from imutils.face_utils import FaceAligner
from imutils import face_utils
from imutils import paths
import numpy as np
import mxnet as mx
import argparse
import pickle
import imutils
import json
import dlib
import os
from pyimagesearch.utils.pltutils import plot_prob

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image (or directory)")
args = vars(ap.parse_args())

# load the label encoders and mean files
print('[INFO] loading label encoder and mean files...')
age_le = pickle.loads(open(deploy.AGE_LABEL_ENCODER, 'rb').read())
gender_le = pickle.loads(open(deploy.GENDER_LABEL_ENCODER, 'rb').read())
age_means = json.loads(open(deploy.AGE_MEANS).read())
gender_means = json.loads(open(deploy.GENDER_MEANS).read())

# load the models from disk
print('[INFO] loading models...')
age_path = os.path.join(deploy.AGE_NETWORK_PATH, deploy.AGE_PREFIX)
gender_path = os.path.join(deploy.GENDER_NETWORK_PATH, deploy.GENDER_PREFIX)
age_model = mx.model.FeedForward.load(age_path, deploy.AGE_EPOCH)
gender_model = mx.model.FeedForward.load(gender_path, deploy.GENDER_EPOCH)

# compile model
age_model = mx.model.FeedForward(
    ctx=[mx.cpu(0)],
    symbol=age_model.symbol,
    arg_params=age_model.arg_params,
    aux_params=age_model.aux_params
)
gender_model = mx.model.FeedForward(
    symbol=gender_model.symbol,
    ctx=[mx.cpu(0)],
    arg_params=gender_model.arg_params,
    aux_params=gender_model.aux_params
)

# initialize the image pre-processors
sp = SimplePreprocessor(width=256, height=256, inter=cv2.INTER_CUBIC)
cp = CropPreprocessor(width=227, height=227, horizontal=True)
age_mp = MeanPreprocessor(age_means['R'], age_means['G'], age_means['B'])
gender_mp = MeanPreprocessor(gender_means['R'], gender_means['G'], gender_means['B'])
iap = ImageToArrayPreprocessor(data_format='channels_first')

# initialize dlib's face detector (HOG-based), then create the facial landmark predictor and face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(deploy.DLIB_LANDMARK_PATH)
fa = FaceAligner(predictor)

# initialize the list of image paths as just a single image
image_paths = [args['image']]

# if the input path is actually a directory, then list all image paths in the directory
if os.path.isdir(args['image']):
    image_paths = sorted(list(paths.list_images(args['image'])))

for image_path in image_paths:
    # load the image, resize it, and convert it to grayscale
    print('[INFO] processing {}'.format(image_path))
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    for rect in rects:
        # determine the facial landmarks for the face region,
        # then align the face
        shape = predictor(gray, rect)
        face = fa.align(image, gray, rect)

        # resize the face to a fixed size, then extract 10-crop patches from it
        face = sp.preprocess(face)
        patches = cp.preprocess(face)

        # allocate memory for the age and gender patches
        age_patches = np.zeros((patches.shape[0], 3, 227, 227), dtype='float')
        gender_patches = np.zeros((patches.shape[0], 3, 227, 227), dtype='float')

        for j in np.arange(0, patches.shape[0]):
            # perform mean subtraction on the patch
            age_patch = age_mp.preprocess(patches[j])
            gender_patch = gender_mp.preprocess(patches[j])
            age_patch = iap.preprocess(age_patch)
            gender_patch = iap.preprocess(gender_patch)

            # update the respective patches lists
            age_patches[j] = age_patch
            gender_patches[j] = gender_patch

        # make predictions on age and gender based on the extracted patches
        age_preds = age_model.predict(age_patches)
        gender_preds = gender_model.predict(gender_patches)

        # compute the average for each class label based on the predictions for the patches
        age_preds = age_preds.mean(axis=0)
        gender_preds = gender_preds.mean(axis=0)

        # visualize the age and gender predictions
        age_canvas = plot_prob(age_preds, age_le)
        gender_canvas = plot_prob(gender_preds, gender_le)

        # draw the bounding box around the face
        clone = image.copy()
        x, y, w, h = face_utils.rect_to_bb(rect)
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the output image
        cv2.imshow('Input', clone)
        cv2.imshow('Face', face)
        cv2.imshow('Age Probabilities', age_canvas)
        cv2.imshow('Gender Probabilities', gender_canvas)
        cv2.waitKey()
