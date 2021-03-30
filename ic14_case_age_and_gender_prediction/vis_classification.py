import argparse
import json
import os
import pickle

import cv2
import imutils
import mxnet as mx
import numpy as np

from ic14_case_age_and_gender_prediction.config import age_gender_config as config
from ic14_case_age_and_gender_prediction.config import age_gender_deploy as deploy
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.utils.pltutils import plot_prob

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--sample-size", type=int, default=10, help="number of images to sample from Adience testing set")
args = vars(ap.parse_args())

# load the label encoder and mean files
print('[INFO] loading label encoders and mean files...')
age_le = pickle.loads(open(deploy.AGE_LABEL_ENCODER, 'rb').read())
gender_le = pickle.loads(open(deploy.GENDER_LABEL_ENCODER, 'rb').read())
age_means = json.loads(open(deploy.AGE_MEANS).read())
gender_means = json.loads(open(deploy.GENDER_MEANS).read())

# load the model
age_path = os.path.join(deploy.AGE_NETWORK_PATH, deploy.AGE_PREFIX)
gender_path = os.path.join(deploy.GENDER_NETWORK_PATH, deploy.GENDER_PREFIX)
age_model = mx.model.FeedForward.load(age_path, deploy.AGE_EPOCH)
gender_model = mx.model.FeedForward.load(gender_path, deploy.GENDER_EPOCH)

# compile the model
age_model = mx.model.FeedForward(
    ctx=[mx.cpu(0)],
    symbol=age_model.symbol,
    arg_params=age_model.arg_params,
    aux_params=age_model.aux_params
)
gender_model = mx.model.FeedForward(
    ctx=[mx.cpu(0)],
    symbol=gender_model.symbol,
    arg_params=gender_model.arg_params,
    aux_params=gender_model.aux_params
)

# initialize the image pre-processor
sp = SimplePreprocessor(width=227, height=227, inter=cv2.INTER_CUBIC)
age_mp = MeanPreprocessor(age_means['R'], age_means['G'], age_means['B'])
gender_mp = MeanPreprocessor(gender_means['R'], gender_means['G'], gender_means['B'])
iap = ImageToArrayPreprocessor(data_format='channels_first')

# load a sample of testing images
rows = open(config.TEST_MX_LIST).read().strip().split('\n')
rows = np.random.choice(rows, size=args['sample_size'])

for row in rows:
    _, label, image_path = row.strip().split('\t')
    image = cv2.imread(image_path)

    # pre-process the image
    age_image = iap.preprocess(age_mp.preprocess(sp.preprocess(image)))
    gender_image = iap.preprocess(gender_mp.preprocess(sp.preprocess(image)))
    age_image = np.expand_dims(age_image, axis=0)
    gender_image = np.expand_dims(gender_image, axis=0)

    # pass the ROIs through their respective models
    age_preds = age_model.predict(age_image)[0]
    gender_preds = gender_model.predict(gender_image)[0]

    # sort the predictions according to their probability
    age_idxs = np.argsort(age_preds)[::-1]
    gender_idxs = np.argsort(gender_preds)[::-1]

    # visualize the age and gender predictions
    # age_canvas = AgeGenderHelper.visualize_age(age_preds, age_le)
    # gender_canvas = AgeGenderHelper.visualize_gender(gender_preds, gender_le)
    age_canvas = plot_prob(age_preds, age_le)
    gender_canvas = plot_prob(gender_preds, gender_le, height=50)

    image = imutils.resize(image, width=400)

    # draw the actual prediction on the image
    label = age_le.inverse_transform([int(label)])[0]
    print(label)
    print(age_le.classes_)
    print(row)
    text = 'Acutal: {}-{}'.format(*label.split('_'))
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

    # show the output image
    cv2.imshow('Image', image)
    cv2.imshow('Age Probabilities', age_canvas)
    cv2.imshow('Gender Probabilities', gender_canvas)
    cv2.waitKey()
