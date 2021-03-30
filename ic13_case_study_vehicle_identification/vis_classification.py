import argparse
import os
import pickle

import cv2
import imutils
import mxnet as mx
import numpy as np

from ic13_case_study_vehicle_identification.config import car_config as config
from pyimagesearch.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to the checkpoint directory")
ap.add_argument("-p", "--prefix", required=True, help="name of model prefix")
ap.add_argument("-e", "--epoch", type=int, required=True, help="epoch # to load")
ap.add_argument("-s", "--sample-size", type=int, default=10,
                help="number of images to sample from dataset for classification")
args = vars(ap.parse_args())

# load the label encoder
le = pickle.loads(open(config.LABEL_ENCODER_PATH, 'rb').read())

# load the testing dataset file the sample the testing set
rows = open(config.TEST_MX_LIST).read().strip().split('\n')
rows = np.random.choice(rows, size=args['sample_size'])

# load pre-trained model
checkpoints_path = os.path.join(args['checkpoints'], args['prefix'])
model = mx.model.FeedForward.load(checkpoints_path, args['epoch'])  # just parameter

# compile the model
model = mx.model.FeedForward(
    symbol=model.symbol,
    ctx=[mx.cpu(0)],
    arg_params=model.arg_params,
    aux_params=model.aux_params
)

# initialize the image pre-processors
aap = AspectAwarePreprocessor(width=224, height=224)
mp = MeanPreprocessor(r_mean=config.R_MEAN, g_mean=config.G_MEAN, b_mean=config.B_MEAN)
iap = ImageToArrayPreprocessor(data_format='channels_first')  # just correct the data format

for row in rows:
    # grab the target class label and the image path
    target, image_path = row.split('\t')[1:]  # the index 0 is the line number
    target = int(target)

    # load the image from and preprocess it
    image = cv2.imread(image_path)
    orig = image.copy()  # used for show
    orig = imutils.resize(orig, width=min(500, orig.shape[1]))  # resize to 500 if width > 500
    image = iap.preprocess(mp.preprocess(aap.preprocess(image)))
    image = np.expand_dims(image, axis=0)  # used for input to the model

    # classify the image and grab the indexes of the top-5 predictions
    preds = model.predict(image)[0]  # omit the sample num dimension
    idxs = np.argsort(preds)[::-1][:5]  # 5 most possible classes

    # show the true class label
    true_class = le.inverse_transform([target])  # notice: the input should be array-like
    print('[INFO] actual={}'.format(true_class[0]))

    # format and display the top predicted class label
    label = le.inverse_transform([idxs[0]])[0]
    # label = label.replace(':', ' ')
    label = '{}: {:.2f}%'.format(label, preds[idxs[0]] * 100)
    cv2.putText(
        img=orig,
        text=label,
        org=(10, 30),  # bottom-left corner of the text string in the image
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=.6,
        color=(0, 255, 0),
        thickness=2
    )

    # loop over the top 5 predictions and display them
    for i, prob in zip(idxs, preds):  # the long part in preds is discarded
        print('\t[INFO] predicted={}, probability={:.2f}%'.format(le.inverse_transform([i])[0], preds[i] * 100))

    # show the image
    cv2.imshow('Image', orig)
    cv2.waitKey()
