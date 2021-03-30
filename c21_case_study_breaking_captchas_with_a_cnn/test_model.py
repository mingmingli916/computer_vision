from keras.preprocessing.image import img_to_array
from keras.models import load_model
from pyimagesearch.utils.captchahelper import preprocess
from imutils import contours, paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input directory of images")
ap.add_argument("-m", "--model", required=True, help="path to input model")
args = vars(ap.parse_args())

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args['model'])

# randomly sample a few of the input image
image_paths = list(paths.list_images(args['input']))
image_paths = np.random.choice(image_paths, size=(10,), replace=False)

for image_path in image_paths:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    # threshold the image to reveal the digits
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find the contours in the image, keeping only the four largest ones,
    # then sort them from left-to-right
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1] if imutils.is_cv3() else cnts[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
    cnts = contours.sort_contours(cnts)[0]

    # initialize the output image as a 'grayscale' image with 3 channels
    # along with the output predictions
    output = cv2.merge([gray] * 3)
    predictions = []

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

        # pre-process the ROI and classify it
        roi = preprocess(roi, 28, 28)
        roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
        pred = model.predict(roi).argmax(axis=1)[0] + 1
        predictions.append(str(pred))

        # draw the prediction on the output image
        cv2.rectangle(output, (x - 1, y - 2), (x + 2 + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, str(pred), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 2)

    print('[INFO] captcha: {}'.format(''.join(predictions)))
    cv2.imshow('Output', output)
    cv2.waitKey()
