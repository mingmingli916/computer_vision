from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import imutils
import cv2
import os
from chyson.ai.utils.pbar_utils import build_widgets

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', required=True, help='path to input directory of images')
parser.add_argument('-o', '--output', required=True, help='path to output directory of rotated images')
args = vars(parser.parse_args())

# grab the paths to the input images (limiting to 10000 images)
# and shuffle them to make creating a training and testing split easier
image_paths = list(paths.list_images(args['dataset']))[:10000]
random.shuffle(image_paths)

# initialize a dictionary to keep track of the number of each angle
# chosen so far, then initialize the progress bar
angles = {}
widgets = build_widgets('Building Dataset: ')
pbar = progressbar.ProgressBar(maxval=len(image_paths), widgets=widgets).start()

for i, image_path in enumerate(image_paths):
    # determine the rotation angle
    angle = np.random.choice([0, 90, 180, 270])
    image = cv2.imread(image_path)

    # meaning there was a issue loading the image from disk
    if image is None:
        continue

    # rotate the image
    image = imutils.rotate_bound(image, angle)
    base = os.path.join(args['output'], str(angle))

    # if the base path does not exist, create it
    if not os.path.exists(base):
        os.mkdir(base)

    # extract the image file extension, then construct the full path to the output file
    ext = image_path[image_path.rfind('.'):]
    output_path = [base, 'image_{}{}'.format(str(angles.get(angle, 0)).zfill(5), ext)]
    output_path = os.path.sep.join(output_path)

    cv2.imwrite(output_path, image)

    # update the count for the angle
    c = angles.get(angle, 0)
    angles[angle] = c + 1
    pbar.update(i)

pbar.finish()

# display counts for each of them
for angle in sorted(angles.keys()):
    print('[INFO] angle={}: {:,}'.format(angle, angles[angle]))

# [INFO] angle=0: 2,494
# [INFO] angle=90: 2,512
# [INFO] angle=180: 2,423
# [INFO] angle=270: 2,547
# Building Dataset: 100% |#######################################| Time:  0:01:36