from ic14_case_age_and_gender_prediction.config import age_gender_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.utils.agegenderhelper import AgeGenderHelper
import numpy as np
import progressbar
import pickle
import json
import cv2
from pyimagesearch.utils.pbarutils import build_widgets

print('[INFO] building paths and labels...')
agh = AgeGenderHelper(config)
train_paths, train_labels = agh.build_path_and_labels()

# construct the validation and test dataset
num_val = int(len(train_paths) * config.NUM_VAL_IMAGES)
num_test = int(len(train_paths) * config.NUM_TEST_IMAGES)

print('[INFO] encoding labels...')
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)

print('[INFO] constructing validation data...')
split = train_test_split(train_paths, train_labels, test_size=num_val, stratify=train_labels)
train_paths, val_paths, train_labels, val_labels = split

print('[INFO] constructing testing data...')
split = train_test_split(train_paths, train_labels, test_size=num_test, stratify=train_labels)
train_paths, test_paths, train_paths, test_labels = split

datasets = [
    ('train', train_paths, train_labels, config.TRAIN_MX_LIST),
    ('val', val_paths, val_labels, config.VAL_MX_LIST),
    ('test', test_paths, test_labels, config.TEST_MX_LIST)
]

# initialize the lists of RGB channel averages
R, G, B = [], [], []

for dtype, paths, labels, output_path in datasets:
    print('[INFO] building {}...'.format(output_path))
    f = open(output_path, 'w')

    # initialize the progress bar
    widgets = build_widgets('Building List: ')
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    for i, (path, label) in enumerate(zip(paths, labels)):
        # if we are building the training dataset, the compute the mean of each channel
        # in the image, then update the respective lists
        if dtype == 'train':
            image = cv2.imread(path)
            b, g, r = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # write the image index, label, and output path to file
        row = '\t'.join([str(i), str(label), str(path)])
        f.write('{}\n'.format(row))
        pbar.update(i)

    pbar.finish()
    f.close()

# construct a directory of averages, then serialize the means to a JSON file
print('[INFO] serializing means...')
directory = {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
with open(config.DATASET_MEAN, 'w') as f:
    f.write(json.dumps(directory))

# serialize the label encoder
print('[INFO] serializing the label encoder...')
with open(config.LABEL_ENCODER_PATH, 'wb') as f:
    f.write(pickle.dumps(le))
