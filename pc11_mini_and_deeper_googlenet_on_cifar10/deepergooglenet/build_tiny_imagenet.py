from config import tiny_imagenet_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.io.hdf5datasewriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os
from pyimagesearch.utils import pbarutils

# grab the paths to the training images, then extract the training
# class labels and encode them
train_paths = list(paths.list_images(config.TRAIN_IMAGES))
# tiny-imagenet-200/train/{wordnet_id}/{unique_filename}.JPG
train_labels = [p.split(os.path.sep)[-3] for p in train_paths]
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)

# train test split
split = train_test_split(train_paths, train_labels, test_size=config.NUM_TEST_IMAGES, stratify=train_labels,
                         random_state=42)
train_paths, test_paths, train_labels, test_labels = split

# load the validation filename => class mapping from file
# and then use these mappings to build the validation paths
# and label lists
mappings = open(config.VAL_MAPPINGS).read().strip().split('\n')
# val_0.JPEG	n03444034	0	32	44	62
mappings = [r.split('\t')[:2] for r in mappings]
val_paths = [os.path.join(config.VAL_IMAGES, m[0]) for m in mappings]
val_labels = le.transform([m[1] for m in mappings])

# construct a list pairing the training, validation, and testing
# images paths along with their corresponding labels and output
# HDF5 files
datasets = [
    ('train', train_paths, train_labels, config.TRAIN_HDF5),
    ('val', val_paths, val_labels, config.VAL_HDF5),
    ('test', test_paths, test_labels, config.TEST_HDF5)
]

# initialize the lists of RGB channel averages
R, G, B = [], [], []

# loop over the dataset tuples
for dtype, paths, labels, output_path in datasets:
    print('[INFO] building {}...'.format(output_path))
    writer = HDF5DatasetWriter((len(paths), 64, 64, 3), output_path)

    # initialize the progress bar
    widgets = pbarutils.build_widgets('Building Datasets: ', ' ')
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    # loop over the image paths
    for i, (path, label) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)

        if dtype == 'train':
            b, g, r = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # add the image and label to the HDF5 dataset
        writer.add([image], [label])
        pbar.update(i)

    pbar.finish()
    writer.close()

# construct a dictionary of average, then serialize the means
# to a json file
print('[INFO] serializing means...')
D = {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
with open(config.DATASET_MEAN, 'w') as f:
    f.write(json.dumps(D))
