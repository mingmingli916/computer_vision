from config import imagenet_alexnet_config as config
from sklearn.model_selection import train_test_split
from pyimagesearch.utils.imagenethelper import ImageNetHelper
import numpy as np
import progressbar
from pyimagesearch.utils import pbarutils
import json
import cv2

# initialize the ImageNet helper and use it to construct the set of training and testing data
print('[INFO] loading image paths...')
inh = ImageNetHelper(config)
train_paths, train_labels = inh.build_training_set()
val_paths, val_labels = inh.build_validation_set()

# perform sampling from the training set to construct a test set
print('[INFO] constructing splits...')
split = train_test_split(train_paths, train_labels, test_size=config.NUM_TEST_IMAGES, stratify=train_labels,
                         random_state=42)
train_paths, test_paths,train_labels, test_labels = split

# construct a list paring the training, validation, and testing image paths
# along with their corresponding labels and output list files
datasets = [
    ('train', train_paths, train_labels, config.TRAIN_MX_LIST),
    ('val', val_paths, val_labels, config.VAL_MX_LIST),
    ('test', test_paths, test_labels, config.TEST_MX_LIST)
]

# initialize the list of red, green, and blue channel average
R, G, B = [], [], []

for dtype, paths, labels, output_path in datasets:
    print('[INFO] building {}...'.format(output_path))
    f = open(output_path, 'w')

    # initialize the progress bar
    widgets = pbarutils.build_widgets('Building List: ', ' ')
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    for i, (path, label) in enumerate(zip(paths, labels)):
        # write the image index, label, and output path to file
        row = '\t'.join([str(i), str(label), path])
        f.write('{}\n'.format(row))

        # if we are building the training dataset, then compute the mean of each channel in the image,
        # then update the respective lists
        if dtype == 'train':
            image = cv2.imread(path)
            b, g, r = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # update the progress bar
        pbar.update(i)

    pbar.finish()
    f.close()

# construct a dictionary of average, then serialize the means to a JSON file
print('[INFO] serializing means...')
dictionary = {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
with open(config.DATASET_MEAN, 'w') as f:
    f.write(json.dumps(dictionary))

