from ic13_case_study_vehicle_identification.config import car_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import progressbar
import pickle
from imutils import paths
from chyson.ai.utils.label_utils import get_labels
from chyson.ai.utils.pbar_utils import build_widgets

print('[INFO] loading image paths and labels...')

train_paths = list(paths.list_images(config.TRAIN_IMAGES_PATH))
full_labels = get_labels(train_paths)
train_labels = [label[:label.rfind(' ')] for label in full_labels]

test_paths = list(paths.list_images(config.TEST_IMAGES_PATH))
full_labels = get_labels(test_paths)
test_labels = [label[:label.rfind(' ')] for label in full_labels]

# for i in train_paths[:10]:
#     print(i)
# for i in train_labels[:10]:
#     print(i)

num_val = int(len(train_paths) * config.NUM_VAL_IMAGES)
num_test = len(test_paths)

# encode the class labels
print('[INFO] encoding labels...')
le = LabelEncoder()
le.fit(train_labels)
train_labels = le.transform(train_labels)
test_labels = le.transform(test_labels)

# split training dataset into training and validation
print('[INFO] constructing validation data...')
split = train_test_split(train_paths, train_labels, test_size=num_val, stratify=train_labels)
train_paths, val_paths, train_labels, val_labels = split

# construct a directory for clear loop
datasets = [
    ('train', train_paths, train_labels, config.TRAIN_MX_LIST),
    ('val', val_paths, val_labels, config.VAL_MX_LIST),
    ('test', test_paths, test_labels, config.TEST_MX_LIST)
]

# build the mxnet list
for dtype, paths, labels, output_path in datasets:
    print('[INFO] building {}...'.format(output_path))
    with open(output_path, 'w') as f:
        # progress bar
        widgets = build_widgets('Building List: ')
        pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

        for i, (path, label) in enumerate(zip(paths, labels)):
            # write the image index, label, and output path to file
            row = '\t'.join([str(i), str(label), path])
            f.write('{}\n'.format(row))
            pbar.update(i)

        pbar.finish()

# write the label encoder to file
print('[INFO] serializing label encoder...')
with open(config.LABEL_ENCODER_PATH, 'wb') as f:
    f.write(pickle.dumps(le))
