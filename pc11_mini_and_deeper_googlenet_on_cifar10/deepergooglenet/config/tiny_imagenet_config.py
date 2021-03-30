from os import path

# datasets base path
DATASETS_PATH = '../datasets/tiny-imagenet-200'

# define the paths to the training and validation directories
TRAIN_IMAGES = path.join(DATASETS_PATH, 'train')
VAL_IMAGES = path.join(DATASETS_PATH, 'val/images')

# define the path to the file that maps validation filenames
# to their corresponding class labels
VAL_MAPPINGS = path.join(DATASETS_PATH, 'val/val_annotations.txt')

# define the paths to the WordNet hierarchy
# which are used to generate our class labels
WORDNET_IDS = path.join(DATASETS_PATH, 'wnids.txt')
WORD_LABELS = path.join(DATASETS_PATH, 'words.txt')

# since we do not have access to the testing data
# we need to take a number of images from the training data
# and use it instead
NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES

# define the path to the output training, validation,
# and testing HDF5 files
TRAIN_HDF5 = path.join(DATASETS_PATH, 'hdf5/train.hdf5')
VAL_HDF5 = path.join(DATASETS_PATH, 'hdf5/val.hdf5')
TEST_HDF5 = path.join(DATASETS_PATH, 'hdf5/test.hdf5')

# define the path to the dataset mean
DATASET_MEAN = 'output/tiny-image-net-200-mean.json'

# define the path to the output directory used for storing
# plots, classification reports, etc.
OUTPUT_PATH = 'output'
MODEL_PATH = path.join(OUTPUT_PATH, 'checkpoints/checkpoints-010-3.6047.hdf5')
FIG_PATH = path.join(OUTPUT_PATH, 'deepergooglenet_tinyimagenet.png')
JSON_PATH = path.join(OUTPUT_PATH, 'deepergooglenet_tinyimagenet.json')

# batch size
BATCH_SIZE = 32
