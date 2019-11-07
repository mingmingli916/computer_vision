import os
from os import path

#  Annotations/
#    CLS-LOC/
#  Data/
#    CLS-LOC/
#      test/
#      train/
#      val/
#  ImageSets/
#    CLS-LOC/
#      test.txt
#      train_cls.txt
#      train_loc.txt
#      val.txt
#  devkit
#    map_clsloc.txt
#    ILSVRC2015_clsloc_validation_ground_truth.txt
#    ILSVRC2015_clsloc_validation_blacklist.txt

# define the base path to where the ImageNet dataset and devkit are stored on disk
# containing Annotations, Data, ImageSets and devkit
BASE_PATH = '/media/hack/b0802883-dbd6-4644-8e87-1a0350b87ac7/datasets/imagenet_object_localization/ILSVRC'

# based on the base path, derive the images base path, image sets path, and devkit path
IMAGES_PATH = path.sep.join([BASE_PATH, 'Data/CLS-LOC'])  # Data, train, val, test
IMAGE_SETS_PATH = path.sep.join([BASE_PATH, 'ImageSets/CLS-LOC'])  # ImageSets, train_cls.txt, val.txt
DEVKIT_PATH = path.sep.join([BASE_PATH, 'devkit/data'])  # devkit

# define the path that maps the 1000 possible WordNet IDs to the class label integers
WORD_IDS = path.sep.join([DEVKIT_PATH, 'map_clsloc.txt'])

# define the paths to the training file that maps the images filenames to integer class label
TRAIN_LIST = path.sep.join([IMAGE_SETS_PATH, 'train_cls.txt'])

# define the paths to the validation filenames along with the file
# that contains the ground-truth validation labels
VAL_LIST = path.sep.join([IMAGE_SETS_PATH, 'val.txt'])
VAL_LABELS = path.sep.join([DEVKIT_PATH, 'ILSVRC2015_clsloc_validation_ground_truth.txt'])

# define the path to the validation files that are blacklisted
VAL_BLACKLIST = path.sep.join([DEVKIT_PATH, 'ILSVRC2015_clsloc_validation_blacklist.txt'])

# define the path to the testing file
TEST_LIST = path.sep.join([IMAGES_PATH, 'test.txt'])

# define classes
NUM_CLASSES = 1000
NUM_TEST_IMAGES = 50 * NUM_CLASSES

# define the path to the output training, validation, and testing lists
MX_OUTPUT = '/home/hack/PycharmProjects/computer_vision/dataset/imagenet'

MX_LIST = path.sep.join([MX_OUTPUT, 'lists'])
if not path.exists(MX_LIST):
    os.mkdir(MX_LIST)

TRAIN_MX_LIST = path.sep.join([MX_LIST, 'train.lst'])
VAL_MX_LIST = path.sep.join([MX_LIST, 'val.lst'])
TEST_MX_LIST = path.sep.join([MX_LIST, 'test.lst'])

# define the path to the output training, validation, and testing image records
MX_REC = path.sep.join([MX_OUTPUT, 'rec'])
if not path.exists(MX_REC):
    os.mkdir(MX_REC)
TRAIN_MX_REC = path.sep.join([MX_REC, 'train.rec'])
VAL_MX_REC = path.sep.join([MX_REC, 'val.rec'])
TEST_MX_REC = path.sep.join([MX_REC, 'test.rec'])

# define the path to the dataset mean
DATASET_MEAN = 'output/imagenet_mean.json'

# define the batch size and number of devices used for training
BATCH_SIZE = 32
NUM_DEVICES = 1  # CPUs, GPUs etc
