import os

# base path for front/rear vehicle dataset
BASE_PATH = '/home/hack/PycharmProjects/computer_vision/dataset/dlib_front_and_rear_vechicles_v1'

# path to training and testing xml file
TRAIN_XML = os.path.sep.join([BASE_PATH, 'training.xml'])
TEST_XML = os.path.sep.join([BASE_PATH, 'testing.xml'])

# record path
TRAIN_RECORD = os.path.sep.join([BASE_PATH, 'records/training.record'])
TEST_RECORD = os.path.sep.join([BASE_PATH, 'records/testing.record'])
CLASSES_FILE = os.path.sep.join([BASE_PATH, 'records/classes.pbtxt'])

# class label directory
# ID 0 is reserved for the background class
CLASSES = {'rear': 1, 'front': 2}
