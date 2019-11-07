import os

BASE_PATH = '/home/hack/PycharmProjects/computer_vision/dataset/lisa'
ANNOT_PATH = os.path.sep.join([BASE_PATH, 'allAnnotations.csv'])

RECORD_PATH = '/home/hack/PycharmProjects/computer_vision/ic16_training_a_faster_rcnn_from_scratch/records'
TRAIN_RECORD = os.path.sep.join([RECORD_PATH, "training.record"])
TEST_RECORD = os.path.sep.join([RECORD_PATH, "testing.record"])
CLASSES_FILE = os.path.sep.join([RECORD_PATH, "classes.pbtxt"])

TEST_SIZE = .25
CLASSES = {"pedestrianCrossing": 1, "signalAhead": 2, "stop": 3}
