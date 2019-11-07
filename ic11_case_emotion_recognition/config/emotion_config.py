from os import path

BASE_PATH = '/home/hack/PycharmProjects/computer_vision/ic11_case_emotion_recognition/fer2013'
INPUT_PATH = path.join(BASE_PATH, 'fer2013/fer2013.csv')
# define the number of classes (set to 6 if you are ignoring the "disgust" class
# there are seven classes: angry, disgust, fear, happy, sad, surprise, and neutral.
# However, there is heavy class imbalance with the “disgust” class, as it has only 113 image
# samples (the rest have over 1,000 images per class).
NUM_CLASSES = 6

TRAIN_HDF5 = path.join(BASE_PATH, 'hdf5/train.hdf5')
VAL_HDF5 = path.join(BASE_PATH, 'hdf5/val.hdf5')
TEST_HDF5 = path.join(BASE_PATH, 'hdf5/test.hdf5')

BATCH_SIZE = 64

OUTPUT_PATH = path.join(BASE_PATH, 'output')

# cv2 show height
SHOW_HEIGHT = 600
