from os import path

# define the type of dataset we are training (i.e. either 'age' or 'gender')
DATASET_TYPE = 'gender'

BASE_PATH = '/home/hack/PycharmProjects/computer_vision/ic14_case_age_and_gender_prediction/adience'
OUTPUT_PATH = '/home/hack/PycharmProjects/computer_vision/ic14_case_age_and_gender_prediction/output'
MX_OUTPUT = BASE_PATH

IMAGES_PATH = path.join(BASE_PATH, 'aligned')
LABELS_PATH = path.join(BASE_PATH, 'fold')

# define the percentage of validation and testing images relative
# to the number of training images
NUM_VAL_IMAGES = 0.15
NUM_TEST_IMAGES = 0.15

# define the batch size
BATCH_SIZE = 64
NUM_DEVICES = 2

# check to see if we are working with the 'age' portion of the dataset
if DATASET_TYPE == 'age':
    # define the number of labels for the "gender" dataset, along
    # with the path to the label encoder
    NUM_CLASSES = 8
    LABEL_ENCODER_PATH = path.join(BASE_PATH, 'encoder/age_le.pickle')

    # define the path to the output training, validation, and testing
    # lists
    TRAIN_MX_LIST = path.sep.join([MX_OUTPUT, "list/age_train.lst"])
    VAL_MX_LIST = path.sep.join([MX_OUTPUT, "list/age_val.lst"])
    TEST_MX_LIST = path.sep.join([MX_OUTPUT, "list/age_test.lst"])

    # define the path to the output training, validation, and testing
    # image records
    TRAIN_MX_REC = path.sep.join([MX_OUTPUT, "rec/age_train.rec"])
    VAL_MX_REC = path.sep.join([MX_OUTPUT, "rec/age_val.rec"])
    TEST_MX_REC = path.sep.join([MX_OUTPUT, "rec/age_test.rec"])

    # derive the path to the mean pixel file
    DATASET_MEAN = path.sep.join([BASE_PATH, "mean/age_adience_mean.json"])
elif DATASET_TYPE == "gender":
    # define the number of labels for the "gender" dataset, along
    # with the path to the label encoder
    NUM_CLASSES = 2
    LABEL_ENCODER_PATH = path.sep.join([BASE_PATH, "encoder/gender_le.pickle"])

    # define the path to the output training, validation, and testing
    # lists
    TRAIN_MX_LIST = path.sep.join([MX_OUTPUT, "list/gender_train.lst"])
    VAL_MX_LIST = path.sep.join([MX_OUTPUT, "list/gender_val.lst"])
    TEST_MX_LIST = path.sep.join([MX_OUTPUT, "list/gender_test.lst"])

    # define the path to the output training, validation, and testing
    # image records
    TRAIN_MX_REC = path.sep.join([MX_OUTPUT, "rec/gender_train.rec"])
    VAL_MX_REC = path.sep.join([MX_OUTPUT, "rec/gender_val.rec"])
    TEST_MX_REC = path.sep.join([MX_OUTPUT, "rec/gender_test.rec"])

    # derive the path to the mean pixel file
    DATASET_MEAN = path.sep.join([BASE_PATH, "mean/gender_adience_mean.json"])
