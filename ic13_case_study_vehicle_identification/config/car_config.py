from os import path

# stanford dataset
# https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder/download

BASE_PATH = '/home/hack/PycharmProjects/computer_vision/ic13_case_study_vehicle_identification/stanford_car'
TRAIN_IMAGES_PATH = path.join(BASE_PATH, 'car_data/train')
TEST_IMAGES_PATH = path.join(BASE_PATH, 'car_data/test')

MX_PATH = path.join(BASE_PATH, 'lists')
TRAIN_MX_LIST = path.sep.join([MX_PATH, "train.lst"])
VAL_MX_LIST = path.sep.join([MX_PATH, "val.lst"])
TEST_MX_LIST = path.sep.join([MX_PATH, "test.lst"])

REC_PATH = path.join(BASE_PATH, 'rec')
TRAIN_MX_REC = path.sep.join([REC_PATH, "train.rec"])
VAL_MX_REC = path.sep.join([REC_PATH, "val.rec"])
TEST_MX_REC = path.sep.join([REC_PATH, "test.rec"])

# human readable label <=> number label
LABEL_ENCODER_PATH = path.sep.join([BASE_PATH, 'label_encoder/le.pickle'])

# define the RGB means from the ImageNet dataset
# Because we are fine-tuning VGG, we must use the RGB means from the ImageNet dataset
R_MEAN = 123.68
G_MEAN = 116.779
B_MEAN = 103.939


def get_num_classes():
    names_path = path.join(BASE_PATH, 'names.csv')
    with open(names_path) as f:
        names = f.read().strip().split('\n')
        maker_models = [name[:name.rfind(' ')] for name in names]
        mm_set = set(maker_models)
        return len(mm_set)


NUM_CLASSES = get_num_classes()
NUM_VAL_IMAGES = 0.15

BATCH_SIZE = 32
NUM_DEVICES = 1


