from ic14_case_age_and_gender_prediction.config.age_gender_config import OUTPUT_PATH, BASE_PATH
from os import path

# In order to obtain higher accuracy when predicting age and gender, it’s often helpful to crop and
# align a face from a given image. The DLIB_LANDMARK_PATH variables provideacc the path to a
# pre-trained facial landmark predictor that will enable us to align faces in input images.

# define the path to the dlib facial landmark predictor
# dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
DLIB_LANDMARK_PATH = "dlib/shape_predictor_68_face_landmarks.dat"

# define the age path
AGE_NETWORK_PATH = 'checkpoints/age'
AGE_PREFIX = 'agenet'
AGE_EPOCH = 110  # 150
AGE_LABEL_ENCODER = path.join(BASE_PATH, 'encoder/age_le.pickle')
AGE_MEANS = path.join(BASE_PATH, 'mean/age_adience_mean.json')

# define the gender path
GENDER_NETWORK_PATH = 'checkpoints/gender'
GENDER_PREFIX = 'gendernet'
GENDER_EPOCH = 110
GENDER_LABEL_ENCODER = path.join(BASE_PATH, 'encoder/gender_le.pickle')
GENDER_MEANS = path.join(BASE_PATH, 'mean/gender_adience_mean.json')
