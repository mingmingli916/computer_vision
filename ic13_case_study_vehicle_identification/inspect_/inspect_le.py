import pickle
from ic13_case_study_vehicle_identification.config import car_config as config

le = pickle.loads(open(config.LABEL_ENCODER_PATH, 'rb').read())
class_label = le.inverse_transform([1])
print(class_label)
