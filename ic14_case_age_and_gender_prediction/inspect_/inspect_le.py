import pickle

le_path = '/home/hack/PycharmProjects/computer_vision/ic14_case_age_and_gender_prediction/adience/encoder/age_le.pickle'
le = pickle.loads(open(le_path, 'rb').read())
print(le)
print(le.classes_)
print(type(le.classes_))
print(len(le.classes_))

classes = sorted(le.classes_, key=lambda x: int(x.split('_')[0]))
print(classes)
