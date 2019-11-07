import numpy as np
import pickle


def build_one_off_mappings(le):
    # sort the class labels in ascending order and
    # initialize the one-off mappings for computing accuracy
    print('le.classes: {}'.format(le.classes_))
    classes = sorted(le.classes_, key=lambda x: int(x.split('_')[0]))
    print('sorted classes: {}'.format(classes))
    one_off = {}

    for i, name in enumerate(classes):
        # determine the index of the *current* class label name
        # in the *label encoder* (unordered) list, then
        # initialize the index of the previous and next
        # age groups adjacent to the current label
        current = np.where(le.classes_ == name)[0][0]
        print(np.where(le.classes_ == name), name)
        prev = -1
        next_ = -1

        # check to see if we should compute previous adjacent age group
        if i > 0:
            prev = np.where(le.classes_ == classes[i - 1])[0][0]

        # check to see if we should compute the next adjacent age group
        if i < len(classes) - 1:
            next_ = np.where(le.classes_ == classes[i + 1])[0][0]

        one_off[current] = (current, prev, next_)
    return one_off


if __name__ == '__main__':
    le_path = '/home/hack/PycharmProjects/computer_vision/ic14_case_age_and_gender_prediction/adience/encoder/age_le.pickle'
    le = pickle.loads(open(le_path, 'rb').read())

    one_off = build_one_off_mappings(le)
    print(one_off)
