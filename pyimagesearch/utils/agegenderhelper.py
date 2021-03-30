import numpy as np
import glob
import os


class AgeGenderHelper:
    def __init__(self, config):
        self.config = config
        self.age_bins = self.build_age_bins()

    def build_age_bins(self):
        age_bins = [(0, 2), (4, 6), (8, 13), (15, 20), (25, 32), (38, 43), (48, 53), (60, np.inf)]
        return age_bins

    # like a factory
    def to_label(self, age, gender):
        if self.config.DATASET_TYPE == 'age':
            return self.to_age_label(age)
        return self.to_gender_label(gender)

    def to_age_label(self, age):
        label = None

        # break the age tuple into integers
        # for example '(0, 2)'
        age = age.replace('(', '').replace(')', '').split(', ')
        age_lower, age_upper = np.array(age, dtype='int')

        for lower, upper in self.age_bins:
            if age_lower >= lower and age_upper <= upper:
                label = '{}_{}'.format(lower, upper)
                break
        return label

    def to_gender_label(self, gender):
        # return 0 if the gender is male, 1 if the gender is female
        return 0 if gender == 'm' else 1

    # When evaluating one-off accuracy, we need an efficient, fast method to determine if the
    # predicted class label is equal to or adjacent to the ground-truth label. The easiest way to accomplish
    # this task is to define a dictionary that maps a ground-truth label to its corresponding adjacent labels.
    # For example, if we knew the ground-truth label to a given data point is 8_13, we could use this
    # value as a key to our dictionary. The value would then be ["4_6", "8_13", "15_20"] – the set
    # of adjacent labels. Simply checking if 8_13 exists in this list would enable us to quickly evaluate
    # one-off accuracy.

    # determine the adjacent index in the original label encoder's classes_ metadata
    def build_one_off_mappings(self, le):
        # sort the class labels in ascending order and
        # initialize the one-off mappings for computing accuracy
        classes = sorted(le.classes_, key=lambda x: int(x.split('_')[0]))
        one_off = {}

        for i, name in enumerate(classes):
            # determine the index of the *current* class label name
            # in the *label encoder* (unordered) list, then
            # initialize the index of the previous and next
            # age groups adjacent to the current label

            # determine the current label position in the original label encoder's classes_
            current = np.where(le.classes_ == name)[0][0]  # where return like (array([5]),) => 5
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

    def build_path_and_labels(self):
        paths = []
        labels = []

        # grab the paths in the folds files
        fold_paths = os.path.join(self.config.LABELS_PATH, '*.txt')
        fold_paths = glob.glob(fold_paths)

        for fold_path in fold_paths:
            rows = open(fold_path).read()
            rows = rows.strip().split('\n')[1:]  # skip the header

            for row in rows:
                row = row.split('\t')
                user_id, image_path, face_id, age, gender = row[:5]

                # if the age or gender is invalid, ignore the example
                if age[0] != '(' or gender not in ('m', 'f'):
                    continue

                # construct the path to the image path and build the class label
                p = 'landmark_aligned_face.{}.{}'.format(face_id, image_path)
                p = os.path.join(self.config.IMAGES_PATH, user_id, p)
                # adience/aligned/100003415@N08/landmark_aligned_face.2174.9523333835_c7887c3fde_o.jpg
                label = self.to_label(age, gender)

                if label is None:
                    continue

                paths.append(p)
                labels.append(label)

        return paths, labels


