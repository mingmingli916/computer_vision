import os

import numpy as np


class ImageNetHelper:
    def __init__(self, config):
        self.config = config

        # word id -> integer label
        self.label_mappings = self.build_class_labels()
        self.val_blacklist = self.build_blacklist()

    def build_class_labels(self):
        # load the contents of the file that maps the WordNet IDs to
        # integers, then initialize the label mappings dictionary
        # n02119789 1 kit_fox
        # n02100735 2 English_setter
        rows = open(self.config.WORD_IDS).read().strip().split('\n')
        label_mappings = {}

        for row in rows:
            # split the row into the WordNet ID, label integer, and human readable label
            word_id, label, hr_label = row.split(' ')

            # update the label mappings dictionary using the word ID as the key
            # and the label as the value, subtracting '1' from the label
            # since MATLAB is one-indexed with Python is zero-indexed
            label_mappings[word_id] = int(label) - 1

        return label_mappings

    def build_blacklist(self):
        # 36
        # 50
        # 56
        # 103
        rows = open(self.config.VAL_BLACKLIST).read().strip().split('\n')
        # A set object will allow us to determine if a given validation image
        # is part of the the blacklist in O(1) time.
        rows = set(rows)

        return rows

    def build_training_set(self):
        paths = []
        labels = []

        # load the contents of the training input file that lists the partial image ID
        # and image number, then initialize the list of image paths and class labels
        # n01440764/n01440764_10026 1
        # n01440764/n01440764_10027 2
        # n01440764/n01440764_10029 3
        rows = open(self.config.TRAIN_LIST).read().strip().split('\n')

        for row in rows:
            # break the row into the partial path and image number
            # (the image number is sequential and is useless to us)
            partial_path, image_no = row.strip().split(' ')  # without JPEG suffix

            # construct the full path to the training image, then grab the word ID from the path
            # and use it to determine the integer class label
            # {imagenet}Data/CLS-LOC/train/n02097130/n02097130_4602.JPEG
            path = os.path.sep.join([self.config.IMAGES_PATH, 'train', '{}.JPEG'.format(partial_path)])
            word_id = partial_path.split('/')[0]
            label = self.label_mappings[word_id]

            paths.append(path)
            labels.append(label)

        return np.array(paths), np.array(labels)

    def build_validation_set(self):
        paths = []
        labels = []

        # load the contents of the file that lists the partial validation image filenames
        # ILSVRC2012_val_00000001 1
        # ILSVRC2012_val_00000002 2
        # ILSVRC2012_val_00000003 3
        val_filenames = open(self.config.VAL_LIST).read().strip().split('\n')

        # load the content of the file that contains
        # the actual ground-truth integers class labels for the validation set
        # 490
        # 361
        # 171
        val_labels = open(self.config.VAL_LABELS).read().strip().split('\n')

        for row, label in zip(val_filenames, val_labels):
            partial_path, image_no = row.strip().split(' ')

            # if the image number is in the blacklist set then we should
            # ignore this validation image
            if image_no in self.val_blacklist:
                continue

            # construct the full path to the validation image, then update the respective paths and labels lists
            path = os.path.sep.join([self.config.IMAGES_PATH, 'val', '{}.JPEG'.format(partial_path)])
            paths.append(path)
            labels.append(int(label) - 1)

        return np.array(paths), np.array(labels)

    def build_test_set(self):
        paths = []

        # load the contents of the file that contains the partial test image filenames
        # ILSVRC2012_test_00000001 1
        # ILSVRC2012_test_00000002 2
        # ILSVRC2012_test_00000003 3
        test_filenames = open(self.config.TEST_LIST).read().strip().split('\n')

        for row in test_filenames:
            partial_path, image_no = row.strip().split(' ')
            # construct the full path to the test image
            path = os.path.sep.join([self.config.IMAGES_PATH, 'test', '{}.JPEG'.format(partial_path)])
            paths.append(path)

        return np.array(paths)
