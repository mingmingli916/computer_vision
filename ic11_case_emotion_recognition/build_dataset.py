from ic11_case_emotion_recognition.emotion_recognition.config import emotion_config as config
from pyimagesearch.io.hdf5datasewriter import HDF5DatasetWriter
import numpy as np

print('[INFO] loading input data...')
with open(config.INPUT_PATH) as f:
    f.__next__()  # skip the head
    train_images, train_labels = [], []
    val_images, val_labels = [], []
    test_images, test_labels = [], []

    for row in f:
        # extract the label, image, and usage from the row
        # emotion,pixels,Usage
        # 0,70 80 82 72 58...106 109 82,Training
        label, image, usage = row.strip().split(',')
        label = int(label)

        # if 'disgust' is ignored
        if config.NUM_CLASSES == 6:
            # merge together the 'anger' and 'disgust' classes
            if label == 1:
                label = 0

            # if label has a value greater than zero, subtract one from
            # it to make all labels sequential (not required, but helps
            # when interpreting results)
            # angry, disgust, fear, happy, sad, surprise, and neutral.
            # disgust is merged into angry
            if label > 0:
                label -= 1

        # reshape the flattened pixel list into a 48x48 grayscale image
        image = np.array(image.split(' '), dtype='uint8')
        image = image.reshape((48, 48))

        # add them into training, validation, and test
        if usage == 'Training':
            train_images.append(image)
            train_labels.append(label)
        elif usage == 'PrivateTest':
            val_images.append(image)
            val_labels.append(label)
        else:
            test_images.append(image)
            test_labels.append(label)

    # for clarity, define a directory
    datasets = [
        (train_images, train_labels, config.TRAIN_HDF5),
        (val_images, val_labels, config.VAL_HDF5),
        (test_images, test_labels, config.TEST_HDF5)
    ]

    for images, labels, output_path in datasets:
        print('[INFO] building {}...'.format(output_path))
        writer = HDF5DatasetWriter((len(images), 48, 48), output_path)

        for image, label in zip(images, labels):
            writer.add([image], [label])
        writer.close()
