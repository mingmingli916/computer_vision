from keras.utils import np_utils
import numpy as np
import h5py


class HDF5DatasetGenerator:
    def __init__(self, db_path, batch_size, preprocessors=None, aug=None, binarize=True, classes=2):
        """
        Read data from HDF5 file, apply generator if there exist one.
        Return a generator for Keras.

        :param db_path: the path to HDF5 dataset
        :param batch_size: the size of mini-batches to yield
        :param preprocessors: a list of preprocessors
        :param aug: data augmentation
        :param binarize: binarize the labels as one-hot encoded vectors
        :param classes: the number of unique class labels in the dataset
        """
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        # open the HDF5 dataset for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(db_path)
        self.num_images = self.db['labels'].shape[0]

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0

        # keep looping infinitely -- the model will stop
        # once we have reach the desired number of epochs
        while epochs < passes:
            for i in range(0, self.num_images, self.batch_size):
                # extract the images and labels from the HDF dataset
                images = self.db['images'][i:i + self.batch_size]
                labels = self.db['labels'][i:i + self.batch_size]

                # check to see if the labels should be binarized
                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)

                # check to see if our preprocessors are not None
                if self.preprocessors:
                    # initialize the list of processed images
                    processed_images = []

                    for image in images:
                        # loop over the preprocessors and
                        # apply each to the image
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        # update the list of processed images
                        processed_images.append(image)

                    # update the images array to be
                    # the processed images
                    images = np.array(processed_images)

                # if the data augmentation exists, apply it
                if self.aug:
                    # private next method
                    images, labels = next(self.aug.flow(images, labels, batch_size=self.batch_size))

                    # yield a tuple of images and labels
                yield images, labels

            # increment the total images of epochs
            epochs += 1

    def close(self):
        self.db.close()
