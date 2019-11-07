import h5py
import os


class HDF5DatasetWriter:
    def __init__(self, dimension, output_path, data_key='images', buf_size=1000):
        """
        :param dimension: controls the dimension or shape of the data will be stored in the dataset.
        :param output_path: path to write HDF5 file.
        :param data_key: the name of the dataset that will store the data.
        :param buf_size: controls the size of in-memory buffer.
        """
        # check to see if the output path exists, and if so, raise an exception
        if os.path.exists(output_path):
            raise ValueError('The supplied "output_path" already '
                             'exists and cannot be overwritten. Manually delete '
                             'the file before continuing.', output_path)

        # open the HDF5 database for writing and create two datasets:
        # one to store the image/features and another to store
        # the class labels
        self.db = h5py.File(output_path, 'w')
        self.data = self.db.create_dataset(data_key, dimension, dtype='float')
        self.labels = self.db.create_dataset('labels', (dimension[0],), dtype='int')

        # store the buffer size, then initialize the buffer itself
        # along with the index into the datasets
        self.buf_size = buf_size
        self.buffer = {'data': [], 'labels': []}
        self.idx = 0

    def add(self, rows, labels):
        # add the rows and labels to the buffer
        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)

        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer['data']) >= self.buf_size:
            self.flush()

    def flush(self):
        # write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer['data'])
        self.data[self.idx:i] = self.buffer['data']
        self.labels[self.idx:i] = self.buffer['labels']
        self.idx = i
        self.buffer = {'data': [], 'labels': []}

    def store_class_labels(self, class_labels):
        # create a dataset to store the actual class label names,
        # then store the class labels
        # dt = h5py.special_dtype(vlen=unicode) # this is the python 2.x
        dt = h5py.special_dtype(vlen=str)  # vlen – Base type for HDF5 variable-length datatype.
        label_set = self.db.create_dataset('label_names', (len(class_labels),), dtype=dt)
        label_set[:] = class_labels

    def close(self):
        # check to see if there are any other entries in the buffer
        # that need to be flushed to disk
        if len(self.buffer['data']) > 0:
            self.flush()

        # close the dataset
        self.db.close()
