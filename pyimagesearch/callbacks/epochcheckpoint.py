from keras.callbacks import Callback
import os


class EpochCheckpoint(Callback):
    def __init__(self, output_path, every=5, start_at=0):
        # call the parent constructor
        super(Callback, self).__init__()

        self.output_path = output_path
        self.every = every
        self.int_epoch = start_at

    def on_epoch_end(self, epoch, logs={}):
        # check to see if the model should be serialized to disk
        if (self.int_epoch + 1) % self.every == 0:
            p = os.path.sep.join([self.output_path, 'epoch_{}.hdf5'.format(self.int_epoch + 1)])
            self.model.save(p, overwrite=True)

        # increment the internal epoch counter
        self.int_epoch += 1
