from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainingMonitor(BaseLogger):
    def __init__(self, fig_path, json_path=None, start_at=0):
        # store the output path for the figure, the path to the JSON
        # serialized file, and the starting epoch
        super(TrainingMonitor, self).__init__()  # todo what does thi mean?
        self.fig_path = fig_path
        self.json_path = json_path
        self.start_at = start_at

    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}

        # if the JSON history path exist, load the training history
        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.H = json.loads(open(self.json_path).read())

                # check to see if a starting epoch was supplied
                if self.start_at > 0:
                    # loop over the entries in the history log and
                    # trim any entries that are past the starting epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.start_at]

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        for k, v in logs.items():
            l = self.H.get(k, [])
            l.append(float(v))
            self.H[k] = l

        # check to see if the training history should be serialized to a file
        if self.json_path is not None:
            f = open(self.json_path, 'w')
            f.write(json.dumps(self.H))
            f.close()

        # ensure at least two epochs have passed before plotting
        # (epoch starts at zero)
        if len(self.H['loss']) > 1:
            N = np.arange(0, len(self.H['loss']))
            plt.style.use('ggplot')
            plt.figure()
            plt.plot(N, self.H['loss'], label='train_loss')
            plt.plot(N, self.H['val_loss'], label='val_loss')
            plt.plot(N, self.H['acc'], label='acc')
            plt.plot(N, self.H['val_acc'], label='val_acc')
            # plt.plot(N, self.H['accuracy'], label='accuracy')  # matching your Keras version
            # plt.plot(N, self.H['val_accuracy'], label='val_accuracy')
            plt.title('Training Loss and Accuracy [Epoch {}]'.format(len(self.H['loss'])))
            plt.xlabel('Epoch #')
            plt.ylabel('Loss/Accuracy')
            plt.legend()

            # save the figure
            plt.savefig(self.fig_path)
            plt.close()
