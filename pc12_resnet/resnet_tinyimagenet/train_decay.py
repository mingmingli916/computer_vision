import matplotlib
from config import tiny_imagenet_config as config
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyimagesearch.callbacks.epochcheckpoint import EpochCheckpoint
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from keras.callbacks import LearningRateScheduler
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from pyimagesearch.nn.conv.resnet import ResNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import load_model
import keras.backend as K
import argparse
import json
import os
from pyimagesearch.messages import info

matplotlib.use('Agg')
NUM_EPOCHS = 100
INIT_LR = 1e-1


def poly_decay(epoch):
    max_epochs = NUM_EPOCHS
    base_lr = INIT_LR
    power = 1.0

    alpha = base_lr * (1 - (epoch / float(max_epochs))) ** power
    return alpha


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
                help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
args = vars(ap.parse_args())

# construct the training image generator for data aumentation
aug = ImageDataGenerator(rotation_range=18,
                         zoom_range=.15,
                         width_shift_range=.2,
                         height_shift_range=.2,
                         shear_range=.15,
                         horizontal_flip=True,
                         fill_mode='nearest')  # default value is nearest

# load RGB means
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
train_gen = HDF5DatasetGenerator(config.TRAIN_HDF5,
                                 batch_size=config.BATCH_SIZE,
                                 preprocessors=[sp, mp, iap], aug=aug,
                                 classes=config.NUM_CLASSES)
val_gen = HDF5DatasetGenerator(config.VAL_HDF5,
                               batch_size=config.BATCH_SIZE,
                               preprocessors=[sp, mp, iap],
                               classes=config.NUM_CLASSES)

# if there is no specific model checkpoint supplied,
# then initialize the network and compile the model
if args['model'] is None:
    print(info.compiling_model)
    model = ResNet(height=64,
                   width=64,
                   depth=3,
                   classes=config.NUM_CLASSES,
                   stages=(3, 4, 6),
                   filters=(64, 128, 256, 512),
                   reg=.0005,
                   dataset='tiny_imagenet')
    opt = SGD(lr=1e-1, momentum=.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# otherwise, load the checkpoint
else:
    print('[INFO] loading {}...'.format(args['model']))
    model = load_model(args['model'])

    # update the learning rate
    print('[INFO] old learning rate: {}'.format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-5)
    print('[INFO] new learning rate: {}'.format(K.get_value(model.optimizer.lr)))

# construct the set of callbacks
callbacks = [
    EpochCheckpoint(args['checkpoints'], every=5, start_at=args['start_epoch']),
    TrainingMonitor(config.FIG_PATH, json_path=config.JSON_PATH, start_at=args['start_epoch']),
    LearningRateScheduler(poly_decay)
]

# train the network
print(info.training_model)
model.fit_generator(train_gen.generator(),
                    steps_per_epoch=train_gen.num_images // config.BATCH_SIZE,
                    validation_data=val_gen.generator(),
                    validation_steps=val_gen.num_images // config.BATCH_SIZE,
                    epochs=50,
                    max_queue_size=config.BATCH_SIZE * 2,
                    callbacks=callbacks,
                    verbose=1)

print(info.saving_model)
model.save(args['model'])

# close the resources
train_gen.close()
val_gen.close()
