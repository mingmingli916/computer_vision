import matplotlib
from ic11_case_emotion_recognition.emotion_recognition.config import emotion_config as config
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.callbacks.epochcheckpoint import EpochCheckpoint
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from pyimagesearch.nn.conv.emotionvggnet import EmotionVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as K
import argparse
import os

matplotlib.use('Agg')
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
parser.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
parser.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
args = vars(parser.parse_args())

# image generator for data augmentation
train_aug = ImageDataGenerator(rotation_range=10,
                               zoom_range=.1,
                               horizontal_flip=True,
                               rescale=1 / 255.0,
                               fill_mode='nearest')  # in range [0,1]
val_aug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
train_gen = HDF5DatasetGenerator(db_path=config.TRAIN_HDF5,
                                 batch_size=config.BATCH_SIZE,
                                 aug=train_aug,
                                 preprocessors=[iap],
                                 classes=config.NUM_CLASSES)
val_gen = HDF5DatasetGenerator(db_path=config.VAL_HDF5,
                               batch_size=config.BATCH_SIZE,
                               aug=val_aug,
                               preprocessors=[iap],
                               classes=config.NUM_CLASSES)

# if there is no specific model checkpoint supplied, the initialize the network and compile the model
if args['model'] is None:
    print('[INFO] compiling model...')
    model = EmotionVGGNet(width=48, height=48, depth=1, classes=config.NUM_CLASSES)
    opt = Adam(lr=1e-3)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
else:
    print('[INFO] loading {}...'.format(args['model']))
    model = load_model(args['model'])

    # update the learning rate
    print('[INFO] old learning rate: {}'.format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-4)
    print('[INFO] new learning rate: {}'.format(K.get_value(model.optimizer.lr)))

# callbacks used to save model and monitor
fig_path = os.path.join(config.OUTPUT_PATH, 'vggnet_emotion.png')
json_path = os.path.join(config.OUTPUT_PATH, 'vggnet_emotion.json')
callbacks = [
    EpochCheckpoint(output_path=args['checkpoints'],
                    every=5,
                    start_at=args['start_epoch']),
    TrainingMonitor(fig_path=fig_path,
                    json_path=json_path,
                    start_at=args['start_epoch'])
]

# train
model.fit_generator(generator=train_gen.generator(),
                    steps_per_epoch=train_gen.num_images // config.BATCH_SIZE,
                    validation_data=val_gen.generator(),
                    validation_steps=val_gen.num_images // config.BATCH_SIZE,
                    epochs=15,
                    max_queue_size=config.BATCH_SIZE * 2,
                    callbacks=callbacks,
                    verbose=1)

train_gen.close()
val_gen.close()
