# set the matplotlib backend so figures can be saved in the background
import matplotlib
from pyimagesearch.messages import info
from .config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.patchpreprocessor import PatchPreprocessor
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from pyimagesearch.nn.conv.alexnet import AlexNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import os

matplotlib.use("Agg")

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=.15,
                         width_shift_range=.2,
                         height_shift_range=.2,
                         shear_range=.15,
                         horizontal_flip=True,
                         fill_mode='nearest')

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(227, 227)  # used in the validation data generator
pp = PatchPreprocessor(227, 227)  # used in training time
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
train_gen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=aug, preprocessors=[pp, mp, iap], classes=2)
# done misspell TRAIN_HDF5 into TEST_HDF5, resulting in acc and val_acc 50% forever
val_gen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, preprocessors=[sp, mp, iap], classes=2)

# initialize the optimizer
print(info.compiling_model)
opt = Adam(lr=1e-3)
model = AlexNet.build(width=227, height=227, depth=3, classes=2, reg=.0002)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# construct the set of callbacks
path = os.path.join(config.OUTPUT_PATH, '{}.png'.format(os.getpid()))
callbacks = [TrainingMonitor(path)]

# train the network
model.fit_generator(train_gen.generator(),
                    steps_per_epoch=train_gen.num_images // config.BATCH_SIZE,
                    validation_data=val_gen.generator(),
                    validation_steps=val_gen.num_images // config.BATCH_SIZE,
                    epochs=75,
                    max_queue_size=config.BATCH_SIZE * 2,
                    callbacks=callbacks,
                    verbose=1)

# save the model to file
print(info.saving_model)
model.save(config.MODEL_PATH, overwrite=True)

# close the HDF5 datasets
train_gen.close()
val_gen.close()

# Process finished with exit code 137 (interrupted by signal 9: SIGKILL)
# adjust batch size

# todo debug 50% forever
