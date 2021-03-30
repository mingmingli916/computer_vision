from pyimagesearch.messages import info
from config import tiny_imagenet_config as config
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyimagesearch.utils.ranked import rank5_accuracy
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from keras.models import load_model
import json

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
iap = ImageToArrayPreprocessor()

# initialize the testing dataset generator
test_gen = HDF5DatasetGenerator(config.TEST_HDF5, batch_size=config.BATCH_SIZE, preprocessors=[sp, mp, iap],
                                classes=config.NUM_CLASSES)

# load the pre-trained network
print(info.loading_model)
model = load_model(config.MODEL_PATH)
print(model.summary())

# make predictions on the testing data
print('[INFO] predicting on test data...')
preds = model.predict_generator(test_gen.generator(),
                                steps=test_gen.num_images // config.BATCH_SIZE,
                                max_queue_size=config.BATCH_SIZE * 2)

# compute the rank-1 and rank-5 accuracies
rank1, rank5 = rank5_accuracy(preds, test_gen.db['labels'])
print('[INFO] rank-1: {:.2f}%'.format(rank1 * 100))
print('[INFO] rank-5: {:.2f}%'.format(rank5 * 100))

test_gen.close()
