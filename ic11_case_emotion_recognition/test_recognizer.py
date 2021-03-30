from ic11_case_emotion_recognition.config import emotion_config as config
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, help="path to model checkpoint to load")
args = vars(ap.parse_args())

# initialize dataset generator and image preprocessor
test_aug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()
test_gen = HDF5DatasetGenerator(db_path=config.TEST_HDF5,
                                batch_size=config.BATCH_SIZE,
                                aug=test_aug,
                                preprocessors=[iap],
                                classes=config.NUM_CLASSES)

# load the model
print('[INFO] loading {}'.format(args['model']))
model = load_model(args['model'])

# evaluate the network
loss, acc = model.evaluate_generator(generator=test_gen.generator(),
                                     steps=test_gen.num_images // config.BATCH_SIZE,
                                     max_queue_size=config.BATCH_SIZE * 2)
print('[INFO] accuracy: {:.2f}'.format(acc * 100))

test_gen.close()
