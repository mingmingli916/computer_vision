from pyimagesearch.messages import info
from pyimagesearch.utils.ranked import rank5_accuracy
import argparse
import pickle
import h5py

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True,
                help="path HDF5 database")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained model")
args = vars(ap.parse_args())

# load the pre-trained model
print(info.loading_model)
model = pickle.loads(open(args['model'], 'rb').read())

# open the HDF5 database for reading then determine the index of
# the training and testing split, provided that this data was
# already shuffled prior to writing it to disk
db = h5py.File(args['db'], 'r')
i = int(db['features'].shape[0] * .75)
features = db['features']
labels = db['labels']
label_names = db['label_names']

# make predictions on the testing set then computer the
# rank-1 and rank-5 accuracy
print(info.predicting)
preds = model.predict_proba(features[i:])
rank1, rank5 = rank5_accuracy(preds, labels[i:])

# display
print('[INFO] rank-1: {:.2f}%'.format(rank1 * 100))
print('[INFO] rank-5: {:.2f}%'.format(rank5 * 100))

# flowers17
# [INFO] rank-1: 90.59%
# [INFO] rank-5: 100.00%


# caltech101
# [INFO] rank-1: 93.31%
# [INFO] rank-5: 98.82%
