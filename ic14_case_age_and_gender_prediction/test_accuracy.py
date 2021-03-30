from pyimagesearch.utils.agegenderhelper import AgeGenderHelper

from ic14_case_age_and_gender_prediction.config import age_gender_config as config
from pyimagesearch.mxcallbacks.mxmetrics import _compute_one_off
import mxnet as mx
import argparse
import pickle
import json
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True, help="name of model prefix")
ap.add_argument("-e", "--epoch", type=int, required=True, help="epoch # to load")
args = vars(ap.parse_args())

# load RGB means
means = json.loads(open(config.DATASET_MEAN).read())

# construct the testing image iterator
test_iter = mx.io.ImageRecordIter(
    path_imgrec=config.TEST_MX_REC,
    data_shape=(3, 227, 227),
    batch_size=config.BATCH_SIZE,
    mean_r=means['R'],
    mean_g=means['G'],
    mean_b=means['B']
)

# load the checkpoint
print('[INFO] loading model...')
checkpoints_path = os.path.join(args['checkpoints'], args['prefix'])
model = mx.model.FeedForward.load(checkpoints_path, epoch=args['epoch'])

# compile the model
model = mx.model.FeedForward(
    ctx=[mx.cpu(0)],
    symbol=model.symbol,
    arg_params=model.arg_params,
    aux_params=model.aux_params
)

# make predictions on the testing data
print('[INFO] predicting on "{}" test data...'.format(config.DATASET_TYPE))
metrics = [mx.metric.Accuracy()]
acc = model.score(test_iter, eval_metric=metrics)  # this method

# display the rank-1 accuracy
print('[INFO] rank-1: {:.2f}%'.format(acc[0] * 100))

# if we are working with the age dataset, we also need to compute the one-off accuracy # as well
if config.DATASET_TYPE == 'age':
    # recompile the model so that we can compute our custom one-off evaluation metric
    arg = model.arg_params
    aux = model.aux_params
    # todo what is the difference of this compile manner and the above compile manner
    model = mx.mod.Module(symbol=model.symbol, context=[mx.cpu(1)])
    model.bind(data_shapes=test_iter.provide_data, label_shapes=test_iter.provide_label)
    model.set_params(arg_params=arg, aux_params=aux)

    # load the label encoder and build the one-off mapping
    le = pickle.loads(open(config.LABEL_ENCODER_PATH, 'rb').read())
    agh = AgeGenderHelper(config)
    one_off = agh.build_one_off_mappings(le)

    # compute the acc
    acc = _compute_one_off(model, test_iter, one_off)
    print('[INFO] one-off: {:.2f}%'.format(acc * 100))
