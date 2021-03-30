from ic13_case_study_vehicle_identification.config import car_config as config
from pyimagesearch.utils.ranked import rank5_accuracy
import mxnet as mx
import argparse
import pickle
import os

# commandline arguments
ap = argparse.ArgumentParser()
# notice: if the path is wrong, it will produce the error: MXNetError(py_str(_LIB.MXGetLastError()))
ap.add_argument('-c', '--checkpoints', required=True, help='path to output checkpoint directory')
ap.add_argument('-p', '--prefix', required=True, help='name of model prefix')
ap.add_argument('-e', '--epoch', type=int, required=True, help='epoch to load')
args = vars(ap.parse_args())

# load the encoder
le = pickle.loads(open(config.LABEL_ENCODER_PATH, 'rb').read())

# construct the validation image iterator
test_iter = mx.io.ImageRecordIter(
    path_imgrec=config.TEST_MX_REC,
    data_shape=(3, 224, 224),  # todo 256 and 224 without crop?
    batch_size=config.BATCH_SIZE,
    mean_r=config.R_MEAN,
    mean_g=config.G_MEAN,
    mean_b=config.B_MEAN
)

# load pre-trained model
print('[INFO] loading pre-trained model...')
checkpoints_path = os.path.join(args['checkpoints'], args['prefix'])
# determine load which model, the model is different from each other with epoch name
symbol, arg_params, aux_params = mx.model.load_checkpoint(checkpoints_path, args['epoch'])

# construct the model
model = mx.mod.Module(symbol=symbol, context=[mx.cpu(0)])
model.bind(data_shapes=test_iter.provide_data, label_shapes=test_iter.provide_label)
model.set_params(arg_params=arg_params, aux_params=aux_params)

# initialize the list of predictions and targets
print('[INFO] evaluating model...')
predictions = []
targets = []

# loop over the predictions in batches
#         >>> for pred, i_batch, batch in module.iter_predict(eval_data):
#         ...     # pred is a list of outputs from the module
#         ...     # i_batch is a integer
#         ...     # batch is the data batch from the data iterator
for preds, _, batch in model.iter_predict(test_iter):
    # convert the batch of predictions and labels to NumPy arrays
    preds = preds[0].asnumpy()
    labels = batch.label[0].asnumpy().astype('int')

    # update the predictions and targets list
    predictions.extend(preds)
    targets.extend(labels)

# ensures that both the targets and predictions lists are the same length. This line
# is a requirement as the iter_predict function will only return batch in sizes of powers of two
# for efficiency reasons (or it could also be a small bug in the function). Thus, it’s nearly always the
# case that the targets list is longer than the predictions list. We can easily fix the discpreancy
# by applying array slicing.
targets = targets[:len(predictions)]

rank1, rank5 = rank5_accuracy(predictions, targets)
print('[INFO] rank-1: {:.2f}%'.format(rank1 * 100))
print('[INFO] rank-5: {:.2f}%'.format(rank5 * 100))
