from ic14_case_age_and_gender_prediction.config import age_gender_config as config
from pyimagesearch.nn.mxconv.agegendernet import MxAgeGenderNet
from pyimagesearch.utils.agegenderhelper import AgeGenderHelper
from pyimagesearch.mxcallbacks.mxmetrics import one_off_callback
import mxnet as mx
import argparse
import logging
import pickle
import json
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True, help="name of model prefix")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

# set the logging level and output file
logging.basicConfig(
    level=logging.DEBUG,
    filename='training_{}.log'.format(args['start_epoch']),
    filemode='w'
)

# determine the batch and load the mean pixel values
batch_size = config.BATCH_SIZE * config.NUM_DEVICES
means = json.loads(open(config.DATASET_MEAN).read())

# construct the training image iterator
train_iter = mx.io.ImageRecordIter(
    path_imgrec=config.TRAIN_MX_REC,
    data_shape=(3, 227, 227),
    batch_size=batch_size,
    rand_crop=True,
    rand_mirror=True,
    rotate=7,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"],
    preprocess_threads=config.NUM_DEVICES * 2
)

# construct the validation image iterator
val_iter = mx.io.ImageRecordIter(
    path_imgrec=config.VAL_MX_REC,
    data_shape=(3, 227, 227),
    batch_size=batch_size,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"]
)

# optimizer
opt = mx.optimizer.SGD(
    learning_rate=1e-3,
    momentum=.9,
    wd=0.0005,
    rescale_grad=1.0 / batch_size
)

# construct the checkpoints path, initialize the model argument and auxiliary parameters
checkpoints_path = os.path.join(args['checkpoints'], args['prefix'])
arg_params = None
aux_params = None

# if there is no specific model starting epoch supplied, then
# initialize the network
if args['start_epoch'] <= 0:
    print('[INFO] building network...')
    symbol = MxAgeGenderNet(config.NUM_CLASSES)
# otherwise, a specific checkpoint was supplied
else:
    print('[INFO] loading epoch {}...'.format(args['start_epoch']))
    symbol, arg_params, aux_params = mx.model.load_checkpoint(checkpoints_path, args['start_epoch'])

# compile the model
model = mx.model.FeedForward(
    ctx=[mx.cpu(2), mx.cpu(3)],
    symbol=symbol,
    initializer=mx.initializer.Xavier(),
    arg_params=arg_params,
    aux_params=aux_params,
    optimizer=opt,
    num_epoch=110,
    begin_epoch=args['start_epoch']
)

# initialize the callbacks and evaluation metrics
batch_end_cb = [mx.callback.Speedometer(batch_size, frequent=10)]
epoch_end_cb = [mx.callback.do_checkpoint(checkpoints_path)]
metric = [mx.metric.Accuracy(), mx.metric.CrossEntropy()]  # note: CrossEntropy() not CrossEntropy
# unsupported operand type(s) for +=: 'float' and 'CrossEntropy'

# check to see if the one-off accuracy callback should be used
if config.DATASET_TYPE == 'age':
    le = pickle.loads(open(config.LABEL_ENCODER_PATH, 'rb').read())
    agh = AgeGenderHelper(config)
    one_off = agh.build_one_off_mappings(le)
    epoch_end_cb.append(one_off_callback(train_iter, val_iter, one_off, mx.cpu(0)))

# train the network
print('[INFO] training network...')
model.fit(
    X=train_iter,
    eval_data=val_iter,
    eval_metric=metric,
    batch_end_callback=batch_end_cb,
    epoch_end_callback=epoch_end_cb
)
