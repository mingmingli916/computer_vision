# a blueprint for training CNN on the ImageNet
# the change is:
# 1. configuration file
# 2. data_shape
# 3. optimizer
# 4. name of the model

from config import imagenet_alexnet_config as config
from pyimagesearch.nn.mxconv.alexnet import AlexNet
import mxnet as mx
import argparse
import logging
import json
import os

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--checkpoints', required=True, help='path to output checkpoint directory')
ap.add_argument('-p', '--prefix', required=True, help='name of model prefix')
ap.add_argument('-s', '--start-epoch', type=int, default=0, help='epoch to restart training at')
args = vars(ap.parse_args())

# set the logging level and output file
logging.basicConfig(level=logging.DEBUG, filename='training_{}.log'.format(args['start_epoch']), filemode='w')

# load RGB and determine the batch size
means = json.loads(open(config.DATASET_MEAN).read())
batch_size = config.BATCH_SIZE * config.NUM_DEVICES

# training image iterator
train_iter = mx.io.ImageRecordIter(path_imgrec=config.TRAIN_MX_REC,
                                   data_shape=(3, 227, 227),
                                   batch_size=batch_size,
                                   rand_crop=True,
                                   rand_mirror=True,
                                   rotate=15,
                                   max_shear_ratio=.1,
                                   mean_r=means['R'],
                                   mean_g=means['G'],
                                   mean_b=means['B'],
                                   preprocess_threads=config.NUM_DEVICES * 2)  # 2 is experience

# validation image iterator
val_iter = mx.io.ImageRecordIter(path_imgrec=config.VAL_MX_REC,
                                 data_shape=(3, 227, 227),
                                 batch_size=batch_size,
                                 mean_r=means['R'],
                                 mean_g=means['G'],
                                 mean_b=means['B'])

# optimizer
# note rescale_grad when run in parallel
opt = mx.optimizer.SGD(learning_rate=.01, momentum=.9, wd=.0005, rescale_grad=1. / batch_size)

# construct the checkpoints path, initialize the model argument and auxiliary parameters
checkpoints_path = os.path.sep.join([args['checkpoints'], args['prefix']])
arg_params = None
aux_params = None

# if there is no specific model starting epoch supplied, then initialize the network
if args['start_epoch'] <= 0:
    print('[INFO] building network...')
    model = AlexNet(config.NUM_CLASSES)
else:
    print('[INFO] loading epoch {}...'.format(args['start_epoch']))
    model = mx.model.FeedForward.load(checkpoints_path, args['start_epoch'])

    # update parameters and the model
    arg_params = model.arg_params
    aux_params = model.aux_params
    model = model.symbol

# compile the model
model = mx.model.FeedForward(symbol=model,
                             # ctx=[mx.cpu(0), mx.cpu(1), mx.cpu(2), mx.cpu(3)],
                             initializer=mx.initializer.Xavier(),
                             arg_params=arg_params,
                             aux_params=aux_params,
                             optimizer=opt,
                             num_epoch=70,
                             begin_epoch=args['start_epoch'])

# initialize the callbacks and evaluation metrics
batch_end_callbacks = [mx.callback.Speedometer(batch_size, 500)]
epoch_end_callbacks = [mx.callback.do_checkpoint(checkpoints_path)]
# monitoring rank-1, rank-5, as well as cross-entropy
metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5), mx.metric.CrossEntropy()]

# train the network
print('[INFO] training network...')
model.fit(X=train_iter,
          eval_data=val_iter,
          eval_metric=metrics,
          batch_end_callback=batch_end_callbacks,
          epoch_end_callback=epoch_end_callbacks)
