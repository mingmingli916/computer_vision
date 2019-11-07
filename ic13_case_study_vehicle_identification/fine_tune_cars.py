from ic13_case_study_vehicle_identification.config import car_config as config
import mxnet as mx
import argparse
import logging
import os

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--vgg', required=True, help='path to pre-trained VGGNet for fine-tuning')
ap.add_argument('-c', '--checkpoints', required=True, help='path to output checkpoint directory')
ap.add_argument('-p', '--prefix', required=True, help='name of model prefix')
ap.add_argument('-s', '--start-epoch', type=int, default=0, help='epoch to restart training at')
args = vars(ap.parse_args())

# set the logging level and output file
# the logging filename change according to the start epoch (good practice)
logging.basicConfig(level=logging.DEBUG, filename='training_{}.log'.format(args['start_epoch']), filemode='w')

# determine the batch size with relation to num devices
batch_size = config.BATCH_SIZE * config.NUM_DEVICES

# similar to data generator with augmentation
train_iter = mx.io.ImageRecordIter(
    path_imgrec=config.TRAIN_MX_REC,
    data_shape=(3, 224, 224),
    batch_size=batch_size,
    # randomly crop 224x224 regions from the 256x256 input image
    rand_crop=True,
    rand_mirror=True,
    rotate=15,
    max_shear_ratio=0.1,
    mean_r=config.R_MEAN,
    mean_g=config.G_MEAN,
    mean_b=config.B_MEAN,
    preprocess_threads=config.NUM_DEVICES * 2
)

# val generator without augmentation
val_iter = mx.io.ImageRecordIter(
    path_imgrec=config.VAL_MX_REC,
    data_shape=(3, 224, 224),
    batch_size=batch_size,
    mean_r=config.R_MEAN,
    mean_g=config.G_MEAN,
    mean_b=config.B_MEAN
)

# initialize the optimizer and the training contexts
# opt = mx.optimizer.SGD(
#     learning_rate=1e-5,
#     momentum=.9,
#     wd=.0005,
#     rescale_grad=1.0 / batch_size
# )
opt = mx.optimizer.Adam()
ctx = [mx.cpu(3)]

# construct the checkpoints path and
# initialize the arguments
checkpoints_path = os.path.join(args['checkpoints'], args['prefix'])
arg_params = None
aux_params = None
# 'missing parameters' are parameters that have not been initialized in the network
allow_missing = False

# if there is no specific model starting epoch supplied,
# then we need to build the network architecture
if args['start_epoch'] <= 0:
    print('[INFO] loading the pre-trained model...')
    symbol, arg_params, aux_params = mx.model.load_checkpoint(args['vgg'], 0)
    # Normally we would not allow uninitialized parameters; however, recall that fine-tuning
    # requires us to slice off the head of the network and replace it with a new, uninitialized fully-
    # connected head. Therefore, if we are training from epoch zero, we will allow missing parameters.
    allow_missing = True

    # grab the layers from the pre-trained model, then find the dropout layer prior to the final FC layer
    # HINT: you can find layer names like this:
    # for layer in layers:
    #     print(layer.name)
    # then, append the string '_output' to the layer name
    layers = symbol.get_internals()
    net = layers['drop7_output']

    # construct the new FC layer using the desired number of output class labels
    net = mx.sym.FullyConnected(
        data=net,
        num_hidden=config.NUM_CLASSES,
        name='fc8'
    )
    net = mx.sym.SoftmaxOutput(data=net, name='softmax')

    # construct a new set of network arguments, removing any previous arguments pertaining to FC8
    # i.e. delete any parameter entries for fc8,
    # the FC layer we just surgically removed from the network.
    # (this will allow us to train the final layer)
    arg_params = dict({k: arg_params[k] for k in arg_params if 'fc8' not in k})
# In the case that we are restarting our fine-tuning from a specific epoch,
# we simply need to load the respective weights
else:
    print('[INFO] loading epoch {}...'.format(args['start_epoch']))
    net, arg_params, aux_params = mx.model.load_checkpoint(checkpoints_path, args['start_epoch'])

# initialize the callbacks and evaluation metrics
batch_end_cbs = [mx.callback.Speedometer(batch_size, frequent=50)]
epoch_end_cbs = [mx.callback.do_checkpoint(checkpoints_path)]
metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5), mx.metric.CrossEntropy()]

# construct the model and train it
print('[INFO] training network...')
model = mx.mod.Module(symbol=net, context=ctx)
model.fit(
    train_data=train_iter,
    eval_data=val_iter,
    num_epoch=65,
    begin_epoch=args['start_epoch'],
    initializer=mx.initializer.Xavier(),
    arg_params=arg_params,
    aux_params=aux_params,
    optimizer=opt,
    allow_missing=allow_missing,
    eval_metric=metrics,
    batch_end_callback=batch_end_cbs,
    epoch_end_callback=epoch_end_cbs
)
