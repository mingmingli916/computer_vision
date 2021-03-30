import mxnet as mx

prefix_path = '/home/hack/PycharmProjects/computer_vision/ic13_case_study_vehicle_identification/vgg16/vgg16'
symbol, arg_params, aux_params = mx.model.load_checkpoint(prefix_path, 0)
layers = symbol.get_internals()
for layer in layers:
    print(layer.name)

# data
# conv1_1_weight
# conv1_1_bias
# conv1_1
# relu1_1
# conv1_2_weight
# conv1_2_bias
# conv1_2
# relu1_2
# pool1
# conv2_1_weight
# conv2_1_bias
# conv2_1
# relu2_1
# conv2_2_weight
# conv2_2_bias
# conv2_2
# relu2_2
# pool2
# conv3_1_weight
# conv3_1_bias
# conv3_1
# relu3_1
# conv3_2_weight
# conv3_2_bias
# conv3_2
# relu3_2
# conv3_3_weight
# conv3_3_bias
# conv3_3
# relu3_3
# pool3
# conv4_1_weight
# conv4_1_bias
# conv4_1
# relu4_1
# conv4_2_weight
# conv4_2_bias
# conv4_2
# relu4_2
# conv4_3_weight
# conv4_3_bias
# conv4_3
# relu4_3
# pool4
# conv5_1_weight
# conv5_1_bias
# conv5_1
# relu5_1
# conv5_2_weight
# conv5_2_bias
# conv5_2
# relu5_2
# conv5_3_weight
# conv5_3_bias
# conv5_3
# relu5_3
# pool5
# flatten_0
# fc6_weight
# fc6_bias
# fc6
# relu6
# drop6
# fc7_weight
# fc7_bias
# fc7
# relu7
# drop7
# fc8_weight
# fc8_bias
# fc8
# prob_label
# prob