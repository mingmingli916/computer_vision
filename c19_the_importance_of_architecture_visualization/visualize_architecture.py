from pyimagesearch.nn.conv.lenet import LeNet
from pyimagesearch.nn.conv.shallownet import ShallowNet
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.utils import plot_model

# initialize LeNet and then write the network architecture
# visualization graph to disk
model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file='lenet.png', show_shapes=True)

model = ShallowNet.build(32, 32, 3, 10)
plot_model(model, to_file='shallownet.png', show_shapes=True)

model = MiniVGGNet.build(32, 32, 3, 10)
plot_model(model, to_file='minivgg.png', show_shapes=True)
