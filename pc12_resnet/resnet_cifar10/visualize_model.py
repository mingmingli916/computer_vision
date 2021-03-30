from pyimagesearch.nn.conv.resnet import ResNet
from keras.utils import plot_model

model = ResNet(width=32,
               height=32,
               depth=3,
               classes=10,
               stages=(9, 9, 9),
               filters=(64, 64, 128, 256),
               reg=.0005)
plot_model(model, to_file='resnet.png', show_shapes=True)
