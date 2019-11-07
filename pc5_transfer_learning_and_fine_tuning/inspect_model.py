from keras.applications import VGG16
import argparse
from pyimagesearch.messages import info

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--include-top', type=int, default=-1, help='whether or not to include top of CNN')
args = vars(ap.parse_args())

# load model
print(info.loading_model)
model = VGG16(weights='imagenet', include_top=args['include_top'] > 0)

print('[INFO] showing layers...')
for i, layer in enumerate(model.layers):
    print('[INFO] {}\t{}'.format(i, layer.__class__.__name__))
