import numpy as np
import os


def get_labels(image_paths):
    # suppose {dir}/{label}/{image}
    return [p.split(os.path.sep)[-2] for p in image_paths]


def unique_labels(image_paths):
    class_names = get_labels(image_paths)
    class_names = [str(x) for x in np.unique(class_names)]
    return class_names
