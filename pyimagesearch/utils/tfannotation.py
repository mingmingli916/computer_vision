# used to serialize and encode Python attributes and objects
from object_detection.utils.dataset_util import bytes_list_feature, float_list_feature, int64_list_feature
from object_detection.utils.dataset_util import bytes_feature, int64_feature


class TFAnnotation:
    """
    TensorFlow Object Detection API compatible annotation.
    """

    def __init__(self):
        # initialize the bounding box + label lists
        self.xmins = []
        self.xmaxs = []
        self.ymins = []
        self.ymaxs = []
        self.text_labels = []  # human readable
        self.classes = []  # integer
        self.difficult = []

        self.image = None
        self.width = None
        self.height = None
        self.encoding = None
        self.filename = None

    def build(self):
        # encode the attribute using their respective TF encoding function
        xmins = float_list_feature(self.xmins)
        xmaxs = float_list_feature(self.xmaxs)
        ymins = float_list_feature(self.ymins)
        ymaxs = float_list_feature(self.ymaxs)
        text_labels = bytes_list_feature(self.text_labels)
        classes = int64_list_feature(self.classes)
        difficult = int64_list_feature(self.difficult)

        image = bytes_feature(self.image)
        w = int64_feature(self.width)
        h = int64_feature(self.height)
        encoding = bytes_feature(self.encoding.encode('utf8'))
        filename = bytes_feature(self.filename.encode('utf8'))

        # construct the TF-compatible data dictionary
        data = {
            'image/height': h,
            'image/width': w,
            'image/filename': filename,
            'image/source_id': filename,
            'image/encoded': image,
            'image/format': encoding,
            'image/object/bbox/xmin': xmins,
            'image/object/bbox/xmax': xmaxs,
            'image/object/bbox/ymin': ymins,
            'image/object/bbox/ymax': ymaxs,
            'image/object/class/text': text_labels,
            'image/object/class/label': classes,
            'image/object/difficult': difficult
        }

        return data
