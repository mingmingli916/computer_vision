import imutils
import cv2


class AspectAwarePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # grab the dimensions of the image and then initializes
        # the delta to use when cropping
        h, w = image.shape[:2]
        dw = 0
        dh = 0

        # if the width is smaller than the height, then resize
        # along the width (i.e., the smaller dimension) and then
        # update the deltas to crop the height to the desired dimension
        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dh = int((image.shape[0] - self.height) / 2)
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dw = int((image.shape[1] - self.width) / 2)

        # now that our images have been resized, we need to
        # re-grab the width and height, followed by performing
        # the crop
        h, w = image.shape[:2]
        image = image[dh:h - dh, dw:w - dw]

        # When cropping (due to rounding errors), our image dimensions
        # may be off by +- one pixel.
        # finally resize the image to the provided spatial
        # dimensions to ensure our output image is always
        # a fixed size
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
