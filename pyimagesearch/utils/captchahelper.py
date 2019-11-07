import imutils
import cv2


def preprocess(image, width, height):
    # grab the dimensions of the image, height and width
    h, w = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width, keep the aspect ratio
    if w > h:
        #         r = width / float(w)
        #         dim = (width, int(h * r))
        image = imutils.resize(image, width=width)
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height
    # to obtain the target dimensions
    pad_w = int((width - image.shape[1]) / 2.0)
    pad_h = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle
    # any rounding issues
    image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    return image



