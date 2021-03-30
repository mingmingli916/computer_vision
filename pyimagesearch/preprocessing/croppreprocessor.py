import numpy as np
import cv2


class CropPreprocessor:
    def __init__(self, width, height, horizontal=True, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.horizontal = horizontal
        self.inter = inter

    def preprocess(self, image):
        # initialize the list of crops
        crops = []

        # grab the width and height of the image then
        # use these dimensions to define the corners
        # of the image based
        h, w = image.shape[:2]
        corrds = [
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [w - self.width, h - self.height, w, h],
            [0, h - self.height, self.width, h]
        ]

        # compute the center crop of the image as well
        dw = int(.5 * (w - self.width))
        dh = int(.5 * (h - self.height))
        corrds.append([dw, dh, w - dw, h - dh])

        # loop over the coordinates, extract each of the crops,
        # and resize each of them to a fixed size
        for startx, starty, endx, endy in corrds:
            crop = image[starty:endy, startx:endx]
            crop = cv2.resize(crop, (self.width, self.height), interpolation=self.inter)
            crops.append(crop)

        # check to see if the horizontal flips should be taken
        if self.horizontal:
            # compute the horizontal mirror flips for each crop
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)

        return np.array(crops)
