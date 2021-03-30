import cv2


class MeanPreprocessor:
    def __init__(self, r_mean, g_mean, b_mean):
        self.r_mean = r_mean
        self.g_mean = g_mean
        self.b_mean = b_mean

    def preprocess(self, image):
        # split the image into its respective red,
        # green, and blue channels
        B, G, R = cv2.split(image.astype('float'))

        # subtract the means for each channel
        R -= self.r_mean
        G -= self.g_mean
        B -= self.b_mean

        return cv2.merge([B, G, R])
