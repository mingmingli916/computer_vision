from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2


# grayscale image
def convolve(image, K):
    # grab the spatial dimensions of the image and kernel
    i_h, i_w = image.shape[:2]
    k_h, k_w = K.shape[:2]

    # allocate memory for the output image, taking care to 'pad'
    # the borders of the input image so the spatial size (i.e.,
    # width and height) are not reduced
    pad = (k_w - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((i_h, i_w), dtype='float')

    # loop over the input image, 'sliding' the kernel across
    # each (x, y)-coordinate from left-to-right and top-to-bottom
    for y in range(pad, i_h + pad):
        for x in range(pad, i_w + pad):
            # extract the ROI of the image by extracting the
            # center region of the current (x, y)-coordinates
            # dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]  # notice the x and y order

            # perform the actual convolution by taking the element-wise multiplication
            # between the ROI and the kernel, then summing the matrix
            k = (roi * K).sum()

            # store the convolved value in the output (x, y)-coordinate of the output image
            output[y - pad, x - pad] = k

    # rescale the output image to be in the range [0, 255]
    # because the convolution computation can change the value out of
    # the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype('uint8')
    return output


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the input image')
args = vars(ap.parse_args())

# construct average blurring kernels used to smooth an image
small_blur = np.ones((7, 7), dtype='float') * (1.0 / (7 * 7))
large_blur = np.ones((21, 21), dtype='float') * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array(([0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]), dtype='int')

# construct the Laplacian kernel used to detect edge-like regions of an image
laplacian = np.array(([0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]), dtype="int")

# construct the Sobel x-axis kernel
sobel_x = np.array(([-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]), dtype="int")

# construct the Sobel y-axis kernel
sobel_y = np.array(([-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]), dtype="int")

# construct an emboss kernel
emboss = np.array(([-2, -1, 0],
                   [-1, 1, 1],
                   [0, 1, 2]), dtype="int")

# construct the kernel bank, a list of kernels we're going to apply
# using both our custom 'convolve' function and OpenCV's 'filter2D function
kernel_bank = (('small_blur', small_blur),
               ('large_blur', large_blur),
               ('sharpen', sharpen),
               ('laplacian', laplacian),
               ('sobel_x', sobel_x),
               ('sobel_y', sobel_y),
               ('emboss', emboss))

# load the input image and convert it to grayscale
image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', gray)

for kernel_name, K in kernel_bank:
    # apply the kernel to the grayscale image using both our custom 'convolve'
    # function and OpenCV's 'filter2D' function
    print('[INFO] applying {} kernel'.format(kernel_name))
    convolve_output = convolve(gray, K)
    opencv_output = cv2.filter2D(gray, -1, K)

    # show the output images
    cv2.imshow('{} - convolve'.format(kernel_name), convolve_output)
    # your can comment this after the demonstration
    # cv2.imshow('{} - opencv'.format(kernel_name), opencv_output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
