from imutils import paths
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='path to input directory of images')
ap.add_argument('-a', '--annotation', required=True, help='path to output directory of annotations')
args = vars(ap.parse_args())

# grab the image paths then initialize the directory of character counts
image_paths = list(paths.list_images(args['input']))
counts = {}

for i, image_path in enumerate(image_paths):
    print('[INFO] processing image {}/{}'.format(i + 1, len(image_paths)))
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # pad the image to ensure digits caught on the border of the image
        # are retained
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        # threshold the image to reveal the digits
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # find contours in the image, keeping only the four largest ones
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cnts = cnts[0] if imutils.is_cv3() else cnts[1]
        # error: (-215:Assertion failed) npoints >= 0 && (depth == CV_32F || depth == CV_32S) in function 'pointSetBoundingRect'
        # this bug is caused by the OpenCV version
        cnts = cnts[1] if imutils.is_cv3() else cnts[0]

        # Just in case there is "noise" in the image, we sort the contours by
        # their area, keeping only the four largest one
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]
            cv2.imshow('ROI', imutils.resize(roi, width=28))
            key = cv2.waitKey(0)

            # if the "'" key is pressed, then ignore the character
            if key == ord("'"):
                print('[INFO] ignoring character')
                continue

            # grab the key that was pressed and construct the path
            # the output directory
            key = chr(key).upper()
            dir_path = os.path.join(args['annotation'], key)

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            count = counts.get(key, 1)
            p = os.path.sep.join([dir_path, '{}.png'.format(str(count).zfill(6))])
            cv2.imwrite(p, roi)

            # increment the count for the current key
            counts[key] = count + 1
    # we are trying to control-c out of the script, so break from the loop
    except KeyboardInterrupt:
        print('[INFO] manually leaving script')
        break
    except Exception as err:
        print(err)
        print('[INFO] skipping image...')
