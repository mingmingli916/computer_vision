from ic16_training_a_faster_rcnn_from_scratch.config import lias_config as config
from pyimagesearch.utils.tfannotation import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import os
import cv2


def main(_):
    # open the classes output file
    f = open(config.CLASSES_FILE, 'w')

    for k, v in config.CLASSES.items():
        # construct the class information and write to file
        item = ("item {\n\tid: " + str(v) + "\n\tname: '" + k + "'\n}\n")
        f.write(item)
    f.close()

    # initialize a data dictionary used to map each image filename
    # to all bounding boxes associated with the image, then load
    # the contents of the annotations file
    D = {}
    rows = open(config.ANNOT_PATH).read().strip().split('\n')
    # loop over the individual rows, skipping the header
    for row in rows[1:]:
        # Filename;Annotation tag;Upper left corner X;Upper left corner Y;Lower right corner X;Lower right corner Y;Occluded,
        # aiua120214-0/frameAnnotations-DataLog02142012_external_camera.avi_annotations/stop_1330545910.avi_image0.png;stop;862;104;916;158;0,
        row = row.split(',')[0].split(';')
        image_path, label, startx, starty, endx, endy, _ = row
        startx, starty, endx, endy = float(startx), float(starty), float(endx), float(endy)

        # if we are not interested in the label, ignore it
        if label not in config.CLASSES:
            continue

        # Since an image can contain multiple traffic signs, and therefore multiple bounding boxes,
        # we need to utilize a Python dictionary to map the image path (as the key) to a list of labels and
        # associated bounding boxes (the value).
        # Our goal here is to ensure that if a particular image is labeled as “training”, then all bounding
        # boxes for that image is included in the training set.
        # We want to avoid situations where an image contains bounding boxes for both training and
        # testing sets. Not only is this behavior inefficient, it creates a larger problem — some object detection
        # algorithms utilize hard-negative mining to increase their accuracy by taking non-labeled areas of
        # the image and treating them as negatives.
        p = os.path.sep.join([config.BASE_PATH, image_path])
        b = D.get(p, [])
        b.append((label, (startx, starty, endx, endy)))
        D[p] = b

    train_keys, test_keys = train_test_split(list(D.keys()), test_size=config.TEST_SIZE, random_state=42)

    # initialize the data split files
    datasets = [
        ('train', train_keys, config.TRAIN_RECORD),
        ('test', test_keys, config.TEST_RECORD)
    ]

    for dtype, keys, output_path in datasets:
        print('[INFO] processing "{}"...'.format(dtype))
        writer = tf.io.TFRecordWriter(output_path)

        total = 0
        for k in keys:
            # load the input image from disk as a TF object
            encoded = tf.io.gfile.GFile(k, 'rb').read()
            encoded = bytes(encoded)

            # load the image from disk again, this time as a PIL object
            pil_image = Image.open(k)
            w, h = pil_image.size[:2]

            # parse the filename and encoding from the input path
            filename = k.split(os.path.sep)[-1]
            encoding = filename[filename.rfind('.') + 1:]  # like png, jpg

            # initialize the annotation object
            tf_annot = TFAnnotation()
            tf_annot.image = encoded
            tf_annot.encoding = encoding
            tf_annot.filename = filename
            tf_annot.width = w
            tf_annot.height = h

            # loop over the bounding boxes + labels associated with the image
            for label, (startx, starty, endx, endy) in D[k]:
                # TF assumes all bounding boxes are in the range [0,1]
                xmin = startx / w
                xmax = endx / w
                ymin = starty / h
                ymax = endy / h

                # # This visual validation is critical and should not be skipped under any circumstance!!!!!!
                # # load the input image from disk and denormalize the bounding box coordinates
                # image = cv2.imread(k)
                # startx = int(xmin * w)
                # starty = int(ymin * h)
                # endx = int(xmax * w)
                # endy = int(ymax * h)
                # # draw the bounding box on the image
                # cv2.rectangle(image, (startx, starty), (endx, endy), (0, 255, 0), 2)
                # # show the output image
                # cv2.imshow('Image', image)
                # cv2.waitKey(0)

                # update the bounding boxes + labels lists
                tf_annot.xmins.append(xmin)
                tf_annot.xmaxs.append(xmax)
                tf_annot.ymins.append(ymin)
                tf_annot.ymaxs.append(ymax)
                tf_annot.text_labels.append(label.encode('utf8'))
                tf_annot.classes.append(config.CLASSES[label])
                tf_annot.difficult.append(0)

                total += 1

            # encode the data point attributes using the TensorFlow
            # helper functions
            features = tf.train.Features(feature=tf_annot.build())
            example = tf.train.Example(features=features)

            writer.write(example.SerializeToString())
        writer.close()
        print('[INFO] {} examples saved for "{}"'.format(total, dtype))


if __name__ == '__main__':
    # tf.app.run()
    tf.compat.v1.app.run()
