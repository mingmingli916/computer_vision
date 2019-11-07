from ic18_ssd.config import dlib_front_rear_config as config
from pyimagesearch.utils.tfannotation import TFAnnotation
from bs4 import BeautifulSoup
from PIL import Image
import tensorflow as tf
import os


def main(_):
    with open(config.CLASSES_FILE, 'w') as f:
        for k, v in config.CLASSES.items():
            item = ("item {\n" +
                    "\tid: " + str(v) + "\n" +
                    "\tname: '" + k + "'\n" +
                    "}\n")
            f.write(item)

    datasets = [
        ('train', config.TRAIN_XML, config.TRAIN_RECORD),
        ('test', config.TEST_XML, config.TEST_RECORD)
    ]

    for dtype, input_path, output_path in datasets:
        print("[INFO] processing '{}'...".format(dtype))
        contents = open(input_path).read()
        # soup = BeautifulSoup(contents, 'html.parser')
        # note: Do not use the html.parser parameter!
        soup = BeautifulSoup(contents, 'lxml')

        # initialize the TensorFlow writer and initialize
        # the total number of examples written to file
        writer = tf.python_io.TFRecordWriter(output_path)
        total = 0

        for image in soup.find_all('image'):
            # load the input image from disk as a TensorFlow object
            #   <image file='la_hill_st/la_hill_st_000001.jpg'>
            #     <box top='1406' left='668' width='203' height='134'>
            #       <label>front</label>
            #     </box>
            p = os.path.join(config.BASE_PATH, image['file'])
            encoded = tf.gfile.GFile(p, 'rb').read()
            encoded = bytes(encoded)

            # load the image again, this time as a PIL object
            pil_image = Image.open(p)
            w, h = pil_image.size[:2]

            # parse the filename and encoding from the input path
            # file='la_hill_st/la_hill_st_000001.jpg
            filename = image['file'].split(os.path.sep)[-1]
            encoding = filename[filename.rfind('.') + 1:]  # jpg

            # initialize the annotation object used to store information
            # regarding the bounding box + labels
            tf_annot = TFAnnotation()
            tf_annot.image = encoded
            tf_annot.encoding = encoding
            tf_annot.filename = filename
            tf_annot.width = w
            tf_annot.height = h

            # bounding boxes associated with the image
            boxes = image.find_all('box')
            for box in boxes:
                if box.has_attr('ignore'):
                    continue

                # extract the bounding box information + label,
                # ensuring that all bounding box dimensions fit inside the image
                # The dlib library requires that objects have similar bounding box aspect ratios during training —
                # if bounding boxes do not have similar aspect ratios, the algorithm will error out. Since vehicles can
                # appear at the borders of an image, in order to maintain similar aspect ratios, the bounding boxes
                # can actually extend outside the boundaries of the image. The aspect ratio issue isn’t a problem with
                # the TFOD API so we simply clip those values.
                #     <box top='1406' left='668' width='203' height='134'>
                #       <label>front</label>
                #     </box>
                start_x = max(0., float(box['left']))
                start_y = max(0., float(box['top']))
                end_x = min(w, float(box['width']) + start_x)
                end_y = min(h, float(box['height']) + start_y)
                label = box.find('label').text

                # TensorFlow assumes all bounding boxes are in the range [0,1]
                xmin = start_x / w
                xmax = end_x / w
                ymin = start_y / h
                ymax = end_y / h

                # duo to errors in annoation, it may be possible that
                # the minimum values are larger than the maximum values --
                # in this case, treat it as an error during annotation
                # and ignore the bounding box
                # this is important
                # the correctness of the data is the first step to make a correct result
                if xmin > xmax or ymin > ymax:
                    continue

                tf_annot.xmins.append(xmin)
                tf_annot.xmaxs.append(xmax)
                tf_annot.ymins.append(ymin)
                tf_annot.ymaxs.append(ymax)
                tf_annot.text_labels.append(label.encode('utf8'))
                tf_annot.classes.append(config.CLASSES[label])
                tf_annot.difficult.append(0)

                total += 1
            # encode the data point attributes using the TensorFlow helper functions
            features = tf.train.Features(feature=tf_annot.build())
            example = tf.train.Example(features=features)

            # add the example to the writer
            writer.write(example.SerializeToString())
        writer.close()
        print('[INFO] {} examples saved for "{}"'.format(total, dtype))


if __name__ == '__main__':
    tf.app.run()
