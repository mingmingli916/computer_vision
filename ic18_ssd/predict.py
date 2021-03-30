from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import argparse
import imutils
import cv2
import os
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="base path for frozen checkpoint detection graph")
ap.add_argument("-l", "--labels", required=True, help="labels file")
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-n", "--num-classes", type=int, required=True, help="# of class labels")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                help="minimum probability used to filter weak detections")
args = vars(ap.parse_args())

# initialize a set of colors for class labels
np.random.seed(3)
COLORS = np.random.uniform(0, 255, size=(args['num_classes'], 3))

# initialize the model
model = tf.Graph()

# create a context manager that makes this model the default one for execution
with model.as_default():
    # initialize the graph definition
    graph_def = tf.GraphDef()

    # load the graph from disk
    with tf.gfile.GFile(args['model'], 'rb') as f:
        serialized_graph = f.read()
        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='')

# load the class labels from disk
label_map = label_map_util.load_labelmap(args['labels'])  # load pbtxt
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=args['num_classes'],
                                                            use_display_name=True)
category_idx = label_map_util.create_category_index(categories)  # id -> category

# In order to predict bounding boxes for our input image we first need to create a TensorFlow
# session and grab references to each of the image, bounding box, probability, and classes tensors
# inside the network
# create a session to perform inference
with model.as_default():
    with tf.Session(graph=model) as sess:
        # grab a reference to the input image tensor and the boxes tensor
        image_tensor = model.get_tensor_by_name('image_tensor:0')
        boxes_tensor = model.get_tensor_by_name('detection_boxes:0')

        # for each bounding box we would like to know the score and class label
        scores_tensor = model.get_tensor_by_name('detection_scores:0')
        classes_tensor = model.get_tensor_by_name('detection_classes:0')
        num_detections = model.get_tensor_by_name('num_detections:0')
        # These references will enable us to access their associated values
        # after passing the image through the network.

        # initialize the list of image paths as just a single image
        image_paths = [args['image']]

        # if the input path is actually a directory, then list all image paths in the directory
        if os.path.isdir(args['image']):
            image_paths = sorted(list(paths.list_images(args['image'])))

        for image_path in image_paths:
            # load image
            image = cv2.imread(image_path)
            h, w = image.shape[:2]

            # check to see if we should resize along the width
            if w > h and w > 1000:
                image = imutils.resize(image, width=1000)
            elif h > w and h > 1000:
                image = imutils.resize(image, height=1000)

            # prepare the image for detection
            h, w = image.shape[:2]
            output = image.copy()
            image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)  # RGB in PIL; BGR in cv2
            image = np.expand_dims(image, axis=0)

            # perform inference and compute the bounding boxes,
            # probabilities, and class labels
            boxes, scores, labels, N = sess.run([boxes_tensor, scores_tensor, classes_tensor, num_detections],
                                                feed_dict={image_tensor: image})

            # squeeze the lists into a single dimension (expand_dims previously)
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            labels = np.squeeze(labels)

            # loop over the bounding box predictions
            for box, score, label in zip(boxes, scores, labels):
                if score < args['min_confidence']:
                    continue

                # scale the bounding box from the range [0,1] to [w,h]
                starty, startx, endy, endx = box
                startx = int(startx * w)
                starty = int(starty * h)
                endx = int(endx * w)
                endy = int(endy * h)

                # draw the prediction on the output image
                label = category_idx[label]
                idx = int(label['id']) - 1
                label = '{}: {:.2f}'.format(label['name'], score)
                cv2.rectangle(output, (startx, starty), (endx, endy), COLORS[idx], 3)
                y = starty - 10 if starty - 10 > 10 else starty + 10
                cv2.putText(output, label, (startx, y), cv2.FONT_HERSHEY_SIMPLEX, .6, COLORS[idx], 2)
                cv2.putText(output, 'MINGMING LI', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)

            cv2.imshow('Output', output)

            # if the 'q' key is pressed, stop the loop
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
