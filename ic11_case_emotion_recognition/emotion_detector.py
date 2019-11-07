from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from ic11_case_emotion_recognition.emotion_recognition.config import emotion_config as config

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained emotion detector CNN")
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

# load the face detector cascade, emotion detection CNN, then define the list of emotion labels
detector = cv2.CascadeClassifier(args['cascade'])
model = load_model(args['model'])
EMOTIONS = ['argry', 'scared', 'happy', 'sad', 'surprised', 'neutral']

# if a video path was not supplied, grab the reference to the webcam
if not args.get('video', False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args['video'])

while True:
    grabbed, frame = camera.read()
    # if we are viewing a video and we did not grab a frame,
    # then we have reached end of the video
    if args.get('video') and not grabbed:
        break

    frame = imutils.resize(frame, height=config.SHOW_HEIGHT)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas_width = frame.shape[1] // 4
    canvas = np.zeros((config.SHOW_HEIGHT, canvas_width, 3), dtype='uint8')
    frame_clone = frame.copy()

    rects = detector.detectMultiScale(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    if len(rects) > 0:
        # determine the largest face area
        rect = sorted(rects,
                      reverse=True,
                      key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        fx, fy, fw, fh = rect

        # extract the face ROI from the image and pre-process it for the network
        roi = gray[fy:fy + fh, fx:fx + fw]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]

        for i, (emotion, prob) in enumerate(zip(EMOTIONS, preds)):
            text = '{}: {:.2f}%'.format(emotion, prob * 100)

            # draw the label + probability bar on the canvas
            w = int(prob * 300)
            bar_height = config.SHOW_HEIGHT // config.NUM_CLASSES
            cv2.rectangle(canvas, (0, (i * bar_height)), (w, (i * bar_height) + bar_height), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * bar_height) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        # draw the label on the frame
        cv2.putText(frame_clone, label, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        cv2.rectangle(frame_clone, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

        # show
        concat = np.hstack([frame_clone, canvas])
        cv2.imshow('Emotion', concat)
        # cv2.imshow('Face', frame_clone)
        # cv2.imshow('Probability', canvas)

        # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
camera.release()
cv2.destroyAllWindows()
