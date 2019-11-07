python ../predict.py --model ~/PycharmProjects/computer_vision/ic18_ssd/dlib_front_and_rear_vechicles_v1/experiments/exported_model/frozen_inference_graph.pb \
--labels ~/PycharmProjects/computer_vision/ic18_ssd/dlib_front_and_rear_vechicles_v1/records/classes.pbtxt \
--num-classes 2 \
--image ~/PycharmProjects/computer_vision/ic18_ssd/dlib_front_and_rear_vechicles_v1/youtube_frames/crazy-000010.jpg
