# source setup.sh first
cd ~/models/research/

python object_detection/legacy/eval.py --logtostderr \
--pipeline_config_path ~/PycharmProjects/computer_vision/ic18_ssd/dlib_front_and_rear_vechicles_v1/experiments/training/ssd_vehicles.config \
--checkpoint_dir  ~/PycharmProjects/computer_vision/ic18_ssd/dlib_front_and_rear_vechicles_v1/experiments/training \
--eval_dir ~/PycharmProjects/computer_vision/ic18_ssd/dlib_front_and_rear_vechicles_v1/experiments/

