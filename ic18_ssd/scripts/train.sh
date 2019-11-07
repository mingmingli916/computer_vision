# run setup.sh first
cd ~/models/research/

python object_detection/legacy/train.py --logtostderr \
--pipeline_config_path ~/PycharmProjects/computer_vision/ic18_ssd/dlib_front_and_rear_vechicles_v1/experiments/training/ssd_vehicles.config \
--train_dir  ~/PycharmProjects/computer_vision/ic18_ssd/dlib_front_and_rear_vechicles_v1/experiments/training/