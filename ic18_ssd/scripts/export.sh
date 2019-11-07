cd ~/models/research/

python ~/models/research/object_detection/export_inference_graph.py --input_type image_tensor \
--pipeline_config_path ~/PycharmProjects/computer_vision/ic18_ssd/dlib_front_and_rear_vechicles_v1/experiments/training/ssd_vehicles.config \
--trained_checkpoint_prefix ~/PycharmProjects/computer_vision/ic18_ssd/dlib_front_and_rear_vechicles_v1/experiments/training/model.ckpt-1791 \
--output_directory ~/PycharmProjects/computer_vision/ic18_ssd/dlib_front_and_rear_vechicles_v1/experiments/exported_model/