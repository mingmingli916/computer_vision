cd ~/PycharmProjects/computer_vision/ic18_ssd/dlib_front_and_rear_vechicles_v1/experiments/training/

# download model
if [ ! -e ssd_inception_v2_coco_2018_01_28.tar.gz]
then
    wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
    tar -zxvf ssd_inception_v2_coco_2018_01_28.tar.gz
fi

# download configuration file
if [ ! -e ssd_vehicles.config ]
then
    wget https://github.com/tensorflow/models/raw/master/research/object_detection/samples/configs/ssd_inception_v2_pets.config
    mv ssd_inception_v2_pets.config ssd_vehicles.config
fi

# alter the config


