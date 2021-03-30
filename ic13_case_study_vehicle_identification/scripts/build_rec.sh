cd ~/PycharmProjects/computer_vision/ic13_case_study_vehicle_identification/stanford_car/
for i in train test val;
do
    if [ ! -e rec/${i}.idx ] && [ ! -e rec/${i}.rec ] ;
    then
        python ~/anaconda3/lib/python3.7/site-packages/mxnet/tools/im2rec.py lists/${i}.lst ''  --resize=256 --encoding='.jpg' --quality=100
        mv lists/${i}.idx lists/${i}.rec rec/
    fi
done
