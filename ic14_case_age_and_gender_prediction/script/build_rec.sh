cd ~/PycharmProjects/computer_vision/ic14_case_age_and_gender_prediction/adience/
for i in age_train age_test age_val gender_train gender_test gender_val;
do
    if [ ! -e rec/${i}.idx ] && [ ! -e rec/${i}.rec ] ;
    then
        python ~/anaconda3/lib/python3.7/site-packages/mxnet/tools/im2rec.py list/${i}.lst ''  --resize=256 --encoding='.jpg' --quality=100
        mv list/${i}.idx list/${i}.rec rec/
    fi
done
