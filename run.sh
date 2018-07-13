# #!/usr/bin/env bash
# python experiment_VOC.py --model_name Res50_Deeplab2 --dump_root ~/summary/SS --is_train --is_predict --device 0 --resize_shape 473
# python experiment_VOC.py --model_name Res50_Deeplab2_2 --dump_root G:\tmp --is_train --is_predict --device 0 --resize_shape 473 --learning_rate 1e-2 batch_size 8

python run_ISPRS_post.py trainaug#43#10#0.007#1e-06#45500#0
python run_ISPRS_post.py trainaug#43#10#0.007#1e-06#45500#1
python run_ISPRS_post.py trainaug#43#10#0.007#1e-06#45500#2

