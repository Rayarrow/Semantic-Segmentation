# python experiment_VOC.py --model_name Res50_Deeplab2 --dump_root ~/summary/SS --is_train --is_predict --device 0 --resize_shape 473
# python experiment_VOC.py --model_name Res50_Deeplab2_2 --dump_root G:\tmp --is_train --is_predict --device 0 --resize_shape 473 --learning_rate 1e-2 batch_size 8

python experiment_VOC.py --model_name Res50_PSPNet_VOC --dump_root ~/summary/SS --ignore 255 --resize_shape 473 --lr_decay none --weight_decay 1e-4 --nr_iter 60000 --batch_size 8 --is_train --is_predict --devices 2
