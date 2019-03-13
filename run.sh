#!/usr/bin/env bash
# #!/usr/bin/env bash
# python experiment_VOC.py --model_name Res50_Deeplab2 --dump_root ~/summary/SS --is_train --is_predict --device 0 --resize_shape 473
# python experiment_VOC.py --model_name Res50_Deeplab2_2 --dump_root G:\tmp --is_train --is_predict --device 0 --resize_shape 473 --learning_rate 1e-2 batch_size 8

# segmentation
 C:/Users/admin/Anaconda3/python.exe run_estimator.py --dataset Vaihingen --encoder SRes101@16 --decoder Deeplabv3@2x4x6 --crop_size 513,513 --batch_size 8 --epochs 1000 --learning_rate 0.007 --mode t --datalist_train train.txt --datalist_val val.txt

# C:/Anaconda3/python.exe run_estimator.py --front_end VGG16 --model FCN16s --crop_size 800,174 --batch_size 6 --epochs 600 --learning_rate 0.0002 --mode t --extra debug_bndecay9997 --datalist_train train.txt --siamese --init_model_path "D:\tmp\TSUNAMI\VGG16_FCN32s#TSUNAMI#train#1000#6#0.001#1e-06#14000#800,174#False#255#True\model.ckpt-14000"
# C:/Anaconda3/python.exe run_estimator.py --front_end SRes101@32 --model FCN32s --crop_size 513 --batch_size 8 --epochs 50 --learning_rate 0.007 --mode t --extra debug_bndecay9997 --datalist_train trainaug.txt

# inference
# C:/Anaconda3/python.exe run_estimator.py --mode e --eval_dir "D:\tmp\TSUNAMI\SimpleConv_SimpleDeconv#TSUNAMI#train#600#8#0.001#1e-06#6600#800,174#False#255#sup#" --infer_eval_dir --inference_root "D:\tmp\TSUNAMI\inference" --datalist_val val1.txt --structure_mode siamese
#C:/Anaconda3/python.exe run_estimator.py --dataset temp_test --mode p --eval_dir "D:\tmp\fujian\SRes101@16_SimpleDeconv84#fujian#train17#9000#8#0.007#1e-06#54000#513,513#False#255#seg#" --infer_eval_dir  --datalist_val temp.txt
# C:/Anaconda3/python.exe run_estimator.py --front_end VGG16 --model FCN32s --crop_size 800,174 --batch_size 8 --epochs 600 --learning_rate 0.001 --mode t --datalist_train train1.txt --datalist_val val1.txt --structure_mode siamese

# train
# C:/Anaconda3/python.exe run_estimator.py --front_end SimpleConv --model SimpleDeconv --crop_size 800,174 --batch_size 8 --epochs 600 --learning_rate 0.001 --mode t --datalist_train train.txt     --datalist_val val.txt --structure_mode sup

