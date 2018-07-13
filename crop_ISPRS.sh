export PYTHONWARNINGS="ignore"
export PYTHONPATH=.:$PYTHONPATH
python test_files/crop_ISPRS.py --input /media/mass/dataset/zisprs/original_images --output /media/mass/dataset/zisprs/cropped
