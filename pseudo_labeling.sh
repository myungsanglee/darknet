#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
./darknet detector test \
custom_train/focus_yolov4-tiny-custom-v2_608/focus.data \
custom_train/focus_yolov4-tiny-custom-v2_608/yolov4-tiny-custom-v2.cfg \
custom_train/focus_yolov4-tiny-custom-v2_608/weights/yolov4-tiny-custom-v2_best.weights \
-thresh 0.25 \
-dont_show \
-save_labels < focus_video_test/tmp/ch04.txt