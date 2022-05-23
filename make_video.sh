#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

./darknet detector demo \
custom_train/yolov4-tiny-3l-voc/voc.data \
custom_train/yolov4-tiny-3l-voc/yolov4-tiny-3l.cfg \
custom_train/yolov4-tiny-3l-voc/weights/yolov4-tiny-3l_best.weights \
crowd_skywalk.mp4 \
-out_filename res.avi
