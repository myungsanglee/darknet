#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
./darknet detector train \
custom_train/yolov1-voc/voc.data \
custom_train/yolov1-voc/yolov1-voc.cfg \
-map