#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
./darknet detector map \
custom_train/yolov2-voc/voc.data \
custom_train/yolov2-voc/yolov2-voc.cfg \
custom_train/yolov2-voc/weights/yolov2-voc_best.weights
