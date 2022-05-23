#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
./darknet yolo train \
custom_train/yolov1-voc/yolov1-voc.cfg