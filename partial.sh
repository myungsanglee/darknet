#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
./darknet partial \
custom_train/yolov4-tiny-3l-custom-05-voc/yolov4-tiny-3l-custom.cfg \
custom_train/yolov4-tiny-3l-custom-05-voc/weights/yolov4-tiny-3l-custom_best.weights \
custom_train/yolov4-tiny-3l-custom-05-voc/yolov4-tiny-3l-custom-05-voc.conv.27 \
27