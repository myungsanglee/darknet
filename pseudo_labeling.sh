#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
./darknet detector test \
custom_train/focus/221024_C/front/yolov4-tiny-custom-v4_416/focus.data \
custom_train/focus/221024_C/front/yolov4-tiny-custom-v4_416/yolov4-tiny-custom-v4_416.cfg \
custom_train/focus/221024_C/front/yolov4-tiny-custom-v4_416/weights/yolov4-tiny-custom-v4_416_best.weights \
-thresh 0.4 \
-dont_show \
-save_labels < /home/fssv2/myungsang/datasets/focus/221102_D/front/test.txt