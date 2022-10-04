#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

#./darknet detector map \
custom_train/focus/220812_B/front/yolov4-tiny-custom-v2_416/focus.data \
custom_train/focus/220812_B/front_v2/yolov4-tiny-custom-v2_416/yolov4-tiny-custom-v2_416.cfg \
custom_train/focus/220812_B/front_v2/yolov4-tiny-custom-v2_416/weights/yolov4-tiny-custom-v2_416_best.weights \
-thresh 0.25 \
-points 0

./darknet detector map \
custom_train/yolov2-voc/voc.data \
custom_train/yolov2-voc/yolov2-voc.cfg \
custom_train/yolov2-voc/weights_random=0/yolov2-voc_best.weights \
-points 101

#./darknet detector map \
custom_train/yolov3-tiny-3l-custom-coco/coco.data \
cfg/yolov4.cfg \
yolov4.weights \
-points 101