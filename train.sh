#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

./darknet detector train \
custom_train/yolov4-tiny-custom-v4_coco-person_416/coco.data \
custom_train/yolov4-tiny-custom-v4_coco-person_416/yolov4-tiny-custom-v4_coco-person_416.cfg \
-map

#./darknet detector train \
custom_train/focus/220812_B/front_v2/yolov4-tiny-custom-v2_416/focus.data \
custom_train/focus/220812_B/front_v2/yolov4-tiny-custom-v2_416/yolov4-tiny-custom-v2_416.cfg \
custom_train/focus/220812_B/front_v2/yolov4-tiny-custom-v2_416/yolov4-tiny-custom-v2_coco-person_416.conv.31 \
-map