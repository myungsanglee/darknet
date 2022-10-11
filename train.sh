#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

#./darknet detector train \
custom_train/yolov4-tiny-custom-v4_coco-person_416/coco.data \
custom_train/yolov4-tiny-custom-v4_coco-person_416/yolov4-tiny-custom-v4_coco-person_416.cfg \
-map

./darknet detector train \
custom_train/focus/221006_B_rev/front/yolov4-tiny-custom-v4_416/focus.data \
custom_train/focus/221006_B_rev/front/yolov4-tiny-custom-v4_416/yolov4-tiny-custom-v4_416.cfg \
custom_train/focus/221006_B_rev/front/yolov4-tiny-custom-v4_416/yolov4-tiny-custom-v4_coco-person_416.conv.28 \
-map