#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

#./darknet detector train \
custom_train/yolov4-tiny-3l-custom-05-voc/voc.data \
custom_train/yolov4-tiny-3l-custom-05-voc/yolov4-tiny-3l-custom.cfg \
-map

./darknet detector train \
custom_train/yolov4-tiny-custom-v5_coco-person_416/coco.data \
custom_train/yolov4-tiny-custom-v5_coco-person_416/yolov4-tiny-custom-v5_coco-person_416.cfg \
-map

#./darknet detector train \
custom_train/focus/221014_B_rev_3/front/yolov4-tiny-custom-v4_416/focus.data \
custom_train/focus/221014_B_rev_3/front/yolov4-tiny-custom-v4_416/yolov4-tiny-custom-v4_416.cfg \
custom_train/focus/221014_B_rev_3/front/yolov4-tiny-custom-v4_416/yolov4-tiny-custom-v4_coco-person_416.conv.28 \
-map
