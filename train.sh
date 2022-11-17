#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

#./darknet detector train \
custom_train/yolov4-tiny-3l-custom-04-voc/voc.data \
custom_train/yolov4-tiny-3l-custom-04-voc/yolov4-tiny-3l-custom.cfg \
-map

#./darknet detector train \
custom_train/yolov4-tiny-custom-v4_inria-person_416/inria.data \
custom_train/yolov4-tiny-custom-v4_inria-person_416/yolov4-tiny-custom-v4_inria-person_416.cfg \
custom_train/yolov4-tiny-custom-v4_inria-person_416/yolov4-tiny-custom-v4_coco-person_416.conv.27 \
-map

./darknet detector train \
custom_train/focus/221102_D/front/yolov4-tiny-custom-v4_416/focus.data \
custom_train/focus/221102_D/front/yolov4-tiny-custom-v4_416/yolov4-tiny-custom-v4_416.cfg \
custom_train/focus/221102_D/front/yolov4-tiny-custom-v4_416/yolov4-tiny-custom-v4_coco-person_416.conv.27 \
-map