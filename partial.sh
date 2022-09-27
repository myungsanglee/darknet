#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
./darknet partial \
custom_train/yolov4-tiny-custom-v4_coco-person_416/yolov4-tiny-custom-v4_coco-person_416.cfg \
custom_train/yolov4-tiny-custom-v4_coco-person_416/weights/yolov4-tiny-custom-v4_coco-person_416_best.weights \
custom_train/yolov4-tiny-custom-v4_coco-person_416/yolov4-tiny-custom-v4_coco-person_416.conv.28 \
28