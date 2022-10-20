#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

./darknet detector map \
custom_train/focus/221014_B_rev_3/front/yolov4-tiny-custom-v4_416/focus.data \
custom_train/focus/221014_B_rev_3/front/yolov4-tiny-custom-v4_416/yolov4-tiny-custom-v4_416.cfg \
custom_train/focus/221014_B_rev_3/front/yolov4-tiny-custom-v4_416/weights/yolov4-tiny-custom-v4_416_best.weights \
-points 0

#./darknet detector map \
custom_train/yolov4-tiny-custom-v4_coco-person_224/coco.data \
custom_train/yolov4-tiny-custom-v4_coco-person_224/yolov4-tiny-custom-v4_coco-person_224.cfg \
custom_train/yolov4-tiny-custom-v4_coco-person_224/weights/yolov4-tiny-custom-v4_coco-person_224_best.weights \
-points 0

#./darknet detector map \
custom_train/yolov3-tiny-3l-custom-coco/coco.data \
cfg/yolov4.cfg \
yolov4.weights \
-points 101