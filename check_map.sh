#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

./darknet detector map \
custom_train/focus/230102_E/front/version_01/focus.data \
custom_train/focus/230102_E/front/version_01/yolov4-tiny-custom-v4_416.cfg \
custom_train/focus/230102_E/front/version_01/weights/yolov4-tiny-custom-v4_416_best.weights \
-points 0

#./darknet detector map \
custom_train/yolov4-tiny-custom-v4_inria-person_416/inria.data \
custom_train/yolov4-tiny-custom-v4_inria-person_416/yolov4-tiny-custom-v4_inria-person_416.cfg \
custom_train/yolov4-tiny-custom-v4_inria-person_416/weights/yolov4-tiny-custom-v4_inria-person_416_best.weights \
-thresh 0.3 \
-iou_thresh 0.3 \
-points 0

#./darknet detector map \
custom_train/yolov4-tiny-3l-custom-04-voc/voc.data \
custom_train/yolov4-tiny-3l-custom-04-voc/yolov4-tiny-3l-custom.cfg \
custom_train/yolov4-tiny-3l-custom-04-voc/weights/yolov4-tiny-3l-custom_best.weights \
-thresh 0.005 \
-points 0