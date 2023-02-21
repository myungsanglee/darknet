#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

#./darknet detector map \
custom_train/focus/230102_E/front/version_01/focus.data \
custom_train/focus/230102_E/front/version_01/yolov4-tiny-custom-v4_416.cfg \
custom_train/focus/230102_E/front/version_01/weights/yolov4-tiny-custom-v4_416_best.weights \
-points 0

./darknet detector map \
custom_train/focus/crowded_people/416/focus.data \
custom_train/focus/crowded_people/416/yolov4-tiny-custom-v4_416.cfg \
custom_train/focus/crowded_people/416/weights/yolov4-tiny-custom-v4_416_best.weights \
-points 0