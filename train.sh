#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

#./darknet detector train \
custom_train/focus/crowded_people/576/focus.data \
custom_train/focus/crowded_people/576/yolov4-tiny-custom-v4_576.cfg \
custom_train/focus/crowded_people/576/weights/yolov4-tiny-custom-v4_576_last.weights \
-map

#./darknet detector train \
custom_train/focus/crowded_people/416/focus.data \
custom_train/focus/crowded_people/416/yolov4-tiny-custom-v4_416.cfg \
custom_train/focus/crowded_people/416/yolov4-tiny-custom-v4_coco-person_416.conv.27 \
-map

#./darknet detector train \
custom_train/yolov4-tiny-3l-custom-05-lpr/lpr.data \
custom_train/yolov4-tiny-3l-custom-05-lpr/yolov4-tiny-3l-custom.cfg \
custom_train/yolov4-tiny-3l-custom-05-lpr/yolov4-tiny-3l-custom-05-voc.conv.27 \
-map

#./darknet detector train \
custom_train/yolov4-tiny-custom-v4_inria-person_416/inria.data \
custom_train/yolov4-tiny-custom-v4_inria-person_416/yolov4-tiny-custom-v4_inria-person_416.cfg \
custom_train/yolov4-tiny-custom-v4_inria-person_416/yolov4-tiny-custom-v4_coco-person_416.conv.27 \
-map

./darknet detector train \
custom_train/focus/230717_F/rear/yolov4-tiny-custom-v4_416/focus.data \
custom_train/focus/230717_F/rear/yolov4-tiny-custom-v4_416/yolov4-tiny-custom-v4_416.cfg \
custom_train/focus/230717_F/rear/yolov4-tiny-custom-v4_416/yolov4-tiny-custom-v4_coco-person_416.conv.27 \
-map
