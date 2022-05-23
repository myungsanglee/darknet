#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# include video_capturing + NMS + drawing_bboxes
#./darknet detector demo \
custom_train/yolov4-tiny-3l-custom-03-voc/voc.data \
custom_train/yolov4-tiny-3l-custom-03-voc/yolov4-tiny-3l-custom.cfg \
custom_train/yolov4-tiny-3l-custom-03-voc/weights/yolov4-tiny-3l-custom_best.weights \
crowd_skywalk.mp4 \
-dont_show \
-ext_output

#./darknet detector demo \
custom_train/yolov4-tiny-3l-voc/voc.data \
custom_train/yolov4-tiny-3l-voc/yolov4-tiny-3l.cfg \
custom_train/yolov4-tiny-3l-voc/weights/yolov4-tiny-3l_best.weights \
crowd_skywalk.mp4 \
-dont_show \
-ext_output

# exclude video_capturing + NMS + drawing_bboxes
./darknet detector demo \
custom_train/yolov4-tiny-3l-custom-03-voc/voc.data \
custom_train/yolov4-tiny-3l-custom-03-voc/yolov4-tiny-3l-custom.cfg \
custom_train/yolov4-tiny-3l-custom-03-voc/weights/yolov4-tiny-3l-custom_best.weights \
crowd_skywalk.mp4 \
-benchmark

#./darknet detector demo \
custom_train/yolov4-tiny-3l-voc/voc.data \
custom_train/yolov4-tiny-3l-voc/yolov4-tiny-3l.cfg \
custom_train/yolov4-tiny-3l-voc/weights/yolov4-tiny-3l_best.weights \
crowd_skywalk.mp4 \
-benchmark
