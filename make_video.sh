#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

#./darknet detector demo \
custom_train/yolov4-tiny-custom-v4_coco-person_224/coco.data \
custom_train/yolov4-tiny-custom-v4_coco-person_224/yolov4-tiny-custom-v4_coco-person_224.cfg \
custom_train/yolov4-tiny-custom-v4_coco-person_224/weights/yolov4-tiny-custom-v4_coco-person_224_best.weights \
video_test/raw_video/fssolution_moh_10.avi \
-out_filename video_test/result_video/yolov4-tiny-custom-v4_coco-person_224/fssolution_moh_10.avi \
-thresh 0.4 \
-dont_show

./darknet detector demo \
custom_train/focus/230102_E/front/version_09/focus.data \
custom_train/focus/230102_E/front/version_09/yolov4-tiny-custom-v4_416.cfg \
custom_train/focus/230102_E/front/version_09/weights/yolov4-tiny-custom-v4_416_best.weights \
video_test/raw_video/23.01.13_ottugi_test_video.avi \
-out_filename video_test/result_video/230102_E/version_09/23.01.13_ottugi_test_video.avi \
-thresh 0.4 \
-dont_show

#./darknet detector demo \
cfg/coco.data \
cfg/yolov4.cfg \
yolov4.weights \
focus_video_test/raw/focus_outside_rear/20220812_135624_ch03.avi \
-out_filename focus_video_test/result/20220812_135624_ch03_yolov4.avi \
-thresh 0.3 \
-dont_show
