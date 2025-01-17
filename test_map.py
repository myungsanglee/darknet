import argparse
import os
import glob
import random
import sys
from tqdm import tqdm
import time
import json

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import ImageFont, ImageDraw, Image
import torch
import albumentations
import albumentations.pytorch

import darknet

sys.path.append('/home/fssv2/myungsang/my_projects/PyTorch-Object-Detection')
from utils.yolo_utils import mean_average_precision, metrics_per_class

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))


def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)


def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh, nms=0.1)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = os.path.splitext(name)[0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def batch_detection_example():
    args = parser()
    check_arguments_errors(args)
    batch_size = 3
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )
    image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    images = [cv2.imread(image) for image in image_names]
    images, detections,  = batch_detection(network, images, class_names,
                                           class_colors, batch_size=batch_size)
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)
    print(detections)


def get_detection(image_path, network, class_names, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    if image is None:
        print(f'Can not read this file: \n{image_path}')
        return []
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    # detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    detections = darknet.decode_prediction(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    return detections


def get_detection_for_lpr(image_path, network, class_names, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    # detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    detections = darknet.decode_prediction(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    return detections


def get_detection_for_video(image, network, class_names, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    # detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    detections = darknet.decode_prediction(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    return detections


def get_detection_for_inference_speed(image, network, class_names, thresh, fps):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    # detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    detections = darknet.decode_prediction_and_inference_speed_check(network, class_names, darknet_image, thresh=thresh, fps=fps)
    darknet.free_image(darknet_image)
    return detections


def get_y_true(img_path, img_idx, input_size):
    valid_transform = albumentations.Compose([
        albumentations.Resize(input_size, input_size, always_apply=True),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ], bbox_params=albumentations.BboxParams(format='yolo', min_visibility=0.1))

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes = _get_boxes(img_path.replace('.jpg', '.txt'))

    data = valid_transform(image=img, bboxes=boxes)
    
    bboxes = torch.tensor(data['bboxes'])
    max_num_annots = bboxes.size(0)

    padded_annots = torch.zeros((max_num_annots, 7))
    
    for idx in range(max_num_annots):
        cx, cy, w, h, class_idx = bboxes[idx]
        padded_annots[idx] = torch.FloatTensor([img_idx, cx*input_size, cy*input_size, w*input_size, h*input_size, 1., class_idx])

    return padded_annots


def _get_boxes(label_path):
    boxes = np.zeros((0, 5))
    with open(label_path, 'r') as f:
        annotations = f.read().splitlines()
        for annot in annotations:
            class_id, cx, cy, w, h = map(float, annot.split(' '))
            annotation = np.array([[cx, cy, w, h, class_id]])
            boxes = np.append(boxes, annotation, axis=0)

    return boxes


def check_map_by_custom_map_calculator():
    network, class_names, class_colors = darknet.load_network(
        #######################################################################################################
        # VOC2007
        #######################################################################################################
        
        # 'custom_train/yolov2-voc/yolov2-voc.cfg',
        # 'custom_train/yolov2-voc/voc.data',
        # 'custom_train/yolov2-voc/weights_random=0/yolov2-voc_best.weights',
        # 'custom_train/yolov2-voc/weights_random=1/yolov2-voc_best.weights',
        # 'custom_train/yolov2-voc/weights_with_pretrained/yolov2-voc_best.weights',
        
        # 'custom_train/yolov3-custom-voc/yolov3-custom-voc.cfg',
        # 'custom_train/yolov3-custom-voc/voc.data',
        # 'custom_train/yolov3-custom-voc/weights/yolov3-custom-voc_best.weights',
        
        # 'custom_train/yolov3-tiny-3l-voc/yolov3-tiny-3l.cfg',
        # 'custom_train/yolov3-tiny-3l-voc/voc.data',
        # 'custom_train/yolov3-tiny-3l-voc/weights/yolov3-tiny-3l_best.weights',
        
        # 'custom_train/yolov3-tiny-3l-custom-01-voc/yolov3-tiny-3l-custom.cfg',
        # 'custom_train/yolov3-tiny-3l-custom-01-voc/voc.data',
        # 'custom_train/yolov3-tiny-3l-custom-01-voc/weights/yolov3-tiny-3l-custom_best.weights',
        
        # 'custom_train/yolov4-tiny-3l-voc/yolov4-tiny-3l.cfg',
        # 'custom_train/yolov4-tiny-3l-voc/voc.data',
        # 'custom_train/yolov4-tiny-3l-voc/weights/yolov4-tiny-3l_best.weights',
        
        # 'custom_train/yolov4-tiny-3l-custom-01-voc/yolov4-tiny-3l-custom.cfg',
        # 'custom_train/yolov4-tiny-3l-custom-01-voc/voc.data',
        # 'custom_train/yolov4-tiny-3l-custom-01-voc/weights/yolov4-tiny-3l-custom_best.weights',
        
        # 'custom_train/yolov4-tiny-3l-custom-02-voc/yolov4-tiny-3l-custom.cfg',
        # 'custom_train/yolov4-tiny-3l-custom-02-voc/voc.data',
        # 'custom_train/yolov4-tiny-3l-custom-02-voc/weights/yolov4-tiny-3l-custom_best.weights',
        
        # 'custom_train/yolov4-tiny-3l-custom-03-voc/yolov4-tiny-3l-custom.cfg',
        # 'custom_train/yolov4-tiny-3l-custom-03-voc/voc.data',
        # 'custom_train/yolov4-tiny-3l-custom-03-voc/weights/yolov4-tiny-3l-custom_best.weights',
        
        # 'custom_train/yolov4-tiny-3l-custom-04-voc/yolov4-tiny-3l-custom.cfg',
        # 'custom_train/yolov4-tiny-3l-custom-04-voc/voc.data',
        # 'custom_train/yolov4-tiny-3l-custom-04-voc/weights/yolov4-tiny-3l-custom_best.weights',
        
        # 'custom_train/yolov4-tiny-3l-custom-05-voc/yolov4-tiny-3l-custom.cfg',
        # 'custom_train/yolov4-tiny-3l-custom-05-voc/voc.data',
        # 'custom_train/yolov4-tiny-3l-custom-05-voc/weights/yolov4-tiny-3l-custom_best.weights',
        
        'custom_train/yolov4-tiny-3l-custom-06-voc/yolov4-tiny-3l-custom.cfg',
        'custom_train/yolov4-tiny-3l-custom-06-voc/voc.data',
        'custom_train/yolov4-tiny-3l-custom-06-voc/weights/yolov4-tiny-3l-custom_best.weights',
        
        #######################################################################################################
        # COCO2017 Person
        #######################################################################################################
        
        # 'custom_train/yolov4-tiny-custom-v1_coco-person_224/yolov4-tiny-custom-v1_coco-person_224.cfg',
        # 'custom_train/yolov4-tiny-custom-v1_coco-person_224/coco.data',
        # 'custom_train/yolov4-tiny-custom-v1_coco-person_224/weights/yolov4-tiny-custom-v1_coco-person_224_best.weights',
        
        # 'custom_train/yolov4-tiny-custom-v2_coco-person_224/yolov4-tiny-custom-v2_coco-person_224.cfg',
        # 'custom_train/yolov4-tiny-custom-v2_coco-person_224/coco.data',
        # 'custom_train/yolov4-tiny-custom-v2_coco-person_224/weights/yolov4-tiny-custom-v2_coco-person_224_best.weights',
        
        # 'custom_train/yolov4-tiny-custom-v2_coco-person_416/yolov4-tiny-custom-v2_coco-person_416.cfg',
        # 'custom_train/yolov4-tiny-custom-v2_coco-person_416/coco.data',
        # 'custom_train/yolov4-tiny-custom-v2_coco-person_416/weights/yolov4-tiny-custom-v2_coco-person_416_best.weights',
        
        # 'custom_train/yolov4-tiny-custom-v4_coco-person_224/yolov4-tiny-custom-v4_coco-person_224.cfg',
        # 'custom_train/yolov4-tiny-custom-v4_coco-person_224/coco.data',
        # 'custom_train/yolov4-tiny-custom-v4_coco-person_224/weights/yolov4-tiny-custom-v4_coco-person_224_best.weights',
        
        # 'custom_train/yolov4-tiny-custom-v4_coco-person_416/yolov4-tiny-custom-v4_coco-person_416.cfg',
        # 'custom_train/yolov4-tiny-custom-v4_coco-person_416/coco.data',
        # 'custom_train/yolov4-tiny-custom-v4_coco-person_416/weights/yolov4-tiny-custom-v4_coco-person_416_best.weights',
        
        # 'custom_train/yolov4-tiny-custom-v5_coco-person_416/yolov4-tiny-custom-v5_coco-person_416.cfg',
        # 'custom_train/yolov4-tiny-custom-v5_coco-person_416/coco.data',
        # 'custom_train/yolov4-tiny-custom-v5_coco-person_416/weights/yolov4-tiny-custom-v5_coco-person_416_best.weights',
        
        # 'custom_train/yolov3-tiny-custom-v1_coco-person_416/yolov3-tiny-custom-v1_coco-person_416.cfg',
        # 'custom_train/yolov3-tiny-custom-v1_coco-person_416/coco.data',
        # 'custom_train/yolov3-tiny-custom-v1_coco-person_416/weights/yolov3-tiny-custom-v1_coco-person_416_best.weights',
        
        #######################################################################################################
        # Focus Front
        #######################################################################################################
        
        # 'custom_train/focus/220812_B/front/yolov4-tiny-custom-v2_416/yolov4-tiny-custom-v2_416.cfg',
        # 'custom_train/focus/220812_B/front/yolov4-tiny-custom-v2_416/focus.data',
        # 'custom_train/focus/220812_B/front/yolov4-tiny-custom-v2_416/weights/yolov4-tiny-custom-v2_416_best.weights',
        
        # 'custom_train/focus/221006_B_rev/front/yolov4-tiny-custom-v4_416/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/221006_B_rev/front/yolov4-tiny-custom-v4_416/focus.data',
        # 'custom_train/focus/221006_B_rev/front/yolov4-tiny-custom-v4_416/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/221013_B_rev_2/front/yolov4-tiny-custom-v2_416/yolov4-tiny-custom-v2_416.cfg',
        # 'custom_train/focus/221013_B_rev_2/front/yolov4-tiny-custom-v2_416/focus.data',
        # 'custom_train/focus/221013_B_rev_2/front/yolov4-tiny-custom-v2_416/weights/yolov4-tiny-custom-v2_416_best.weights',
        
        # 'custom_train/focus/221014_B_rev_3/front/yolov4-tiny-custom-v4_416/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/221014_B_rev_3/front/yolov4-tiny-custom-v4_416/focus.data',
        # 'custom_train/focus/221014_B_rev_3/front/yolov4-tiny-custom-v4_416/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/221024_C/front/yolov4-tiny-custom-v2_416/yolov4-tiny-custom-v2_416.cfg',
        # 'custom_train/focus/221024_C/front/yolov4-tiny-custom-v2_416/focus.data',
        # 'custom_train/focus/221024_C/front/yolov4-tiny-custom-v2_416/weights/yolov4-tiny-custom-v2_416_best.weights',
        
        # 'custom_train/focus/221024_C/front/yolov4-tiny-custom-v4_416/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/221024_C/front/yolov4-tiny-custom-v4_416/focus.data',
        # 'custom_train/focus/221024_C/front/yolov4-tiny-custom-v4_416/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/221024_C/front/yolov4-tiny-custom-v5_416/yolov4-tiny-custom-v5_416.cfg',
        # 'custom_train/focus/221024_C/front/yolov4-tiny-custom-v5_416/focus.data',
        # 'custom_train/focus/221024_C/front/yolov4-tiny-custom-v5_416/weights/yolov4-tiny-custom-v5_416_best.weights',
        
        # 'custom_train/focus/221024_C/front/yolov3-tiny-custom-v1_416/yolov3-tiny-custom-v1_416.cfg',
        # 'custom_train/focus/221024_C/front/yolov3-tiny-custom-v1_416/focus.data',
        # 'custom_train/focus/221024_C/front/yolov3-tiny-custom-v1_416/weights/yolov3-tiny-custom-v1_416_best.weights',
        
        # 'custom_train/focus/221102_D/front/yolov4-tiny-custom-v4_416/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/221102_D/front/yolov4-tiny-custom-v4_416/focus.data',
        # 'custom_train/focus/221102_D/front/yolov4-tiny-custom-v4_416/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_01/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_01/focus.data',
        # 'custom_train/focus/230102_E/front/version_01/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_02/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_02/focus.data',
        # 'custom_train/focus/230102_E/front/version_02/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_03/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_03/focus.data',
        # 'custom_train/focus/230102_E/front/version_03/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_04/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_04/focus.data',
        # 'custom_train/focus/230102_E/front/version_04/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_05/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_05/focus.data',
        # 'custom_train/focus/230102_E/front/version_05/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_06/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_06/focus.data',
        # 'custom_train/focus/230102_E/front/version_06/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_07/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_07/focus.data',
        # 'custom_train/focus/230102_E/front/version_07/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_08/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_08/focus.data',
        # 'custom_train/focus/230102_E/front/version_08/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_09/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_09/focus.data',
        # 'custom_train/focus/230102_E/front/version_09/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_10/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_10/focus.data',
        # 'custom_train/focus/230102_E/front/version_10/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_11/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_11/focus.data',
        # 'custom_train/focus/230102_E/front/version_11/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_12/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_12/focus.data',
        # 'custom_train/focus/230102_E/front/version_12/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_07_1/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_07_1/focus.data',
        # 'custom_train/focus/230102_E/front/version_07_1/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        #######################################################################################################
        # LPR
        #######################################################################################################
        
        # 'custom_train/yolov4-tiny-3l-custom-04-lpr/yolov4-tiny-3l-custom.cfg',
        # 'custom_train/yolov4-tiny-3l-custom-04-lpr/lpr.data',
        # 'custom_train/yolov4-tiny-3l-custom-04-lpr/weights/yolov4-tiny-3l-custom_best.weights',
        
        #######################################################################################################
        # Crowded People
        #######################################################################################################
        
        # 'custom_train/focus/crowded_people/416/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/crowded_people/416/focus.data',
        # 'custom_train/focus/crowded_people/416/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        batch_size=1
    )

    # names_txt_path = '/home/fssv2/myungsang/datasets/voc/yolo_format/voc.names'
    # names_txt_path = '/home/fssv2/myungsang/datasets/coco_2017/person/coco.names'
    # with open(names_txt_path, 'r') as f:
    #     name_list = f.read().splitlines()
    
    val_txt_path = '/home/fssv2/myungsang/datasets/voc/yolo_format/val.txt'
    # val_txt_path = '/home/fssv2/myungsang/datasets/coco_2017/person/val.txt'
    # val_txt_path = '/home/fssv2/myungsang/datasets/focus/220812_B/front/val.txt'
    # val_txt_path = '/home/fssv2/myungsang/datasets/focus/221024_C/front/val.txt'
    # val_txt_path = '/home/fssv2/myungsang/datasets/focus/221102_D/front/val.txt'
    # val_txt_path = '/home/fssv2/myungsang/datasets/focus/230102_E/front/val_v1.txt'
    # val_txt_path = '/home/fssv2/myungsang/datasets/focus/230102_E/front/val_v2.txt'
    # val_txt_path = '/home/fssv2/myungsang/datasets/focus/230102_E/front/test_v2.txt'
    # val_txt_path = '/home/fssv2/myungsang/datasets/focus/230102_E/front_crop/version11/crop_val_v1.txt'
    # val_txt_path = '/home/fssv2/myungsang/datasets/focus/230102_E/front_crop/version11/crop_val_v2.txt'
    # val_txt_path = '/home/fssv2/myungsang/datasets/focus/230102_E/front_crop/version11/crop_test_v2.txt'
    # val_txt_path = '/home/fssv2/myungsang/datasets/lpr/val.txt'
    # val_txt_path = '/home/fssv2/fssv2_dataset/Crowd/1. VSCrowd/VSCrowd_Aug2/val.txt'
    
    with open(val_txt_path, 'r') as f:
        img_list = f.read().splitlines()


    all_true_boxes_variable = 0
    all_pred_boxes_variable = 0
    img_idx = 0
    for img_path in tqdm(img_list[:]):
        y_true = get_y_true(img_path, img_idx, darknet.network_width(network))

        detections = get_detection(
            img_path, 
            network, 
            class_names,
            thresh=0.25
        )

        detection_list = []
        for detection in detections:
            class_name, confidence, box = detection
            class_idx = class_names.index(class_name)
            cx = box[0]
            cy = box[1]
            w = box[2]
            h = box[3]
            detection_list.append([img_idx, cx, cy, w, h, confidence, class_idx])
        if not detection_list:
            detection_list = np.zeros((0, 7))

        if img_idx == 0:
            all_pred_boxes_variable = torch.as_tensor(detection_list, dtype=torch.float32)
            all_true_boxes_variable = y_true
        else:
            all_pred_boxes_variable = torch.cat([all_pred_boxes_variable, torch.as_tensor(detection_list, dtype=torch.float32)], dim=0)
            all_true_boxes_variable = torch.cat([all_true_boxes_variable, y_true], dim=0)

        img_idx += 1
    
    # map = mean_average_precision(all_true_boxes_variable, all_pred_boxes_variable, len(class_names))
    # print(f'mAP: {map*100:.2f}%')
    
    start = time.time()
    metrics = metrics_per_class(all_true_boxes_variable, all_pred_boxes_variable, len(class_names))
    taken_time = time.time() - start
    print(f'Done (t={taken_time:.2f}s).')
    
    for idx, (ap, tp, fp, fn) in enumerate(metrics):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f'Name: {class_names[idx]}, AP: {ap*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1 Score: {f1_score*100:.2f}%')
    
    map = metrics[:, 0].mean()
    print(f'mAP: {map*100:.2f}%')


def check_map_by_coco_map_calculator():
    network, class_names, class_colors = darknet.load_network(
        # 'custom_train/yolov3-custom-voc/yolov3-custom-voc.cfg',
        # 'custom_train/yolov3-custom-voc/voc.data',
        # 'custom_train/yolov3-custom-voc/weights/yolov3-custom-voc_best.weights',
        
        #######################################################################################################
        # Crowded People
        #######################################################################################################
        'custom_train/focus/crowded_people/416/yolov4-tiny-custom-v4_416.cfg',
        'custom_train/focus/crowded_people/416/focus.data',
        'custom_train/focus/crowded_people/416/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        batch_size=1
    )

    # json_path = '/home/fssv2/myungsang/datasets/voc/coco_format/val.json'
    # img_dir = '/home/fssv2/myungsang/datasets/voc'
    json_path = '/home/fssv2/fssv2_dataset/Crowd/1. VSCrowd/VSCrowd_Aug2/val.json'
    coco = COCO(json_path)

    imgs = coco.loadImgs(coco.getImgIds())
    cats = coco.loadCats(coco.getCatIds())
    cats_dict = dict([[cat['name'], cat['id']] for cat in cats])

    imgs_info = [[img['id'], img['file_name'], img['width'], img['height']] for img in imgs]
    
    results = []

    for (img_id, file_name, img_width, img_height) in tqdm(imgs_info):
        # img_path = os.path.join(img_dir, file_name)
        img_path = file_name

        detections = get_detection(
            img_path, 
            network, 
            class_names,
            thresh=0.4
        )

        for detection in detections:
            class_name, confidence, box = detection
            cx = box[0] * (img_width / darknet.network_width(network))
            cy = box[1] * (img_height / darknet.network_height(network))
            w = box[2] * (img_width / darknet.network_width(network))
            h = box[3] * (img_height / darknet.network_height(network))

            xmin = int(round((cx - (w / 2))))
            ymin = int(round((cy - (h / 2))))
            w = int(round(w))
            h = int(round(h))
            
            results.append({
                "image_id": img_id,
                "category_id": cats_dict[class_name],
                "bbox": [xmin, ymin, w, h],
                "score": float(confidence)
            })
    
    img_ids = sorted(coco.getImgIds())
    cat_ids = sorted(coco.getCatIds())

    # load detection JSON file from the disk
    # cocovalPrediction = coco.loadRes(results_json_path)
    cocovalPrediction = coco.loadRes(results)
    
    # initialize the COCOeval object by passing the coco object with
    # ground truth annotations, coco object with detection results
    cocoEval = COCOeval(coco, cocovalPrediction, "bbox")
 
    # run evaluation for each image, accumulates per image results
    # display the summary metrics of the evaluation
    cocoEval.params.imgIds  = img_ids
    cocoEval.params.catIds = cat_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def check_map_by_torchmetrics_map_calculator():
    network, class_names, class_colors = darknet.load_network(
        'custom_train/yolov3-custom-voc/yolov3-custom-voc.cfg',
        'custom_train/yolov3-custom-voc/voc.data',
        'custom_train/yolov3-custom-voc/weights/yolov3-custom-voc_best.weights',
        
        batch_size=1
    )

    json_path = '/home/fssv2/myungsang/datasets/voc/coco_format/val.json'
    coco = COCO(json_path)

    imgs = coco.loadImgs(coco.getImgIds())
    cats = coco.loadCats(coco.getCatIds())
    cats_dict = dict([[cat['name'], cat['id']] for cat in cats])

    imgs_info = [[img['id'], img['file_name'], img['width'], img['height']] for img in imgs]
    
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric = MeanAveragePrecision(box_format='xywh')
    
    for (img_id, file_name, img_width, img_height) in tqdm(imgs_info):
        preds_boxes = []
        preds_scores = []
        preds_labels = []
        
        target_boxes = []
        target_labels = []
        
        # update target
        annids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(annids)
        for ann in anns:
            bbox = ann['bbox']
            category_id = ann['category_id']
            
            xmin = bbox[0]
            ymin = bbox[1]
            w = bbox[2]
            h = bbox[3]
            
            target_boxes.append([xmin, ymin, w, h])
            target_labels.append(category_id - 1)
        
        # update preds
        img_path = os.path.join('/home/fssv2/myungsang/datasets/voc', file_name)
        detections = get_detection(
            img_path, 
            network, 
            class_names,
            thresh=0.25
        )

        for detection in detections:
            class_name, confidence, box = detection
            class_idx = class_names.index(class_name)
            cx = box[0] * (img_width / darknet.network_width(network))
            cy = box[1] * (img_height / darknet.network_height(network))
            w = box[2] * (img_width / darknet.network_width(network))
            h = box[3] * (img_height / darknet.network_height(network))

            xmin = int(round((cx - (w / 2))))
            ymin = int(round((cy - (h / 2))))
            w = int(round(w))
            h = int(round(h))
            
            preds_boxes.append([xmin, ymin, w, h])
            preds_scores.append(float(confidence))
            preds_labels.append(class_idx)
            
        # update metric
        preds = [
            dict(
                boxes=torch.tensor(preds_boxes) if preds_boxes else torch.zeros((0, 4)),
                scores=torch.tensor(preds_scores),
                labels=torch.tensor(preds_labels)
            )
        ]
        
        target = [
            dict(
                boxes=torch.tensor(target_boxes) if target_boxes else torch.zeros((0, 4)),
                labels=torch.tensor(target_labels)
            )
        ]
        
        metric.update(preds, target)

    result = metric.compute()
    print(result)


def check_map_by_torchmetrics_yolo_format():
    network, class_names, class_colors = darknet.load_network(
        # 'custom_train/yolov3-custom-voc/yolov3-custom-voc.cfg',
        # 'custom_train/yolov3-custom-voc/voc.data',
        # 'custom_train/yolov3-custom-voc/weights/yolov3-custom-voc_best.weights',
        
        #######################################################################################################
        # Crowded People
        #######################################################################################################
        
        'custom_train/focus/crowded_people/416/yolov4-tiny-custom-v4_416.cfg',
        'custom_train/focus/crowded_people/416/focus.data',
        'custom_train/focus/crowded_people/416/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        batch_size=1
    )

    # val_txt_path = '/home/fssv2/myungsang/datasets/voc/yolo_format/val.txt'
    val_txt_path = '/home/fssv2/fssv2_dataset/Crowd/1. VSCrowd/VSCrowd_Aug2/val.txt'
    with open(val_txt_path, 'r') as f:
        img_list = f.read().splitlines()

    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric = MeanAveragePrecision(box_format='xywh')
    
    for img_path in tqdm(img_list):
        preds_boxes = []
        preds_scores = []
        preds_labels = []
        
        target_boxes = []
        target_labels = []
        
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError('can not open img file')
        img_height, img_width, _ = img.shape
        
        # update target
        label_path = img_path.rsplit('.', 1)[0] + '.txt'
        with open(label_path, 'r') as f:
            anns = f.read().splitlines()
        for ann in anns:
            cls_idx, cx, cy, w, h = ann.split(' ')
            
            cls_idx = int(cls_idx)
            cx = float(cx) * img_width
            cy = float(cy) * img_height
            w = float(w) * img_width
            h = float(h) * img_height
            
            xmin = int(round(cx - (w/2)))
            ymin = int(round(cy - (h/2)))
            w = int(round(w))
            h = int(round(h))
            
            target_boxes.append([xmin, ymin, w, h])
            target_labels.append(cls_idx)
        
        # update preds
        detections = get_detection_for_video(
            img, 
            network, 
            class_names,
            thresh=0.4
        )

        for detection in detections:
            class_name, confidence, box = detection
            class_idx = class_names.index(class_name)
            cx = box[0] * (img_width / darknet.network_width(network))
            cy = box[1] * (img_height / darknet.network_height(network))
            w = box[2] * (img_width / darknet.network_width(network))
            h = box[3] * (img_height / darknet.network_height(network))

            xmin = int(round((cx - (w / 2))))
            ymin = int(round((cy - (h / 2))))
            w = int(round(w))
            h = int(round(h))
            
            preds_boxes.append([xmin, ymin, w, h])
            preds_scores.append(float(confidence))
            preds_labels.append(class_idx)
            
        # update metric
        preds = [
            dict(
                boxes=torch.tensor(preds_boxes) if preds_boxes else torch.zeros((0, 4)),
                scores=torch.tensor(preds_scores),
                labels=torch.tensor(preds_labels)
            )
        ]
        
        target = [
            dict(
                boxes=torch.tensor(target_boxes) if target_boxes else torch.zeros((0, 4)),
                labels=torch.tensor(target_labels)
            )
        ]
        
        metric.update(preds, target)

    start = time.time()
    result = metric.compute()
    taken_time = time.time() - start
    print(f'Done (t={taken_time:.2f}s).')
    print(result)
    

def make_pred_result_file_for_public_map_calculator():
    network, class_names, class_colors = darknet.load_network(
        # 'custom_train/yolov2-voc/yolov2-voc.cfg',
        'custom_train/yolov3-custom-voc/yolov3-custom-voc.cfg',
        # 'custom_train/yolov2-voc/voc.data',
        'custom_train/yolov3-custom-voc/voc.data',
        # 'custom_train/yolov2-voc/weights_random=0/yolov2-voc_best.weights',
        # 'custom_train/yolov2-voc/weights_random=1/yolov2-voc_best.weights',
        # 'custom_train/yolov2-voc/weights_with_pretrained/yolov2-voc_best.weights',
        'custom_train/yolov3-custom-voc/weights/yolov3-custom-voc_best.weights',
    )

    val_txt_path = '/home/fssv2/myungsang/datasets/voc/yolo_format/val.txt'
    with open(val_txt_path, 'r') as f:
        img_list = f.read().splitlines()

    img_idx = 0
    for img_path in tqdm(img_list[:]):
        detections = get_detection(
            img_path, 
            network, 
            class_names,
            thresh=0.5
        )

        img_idx += 1
        pred_txt_fd = open(os.path.join('/home/fssv2/myungsang/my_projects/mAP/input/detection-results', f'{img_idx:05d}.txt'), 'w')

        for detection in detections:
            class_name, confidence, box = detection
            cx = box[0]
            cy = box[1]
            w = box[2]
            h = box[3]

            xmin = int((cx - (w / 2)))
            ymin = int((cy - (h / 2)))
            xmax = int((cx + (w / 2)))
            ymax = int((cy + (h / 2)))

            pred_txt_fd.write(f'{class_name} {confidence} {xmin} {ymin} {xmax} {ymax}\n')
        pred_txt_fd.close()


def show_result():
    network, class_names, class_colors = darknet.load_network(
        #######################################################################################################
        # VOC2007
        #######################################################################################################
        
        # 'custom_train/yolov2-voc/yolov2-voc.cfg',
        # 'custom_train/yolov2-voc/voc.data',
        # 'custom_train/yolov2-voc/weights_random=0/yolov2-voc_best.weights',
        # 'custom_train/yolov2-voc/weights_random=1/yolov2-voc_best.weights',
        # 'custom_train/yolov2-voc/weights_with_pretrained/yolov2-voc_best.weights',
        
        # 'custom_train/yolov3-custom-voc/yolov3-custom-voc.cfg',
        # 'custom_train/yolov3-custom-voc/voc.data',
        # 'custom_train/yolov3-custom-voc/weights/yolov3-custom-voc_best.weights',
        
        # 'custom_train/yolov3-tiny-3l-voc/yolov3-tiny-3l.cfg',
        # 'custom_train/yolov3-tiny-3l-voc/voc.data',
        # 'custom_train/yolov3-tiny-3l-voc/weights/yolov3-tiny-3l_best.weights',
        
        # 'custom_train/yolov3-tiny-3l-custom-01-voc/yolov3-tiny-3l-custom.cfg',
        # 'custom_train/yolov3-tiny-3l-custom-01-voc/voc.data',
        # 'custom_train/yolov3-tiny-3l-custom-01-voc/weights/yolov3-tiny-3l-custom_best.weights',
        
        # 'custom_train/yolov4-tiny-3l-voc/yolov4-tiny-3l.cfg',
        # 'custom_train/yolov4-tiny-3l-voc/voc.data',
        # 'custom_train/yolov4-tiny-3l-voc/weights/yolov4-tiny-3l_best.weights',
        
        # 'custom_train/yolov4-tiny-3l-custom-01-voc/yolov4-tiny-3l-custom.cfg',
        # 'custom_train/yolov4-tiny-3l-custom-01-voc/voc.data',
        # 'custom_train/yolov4-tiny-3l-custom-01-voc/weights/yolov4-tiny-3l-custom_best.weights',
        
        # 'custom_train/yolov4-tiny-3l-custom-02-voc/yolov4-tiny-3l-custom.cfg',
        # 'custom_train/yolov4-tiny-3l-custom-02-voc/voc.data',
        # 'custom_train/yolov4-tiny-3l-custom-02-voc/weights/yolov4-tiny-3l-custom_best.weights',
        
        # 'custom_train/yolov4-tiny-3l-custom-03-voc/yolov4-tiny-3l-custom.cfg',
        # 'custom_train/yolov4-tiny-3l-custom-03-voc/voc.data',
        # 'custom_train/yolov4-tiny-3l-custom-03-voc/weights/yolov4-tiny-3l-custom_best.weights',
        
        # 'custom_train/yolov4-tiny-3l-custom-04-voc/yolov4-tiny-3l-custom.cfg',
        # 'custom_train/yolov4-tiny-3l-custom-04-voc/voc.data',
        # 'custom_train/yolov4-tiny-3l-custom-04-voc/weights/yolov4-tiny-3l-custom_best.weights',
        
        # 'custom_train/yolov4-tiny-3l-custom-05-voc/yolov4-tiny-3l-custom.cfg',
        # 'custom_train/yolov4-tiny-3l-custom-05-voc/voc.data',
        # 'custom_train/yolov4-tiny-3l-custom-05-voc/weights/yolov4-tiny-3l-custom_best.weights',
        
        #######################################################################################################
        # COCO2017 Person
        #######################################################################################################
        
        # 'custom_train/yolov4-tiny-custom-v1_coco-person_224/yolov4-tiny-custom-v1_coco-person_224.cfg',
        # 'custom_train/yolov4-tiny-custom-v1_coco-person_224/coco.data',
        # 'custom_train/yolov4-tiny-custom-v1_coco-person_224/weights/yolov4-tiny-custom-v1_coco-person_224_best.weights',
        
        # 'custom_train/yolov4-tiny-custom-v2_coco-person_224/yolov4-tiny-custom-v2_coco-person_224.cfg',
        # 'custom_train/yolov4-tiny-custom-v2_coco-person_224/coco.data',
        # 'custom_train/yolov4-tiny-custom-v2_coco-person_224/weights/yolov4-tiny-custom-v2_coco-person_224_best.weights',
        
        # 'custom_train/yolov4-tiny-custom-v2_coco-person_416/yolov4-tiny-custom-v2_coco-person_416.cfg',
        # 'custom_train/yolov4-tiny-custom-v2_coco-person_416/coco.data',
        # 'custom_train/yolov4-tiny-custom-v2_coco-person_416/weights/yolov4-tiny-custom-v2_coco-person_416_best.weights',
        
        # 'custom_train/yolov4-tiny-custom-v4_coco-person_224/yolov4-tiny-custom-v4_coco-person_224.cfg',
        # 'custom_train/yolov4-tiny-custom-v4_coco-person_224/coco.data',
        # 'custom_train/yolov4-tiny-custom-v4_coco-person_224/weights/yolov4-tiny-custom-v4_coco-person_224_best.weights',
        
        # 'custom_train/yolov4-tiny-custom-v4_coco-person_416/yolov4-tiny-custom-v4_coco-person_416.cfg',
        # 'custom_train/yolov4-tiny-custom-v4_coco-person_416/coco.data',
        # 'custom_train/yolov4-tiny-custom-v4_coco-person_416/weights/yolov4-tiny-custom-v4_coco-person_416_best.weights',
        
        # 'custom_train/yolov4-tiny-custom-v5_coco-person_416/yolov4-tiny-custom-v5_coco-person_416.cfg',
        # 'custom_train/yolov4-tiny-custom-v5_coco-person_416/coco.data',
        # 'custom_train/yolov4-tiny-custom-v5_coco-person_416/weights/yolov4-tiny-custom-v5_coco-person_416_best.weights',
        
        # 'custom_train/yolov3-tiny-custom-v1_coco-person_416/yolov3-tiny-custom-v1_coco-person_416.cfg',
        # 'custom_train/yolov3-tiny-custom-v1_coco-person_416/coco.data',
        # 'custom_train/yolov3-tiny-custom-v1_coco-person_416/weights/yolov3-tiny-custom-v1_coco-person_416_best.weights',
        
        #######################################################################################################
        # Focus Front
        #######################################################################################################
        
        # 'custom_train/focus/220812_B/front/yolov4-tiny-custom-v2_416/yolov4-tiny-custom-v2_416.cfg',
        # 'custom_train/focus/220812_B/front/yolov4-tiny-custom-v2_416/focus.data',
        # 'custom_train/focus/220812_B/front/yolov4-tiny-custom-v2_416/weights/yolov4-tiny-custom-v2_416_best.weights',
        
        # 'custom_train/focus/221006_B_rev/front/yolov4-tiny-custom-v4_416/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/221006_B_rev/front/yolov4-tiny-custom-v4_416/focus.data',
        # 'custom_train/focus/221006_B_rev/front/yolov4-tiny-custom-v4_416/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/221013_B_rev_2/front/yolov4-tiny-custom-v2_416/yolov4-tiny-custom-v2_416.cfg',
        # 'custom_train/focus/221013_B_rev_2/front/yolov4-tiny-custom-v2_416/focus.data',
        # 'custom_train/focus/221013_B_rev_2/front/yolov4-tiny-custom-v2_416/weights/yolov4-tiny-custom-v2_416_best.weights',
        
        # 'custom_train/focus/221014_B_rev_3/front/yolov4-tiny-custom-v4_416/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/221014_B_rev_3/front/yolov4-tiny-custom-v4_416/focus.data',
        # 'custom_train/focus/221014_B_rev_3/front/yolov4-tiny-custom-v4_416/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/221024_C/front/yolov4-tiny-custom-v2_416/yolov4-tiny-custom-v2_416.cfg',
        # 'custom_train/focus/221024_C/front/yolov4-tiny-custom-v2_416/focus.data',
        # 'custom_train/focus/221024_C/front/yolov4-tiny-custom-v2_416/weights/yolov4-tiny-custom-v2_416_best.weights',
        
        # 'custom_train/focus/221024_C/front/yolov4-tiny-custom-v4_416/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/221024_C/front/yolov4-tiny-custom-v4_416/focus.data',
        # 'custom_train/focus/221024_C/front/yolov4-tiny-custom-v4_416/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/221024_C/front/yolov4-tiny-custom-v5_416/yolov4-tiny-custom-v5_416.cfg',
        # 'custom_train/focus/221024_C/front/yolov4-tiny-custom-v5_416/focus.data',
        # 'custom_train/focus/221024_C/front/yolov4-tiny-custom-v5_416/weights/yolov4-tiny-custom-v5_416_best.weights',
        
        # 'custom_train/focus/221024_C/front/yolov3-tiny-custom-v1_416/yolov3-tiny-custom-v1_416.cfg',
        # 'custom_train/focus/221024_C/front/yolov3-tiny-custom-v1_416/focus.data',
        # 'custom_train/focus/221024_C/front/yolov3-tiny-custom-v1_416/weights/yolov3-tiny-custom-v1_416_best.weights',
        
        # 'custom_train/focus/221102_D/front/yolov4-tiny-custom-v4_416/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/221102_D/front/yolov4-tiny-custom-v4_416/focus.data',
        # 'custom_train/focus/221102_D/front/yolov4-tiny-custom-v4_416/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_01/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_01/focus.data',
        # 'custom_train/focus/230102_E/front/version_01/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_02/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_02/focus.data',
        # 'custom_train/focus/230102_E/front/version_02/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_03/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_03/focus.data',
        # 'custom_train/focus/230102_E/front/version_03/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_04/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_04/focus.data',
        # 'custom_train/focus/230102_E/front/version_04/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_05/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_05/focus.data',
        # 'custom_train/focus/230102_E/front/version_05/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_06/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_06/focus.data',
        # 'custom_train/focus/230102_E/front/version_06/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        'custom_train/focus/230102_E/front/version_07/yolov4-tiny-custom-v4_416.cfg',
        'custom_train/focus/230102_E/front/version_07/focus.data',
        'custom_train/focus/230102_E/front/version_07/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/230102_E/front/version_12/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/230102_E/front/version_12/focus.data',
        # 'custom_train/focus/230102_E/front/version_12/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        batch_size=1
    )

    # val_txt_path = '/home/fssv2/myungsang/datasets/voc/yolo_format/val.txt'
    # val_txt_path = '/home/fssv2/myungsang/datasets/coco_2017/person/val.txt'
    # val_txt_path = '/home/fssv2/myungsang/datasets/focus/220812_B/front/val.txt'
    # val_txt_path = '/home/fssv2/myungsang/datasets/focus/221024_C/front/val.txt'
    # val_txt_path = '/home/fssv2/myungsang/datasets/focus/221102_D/front/val.txt'
    # val_txt_path = '/home/fssv2/myungsang/datasets/focus/test_images/wheel.txt'
    val_txt_path = '/home/fssv2/myungsang/datasets/focus/230102_E/front/test_v2.txt'
    # val_txt_path = '/home/fssv2/myungsang/datasets/focus/230102_E/front_crop/version11/crop_test_v2.txt'
    with open(val_txt_path, 'r') as f:
        img_list = f.read().splitlines()
        
    tmp_num = 0
    color = (0, 255, 0)
    
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 1080, 720)

    for img_path in tqdm(img_list[:]):
        detections = get_detection(
            img_path, 
            network, 
            class_names,
            thresh=0.4
        )
        
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        img = cv2.resize(img, (width, height))

        for detection in detections:
            class_name, confidence_score, box = detection
            class_idx = class_names.index(class_name)
            cx = box[0]
            cy = box[1]
            w = box[2]
            h = box[3]

            xmin = int((cx - (w / 2)))
            ymin = int((cy - (h / 2)))
            xmax = int((cx + (w / 2)))
            ymax = int((cy + (h / 2)))
            
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=1)
            img = cv2.putText(img, "{:s}, {:.2f}".format(class_name, confidence_score), (xmin, ymin + 20),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1,
                            color=color)
            
        img = cv2.resize(img, (img_w, img_h))

        cv2.imshow('Image', img)
        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == ord('c'):
            tmp_num += 1
            img_path = f'captured_images/image_{tmp_num:05d}.jpg'
            cv2.imwrite(img_path, img)
            
    cv2.destroyAllWindows()


def save_result():
    network, class_names, class_colors = darknet.load_network(

        'custom_train/focus/crowded_people/416/yolov4-tiny-custom-v4_416.cfg',
        'custom_train/focus/crowded_people/416/focus.data',
        'custom_train/focus/crowded_people/416/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        # 'custom_train/focus/crowded_people/576/yolov4-tiny-custom-v4_576.cfg',
        # 'custom_train/focus/crowded_people/576/focus.data',
        # 'custom_train/focus/crowded_people/576/weights/yolov4-tiny-custom-v4_576_best.weights',
        
        batch_size=1
    )

    # val_txt_path = '/home/fssv2/fssv2_dataset/Crowd/3. FDST/test.txt'
    # val_txt_path = '/home/fssv2/fssv2_dataset/Crowd/x. Mail_dot/mall_dataset/frames.txt'
    # val_txt_path = '/home/fssv2/fssv2_dataset/Crowd/2. jhu_crowd_v2.0/test.txt'
    val_txt_path = '/home/fssv2/fssv2_dataset/Crowd/1. VSCrowd/100.txt'
    with open(val_txt_path, 'r') as f:
        img_list = f.read().splitlines()
    
    # out_dir = '/home/fssv2/fssv2_dataset/Crowd/3. FDST/test_data_inference_results'
    # out_dir = '/home/fssv2/fssv2_dataset/Crowd/x. Mail_dot/mall_dataset/inference_results'
    # out_dir = '/home/fssv2/fssv2_dataset/Crowd/2. jhu_crowd_v2.0/test_inference_results'
    # out_dir = '/home/fssv2/fssv2_dataset/Crowd/1. VSCrowd/100_inference_result_576'
    out_dir = '/home/fssv2/fssv2_dataset/Crowd/1. VSCrowd/100_inference_result_416'
    
    tmp_num = 0
    color = (0, 255, 0)
    
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    
    # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Image', 1080, 720)

    for img_path in tqdm(img_list[:]):
        detections = get_detection(
            img_path, 
            network, 
            class_names,
            thresh=0.4
        )
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w, _ = img.shape
        img = cv2.resize(img, (width, height))

        for detection in detections:
            class_name, confidence_score, box = detection
            class_idx = class_names.index(class_name)
            cx = box[0]
            cy = box[1]
            w = box[2]
            h = box[3]

            xmin = int((cx - (w / 2)))
            ymin = int((cy - (h / 2)))
            xmax = int((cx + (w / 2)))
            ymax = int((cy + (h / 2)))
            
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=1)
            # img = cv2.putText(img, "{:s}, {:.2f}".format(class_name, confidence_score), (xmin, ymin + 20),
            #                 fontFace=cv2.FONT_HERSHEY_PLAIN,
            #                 fontScale=1,
            #                 color=color)
            
        img = cv2.resize(img, (img_w, img_h))
        
        # new_out_dir = os.path.join(out_dir, img_path.split(os.sep)[-2])
        # if not os.path.isdir(new_out_dir):
        #     os.makedirs(new_out_dir, exist_ok=True)
        new_img_path = os.path.join(out_dir, os.path.basename(img_path))
        cv2.imwrite(new_img_path, img)

    #     cv2.imshow('Image', img)
    #     key = cv2.waitKey(0)
    #     if key == 27:
    #         break
    #     elif key == ord('c'):
    #         tmp_num += 1
    #         img_path = f'captured_images/image_{tmp_num:05d}.jpg'
    #         cv2.imwrite(img_path, img)
            
    # cv2.destroyAllWindows()


def save_videos():
    network, class_names, class_colors = darknet.load_network(

        # 'custom_train/focus/crowded_people/416/yolov4-tiny-custom-v4_416.cfg',
        # 'custom_train/focus/crowded_people/416/focus.data',
        # 'custom_train/focus/crowded_people/416/weights/yolov4-tiny-custom-v4_416_best.weights',
        
        'custom_train/focus/crowded_people/576/yolov4-tiny-custom-v4_576.cfg',
        'custom_train/focus/crowded_people/576/focus.data',
        'custom_train/focus/crowded_people/576/weights/yolov4-tiny-custom-v4_576_best.weights',
        
        batch_size=1
    )

    in_dir = '/home/fssv2/fssv2_dataset/Crowd/1. VSCrowd/test_videos'
    video_list = os.listdir(in_dir)
    
    out_dir = '/home/fssv2/fssv2_dataset/Crowd/1. VSCrowd/test_videos_inference_result_576'
    
    color = (0, 255, 0)
    
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    for video_filename in tqdm(video_list[:]):
        cap = cv2.VideoCapture(os.path.join(in_dir, video_filename))
        if not cap.isOpened():
            print(f'Can not open this video: {video_filename}')
            continue
        
        save_video_path = os.path.join(out_dir, video_filename.rsplit('.', 1)[0] + '.avi')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(save_video_path, fourcc, fps, frame_size)
        
        while True:
            ret, img = cap.read()
            if not ret:
                break
        
            detections = get_detection_for_video(
                img, 
                network, 
                class_names,
                thresh=0.4
            )
            
            img_h, img_w, _ = img.shape
            img = cv2.resize(img, (width, height))

            for detection in detections:
                class_name, confidence_score, box = detection
                class_idx = class_names.index(class_name)
                cx = box[0]
                cy = box[1]
                w = box[2]
                h = box[3]

                xmin = int((cx - (w / 2)))
                ymin = int((cy - (h / 2)))
                xmax = int((cx + (w / 2)))
                ymax = int((cy + (h / 2)))
                
                img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=1)
            
            img = cv2.resize(img, (img_w, img_h))
            out.write(img)
        
        out.release()


def make_video():
    network, class_names, class_colors = darknet.load_network(
        'custom_train/focus/220812_B/front/yolov4-tiny-custom-v2_416/yolov4-tiny-custom-v2_416.cfg',
        'custom_train/focus/220812_B/front/yolov4-tiny-custom-v2_416/focus.data',
        'custom_train/focus/220812_B/front/yolov4-tiny-custom-v2_416/weights/yolov4-tiny-custom-v2_416_best.weights',
        batch_size=1
    )

    video_path = '/mnt/x/focus_dataset/20220816_focus_release_test_video/front/20220816_143754_ch01.avi'
    if not os.path.isfile(video_path):
        print(f'There is no file: {video_path}')
        sys.exit(1)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Can not open video file: {video_path}')
        sys.exit(1)
        
    save_video_path = os.path.join('/home/fssv2/myungsang/darknet/focus_video_test/result', os.path.basename(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    # frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_size = (416, 416)
    out = cv2.VideoWriter(save_video_path, fourcc, fps, frame_size)
    
    color = (0, 255, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = get_detection_for_video(
            frame, 
            network, 
            class_names,
            thresh=0.3
        )
        img = frame.copy()
        img = cv2.resize(img, (416, 416))
        
        for detection in detections:
            class_name, confidence_score, box = detection
            # class_idx = name_list.index(class_name)
            cx = box[0]
            cy = box[1]
            w = box[2]
            h = box[3]

            xmin = int((cx - (w / 2)))
            ymin = int((cy - (h / 2)))
            xmax = int((cx + (w / 2)))
            ymax = int((cy + (h / 2)))
            
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=2)

        out.write(img)

        cv2.imshow('Image', img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    
    out.release()
    cv2.destroyAllWindows()


def check_inference_speed():
    network, class_names, class_colors = darknet.load_network(
        'custom_train/yolov3-tiny-3l-voc/yolov3-tiny-3l.cfg',
        'custom_train/yolov3-tiny-3l-voc/voc.data',
        'custom_train/yolov3-tiny-3l-voc/weights/yolov3-tiny-3l_best.weights',
        batch_size=1
    )

    video_path = './crowd_skywalk.mp4'
    if not os.path.isfile(video_path):
        print(f'There is no file: {video_path}')
        sys.exit(1)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Can not open video file: {video_path}')
        sys.exit(1)
        
    color = (0, 255, 0)
    fps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = get_detection_for_inference_speed(
            frame,
            network, 
            class_names,
            thresh=0.3, 
            fps=fps
        )
        img = frame.copy()
        img = cv2.resize(img, (416, 416))
        
        for detection in detections:
            class_name, confidence_score, box = detection
            # class_idx = name_list.index(class_name)
            cx = box[0]
            cy = box[1]
            w = box[2]
            h = box[3]

            xmin = int((cx - (w / 2)))
            ymin = int((cy - (h / 2)))
            xmax = int((cx + (w / 2)))
            ymax = int((cy + (h / 2)))
            
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=2)

        cv2.imshow('Image', img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    
    print(f'\navg-inference: {sum(fps)/len(fps)}')
    
    cv2.destroyAllWindows()


def check_inference_speed_by_image():
    network, class_names, class_colors = darknet.load_network(
        # 'custom_train/yolov2-voc/yolov2-voc.cfg',
        # 'custom_train/yolov2-voc/voc.data',
        # 'custom_train/yolov2-voc/weights_random=0/yolov2-voc_best.weights',
        
        # 'custom_train/yolov3-custom-voc/yolov3-custom-voc.cfg',
        # 'custom_train/yolov3-custom-voc/voc.data',
        # 'custom_train/yolov3-custom-voc/weights/yolov3-custom-voc_best.weights',
        
        # 'custom_train/yolov3-tiny-3l-voc/yolov3-tiny-3l.cfg',
        # 'custom_train/yolov3-tiny-3l-voc/voc.data',
        # 'custom_train/yolov3-tiny-3l-voc/weights/yolov3-tiny-3l_best.weights',
        
        # 'custom_train/yolov3-tiny-3l-custom-01-voc/yolov3-tiny-3l-custom.cfg',
        # 'custom_train/yolov3-tiny-3l-custom-01-voc/voc.data',
        # 'custom_train/yolov3-tiny-3l-custom-01-voc/weights/yolov3-tiny-3l-custom_best.weights',
        
        # 'custom_train/yolov4-tiny-3l-voc/yolov4-tiny-3l.cfg',
        # 'custom_train/yolov4-tiny-3l-voc/voc.data',
        # 'custom_train/yolov4-tiny-3l-voc/weights/yolov4-tiny-3l_best.weights',
        
        # 'custom_train/yolov4-tiny-3l-custom-01-voc/yolov4-tiny-3l-custom.cfg',
        # 'custom_train/yolov4-tiny-3l-custom-01-voc/voc.data',
        # 'custom_train/yolov4-tiny-3l-custom-01-voc/weights/yolov4-tiny-3l-custom_best.weights',
        
        # 'custom_train/yolov4-tiny-3l-custom-02-voc/yolov4-tiny-3l-custom.cfg',
        # 'custom_train/yolov4-tiny-3l-custom-02-voc/voc.data',
        # 'custom_train/yolov4-tiny-3l-custom-02-voc/weights/yolov4-tiny-3l-custom_best.weights',
        
        # 'custom_train/yolov4-tiny-3l-custom-03-voc/yolov4-tiny-3l-custom.cfg',
        # 'custom_train/yolov4-tiny-3l-custom-03-voc/voc.data',
        # 'custom_train/yolov4-tiny-3l-custom-03-voc/weights/yolov4-tiny-3l-custom_best.weights',
        
        # 'custom_train/yolov4-tiny-3l-custom-04-voc/yolov4-tiny-3l-custom.cfg',
        # 'custom_train/yolov4-tiny-3l-custom-04-voc/voc.data',
        # 'custom_train/yolov4-tiny-3l-custom-04-voc/weights/yolov4-tiny-3l-custom_best.weights',
        
        # 'custom_train/yolov4-tiny-3l-custom-05-voc/yolov4-tiny-3l-custom.cfg',
        # 'custom_train/yolov4-tiny-3l-custom-05-voc/voc.data',
        # 'custom_train/yolov4-tiny-3l-custom-05-voc/weights/yolov4-tiny-3l-custom_best.weights',
        
        'custom_train/yolov4-tiny-3l-custom-06-voc/yolov4-tiny-3l-custom.cfg',
        'custom_train/yolov4-tiny-3l-custom-06-voc/voc.data',
        'custom_train/yolov4-tiny-3l-custom-06-voc/weights/yolov4-tiny-3l-custom_best.weights',
        
        batch_size=1
    )
    
    fps = []
    frame = cv2.imread('data/dog.jpg')
    
    for idx in range(10005):

        detections = get_detection_for_inference_speed(
            frame,
            network, 
            class_names,
            thresh=0.3, 
            fps=fps
        )
        
    print(f'\nAvg Inference: {int(sum(fps[5:])/len(fps[5:]))}')


def show_lpr_result_by_full_img():
    network, class_names, class_colors = darknet.load_network(
        #######################################################################################################
        # LPR
        #######################################################################################################
        
        # 'custom_train/yolov4-tiny-3l-custom-04-lpr/yolov4-tiny-3l-custom.cfg',
        # 'custom_train/yolov4-tiny-3l-custom-04-lpr/lpr.data',
        # 'custom_train/yolov4-tiny-3l-custom-04-lpr/weights/yolov4-tiny-3l-custom_best.weights',
        
        'custom_train/yolov4-tiny-3l-custom-05-lpr/yolov4-tiny-3l-custom.cfg',
        'custom_train/yolov4-tiny-3l-custom-05-lpr/lpr.data',
        'custom_train/yolov4-tiny-3l-custom-05-lpr/weights/yolov4-tiny-3l-custom_best.weights',
        
        batch_size=1
    )

    val_txt_path = '/home/fssv2/myungsang/datasets/lpr/test_v1.txt'
    with open(val_txt_path, 'r') as f:
        img_list = f.read().splitlines()
        
    tmp_num = 0
    color = (0, 255, 0)
    true_num = 0
    total_num = len(img_list)
    
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    
    font = ImageFont.truetype('malgun.ttf', 30)
    with open('/home/fssv2/myungsang/datasets/lpr/lpr_kr.names') as f:
        name_list = f.read().splitlines()
    
    for img_path in tqdm(img_list[:]):
        cv_img = cv2.imread(img_path)
        img_h, img_w, _ = cv_img.shape
        img = Image.fromarray(cv_img)
        draw = ImageDraw.Draw(img)
        
        txt_path = img_path.rsplit('.', 1)[0] + '.txt'
        with open(txt_path, 'r') as f:
            labels = f.read().splitlines()
        # labels = [[float(y) for y in x.split(' ')[1:]] for x in labels if x.split(' ')[0] == '8']
        labels = [[float(y) for y in x.split(' ')[1:]] for x in labels if x.split(' ')[0] == '0']
        label = labels[0]
        
        cx = label[0] * img_w
        cy = label[1] * img_h
        w = label[2] * img_w
        h = label[3] * img_h
        
        xmin = int(cx - (w / 2))
        ymin = int(cy - (h / 2))
        xmax = int(cx + (w / 2))
        ymax = int(cy + (h / 2))
        
        crop_img = cv_img[ymin:ymax, xmin:xmax].copy()
        
        detections = get_detection_for_video(
            crop_img, 
            network, 
            class_names,
            thresh=0.4
        )
        # [cls_idx, confidence, cx, cy, w, h]
        detections = np.array([[class_names.index(x[0]), x[1], x[2][0], x[2][1], x[2][2], x[2][3]] for x in detections])
        
        plate_num, detections = get_plate_number(detections, height, name_list)
        true_label = os.path.basename(img_path).rsplit('.', 1)[0]
        if len(true_label.split('-')) > 1:
            true_label = true_label.split('-')[0]
        if plate_num == true_label:
            true_num += 1
        print(f'True: {true_label}, Pred: {plate_num}')
        
        
        # show image
        draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=1)
        
        txt_w, txt_h = draw.textsize(plate_num, font=font)
        draw.text(((xmin, max(ymin - txt_h, 0))), f'{plate_num}', font=font, fill=color)
        
        # draw plate number
        for detection in detections:
            _, _, cx, cy, w, h = detection
            
            pxmin = xmin + int((cx - (w / 2)) * (xmax - xmin) / width)
            pxmax = xmin + int((cx + (w / 2)) * (xmax - xmin) / width)
            pymin = ymin + int((cy - (h / 2)) * (ymax - ymin) / height)
            pymax = ymin + int((cy + (h / 2)) * (ymax - ymin) / height)
            
            draw.rectangle((pxmin, pymin, pxmax, pymax), outline=color, width=1)
                
        
        img = np.array(img)
        
        # # Save Image
        # new_dir = os.path.join('/home/fssv2/myungsang/datasets/lpr/darknet_test', img_path.split(os.sep)[-2])
        # if not os.path.isdir(new_dir):
        #     os.makedirs(new_dir, exist_ok=True)
        # new_path = os.path.join(new_dir, os.path.basename(img_path))
        # cv2.imwrite(new_path, img)

        # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Image', 1080, 720)
        # cv2.imshow('Image', img)
        # key = cv2.waitKey(0)
        # if key == 27:
        #     break
        # elif key == ord('c'):
        #     tmp_num += 1
        #     img_path = f'captured_images/image_{tmp_num:05d}.jpg'
        #     cv2.imwrite(img_path, img)

    print(f'Accuracy: {true_num} / {total_num} = {(true_num / total_num)*100:.2f}%')


def show_lpr_result():
    network, class_names, class_colors = darknet.load_network(
        #######################################################################################################
        # LPR
        #######################################################################################################
        
        'custom_train/yolov4-tiny-3l-custom-04-lpr/yolov4-tiny-3l-custom.cfg',
        'custom_train/yolov4-tiny-3l-custom-04-lpr/lpr.data',
        'custom_train/yolov4-tiny-3l-custom-04-lpr/weights/yolov4-tiny-3l-custom_best.weights',
        
        batch_size=1
    )

    val_txt_path = '/home/fssv2/myungsang/datasets/lpr/val.txt'
    with open(val_txt_path, 'r') as f:
        img_list = f.read().splitlines()
        
    tmp_num = 0
    color = (0, 255, 0)
    
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    
    font = ImageFont.truetype('malgun.ttf', 20)
    with open('/home/fssv2/myungsang/datasets/lpr/lpr_kr.names') as f:
        name_list = f.read().splitlines()

    for img_path in tqdm(img_list[:]):
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        img = cv2.resize(img, (width, height))
        
        detections = get_detection_for_video(
            img, 
            network, 
            class_names,
            thresh=0.4
        )
        
        # [cls_idx, confidence, cx, cy, w, h]
        detections = np.array([[class_names.index(x[0]), x[1], x[2][0], x[2][1], x[2][2], x[2][3]] for x in detections])
        
        plate_num, detections = get_plate_number(detections, height, name_list)
        print(plate_num)
        
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        for detection in detections:
            class_idx, confidence_score, cx, cy, w, h = detection

            xmin = int((cx - (w / 2)))
            ymin = int((cy - (h / 2)))
            xmax = int((cx + (w / 2)))
            ymax = int((cy + (h / 2)))
            
            draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=1)
            draw.text(((xmin, ymin)), f'{name_list[int(class_idx)]}', font=font, fill=color)
            
        img = cv2.resize(np.array(img), (img_w, img_h))

        cv2.imshow('Image', img)
        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == ord('c'):
            tmp_num += 1
            img_path = f'captured_images/image_{tmp_num:05d}.jpg'
            cv2.imwrite(img_path, img)
            
    
    cv2.destroyAllWindows()


def get_plate_number(detections, network_height, cls_name_list):
    detect_num = len(detections)
    if detect_num < 4:
        return 'None', detections
    elif detect_num > 8:
        detections = np.delete(detections, np.argsort(detections[..., 1])[:detect_num-8], axis=0)
    detections = detections[np.argsort(detections[..., 3])]
    
    thresh = int(network_height / 5)
    y1 = detections[1][3] - detections[0][3]
    y2 = detections[3][3] - detections[2][3]
    
    # 외교 번호판
    if y1 > thresh:
        detections[1:] = detections[1:][np.argsort(detections[1:, 2])]
    
    # 운수/건설 번호판
    elif y2 > thresh:
        detections[:3] = detections[:3][np.argsort(detections[:3, 2])]
        detections[3:] = detections[3:][np.argsort(detections[3:, 2])]
    
    # 일반 가로형 번호판
    else:
        detections = detections[np.argsort(detections[..., 2])]

    # 번호판 포맷에 맞는지 체크
    detections = check_plate(detections)

    plate_num = ''
    for cls_idx in detections[..., 0]:
        plate_num += cls_name_list[int(cls_idx)]
    
    return plate_num, detections


def check_plate(detections):
    # 가, 나, 다, ... 번호는 하나만 존재하고 그 뒤의 번호는 4자리만 올 수 있다.
    str_idx_list = np.where((10<=detections[..., 0]) & (detections[..., 0]<=48))[0]
    if len(str_idx_list):
        if len(str_idx_list) > 1:
            arg_idx = np.argsort(-detections[str_idx_list][..., 1])
            delete_idx = str_idx_list[arg_idx[1:]]
            detections = np.delete(detections, delete_idx, axis=0)
            str_idx = str_idx_list[arg_idx[0]]
        else:
            str_idx = str_idx_list[0]
            
        if len(detections[str_idx+1:]) > 4:
            delete_idx = np.argsort(-detections[str_idx+1:, 1])[4:] + (str_idx + 1)
            detections = np.delete(detections, delete_idx, axis=0)

    # 서울, 경기, ... 지역 번호는 하나만 존재
    area_idx_list = np.where((49<=detections[..., 0]) & (detections[..., 0]<=64))[0]
    if len(area_idx_list) > 1:
        arg_idx = np.argsort(-detections[area_idx_list][..., 1])
        delete_idx = area_idx_list[arg_idx[1:]]
        detections = np.delete(detections, delete_idx, axis=0)
    
    # 외교, 영사, ... 번호는 하나만 존재
    diplomacy_idx_list = np.where(64 < detections[..., 0])[0]
    if len(diplomacy_idx_list) > 1:
        arg_idx = np.argsort(-detections[diplomacy_idx_list][..., 1])
        delete_idx = diplomacy_idx_list[arg_idx[1:]]
        detections = np.delete(detections, delete_idx, axis=0)
    
    return detections


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return image
    
    
if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    
    # check_map_by_custom_map_calculator()
    # check_map_by_coco_map_calculator()
    # check_map_by_torchmetrics_map_calculator()
    # check_map_by_torchmetrics_yolo_format()
    # make_pred_result_file_for_public_map_calculator()
    # show_result()
    # save_result()
    # save_videos()
    # make_video()
    # check_inference_speed()
    # check_inference_speed_by_image()
    show_lpr_result_by_full_img()
    # show_lpr_result()