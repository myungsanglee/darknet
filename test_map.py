import argparse
import os
import glob
import random
from unicodedata import name
import sys
from tqdm import tqdm

from matplotlib.pyplot import axis
import darknet
import time
import cv2
import numpy as np
import darknet
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
import albumentations
import albumentations.pytorch

sys.path.append('/home/fssv2/myungsang/my_projects/PyTorch-Object-Detection')
from dataset.detection.yolov2_utils import mean_average_precision

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


def get_y_true(img_path, img_idx):
    valid_transform = albumentations.Compose([
        albumentations.Resize(416, 416, always_apply=True),
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
        padded_annots[idx] = torch.FloatTensor([img_idx, cx*416, cy*416, w*416, h*416, 1., class_idx])

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
        # 'custom_train/yolov2-voc/yolov2-voc.cfg',
        'custom_train/yolov3-custom-voc/yolov3-custom-voc.cfg',
        # 'custom_train/yolov2-voc/voc.data',
        'custom_train/yolov3-custom-voc/voc.data',
        # 'custom_train/yolov2-voc/weights_random=0/yolov2-voc_best.weights',
        # 'custom_train/yolov2-voc/weights_random=1/yolov2-voc_best.weights',
        # 'custom_train/yolov2-voc/weights_with_pretrained/yolov2-voc_best.weights',
        'custom_train/yolov3-custom-voc/weights/yolov3-custom-voc_best.weights',
        batch_size=1
    )

    names_txt_path = '/home/fssv2/myungsang/datasets/voc/yolo_format/voc.names'
    with open(names_txt_path, 'r') as f:
        name_list = f.read().splitlines()

    val_txt_path = '/home/fssv2/myungsang/datasets/voc/yolo_format/val.txt'
    with open(val_txt_path, 'r') as f:
        img_list = f.read().splitlines()

    all_true_boxes_variable = 0
    all_pred_boxes_variable = 0
    img_idx = 0
    for img_path in tqdm(img_list[:]):
        y_true = get_y_true(img_path, img_idx)

        detections = get_detection(
            img_path, 
            network, 
            class_names,
            thresh=0.005
        )

        detection_list = []
        for detection in detections:
            class_name, confidence, box = detection
            class_idx = name_list.index(class_name)
            cx = box[0]
            cy = box[1]
            w = box[2]
            h = box[3]
            detection_list.append([img_idx, cx, cy, w, h, confidence, class_idx])

        if img_idx == 0:
            all_pred_boxes_variable = torch.as_tensor(detection_list, dtype=torch.float32)
            all_true_boxes_variable = y_true
        else:
            all_pred_boxes_variable = torch.cat([all_pred_boxes_variable, torch.as_tensor(detection_list, dtype=torch.float32)], dim=0)
            all_true_boxes_variable = torch.cat([all_true_boxes_variable, y_true], dim=0)

        img_idx += 1
    
    map = mean_average_precision(all_true_boxes_variable, all_pred_boxes_variable, 20)
    print(f'mAP: {map}')


def check_map_by_coco_map_calculator():
    network, class_names, class_colors = darknet.load_network(
        # 'custom_train/yolov2-voc/yolov2-voc.cfg',
        'custom_train/yolov3-custom-voc/yolov3-custom-voc.cfg',
        # 'cfg/yolov4.cfg',
        # 'custom_train/yolov2-voc/voc.data',
        'custom_train/yolov3-custom-voc/voc.data',
        # 'custom_train/yolov3-tiny-3l-custom-coco/coco.data',
        # 'custom_train/yolov2-voc/weights_random=0/yolov2-voc_best.weights',
        # 'custom_train/yolov2-voc/weights_random=1/yolov2-voc_best.weights',
        # 'custom_train/yolov2-voc/weights_with_pretrained/yolov2-voc_best.weights',
        'custom_train/yolov3-custom-voc/weights/yolov3-custom-voc_best.weights',
        # 'yolov4.weights',
        batch_size=1
    )

    # # names_txt_path = '/home/fssv2/myungsang/datasets/voc/yolo_format/voc.names'
    # names_txt_path = '/home/fssv2/myungsang/darknet/custom_train/yolov3-tiny-3l-custom-coco/coco.names'
    # with open(names_txt_path, 'r') as f:
    #     name_list = f.read().splitlines()

    # names_txt_path = '/home/fssv2/myungsang/datasets/coco_2017/coco.names'
    # with open(names_txt_path, 'r') as f:
    #     org_name_list = f.read().splitlines()

    json_path = '/home/fssv2/myungsang/datasets/voc/coco_format/val.json'
    # json_path = '/home/fssv2/myungsang/datasets/coco_2017/annotations/instances_val2017.json'
    coco = COCO(json_path)

    imgs = coco.loadImgs(coco.getImgIds())
    cats = coco.loadCats(coco.getCatIds())
    cats_dict = dict([[cat['name'], cat['id']] for cat in cats])

    imgs_info = [[img['id'], img['file_name'], img['width'], img['height']] for img in imgs]
    
    results = []
    results_json_path = os.path.join(os.getcwd(), 'results.json')

    for (img_id, img_name, width, height) in tqdm(imgs_info):
        img_path = os.path.join('/home/fssv2/myungsang/datasets/voc/yolo_format/val', img_name)
        # img_path = os.path.join('/home/fssv2/myungsang/datasets/coco_2017/val2017', img_name)

        detections = get_detection(
            img_path, 
            network, 
            class_names,
            thresh=0.25
        )

        for detection in detections:
            class_name, confidence, box = detection
            cx = box[0] * (width / darknet.network_width(network))
            cy = box[1] * (height / darknet.network_height(network))
            w = box[2] * (width / darknet.network_width(network))
            h = box[3] * (height / darknet.network_height(network))

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
    
    # print(f'detection count: {len(results)}')
    
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    img_ids = sorted(coco.getImgIds())
    cat_ids = sorted(coco.getCatIds())

    # load detection JSON file from the disk
    cocovalPrediction = coco.loadRes(results_json_path)
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
        # 'custom_train/yolov2-voc/yolov2-voc.cfg',
        'custom_train/yolov3-custom-voc/yolov3-custom-voc.cfg',
        # 'custom_train/yolov2-voc/voc.data',
        'custom_train/yolov3-custom-voc/voc.data',
        # 'custom_train/yolov2-voc/weights_random=0/yolov2-voc_best.weights',
        # 'custom_train/yolov2-voc/weights_random=1/yolov2-voc_best.weights',
        # 'custom_train/yolov2-voc/weights_with_pretrained/yolov2-voc_best.weights',
        'custom_train/yolov3-custom-voc/weights/yolov3-custom-voc_best.weights',
        batch_size=1
    )

    names_txt_path = '/home/fssv2/myungsang/datasets/voc/yolo_format/voc.names'
    with open(names_txt_path, 'r') as f:
        name_list = f.read().splitlines()

    val_txt_path = '/home/fssv2/myungsang/datasets/voc/yolo_format/val.txt'
    # val_txt_path = '/home/fssv2/myungsang/datasets/tmp/val.txt'
    with open(val_txt_path, 'r') as f:
        img_list = f.read().splitlines()
        
    color = (0, 255, 0)

    for img_path in tqdm(img_list[:]):
        detections = get_detection(
            img_path, 
            network, 
            class_names,
            thresh=0.005
        )
        
        img = cv2.imread(img_path)
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
            
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=1)
            img = cv2.putText(img, "{:s}, {:.2f}".format(class_name, confidence_score), (xmin, ymin + 20),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1,
                            color=color)

        cv2.imshow('Image', img)
        key = cv2.waitKey(0)
        if key == 27:
            break
    
    cv2.destroyAllWindows()


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
        
        'custom_train/yolov4-tiny-3l-custom-05-voc/yolov4-tiny-3l-custom.cfg',
        'custom_train/yolov4-tiny-3l-custom-05-voc/voc.data',
        'custom_train/yolov4-tiny-3l-custom-05-voc/weights/yolov4-tiny-3l-custom_best.weights',
        
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

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= '1'
    
    # check_map_by_custom_map_calculator()
    # check_map_by_coco_map_calculator()
    # make_pred_result_file_for_public_map_calculator()
    # show_result()
    # make_video()
    # check_inference_speed()
    check_inference_speed_by_image()