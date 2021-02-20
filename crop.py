from core.config import cfg
from absl import app, flags, logging
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from PIL import Image
import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import colorsys
import random
import pandas as pd
import sys

def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
    return bboxes

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def crop_objects(img, data, path, allowed_classes, image_name):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    #create dictionary to hold count of objects for image name
    counts = dict()
    for i in range(num_objects):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            # get box coords
            xmin, ymin, xmax, ymax = boxes[i]
            # crop detection from image (take an additional 5 pixels around all edges)
            cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
            # construct image name and join it to path for saving crop properly
            img_name = image_name + '.png'
            img_path = os.path.join(path, img_name )
            # save image
            try:
                cv2.imwrite(img_path, cropped_img)
            except Exception:
                pass
        else:
            continue

def count_vehicals(data, allowed_classes):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    vehicalCount = 0
    for i in range(num_objects):
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            vehicalCount += 1
    return vehicalCount

def main(image_dir, flgOutput):
    flgIou = 0.45
    flgScore = 0.50
    input_size = 416
    flgModel = 'yolov4'
    flgWeights = './core/yolov4-416'
    flgClasses = ['car','motorbike','bus','truck']

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    STRIDES = np.array(cfg.YOLO.STRIDES)
    ANCHORS = np.array(cfg.YOLO.ANCHORS).reshape(3, 3, 2)

    XYSCALE = cfg.YOLO.XYSCALE

    NUM_CLASS = len(flgClasses)

    # load model
    saved_model_loaded = tf.saved_model.load(flgWeights, tags=[tag_constants.SERVING])

    results_df = pd.DataFrame(columns = ['image','class'])
    images = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]

    sys.stdout.write('Total '+str(len(images))+' images')
    # loop through images in list and run Yolov4 model on each
    for count, image_name in enumerate(images, 1):
        sys.stdout.write("\rImages Croped: %i" % count)
        sys.stdout.flush()
        image_path = image_dir+image_name

        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.
        
        # get image name by using split method
        image_name = image_path.split('/')[-1]
        image_name = image_name.split('.')[0]

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        # run non max suppression on detections
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=flgIou,
            score_threshold=flgScore
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = original_image.shape
        bboxes = format_boxes(boxes.numpy()[0], original_h, original_w)
        
        # hold all detection data in one variable
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # custom allowed classes (uncomment line below to allow detections for only people)
        allowed_classes = flgClasses

        vehicalCount = count_vehicals(pred_bbox,allowed_classes)

        if(vehicalCount == 1):
            crop_path = os.path.join(flgOutput, 'crop')
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path, allowed_classes, image_name)
            results_df = results_df.append({'image':image_name, 'class':1}, ignore_index=True)
        else:
            results_df = results_df.append({'image':image_name, 'class':0}, ignore_index=True)
    results_df.to_csv(flgOutput+'/class.csv')
    print('\nClean.csv generated: '+flgOutput+'/class.csv')

if __name__ == '__main__':
    try:
        main("./data/images/", './data/detections/')
    except SystemExit:
        pass