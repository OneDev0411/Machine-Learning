#!/usr/bin/env python3
"""Class that uses the Yolo v3 algorithm to perform object detection"""
from tensorflow import keras as K


class Yolo:
    """ A class that uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ model_path: where a Darknet Keras model is stored
            classes_path: where the list of class names used for the model
            class_t: the box score threshold for the initial filtering step
            nms_t: IOU threshold for non-max suppression
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
                containing all of the anchor boxes """
        self.model = K.models.load_model(filepath=model_path)
        self.class_names = [line.strip() for line in open(classes_path)]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
