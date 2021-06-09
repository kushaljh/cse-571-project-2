#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that listens to std_msgs/Strings published 
## to the 'chatter' topic

import rospy
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image

import torch
import torchvision
import cv2
import numpy as np
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os



class Wrapper:
    def __init__(self):
        print(os.getcwd())
        # TODO Instantiate your model and other class instances here!
        # TODO Don't forget to set your model in evaluation/testing/production mode, and sending it to the GPU (if you have one)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = Model()
        # self.model.load_state_dict(torch.load("model.pt", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, batch_or_image, area_thresh=10.0, cone_size_red=0.8, duck_size_red=0.9):
        # TODO: Make your model predict here!

        # The given batch_or_image parameter will be a numpy array (ie either a 224 x 224 x 3 image, or a
        # batch_size x 224 x 224 x 3 batch of images)
        # These images will be 224 x 224, but feel free to have any model, so long as it can handle these
        # dimensions. You could resize the images before predicting, make your model dimension-agnostic somehow,
        # etc.

        # This method should return a tuple of three lists of numpy arrays. The first is the bounding boxes, the
        # second is the corresponding labels, the third is the scores (the probabilities)

        if len(batch_or_image.shape) == 3:
            batch = [batch_or_image]
        else:
            batch = batch_or_image

        with torch.no_grad():
            preds = self.model([to_tensor(cv2.resize(img, (224,224))).to(device=self.device, dtype=torch.float) for img in batch])
            print((batch[0]).shape)
        boxes = []
        labels = []
        scores = []
        for pred in preds:
            pred_boxes = pred["boxes"].cpu().numpy()
            
            filt_pred_boxes = []
            filt_pred_labels = []
            filt_pred_scores = []
            
            # Filter predictions that we know to be bad
            for i in range(pred_boxes.shape[0]):
                box = pred_boxes[i]
                label = pred["labels"].cpu().numpy()[i]
                area = (box[2] - box[0]) * (box[3] - box[1])
                
                if area > area_thresh:
                    # alter the box
                    new_box = np.copy(box)

                    if label == 1: # duck
                        new_box[2] = box[0] + duck_size_red * (box[2] - box[0]) # alter the width
                        new_box[1] = box[3] + duck_size_red * (box[1] - box[3]) # alter the height
                    elif label == 2: # cone
                        new_box[2] = box[0] + (cone_size_red + 0.1) * (box[2] - box[0]) # alter the width
                        new_box[1] = box[3] + cone_size_red * (box[1] - box[3]) # alter the height more

                    filt_pred_boxes.append(new_box)
                    filt_pred_labels.append(label)
                    filt_pred_scores.append(pred["scores"].cpu().numpy()[i])

            filt_pred_boxes = np.array(filt_pred_boxes)
            filt_pred_labels = np.array(filt_pred_labels)
            filt_pred_scores = np.array(filt_pred_scores)

            boxes.append(filt_pred_boxes)
            labels.append(filt_pred_labels)
            scores.append(filt_pred_scores)

        return boxes, labels, scores

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # TODO Instantiate your weights etc here!
        # Load the model
        self.model = torchvision.models.detection \
                     .fasterrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)

    def forward(self, x, y=None):
        return self.model(x) if y is None else self.model(x, y)


class RunModel:

    def __init__(self) -> None:
        rospy.init_node('detector', anonymous=True)
        self.pub = rospy.Publisher('detection', String, queue_size=10)
        self.wrapper = Wrapper()
        rospy.Subscriber('images', Image, self.callback)
        # self.rate = rospy.Rate(2) 

    def callback(self, data):
        # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
        rospy.loginfo(rospy.get_caller_id() + " received image")
        im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        im = np.float32(im)
        # torch_data = torch.from_numpy(im)
        p_boxes, p_classes, p_scores = self.wrapper.predict(im)
        p_classes = p_classes[0]
        self.pub.publish(np.any(p_classes == 1))


if __name__ == '__main__':
    detector_node = RunModel()
    rospy.spin()


