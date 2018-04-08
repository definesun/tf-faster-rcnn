#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__',
           'logo')

# ./tools/demo.py --net vgg16 --dataset pascal_voc
# ./tools/demo.py --net res101 --dataset pascal_voc
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_5000.ckpt',),'res101': ('res101_faster_rcnn_iter_5000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

class ImgRecog:
    sess = 0
    net = 0
    def __init__(self):
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        demonet = 'res101'
        dataset = 'pascal_voc'
        tfmodel = os.path.join('../output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])

        if not os.path.isfile(tfmodel + '.meta'):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

        # set config
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth=True

        # init session
        self.sess = tf.Session(config=tfconfig)
        self.net = resnetv1(num_layers=101)
        self.net.create_architecture("TEST", 2, tag='default', anchor_scales=[8, 16, 32])
        saver = tf.train.Saver()
        saver.restore(self.sess, tfmodel)

        print('Loaded network {:s}'.format(tfmodel))

    # [{"name": "logo", "score":0.986, "area":[x1, y1, x2, y2]}, ... ]
    def vis_detections(self, im, class_name, dets, thresh=0.5):
        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return

        im = im[:, :, (2, 1, 0)]
        tmp_arr = []
        for i in inds:
            tmp_dict = {}
            bbox = dets[i, :4]
            score = dets[i, -1] 
            print('Rectangle: [{:.3f}, {:.3f}, {:.3f}, {:.3f}]'.format(bbox[0], bbox[1], bbox[2], bbox[3]))
            print('Result: {:s} {:.3f}'.format(class_name, score))
            
            tmp_dict['name'] = class_name
            tmp_dict['score'] = int(score*100)        
            tmp_dict['area'] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]     
            tmp_arr.append(tmp_dict);    
        return tmp_arr;

    def img_rec(self, im):
        """Detect object classes in an image using pre-computed object proposals."""

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(self.sess, self.net, im)
        timer.toc()
        print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    
        # Visualize detections for each class
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
        result = []
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            tmp_arr = self.vis_detections(im, cls, dets, thresh=CONF_THRESH)
            result += tmp_arr
        return result

ir=ImgRecog()
