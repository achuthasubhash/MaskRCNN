# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:24:43 2020

@author: DELL
"""

import cv2
import numpy as np
from visualize_cv2 import model, display_instances, class_names

capture = cv2.VideoCapture('videofile.mp4') #read video
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), #get width, height
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('videofile_masked.avi', codec, 60.0, size) #write video,fps, frame size