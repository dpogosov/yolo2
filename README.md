# YOLO2 

Implementation of YOLO v.2 image objects detector in TF and Keras
Part of the course SOW-MKI20 Capita Selecta AI

## intro
YOLO v2 was selected for implementation and practical studying for the following reasons:
1. It provides high accuracy detection, which is comparable to Faster R-CNN accuracy or even higher
2. It provides fast detection. YOLO2 is capable of real-time detection of video stream

## files description
- run_yolo.py - runs the predictior
- loader.py - loads original weights
- net.py - creates the network
- image_processor - does image read and write (with labels and boxes)

The sample output of the implemented detector is shown in the *test* folder

## requiremets
- Python 3.5+
- Tensorflow 1.+
- Keras 2.+
- Opencv 2.+
- Configparser
- misc: numpy, os, io

## notes
- Detector only
- Checked on COCO weights only

## sources
I consulted with the following sources (20% was adopted):
- https://github.com/pjreddie/darknet
- https://github.com/thtrieu/darkflow
- https://github.com/allanzelener/yad2k
