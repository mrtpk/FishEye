# FishEye
Detecting fish from sequences

+ [Discussion Notes](https://docs.google.com/document/d/1ZsbtSF3w8XVaRTAvtOWZsQTpCUJeGjc9XE1LMQ6Xwks/edit?usp=sharing)

# Visum

+ [visum-2020](https://github.com/visum-summerschool/visum-2020)
+ [visum-competition](https://github.com/visum-summerschool/visum-competition2020)

Competition project - ...

# GT Format for YOLO
+ The format in the `dataset/train/labels.csv` can be converted into Darknet YOLO format by using `yolo_gt_conversion/convert_gt.py` script. The generated example file along with the `labels.csv` is in the `yolo_gt_conversion`.
+ [TODO] Create YOLO format used [yolov3-keras-tf2](https://github.com/emadboctorx/yolov3-keras-tf2).

# Algorithm
+ [Ultralytics yolov5](https://github.com/ultralytics/yolov5)

Main observations:
+ **Dataset**
  + 5040 samples corresponding to snapshots of underwater videos, with and without objects
  + 1 class of objects 'fish'
  + There may be more than one object per sample
  + Annotations (bounding boxes) are in Coco (?) format
  
+ 10-fold cross validation
+ Implement augmentation [Albumentations](https://github.com/albumentations-team)

# Goal

insert..
