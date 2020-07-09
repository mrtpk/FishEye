# Visum2020

+ [visum-2020](https://github.com/visum-summerschool/visum-2020)
+ [visum-competition](https://github.com/visum-summerschool/visum-competition2020): Fish Detection in Underwater Images

**Why is it important?**
+ Offshore aquaculture
+ Environmental studies
+ Before building a new offshore field ( i.e. offshore wind farm)
+ Before decommissioning an old field

### Dataset

+ Train
  + 84 sequences of 60 frames each;
  + Some images contain one or more fish;
  + Some images do not contain any fish.
  
+ Test (no access)
  + 30 sequences of 60 frames each
  + Some images contain one or more fish;
  + Some images do not contain any fish;
  + Daily test: 10 sequences of 60 frames each.
  
+ Annotations format
  + sequence; frame; [(x_min , y_min , x_max , y_max ), x_min , y_min , x_max , y_max
  + sequence; frame;
  
+ Predictions format
  + sequence; frame; [(x_min , y_min , x_max , y_max )]; confidence

### Evaluation metrics
  + IOU
  + Precision
  + Recall
  + Average precision (AP@[0.5:0.95] - averaged over different IoU thresholds
 
## Team FishEye
+ [Discussion Notes](https://docs.google.com/document/d/1ZsbtSF3w8XVaRTAvtOWZsQTpCUJeGjc9XE1LMQ6Xwks/edit?usp=sharing)

### Algorithm
+ [Ultralytics yolov5](https://github.com/ultralytics/yolov5)

### Challenges
+ Find ways to deal with the big variability of visibility conditions underwater;

### Strategy
+ Implemented augmentation [Albumentations](https://github.com/albumentations-team)

