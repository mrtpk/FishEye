import argparse

import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *
import csv

def detect(save_img=False):
    source, weights, imgsz = \
        opt.source, opt.weights, opt.img_size
    device = torch_utils.select_device(opt.device)
    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    imgsz = check_img_size(imgsz, s=model.model[-1].stride.max())  # check img_size

    # TODO: Second-stage classifier
    dataset = LoadImages(source, img_size=imgsz)
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    
    # inti model
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img)
    
    # detect for all images
    predictions = []
    for path, img, im0, _ in dataset:
        _path = Path(path)
        seq = _path.parent.stem.replace("seq", "")
        frame = _path.stem.replace("img", "")
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    xmin = int(xyxy[0])
                    ymin = int(xyxy[1])
                    xmax = int(xyxy[2])
                    ymax = int(xyxy[3])
                    coor = [xmin, ymin, xmax, ymax]
                    score = float(conf)
                    predictions.append([seq, frame, coor, score])
    with open('predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(['seq', 'frame', 'label', 'score'])
        writer.writerows(predictions)
    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='/home/master/dataset/test/', help='source') 
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
