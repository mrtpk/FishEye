'''
[1]
darknet_format
    one row per object
    class x_center y_center width height
    normalised
    class number starts from 0

class x_center y_center width height
1 0.716797 0.395833 0.216406 0.147222
0 0.687109 0.379167 0.255469 0.158333
1 0.420312 0.395833 0.140625 0.166667

../coco/images/train2017/000000109622.jpg  # image
../coco/labels/train2017/000000109622.txt  # label

train.txt contains training data
test.txt contains testing data

[2]
YunYang format
xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20 
xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
# make sure that x_max < width and y_max < height

[3]
csv file
each bbox has following entry
Image	Object Name	Object Index	bx	by	bw	bh	
plan: create a pd with this info
'''

IS_FLATTEN = True

from pathlib import Path
import argparse
import cv2

def write_list(out_path, data):
    with open(out_path, mode='wt', encoding='utf-8') as fp:
        fp.write('\n'.join(data))

def read_labels(input_path):
    with open(input_path) as fp:
        data = fp.read().splitlines() # f.readlines()
    header = data[0]
    data = data[1:]
    assert len(data) == 84 * 60, "Number of files and labels don't match"
    return data, header

def get_info(data):
    data = data.split(";")
    id_seq, id_frame = data[:2]
    b_boxes = None
    if len(data) == 3:
        b_boxes = eval(data[-1])
    return id_seq, id_frame, b_boxes

def convert(im_w, im_h, x_min, x_max, y_min, y_max):
    """
    Converts bbox into centroid coordinates relative to image dimensions
    """
    dw = 1./im_w
    dh = 1./im_h
    x = (x_min + x_max)/2.0
    y = (y_min + y_max)/2.0
    w = x_max - x_min
    h = y_max - y_min
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x,y,w,h]

def deconvert(im_w, im_h, x, y, w, h):
    ox = float(x)
    oy = float(y)
    ow = float(w)
    oh = float(h)
    x = ox*im_w
    y = oy*im_h
    w = ow*im_w
    h = oh*im_h
    xmax = (((2*x)+w)/2)
    xmin = xmax-w
    ymax = (((2*y)+h)/2)
    ymin = ymax-h
    return [int(xmin),int(ymin),int(xmax),int(ymax)]


def data2yolo(data):
    if data[-1] is None:
        return data
    im_w = 840
    im_h = 600
    res = list(map(lambda x : convert(im_w, im_h, x_min=x[0], x_max=x[2], y_min=x[1], y_max=x[3]), data[-1]))
    return [data[0], data[1], res]

def write_yolo_data(data, img_dir):
    global IS_FLATTEN
    img_dir = Path(img_dir)
    for seq, frame, bboxes in data:
        if bboxes is None:
            continue
        
        if IS_FLATTEN is False:
            out_path = img_dir.joinpath("seq{}".format(seq))
            out_path.mkdir(parents=True, exist_ok=True)
            out_path = out_path.joinpath("img{}.txt".format(frame))
        else:
            out_path = img_dir.joinpath("seq{}_img{}.txt".format(seq, frame))

        cvt_bbox = lambda x: "0 {}".format(" ".join(list(map(str, x)))).strip(" ")
        bboxes = list(map(cvt_bbox, bboxes))
        write_list(out_path, bboxes)

def get_labels_format(data):
    seq, frame, bboxes = data
    ln = "{};{}".format(seq, frame)
    if bboxes is not None:
        bboxes = str(bboxes)
        ln = "{};{}".format(ln, bboxes)
    return ln

def covt_ln_darknet_format(data, input_dir, id_class=0):
    id_seq, id_frame, b_boxes = data
    bb_info = ""
    if b_boxes is not None:
        b_boxes = list(map(lambda x: ",".join(map(str, x)) + ",{}".format(id_class), b_boxes))
        bb_info = " ".join(b_boxes)
    if IS_FLATTEN is False:
        img_path = Path(input_dir).joinpath("seq{}".format(id_seq)).joinpath("img{}.jpg".format(id_frame))
    else:
        img_path = Path(input_dir).joinpath("seq{}_img{}.jpg".format(id_seq, id_frame))

    converted_txt = "{} {}".format(img_path, bb_info)
    return converted_txt.strip()


import xml.etree.cElementTree as ET

def create_root(file_prefix, width, height):
    file_base_path = Path(file_prefix)
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = str(file_base_path.parent)
    ET.SubElement(root, "filename").text = "{}.jpg".format(str(file_base_path.stem))
    ET.SubElement(root, "path").text = "{}.jpg".format(str(file_base_path))
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    return root


def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text=str(voc_label[0])
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(voc_label[1])
        ET.SubElement(bbox, "ymin").text = str(voc_label[2])
        ET.SubElement(bbox, "xmax").text = str(voc_label[3])
        ET.SubElement(bbox, "ymax").text = str(voc_label[4])
    return root


def create_file(file_prefix, width, height, voc_labels, output_dir):
    root = create_root(file_prefix, width, height)
    root = create_object_annotation(root, voc_labels)
    tree = ET.ElementTree(root)
    out_path = Path(output_dir)
    out_path = out_path.joinpath(Path(file_prefix).parent)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path = out_path.joinpath(file_prefix.split("/")[-1])
    tree.write("{}.xml".format(out_path))


def create_xml_file(file_path, output_dir):
    file_prefix = file_path.split(".jpg")[0]
    w, h = 840, 600
    try:
        with open("{}.txt".format(file_prefix), 'r') as file:
            lines = file.readlines()
    except Exception as err:
        lines = []

    voc_labels = []
    for line in lines:
        voc = []
        line = line.strip()
        data = line.split()
        voc.append('fish')
        bbox_width = float(data[3]) * w
        bbox_height = float(data[4]) * h
        center_x = float(data[1]) * w
        center_y = float(data[2]) * h
        voc.append(round(center_x - (bbox_width / 2)))
        voc.append(round(center_y - (bbox_height / 2)))
        voc.append(round(center_x + (bbox_width / 2)))
        voc.append(round(center_y + (bbox_height / 2)))
        voc_labels.append(voc)
    create_file(file_prefix, w, h, voc_labels, output_dir)

def create_train_val(data, test_percent, k_fold_idx):
    max_kfold = int(1 / test_percent)
    assert 0 <= k_fold_idx < max_kfold, "k-fold idx should be between 0 and {}".format(max_kfold-1)
    num_samples = len(data)
    num_test_samples = int(num_samples * test_percent)
    start_idx =  k_fold_idx * num_test_samples
    end_idx = start_idx + num_test_samples
    test_samples = data[start_idx:end_idx]
    train_samples = data[0:start_idx] + data[end_idx:]
    return train_samples, test_samples

def generate_gt(input_path, test_percent, k_fold_idx, img_dir, output_dir):    
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_data, header = read_labels(input_path=input_path)
    full_data = list(map(lambda x: get_info(x), full_data))
    filtered_data = [x for x in full_data if x[-1] is not None]

    for data, data_name in zip([full_data, filtered_data], ["full", "filtered"]):
        
        # Each dataset split into train and test
        train_samples, test_samples = create_train_val(data, test_percent, k_fold_idx)

        # Generating evaluation labels.csv
        for samples, sample_category in zip([train_samples, test_samples], ["train", "test"]):
            out_path = "{}/evaluation_labels_{}_{}_split_{}.csv".format(output_dir, data_name, sample_category, test_percent)
            eval_csv_data = list(map(get_labels_format, samples))
            eval_csv_data.insert(0, header)
            write_list(out_path, eval_csv_data)

        # Generating evaluation labels.csv
        for samples, sample_category in zip([train_samples, test_samples], ["train", "test"]):
            out_path = "{}/yun_yang_format_{}_{}_split_{}.txt".format(output_dir, data_name, sample_category, test_percent)
            yun_yang_data = list(map(lambda  x: covt_ln_darknet_format(x, input_dir=img_dir, id_class=0), samples))
            write_list(out_path, yun_yang_data)
        
        # YOLO format
        for samples, sample_category in zip([train_samples, test_samples], ["train", "test"]):
            print("Generating YOLO gt for {} {} dataset.".format(sample_category, data_name))
            yolo_data = list(map(lambda x: data2yolo(x), samples))
            # write the yolo data along with images
            write_yolo_data(yolo_data, img_dir=img_dir)
            if IS_FLATTEN is not True:
                get_img_path = lambda x: str(img_dir.joinpath("seq{}".format(x[0])).joinpath("img{}.jpg".format(x[1])))
            else:
                get_img_path = lambda x: str(img_dir.joinpath("seq{}_img{}.jpg".format(x[0], x[1])))
            sample_img_paths = list(map(get_img_path, samples))
            out_path = "{}/yolo_{}_{}_split_{}.txt".format(output_dir, data_name, sample_category, test_percent)
            write_list(out_path, sample_img_paths)
        
            # VOC Pascal format
            for sample_img_path in sample_img_paths:
                create_xml_file(file_path=sample_img_path, output_dir="{}/xml_labels".format(output_dir))

from glob import glob
from shutil import copyfile

def flattern_dir_struture(input_dir, output_dir):
    input_paths = glob(input_dir +  "/*/*.jpg")
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copyfile(input_dir.joinpath("labels.csv"), output_dir.joinpath("labels.csv"))
    for img_path in input_paths:
        img_path = Path(img_path)
        img_name = img_path.name
        seq = img_path.parent.stem
        dst_path = output_dir.joinpath("{}_{}".format(seq, img_name))
        copyfile(img_path, dst_path)

def get_yolo_label(file_path):
    im_w = 840
    im_h = 600
    with open(file_path, 'r') as fp:
        data = fp.readlines()
    data = list(map(lambda x: x.split(" ")[1: ], data))
    data = list(map(lambda x: deconvert(im_w, im_h, x=float(x[0]), y=float(x[1]), w=float(x[2]), h=float(x[3])), data))
    return data # [int(xmin),int(ymin),int(xmax),int(ymax)]

def write_aug_label(out_path, labels):
    im_w = 840
    im_h = 600
    bboxes = list(map(lambda x : convert(im_w, im_h, x_min=x[0], x_max=x[2], y_min=x[1], y_max=x[3]), labels))
    cvt_bbox = lambda x: "0 {}".format(" ".join(list(map(str, x)))).strip(" ")
    bboxes = list(map(cvt_bbox, bboxes))
    # import pdb; pdb.set_trace()
    write_list(out_path, bboxes)


def identity(img, labels):
    return img, labels

def augment(img, labels):
    img, labels = identity(img, labels)
    return img, labels

import shutil
def create_coco_style_dataset(train_path, test_path, output_dir):
    output_dir = Path(output_dir).joinpath("dataset")
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)

    for file_path, name in zip([train_path, test_path], ["train", "val"]):
        new_data_paths = []
        dataset_image_path = output_dir.joinpath("images").joinpath(name)
        dataset_label_path = output_dir.joinpath("labels").joinpath(name)
        dataset_label_path.mkdir(parents=True, exist_ok=True)
        dataset_image_path.mkdir(parents=True, exist_ok=True)
        with open(file_path) as fp:
            data = fp.read().splitlines()
        for img_path in data:
            img_path = Path(img_path)
            out_file_img_path = dataset_image_path.joinpath(img_path.name).resolve()
            new_data_paths.append(str(out_file_img_path))
            # copyfile(img_path, out_file_img_path)
            # try:
            #     copyfile(str(img_path).replace(".jpg", ".txt"), dataset_label_path.joinpath(img_path.name.replace(".jpg", ".txt")))
            # except Exception as err:
            #     pass
            is_det_available = True
            try:
                cur_labels = get_yolo_label(file_path=str(img_path).replace(".jpg", ".txt"))
            except Exception as err:
                copyfile(img_path, out_file_img_path)
                is_det_available = False
            
            if is_det_available:
                img_data = cv2.imread(str(img_path))
                nw_img_data, nw_labels = augment(img=img_data, labels=cur_labels)
                cv2.imwrite(str(out_file_img_path), nw_img_data)
                write_aug_label(out_path=dataset_label_path.joinpath(img_path.name.replace(".jpg", ".txt")), labels=nw_labels)



        with open(output_dir.joinpath("{}.txt".format(name)), mode='wt', encoding='utf-8') as fp:
            fp.write('\n'.join(new_data_paths))
"""
output dir
    images
        train
        valid
    labels
        train
        valid
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfold-idx', type=int, help='K fold index')
    opt = parser.parse_args()
    
    test_percent = 0.1

    flattern_dir_struture(input_dir="/home/master/dataset/train", output_dir="/home/visum/tp_workspace/yolo_dataset")
    generate_gt(input_path="/home/visum/tp_workspace/yolo_dataset/labels.csv",
                test_percent = test_percent,
                k_fold_idx = opt.kfold_idx,
                img_dir = "/home/visum/tp_workspace/yolo_dataset",
                output_dir = "/home/visum/tp_workspace/yolo_dataset_yolo_gt")
    create_coco_style_dataset(train_path="/home/visum/tp_workspace/yolo_dataset_yolo_gt/yolo_full_train_split_0.1.txt",
                              test_path="/home/visum/tp_workspace/yolo_dataset_yolo_gt/yolo_full_test_split_0.1.txt",
                              output_dir="/home/visum/tp_workspace/yolo_test/yolov5")



    # flattern_dir_struture(input_dir="/home/tpk/workspace/visum_project/tmp/train", output_dir="/home/tpk/workspace/visum_project/tmp/yolo_dataset")
    # generate_gt(input_path="/home/tpk/workspace/visum_project/tmp/yolo_dataset/labels.csv",
    #             test_percent = test_percent,
    #             k_fold_idx = opt.kfold_idx,
    #             img_dir = "/home/tpk/workspace/visum_project/tmp/yolo_dataset",
    #             output_dir = "/home/tpk/workspace/visum_project/tmp/yolo_gt")
    # create_coco_style_dataset(train_path="/home/tpk/workspace/visum_project/tmp/yolo_gt/yolo_filtered_train_split_0.1.txt",
    #                           test_path="/home/tpk/workspace/visum_project/tmp/yolo_gt/yolo_filtered_test_split_0.1.txt",
    #                           output_dir="coco")
    print("Dataset preperation done.")
