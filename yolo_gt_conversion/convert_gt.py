from pathlib import Path

def write_list(out_path, data):
    with open(out_path, mode='wt', encoding='utf-8') as fp:
        fp.write('\n'.join(data))

def read_labels(input_path):
    with open(input_path) as fp:
        data = fp.read().splitlines() # f.readlines()
    header = data[0]
    data = data[1:]
    assert len(data) == 84 * 60, "Number of files and labels don't match"
    return data

def get_info(data):
    data = data.split(";")
    id_seq, id_frame = data[:2]
    b_boxes = None
    if len(data) == 3:
        b_boxes = eval(data[-1])
    return id_seq, id_frame, b_boxes

def covt_ln_darknet_format(data, input_dir, id_class=1, is_filter=False):
    id_seq, id_frame, b_boxes = get_info(data)
    bb_info = ""
    if b_boxes is not None:
        b_boxes = list(map(lambda x: ",".join(map(str, x)) + ",{}".format(id_class), b_boxes))
        bb_info = " ".join(b_boxes)
    elif is_filter is True:
        return ""
    img_path = Path(input_dir).joinpath("seq{}".format(id_seq)).joinpath("img{}.jpg".format(id_frame))
    converted_txt = "{} {}".format(img_path, bb_info)
    return converted_txt.strip()

def convert_to_darknet_format(data, input_dir, id_class, is_filter, out_path):
    """
    Takes in the current label format @param: data and create a darknet
    style GT format with relative path @param:input_dir, class id
    @param: id_class. The output file will be written at @param:
    out_path. If @param: is_filter is True, filters images without
    detections.
    """
    converter = lambda x: covt_ln_darknet_format(x, input_dir, id_class, is_filter)
    result = list(map(converter, data))
    num_imgs = len(result)
    if is_filter:
        result = [x for x in result if x != '']
    num_filtered = len(result)
    write_list(out_path=out_path, data=result)
    print("Created darknet format gt with {} samples. {} samples is filtered out.".format(num_imgs, num_imgs - num_filtered))
    return result

if __name__ == "__main__":
    data = read_labels(input_path="./labels.csv")
    # Use below function to convert to darknet GT format.
    # The input_dir should be the path to dataset train.
    # Created darknet format gt with 5040 samples. 2345 samples is filtered out.
    # Created darknet format gt with 5040 samples. 0 samples is filtered out.
    convert_to_darknet_format(data,
                              input_dir="dataset/train/",
                              id_class=1,
                              is_filter=False,
                              out_path="yolo_darknet_gt.txt")
    
