import os
import json
import glob
import numpy as np

CLASS_LIST = ['duanxiushangyi', 'changxiushangyi', 'duanxiuchenshan', 'changxiuchenshan',
              'beixinshangyi', 'diaodaishangyi', 'wuxiushangyi', 'duanwaitao', 'duanmajia',
              'changxiulianyiqun', 'duanxiulianyiqun', 'wuxiulianyiqun',
              'changmajia', 'changkuanwaitao', 'liantiyi',
              'gufeng', 'duanqun', 'zhongdengbanshenqun', 'changbanshenqun', 'duanku',
              'zhongku', 'changku', 'beidaiku']
def check_invalid(json_dict):
    instance_list = []
    for ann_ in json_dict['annotations']:
        instance_list.append(ann_['instance_id'] == 0)
    instance_list = np.array(instance_list)
    if np.all(instance_list):
        return True
    else:
        return False

def json2coco(json_dir, save_file, is_video=True):
    class_dict = generate_class_dict(CLASS_LIST)
    jsons = glob.glob(os.path.join(json_dir, '*.json'))
    images = []
    type_ = "instance"
    annotations = []
    categories = []

    img_id = 1
    cat_id = 1
    for j in jsons:
        with open(j, 'r') as f:
            ann = json.load(f)

        #check if the instance ids are all 0 for video images
        if is_video and check_invalid(ann):
            continue

        file_name = ann['image_name']
        height = ann['height']
        width = ann['width']
        ann_ = ann['annotations']

        images.append({'file_name': file_name, 'height': height, 'width': width, 'id': img_id})

        for a_ in ann_:
            label = a_['label']
            category_id = class_dict[label]
            bbox = a_['box']
            viewpoint = a_['viewpoint']
            display = a_['display']
            instance_id = a_['instance_id']
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            annotations.append({'area': w * h, 'image_id': img_id, 'bbox': [x, y, w, h],
                                'category_id': category_id, 'id': cat_id, 'ignore': 0,
                                'viewpoint': viewpoint, 'display': display, 'instance_id': instance_id,
                                'segmentation': [], 'iscrowd': 0})
            cat_id += 1
        img_id += 1

    for k, v in class_dict.items():
        categories.append({'supercategory': 'none', 'id': v, 'name': k})
    final_dict = {'images': images, 'type': type_, 'annotations': annotations, 'categories': categories}

    with open(save_file, 'w') as f:
        json.dump(final_dict, f, indent=3)


def generate_class_dict(class_list):
    class_dict = {}
    for idx, c in enumerate(class_list):
        class_dict.setdefault(c, idx + 1)
    return class_dict


if __name__ == '__main__':
    json_dir = r'/data/sdv2/taobao/data/demo_images'
    save_file = r'/data/sdv2/taobao/data/demo_images.json'
    json2coco(json_dir, save_file, is_video=True)
