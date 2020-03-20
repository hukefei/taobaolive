import os
import cv2
import json


def gallery_instances(img_dir, ann_dir, save_dir):
    gallery_ins = []
    for root, _, files in os.walk(img_dir):
        for file in files:
            if file.endswith('jpg'):
                img_dir_name = root.replace('\\', '/').split('/')[-1]
                filename = img_dir_name + '_' + file
                # shutil.copy(os.path.join(root, file), os.path.join(save_dir, filename))
                # img = cv2.imread(os.path.join(root, file))
                # h, w, c = img.shape

                ann_name = file.replace('jpg', 'json')
                ann_path = os.path.join(ann_dir, img_dir_name, ann_name)
                if os.path.exists(ann_path):
                    with open(ann_path, 'r') as f:
                        ann = json.load(f)
                else:
                    continue
                # ann['image_name'] = ann['item_id'] + '_' + ann['img_name']
                # ann['height'] = h
                # ann['width'] = w
                if ann['annotations'] != []:
                    for ann_ in ann['annotations']:
                        gallery_ins.append(ann_['instance_id'])

    gallery_ins_dict = {'gallery_instance': list(set(gallery_ins))}
    with open(os.path.join(save_dir, 'gallery_instances.json'), 'w') as f:
        json.dump(gallery_ins_dict, f, indent=4)

if __name__ == '__main__':
    img_dir = r'G:\Tianchi\train_dataset_part1\image'
    ann_dir = r'G:\Tianchi\train_dataset_part1\image_annotation'
    save_dir = r'G:\Tianchi\part1'
    gallery_instances(img_dir, ann_dir, save_dir)