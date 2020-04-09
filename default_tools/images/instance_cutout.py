import os
import json
import cv2
import glob

def instance_cutout(img_dir, ann_dir, save_dir, mode='category'):
    imgs = glob.glob(os.path.join(img_dir, '*.jpg'))
    img_names = [img.replace('\\', '/').split('/')[-1] for img in imgs]
    for idx, (img_path, img_name) in enumerate(zip(imgs, img_names)):
        img = cv2.imread(img_path)
        ann_name = img_name.replace('jpg', 'json')
        ann_path = os.path.join(ann_dir, ann_name)
        if not os.path.exists(ann_path):
            continue
        with open(ann_path, 'r') as f:
            json_dict = json.load(f)

        for i, ann in enumerate(json_dict['annotations']):
            box = ann['box']
            instance_id = ann['instance_id']
            label = ann['label']

            cut_instance = img[box[1]:box[3], box[0]:box[2], :]
            save_img_name = img_name[:-4] + '_{}.jpg'.format(i)

            if mode == 'category':
                mark = label
            elif mode == 'instance':
                mark = str(instance_id)

            if not os.path.exists(os.path.join(save_dir, mark)):
                os.makedirs(os.path.join(save_dir, mark))

            save_path = os.path.join(save_dir, mark, save_img_name)
            cv2.imwrite(save_path, cut_instance)

if __name__ == '__main__':
    img_dir = r'G:\Tianchi\train_dataset_part3\gallery'
    ann_dir = img_dir
    save_dir = r'G:\Tianchi\classification\val'
    instance_cutout(img_dir, ann_dir, save_dir, mode='category')