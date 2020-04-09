import os
import json
import shutil
import glob

def gallery_classification(imgs, save_dir):
    for img in imgs:
        img_name = img.replace('\\', '/').split('/')[-1]
        ann = img.replace('.jpg', '.json')
        if not os.path.exists(ann):
            continue
        with open(ann, 'r') as f:
            ann_ = json.load(f)
        if ann_['annotations'] == []:
            continue
        else:
            for a_ in ann_['annotations']:
                if a_['instance_id'] != 0:
                    label = a_['label']
                    if not os.path.exists(os.path.join(save_dir, label)):
                        os.makedirs(os.path.join(save_dir, label))
                    shutil.copy(img, os.path.join(save_dir, label, img_name))


if __name__ == '__main__':
    imgs = glob.glob(r'G:\Tianchi\part1\gallery_dataset\*.jpg')
    save_dir = r'G:\Tianchi\classification'
    gallery_classification(imgs, save_dir)
