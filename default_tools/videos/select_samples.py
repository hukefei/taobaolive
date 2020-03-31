import os
import json
import glob
import shutil
import random
import numpy as np

def get_video_dict(ann_dir):
    jsons = glob.glob(os.path.join(ann_dir, '*.json'))
    video_dict = {}
    for j in jsons:
        j = j.replace('\\', '/')
        j_name = j.split('/')[-1]
        v_name = j_name.split('_')[0]
        with open(j, 'r') as f:
            j_ = json.load(f)

        for ann in j_['annotations']:
            j_d = {}
            for k, v in j_.items():
                if not isinstance(v, list):
                   j_d.setdefault(k, v)
            for k, v in ann.items():
                j_d.setdefault(k ,v)

        video_dict.setdefault(v_name, []).append(j_d)

    return video_dict

def select_samples(ann_dir, img_dir, frame_index_range, save_dir, per_video_num=1):
    video_dict = get_video_dict(ann_dir)
    assert per_video_num > 0
    for k, v in video_dict.items():
        print(k)
        try_time = 0
        flag = per_video_num
        while flag:
            selected = random.choice(v)
            try_time += 1
            frame_index = selected['frame_index']
            instance_id = selected['instance_id']
            viewpoint = selected['viewpoint']
            image_name = selected['image_name']
            if try_time > 200:
                print('cannot find a sample meet the conditions for video: {}'.format(k))
                shutil.copy(os.path.join(img_dir, image_name),
                            os.path.join(save_dir, image_name))
                shutil.copy(os.path.join(ann_dir, image_name.replace('.jpg', '.json')),
                            os.path.join(save_dir, image_name.replace('.jpg', '.json')))
                flag -= 1
            elif (frame_index in frame_index_range) and (instance_id != 0):
                shutil.copy(os.path.join(img_dir, image_name),
                            os.path.join(save_dir, image_name))
                shutil.copy(os.path.join(ann_dir, image_name.replace('.jpg', '.json')),
                            os.path.join(save_dir, image_name.replace('.jpg', '.json')))
                flag -= 1


if __name__ == '__main__':
    ann_dir = r'G:/Tianchi/video_images1'
    save_dir = r'G:\Tianchi\part1\train2'
    select_samples(ann_dir, ann_dir, list(range(80, 360, 40)), save_dir)
