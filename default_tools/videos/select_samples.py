import os
import json
import glob
import shutil

def select_samples(ann_dir, save_dir, per_video_num=1, img_dir=None):
    img_dir = ann_dir if img_dir is None else img_dir
    jsons = glob.glob(os.path.join(ann_dir, '*.json'))
    selected = {}
    for j in jsons:
        j = j.replace('\\', '/')
        j_name = j.split('/')[-1]
        v_name = j_name.split('_')[0]
        selected.setdefault(v_name, {'max_size': 0})
        with open(j, 'r') as f:
            j_ = json.load(f)

        total_size = 0
        instances = []
        for ann in j_['annotations']:
            if ann['instance_id'] != 0:
                box = ann['box']
                total_size += (box[2]-box[0])*(box[3]-box[1])
                instances.append(ann['instance_id'])
        if total_size >= selected[v_name]['max_size']:
            selected[v_name]['selected_ann'] = j_name
            selected[v_name]['selected_img'] = j_['image_name']
            selected[v_name]['max_size'] = total_size
            selected[v_name]['instances'] = instances

    for k,v in selected.items():
        print(k, v['instances'])
        shutil.copy(os.path.join(ann_dir, v['selected_ann']), os.path.join(save_dir, v['selected_ann']))
        shutil.copy(os.path.join(img_dir, v['selected_img']), os.path.join(save_dir, v['selected_img']))


if __name__ == '__main__':
    ann_dir = r'G:/Tianchi/video_images1'
    save_dir = r'G:\Tianchi\train_dataset'
    select_samples(ann_dir, save_dir)
