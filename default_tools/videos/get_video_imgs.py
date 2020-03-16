import cv2
import os
import glob
import time
import json
from multiprocessing.pool import Pool
from functools import partial

CLASS_DICT = {'短袖Top': 'duanxiushangyi', '长袖Top': 'changxiushangyi', '短袖衬衫': 'duanxiuchenshan',
              '长袖衬衫': 'changxiuchenshan', '背心上衣': 'beixinshangyi', '吊带上衣': 'diaodaishangyi',
              '无袖上衣': 'wuxiushangyi', '短外套': 'duanwaitao', '短马甲': 'duanmajia',
              '长袖连衣裙': 'changxiulianyiqun', '短袖连衣裙': 'duanxiulianyiqun', '无袖连衣裙': 'wuxiulianyiqun',
              '长马甲': 'changmajia', '长款外套': 'changkuanwaitao', '连体衣': 'liantiyi',
              '古风': 'gufeng', '古装': 'gufeng', '短裙': 'duanqun',
              '中等半身裙（及膝）': 'zhongdengbanshenqun', '长半身裙（到脚）': 'changbanshenqun', '短裤': 'duanku',
              '中裤': 'zhongku', '长裤': 'changku', '背带裤': 'beidaiku'}

def get_video_imgs(pair, save_dir):
    start = time.time()

    video_file = pair[0]
    ann_file = pair[1]

    with open(ann_file, 'r') as f:
        ann_dict = json.load(f)
    ann_video_name = ann_dict['video_id']
    frames = ann_dict['frames']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vc = cv2.VideoCapture(video_file)
    video_name = video_file.replace('\\', '/').split('/')[-1]
    assert (ann_video_name + '.mp4') == video_name
    c = 0
    idx = 0

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    ann_idx = []
    for ann in frames:
        frame_index = ann['frame_index']
        ann_idx.append(frame_index)

    while rval:
        rval, frame = vc.read()
        if c in ann_idx:
            h, w, d = frame.shape

            frame_ann = frames[idx]
            frame_index = frame_ann['frame_index']
            assert frame_index == c
            frame_ann['image_name'] = ann_video_name + '_{}.jpg'.format(c)
            frame_ann['height'] = h
            frame_ann['width'] = w
            ann_name = ann_video_name + '_{}.json'.format(c)
            idx += 1
            if frame_ann['annotations'] != []:
                for ann_ in frame_ann['annotations']:
                    if ann_['label'] in CLASS_DICT.keys():
                        ann_['label'] = CLASS_DICT[ann_['label']]
                    else:
                        print(ann_['label'])
                # save annotation file
                with open(os.path.join(save_dir, ann_name), 'w') as f:
                    json.dump(frame_ann, f, indent=4)
                # save frame image file
                cv2.imwrite(os.path.join(save_dir, video_name.split('.')[0] + '_{}.jpg'.format(c)), frame)

        c += 1
        cv2.waitKey(1)
    vc.release()
    end = time.time()
    cost_time = end - start
    print('{}'.format(cost_time))

def get_multiple_video_images(video_dir, video_ann_dir, save_dir, **kwargs):
    videos = glob.glob(os.path.join(video_dir, '*.mp4'))
    anns = glob.glob(os.path.join(video_ann_dir, '*.json'))
    pair = [[videos[i], anns[i]] for i in range(len(videos))]
    for i in pair:
        get_video_imgs(i, save_dir)


if __name__ == '__main__':
    save_dir = r'G:\Tianchi\Live_demo_20200117\demo_video_images'
    video_dir = r'G:\Tianchi\Live_demo_20200117\video'
    video_ann_dir = r'G:\Tianchi\Live_demo_20200117\video_annotation'
    get_multiple_video_images(video_dir, video_ann_dir, save_dir)