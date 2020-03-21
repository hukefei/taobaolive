import os
from default_tools.videos.get_video_imgs import get_multiple_video_images

dir_list = ['train_dataset_part{}'.format(i+1) for i in range(6)]
base_dir = r'F:\TianChi'

for d_ in dir_list:
    file_dir = os.path.join(base_dir, d_)
    save_dir = os.path.join(file_dir, 'video_images')
    video_dir = os.path.join(file_dir, 'video')
    ann_dir = os.path.join(file_dir, 'video_annotation')
    print(d_)
    get_multiple_video_images(video_dir, ann_dir, save_dir)