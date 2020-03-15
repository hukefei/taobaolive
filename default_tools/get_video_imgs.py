import cv2
import os
import glob
import time
from multiprocessing.pool import Pool
from functools import partial



def get_video_imgs(video_file, save_dir, frame_interval=40):
    start = time.time()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vc = cv2.VideoCapture(video_file)
    video_name = video_file.replace('\\', '/').split('/')[-1]
    c = 0

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    timeF = frame_interval

    while rval:
        rval, frame = vc.read()
        if (c % timeF) == 0:
            cv2.imwrite(os.path.join(save_dir, video_name.split('.')[0] + '_{}.jpg'.format(c)), frame)
        c += 1
        cv2.waitKey(1)
    vc.release()
    end = time.time()
    cost_time = end - start
    print('{}'.format(cost_time))

def get_multiple_video_images(video_dir, save_dir, **kwargs):
    videos = glob.glob(os.path.join(video_dir, '*.mp4'))
    total = len(videos)
    pool = Pool()
    pool.map(partial(get_video_imgs, save_dir=save_dir, **kwargs), videos)
    pool.close()
    pool.join()


if __name__ == '__main__':
    save_dir = r'F:\TianChi\video_images'
    video_dir = r'F:\TianChi\train_dataset_part1\video'
    get_multiple_video_images(video_dir, save_dir)