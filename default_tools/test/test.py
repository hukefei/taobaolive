#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
from mmdet.apis import inference_detector, init_detector
import pickle
from progressbar import ProgressBar

def model_test(imgs,
               cfg_file,
               ckpt_file,
               save_file=None,
               device='cuda:0'):
    assert len(imgs) >= 1, 'There must be at least one image'
    model = init_detector(cfg_file, ckpt_file, device=device)
    final_result = []

    if len(imgs) == 1:
        final_result = inference_detector(model, imgs[0])
    else:
        pbar = ProgressBar().start()
        for i, img in enumerate(imgs):
            pbar.update(int(i / (len(imgs) - 1) * 100))
            result = inference_detector(model, img)
            final_result.append(result)
        pbar.finish()

    if save_file is not None:
        with open(save_file, 'wb') as fp:
            pickle.dump(final_result, fp)

    return final_result

if __name__ == '__main__':
    imgs = glob.glob('/data/sdv2/taobao/data/demo_images/*.jpg')
    cfg_file = '/data/sdv2/taobao/mmdet_taobao/taobao_configs/faster_rcnn_r50_fpn_triplet.py'
    ckpt_file = '/data/sdv2/taobao/work_dirs/epoch_12.pth'
    # save_file = r'/data/sdv1/whtm/result/1GE02/1GE02_1121_test/1GE02_test.pkl'
    results = model_test(imgs, cfg_file, ckpt_file)
    print(results)
