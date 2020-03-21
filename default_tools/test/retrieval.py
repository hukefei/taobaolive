#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import faiss
import time, glob
import pickle, json
import numpy as np
from mmdet.apis import inference_detector, init_detector


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.float32):
            return float(o)


def gallery_results_from_pkl(imgs,
                             pkl_file,
                             score_thr,
                             out_json=None):
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)
    assert len(imgs) == len(results)

    category_dict = {}
    for i, (img, result) in enumerate(zip(imgs, results)):
        filename = img.split('/')[-1][:-4]
        item_id, img_name = filename.split('_')

        for idx, (bboxes, embeds) in enumerate(zip(*result)):
            category_id = idx + 1
            if len(bboxes) != 0:
                for bbox, embed in zip(bboxes, embeds):
                    conf = bbox[4]
                    if conf > score_thr:
                        category_dict.setdefault(category_id, []).append(
                            [item_id] + [img_name] + list(bbox) + [embed]
                        )
    if out_json is not None:
        with open(out_json, 'w') as f:
            json.dump(category_dict, f, indent=4, cls=NumpyEncoder)
    return category_dict


def gallery_results(imgs,
                    cfg_file,
                    ckpt_file,
                    score_thr,
                    out_json=None,
                    device='cuda:0'):
    model = init_detector(cfg_file, ckpt_file, device=device)

    category_dict = {}
    for i, img in enumerate(imgs):
        item_id = img.split('/')[-2]
        img_name = img.split('/')[-1][:-4]
        result = inference_detector(model, img)

        for idx, (bboxes, embeds) in enumerate(zip(*result)):
            category_id = idx + 1
            if len(bboxes) != 0:
                for bbox, embed in zip(bboxes, embeds):
                    conf = bbox[4]
                    if conf > score_thr:
                        category_dict.setdefault(category_id, []).append(
                            [item_id] + [img_name] + list(bbox) + [embed]
                        )
    if out_json is not None:
        with open(out_json, 'w') as f:
            json.dump(category_dict, f, indent=4, cls=NumpyEncoder)
    return category_dict


def capture_video_imgs(video_file, interval=40):
    vc = cv2.VideoCapture(video_file)
    video_id = video_file.split('/')[-1][:-4]
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    inds = np.arange(0, 400, interval)
    c = 0
    imgs_dict = {}
    while rval:
        rval, frame = vc.read()
        if c in inds:
            imgs_dict[c] = frame
        c += 1
        cv2.waitKey(1)
    vc.release()
    return video_id, imgs_dict


def single_video_query(video_imgs,
                       gallery_dict,
                       cfg_file,
                       ckpt_file,
                       score_thr,
                       k_nearest,
                       device='cuda:0'):
    model = init_detector(cfg_file, ckpt_file, device=device)

    frame_res = []
    for frame_index, img in video_imgs.items():
        result = inference_detector(model, img)

        for idx, (bboxes, embeds) in enumerate(zip(*result)):
            category_id = idx + 1
            if len(bboxes) != 0:
                for bbox, embed in zip(bboxes, embeds):
                    conf = bbox[4]
                    if conf > score_thr:
                        frame_res.append(
                            [frame_index] + [category_id] + list(bbox) + [embed]
                        )
    gallery_res = []
    item_id_lst = []
    for res in frame_res:
        query_embed = res[-1][np.newaxis, :].astype('float32')
        gallerys = gallery_dict[res[1]]
        gallery_embeds = [lst[-1] for lst in gallerys]
        gallery_embeds = np.array(gallery_embeds).astype('float32')

        d = gallery_embeds.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(gallery_embeds)
        D, I = index.search(query_embed, k_nearest)   # Distance, Index
        topk = [gallerys[ind] for ind in I[0]]
        item_id_lst.extend([k[0] for k in topk])
        gallery_res.append(topk)

    # choose item_id according to frequency
    item_id = max(item_id_lst, key=item_id_lst.count)
    frame_bbox, item_bbox = [], []
    for frame, gallery in zip(frame_res, gallery_res):
        for item in gallery:
            if item[0] == item_id:
                frame_bbox.append(frame)
                item_bbox.append(item)

    # choose frame_index according to the matched item quantity
    frame_inds = [frame[0] for frame in frame_bbox]
    frame_index = max(frame_inds, key=frame_inds.count)
    keep_inds = np.where(np.array(frame_inds) == frame_index)
    frame_bbox = np.array(frame_bbox)[keep_inds]
    item_bbox = np.array(item_bbox)[keep_inds]

    result_lst = []
    for frame, item in zip(frame_bbox, item_bbox):
        d = {
            "img_name": item[1],
            "item_bbox": list(item[2:6]),
            "frame_bbox": list(frame[2:6])
        }
        result_lst.append(d)
    output = {
        "item_id": item_id,
        "frame_index": frame_index,
        "result": result_lst
    }
    return output


if __name__ == '__main__':
    imgs = glob.glob('/tcdata/test_dataset_3w/image/*/*.jpg')
    videos = glob.glob('/tcdata/test_dataset_3w/video/*.mp4')

    cfg = './taobao_configs/faster_rcnn_r50_fpn_triplet.py'
    ckpt = './taobao_models/published-8429440b.pth'

    print('\nPreparing gallery datasets...')
    st = time.time()
    gallery_dict = gallery_results(imgs, cfg, ckpt, score_thr=0.5,
                                   out_json='gallery_results.json')
    print('Total Cost Time: {}s'.format(time.time() - st))
    # with open('gallery_results.json', 'r') as f:
    #     gallery_dict = json.load(f)

    print('\nStarting video predict...')
    st = time.time()
    final_result = {}
    for i, video in enumerate(videos):
        video_id, video_imgs = capture_video_imgs(video, interval=40)
        output = single_video_query(
            video_imgs, gallery_dict, cfg, ckpt,
            score_thr=0.5, k_nearest=3
        )
        final_result[video_id] = output
    print('Average Cost Time: {}s'.format((time.time() - st) / len(videos)))

    with open('result.json', 'w') as f:
        json.dump(final_result, f, indent=4, cls=NumpyEncoder)
