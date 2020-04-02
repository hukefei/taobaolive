#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import faiss
import time, glob
import pickle, json
import numpy as np
import os
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
                    model,
                    score_thr,
                    out_json=None):
    print('\nPreparing gallery datasets...')
    category_dict = {}
    num = len(imgs)
    st = time.time()
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
                            [item_id] + [img_name] + list(bbox) + [embed])
        if (i + 1) % 1000 == 0:
            avg_time = (time.time() - st) / (i + 1) + 1e-8
            print('gallery average time: {:.2f}s, eta: {:.2f}m'.format(avg_time, (num - i - 1) * avg_time / 60))

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
                       model,
                       score_thr,
                       k_nearest,
                       iou_thr,
                       topk_bbox=8):
    # TODO: multiple videos may match the same gallery item, carry out a method to solve this solution
    frame_res = []
    score_lst = []
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
                        score_lst.append(conf)
    final_frame_res = []
    # only use topk bbox as query
    topk_bbox = min(topk_bbox, len(score_lst))
    topk_index = np.argpartition(np.array(score_lst), -1*topk_bbox)[-1*topk_bbox:]
    for idx, res in enumerate(frame_res):
        if idx in topk_index:
            final_frame_res.append(res)
    frame_res = final_frame_res

    gallery_res = []
    item_id_lst = []
    distance_lst = []
    for res in frame_res:
        if res == []:
            continue
        query_embed = res[-1][np.newaxis, :].astype('float32')
        gallerys = gallery_dict.get(res[1], None)
        if gallerys is None:
            continue
        gallery_embeds = [lst[-1] for lst in gallerys]
        gallery_embeds = np.array(gallery_embeds).astype('float32')

        d = gallery_embeds.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(gallery_embeds)
        D, I = index.search(query_embed, k_nearest)  # Distance, Index
        topk = [gallerys[ind] for ind in I[0]]
        item_id_lst.extend([k[0] for k in topk])
        gallery_res.append(topk)
        distance_lst.extend(D.flatten().tolist())

    # TODO: more flexible method to deal with video with no bbox whose score is larger than threshold
    if item_id_lst == []:
        return None
    # record item id and average distance
    item_arr = np.array(item_id_lst)
    items = np.unique(item_arr)
    distance_lst = np.array(distance_lst)
    dists = []
    for item in items:
        min_distance = np.min(distance_lst[item_arr == item])
        dists.append(min_distance)

    # choose item_id according to frequency
    # item_id = item_id_lst[np.argmin(min_dist_lst)]
    item_id = max(item_id_lst, key=item_id_lst.count)
    print(item_id, items, dists)
    frame_bbox, item_bbox = [], []
    for frame, gallery in zip(frame_res, gallery_res):  # 1 vs k
        matched_items = [item for item in gallery if item[0] == item_id]
        if len(matched_items) > 0:
            scores = [item[6] for item in matched_items]
            inx = int(np.argmax(np.array(scores)))
            best_item = matched_items[inx]
            item_bbox.append(best_item)
            frame_bbox.append(frame)
    assert len(frame_bbox) == len(item_bbox)  # 1 vs 1

    # choose frame_index according to frequency
    frame_inds = [frame[0] for frame in frame_bbox]
    frame_index = max(frame_inds, key=frame_inds.count)
    keep_inds = np.where(np.array(frame_inds) == frame_index)
    frame_bbox = np.array(frame_bbox)[keep_inds]
    item_bbox = np.array(item_bbox)[keep_inds]

    # merge overlap frame_bbox based on IOU
    # sort (frame, item) pairs by frame bbox scores
    pairs = sorted(zip(frame_bbox, item_bbox),
                   key=lambda x: x[0][6], reverse=True)
    best_frame_bbox = []
    best_item_bbox = []
    for i, (frame, item) in enumerate(pairs):
        if i == 0:
            best_frame_bbox.append(frame)
            best_item_bbox.append(item)
        else:
            exist_bboxes = [bbox[2:6] for bbox in best_frame_bbox]
            iou = bboxes_iou(frame[2:6], exist_bboxes)
            if np.all(iou < iou_thr):
                best_frame_bbox.append(frame)
                best_item_bbox.append(item)
            else:
                ind = int(np.argmax(iou))
                # merge frame bbox coordinates by mean value
                best_frame_bbox[ind][2:6] = np.mean(
                    [best_frame_bbox[ind][2:6], frame[2:6]], axis=0)
                # keep the item bbox with highest score
                best_item_bbox[ind] = \
                    best_item_bbox[ind] if best_item_bbox[ind][6] > item[6] else item
    assert len(best_frame_bbox) == len(best_item_bbox)

    result_lst = []
    for frame, item in zip(best_frame_bbox, best_item_bbox):
        d = {
            "img_name": item[1],
            "item_box": list(map(int, item[2:6])),
            "frame_box": list(map(int, frame[2:6]))
        }
        result_lst.append(d)
    output = {
        "item_id": item_id,
        "frame_index": frame_index,
        "result": result_lst
    }
    return output


def bboxes_iou(boxes1, boxes2):
    """
    boxes: [xmin, ymin, xmax, ymax] format coordinates.
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, 0.0)

    return ious

def multiple_video_query(videos, gallery_dict, model, interval=80):
    print('\nStarting video predict...')
    st = time.time()
    final_result = {}
    num = len(videos)
    for i, video in enumerate(videos):
        video_id, video_imgs = capture_video_imgs(video, interval=interval)
        print(video_id)
        output = single_video_query(
            video_imgs, gallery_dict, model,
            score_thr=0.1, k_nearest=3, iou_thr=0.5
        )
        if output is not None:
            final_result[video_id] = output
        if (i + 1) % 1000 == 0:
            avg_time = (time.time() - st) / (i + 1) + 1e-8
            print('video average time: {:.2f}s, eta: {:.2f}m'.format(avg_time, (num - i - 1) * avg_time / 60))
    return final_result

def load_model(cfg, ckpt):
    print('\nLoading model...')
    model = init_detector(cfg, ckpt, device='cuda:0')
    return model

if __name__ == '__main__':
    # imgs = glob.glob('/tcdata/test_dataset_3w/image/*/*.jpg')
    # videos = glob.glob('/tcdata/test_dataset_3w/video/*.mp4')
    imgs = glob.glob('/data/sdv2/taobao/data/val_demo/image/*/*.jpg')
    videos = glob.glob('/data/sdv2/taobao/data/val_demo/video/*.mp4')

    base_dir = r'/data/sdv2/taobao/mmdet_taobao/'
    cfg = 'taobao_configs/faster_rcnn_r50_fpn_triplet.py'
    ckpt = 'models/published_0401-d8037514.pth'

    cfg = os.path.join(base_dir, cfg)
    ckpt = os.path.join(base_dir, ckpt)

    model = load_model(cfg, ckpt)

    gallery_dict = gallery_results(imgs, model, score_thr=0.1,
                                   out_json='gallery_results.json')
    final_result = multiple_video_query(videos, gallery_dict, model, 80)

    print(final_result)

    with open('result.json', 'w') as f:
        json.dump(final_result, f, indent=4, cls=NumpyEncoder)
