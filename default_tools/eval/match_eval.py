import json
import numpy as np
import os
import cv2


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format.
        bboxes2 (Tensor): shape (n, 4) in <x1, y1, x2, y2> format.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = np.maximum(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = np.minimum(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clip(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
                bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                    bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = np.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = np.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
                bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                    bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious


def match_eval(video_list,
               match_result,
               video_ann_dir,
               img_ann_dir,
               img_dir):
    with open(match_result, 'r') as f:
        match_result = json.load(f)

    TP = np.zeros(3)
    FP = np.zeros(3)
    FN = np.zeros(3)
    for video in video_list:
        video_name = video.split('.')[0]
        pred_result = match_result.get(video_name, {})
        with open(os.path.join(video_ann_dir, '{}.json'.format(video_name)), 'r') as f:
            gt = json.load(f)

        if pred_result == {}:
            # no prediction for this video
            frames = gt['frames']
            gt_frame = 0
            gt_matches = 0
            for frame in frames:
                match_count = 0
                annotations = frame['annotations']
                frame_index = frame['frame_index']
                for ann in annotations:
                    if ann['instance_id'] != 0:
                        match_count += 1
                if match_count >= gt_matches:
                    gt_matches = match_count
                    gt_frame = frame_index

            if gt_matches > 0:
                FN += np.full(3, 1)
            else:
                FN += np.full(3, 0)
        else:
            item_id = pred_result['item_id']
            frame_index = pred_result['frame_index']
            results = pred_result['result']

            # check if item id is in the video gt annotations
            flag_s1 = 0
            for frame in gt['frames']:
                for ann in frame['annotations']:
                    video_item_id = str(ann['instance_id'])[:6]
                    if video_item_id != '0':
                        video_item_id = '0' + video_item_id[1:]
                    if item_id == video_item_id:
                        flag_s1 = 1
                        break
            if flag_s1:
                TP += np.array([1, 0, 0])
            else:
                FP += np.array([1, 0, 0])
                FN += np.array([1, 0, 0])

            # check if item id is in the video frame
            flag_s2 = 0
            for frame in gt['frames']:
                if frame['frame_index'] == frame_index:
                    for ann in frame['annotations']:
                        video_item_id = str(ann['instance_id'])[:6]
                        if video_item_id != '0':
                            video_item_id = '0' + video_item_id[1:]
                        if item_id == video_item_id:
                            flag_s2 = 1
                            break
            if flag_s2:
                TP += np.array([0, 1, 0])
            else:
                FP += np.array([0, 1, 0])
                FN += np.array([0, 1, 0])

            # check if bbox is matched
            flag_s3 = 0
            if not flag_s2:
                flag_s3 = 0
            else:
                for result in results:
                    img_name = result['img_name']
                    item_box = result['item_box']
                    img_ann_path = os.path.join(img_ann_dir, item_id, img_name + '.json')
                    img_path = os.path.join(img_dir, item_id, img_name + '.jpg')
                    if os.path.exists(img_ann_path):
                        with open(img_ann_path, 'r') as f:
                            img_annotations = json.load(f)
                    else:
                        raise ValueError

                    if img_annotations['annotations'] == []:
                        # if there are no annotations, set image bbox to the whole image
                        img = cv2.imread(img_path)
                        h, w, c = img.shape
                        box = np.array([0, 0, w, h])
                        if bbox_overlaps(np.array([box]), np.array([item_box]), is_aligned=True) > 0.5:
                            flag_s3 = 1

                    for ann_ in img_annotations['annotations']:
                        box = ann_['box']
                        if bbox_overlaps(np.array([box]), np.array([item_box]), is_aligned=True) > 0.5:
                            flag_s3 = 1

            if flag_s3:
                TP += np.array([0, 0, 1])
            else:
                FP += np.array([0, 0, 1])
                FN += np.array([0, 0, 1])

    P = TP / (TP + FP)
    R = TP / (FN + TP)

    S = 2 * P * R / (P + R)
    score_weigths = [0.2, 0.6, 0.2]
    final_score = sum(S * score_weigths)

    return S, final_score


if __name__ == '__main__':
    video_list = os.listdir(r'F:\TianChi\validation_dataset_part1\video')
    match_result = r'F:\TianChi\test_result.json'
    video_ann_dir = r'F:\TianChi\validation_dataset_part1\video_annotation'
    img_ann_dir = r'F:\TianChi\validation_dataset_part1\image_annotation'
    img = r'F:\TianChi\validation_dataset_part1\image'
    print(match_eval(video_list, match_result, video_ann_dir, img_ann_dir, img))
