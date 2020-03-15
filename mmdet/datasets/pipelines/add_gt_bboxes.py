import os, glob, random, json
import cv2, mmcv
import numpy as np
from ..registry import PIPELINES


@PIPELINES.register_module
class AddGT(object):
    """Add defect bbox in each image."""

    def __init__(self, gt_json, img_path, normal_path,
                 keep_ratio=0.5, max_per_img=10,
                 left_border=1300, right_border=300,
                 top_border=600, bottom_border=600):
        self.gt_json = gt_json
        self.img_path = img_path
        self.normal_path = normal_path
        self.keep_ratio = keep_ratio
        self.max_per_img = max_per_img
        self.left_border = left_border
        self.right_border = right_border
        self.top_border = top_border
        self.bottom_border = bottom_border
        self.normal_imgs = glob.glob(normal_path + '*.jpg')

    def all_gt_bboxes(self):
        with open(self.gt_json, 'r') as f:
            data = json.load(f)
        images = data['images']
        annotations = data['annotations']
        categories = data['categories']
        img_names = []
        for img in images:
            img_names.append(img['file_name'])
        category = []
        for c in categories:
            category.append(c['id'])
        gt_lst = []
        for anno in annotations:
            img_id = anno['image_id']
            img_name = img_names[img_id - 1]
            cate_id = anno['category_id']
            label = category.index(cate_id) + 1
            bbox = anno['bbox']
            gt_lst.append([img_name] + [label] + bbox)
        return gt_lst

    def __call__(self, results):
        img, gt_bboxes, gt_labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        img_h, img_w, _ = results['img_shape']
        gt_lst = self.all_gt_bboxes()
        if np.random.rand() < self.keep_ratio:
            gt_bboxes_ = gt_bboxes.tolist()
            gt_labels_ = gt_labels.tolist()
        else:
            ind = np.random.randint(len(self.normal_imgs))
            normal_path = self.normal_imgs[ind]
            normal_img = mmcv.imread(normal_path)
            normal_img = mmcv.imresize_like(normal_img, img)
            img = normal_img
            gt_bboxes_, gt_labels_ = [], []

        defects = random.choices(gt_lst, k=self.max_per_img)
        defects = sorted(defects, key=(lambda x: x[-1]*x[-2]), reverse=True)
        for det in defects:
            defect_img = cv2.imread(os.path.join(self.img_path, det[0]))
            label = det[1]
            x, y, w, h = list(map(int, det[2:]))
            xmin = np.random.randint(self.left_border, img_w-self.right_border-w)
            ymin = np.random.randint(self.top_border, img_h-self.bottom_border-h)
            xmax = xmin + w
            ymax = ymin + h
            img[ymin:ymax, xmin:xmax, :] = cv2.addWeighted(
                defect_img[y:y+h, x:x+w, :], 1, img[ymin:ymax, xmin:xmax, :], 0, 0)
            gt_bboxes_.append([xmin, ymin, xmax, ymax])
            gt_labels_.append(label)

        results['img'] = img
        results['gt_bboxes'] = np.array(gt_bboxes_, dtype=np.float32)
        results['gt_labels'] = np.array(gt_labels_, dtype=np.int64)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(gt_json={}, img_path={}, normal_path={}, ' \
                    'keep_ratio={}, max_per_img={}, ' \
                    'left_border={}, right_border={},' \
                    'top_border={}, bottom_border={})'.format(
            self.gt_json, self.img_path, self.normal_path,
            self.keep_ratio, self.max_per_img,
            self.left_border, self.right_border,
            self.top_border, self.bottom_border)
        return repr_str


@PIPELINES.register_module
class AddGTByChannel(object):
    """Add defect bbox in each image."""

    def __init__(self,
                 max_per_img=10,
                 shift_range=500,
                 x_border=400,
                 y_border=400):
        self.max_per_img = max_per_img
        self.shift_range = shift_range
        self.x_border = x_border
        self.y_border = y_border

    def __call__(self, results):
        img, gt_bboxes, gt_labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        h, w, _ = results['img_shape']
        gt_dict = {}
        for bbox, label in zip(gt_bboxes, gt_labels):
            gt_dict.setdefault(label, []).append(bbox)
        max_count = max([len(v) for k, v in gt_dict.items()])

        gt_bboxes_ = gt_bboxes.tolist()
        gt_labels_ = gt_labels.tolist()
        if max_count < self.max_per_img:
            for _ in range(self.max_per_img // max_count - 1):
                x_shift = np.random.randint(-self.shift_range, self.shift_range)
                y_shift = np.random.randint(-self.shift_range, self.shift_range)
                img_tmp = []
                for i in range(5):
                    gray = img[:, :, i]
                    bboxes = gt_dict.get(i + 1)
                    if bboxes is None:
                        img_tmp.append(gray[..., np.newaxis])
                    else:
                        for bbox in bboxes:
                            x1 = int(bbox[0])
                            y1 = int(bbox[1])
                            x2 = int(bbox[2])
                            y2 = int(bbox[3])
                            xmin = np.clip(x1 + x_shift, self.x_border, w - self.x_border)
                            ymin = np.clip(y1 + y_shift, self.y_border, h - self.y_border)
                            xmax = np.clip(x2 + x_shift, self.x_border, w - self.x_border)
                            ymax = np.clip(y2 + y_shift, self.y_border, h - self.y_border)
                            try:
                                gray[ymin:ymax, xmin:xmax] = cv2.addWeighted(
                                    gray[y1:y2, x1:x2], 1, gray[ymin:ymax, xmin:xmax], 0, 0)
                                gt_bboxes_.append([xmin, ymin, xmax, ymax])
                                gt_labels_.append(i + 1)
                            except:
                                continue
                        img_tmp.append(gray[..., np.newaxis])
                img = np.concatenate(img_tmp, axis=-1)

            results['img'] = img
            results['gt_bboxes'] = np.array(gt_bboxes_, dtype=np.float32)
            results['gt_labels'] = np.array(gt_labels_, dtype=np.int64)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(max_per_img={}, shift_range={}, ' \
                    'x_border={}, y_border={})'.format(
            self.max_per_img, self.shift_range, self.x_border, self.y_border)
        return repr_str

