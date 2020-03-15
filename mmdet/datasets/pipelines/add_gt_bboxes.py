import os, glob, random, json
import cv2, mmcv
import numpy as np
from ..registry import PIPELINES


@PIPELINES.register_module
class AddGTWithAug(object):
    """Add defect bbox in each image."""

    def __init__(self, gt_json, img_path, normal_path,
                 keep_ratio=0.5, max_per_img=10,
                 left_border=1300, right_border=300,
                 top_border=600, bottom_border=600,
                 max_scale_factor=3, ps_only=True):
        self.gt_json = gt_json
        self.img_path = img_path
        self.normal_path = normal_path
        self.keep_ratio = keep_ratio
        self.max_per_img = max_per_img
        self.left_border = left_border
        self.right_border = right_border
        self.top_border = top_border
        self.bottom_border = bottom_border
        self.max_scale_factor = max_scale_factor
        self.ps_only = ps_only
        self.normal_imgs = glob.glob(normal_path + '*.jpg')

    def all_gt_bboxes(self):
        with open(self.gt_json, 'r') as f:
            data = json.load(f)
        images = data['images']
        annotations = data['annotations']
        img_names = []
        for img in images:
            img_names.append(img['file_name'])
        if self.ps_only:
            category = [12, 13]
        else:
            category = [1, 2, 3, 4, 5, 9, 10, 12, 13]
        gt_lst = []
        for anno in annotations:
            img_id = anno['image_id']
            img_name = img_names[img_id - 1]
            cate_id = anno['category_id']
            label = category.index(cate_id) + 1
            bbox = anno['bbox']
            gt_lst.append([img_name] + [label] + bbox)
        return gt_lst

    def bbox_scale(self, img):
        f = np.random.uniform(1, self.max_scale_factor)
        enlarge = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        return enlarge

    def bbox_rotate(self, corners, h, w):
        # four corners co-ordinates: x1, y1, x2, y2, x3, y3, x4, y4
        corners = corners.reshape(-1, 2)
        corners = np.hstack((corners, np.ones((corners.shape[0], 1))))
        angle = np.random.randint(0, 360)
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = h * sin + w * cos
        nH = h * cos + w * sin
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW - w) * 0.5
        M[1, 2] += (nH - h) * 0.5
        # prepare the vector to be transformed
        corners = np.dot(M, corners.T).T
        corners = corners.reshape(-1, 8)
        # get enclosing box for ratated corners
        x_ = corners[:, [0, 2, 4, 6]]
        y_ = corners[:, [1, 3, 5, 7]]
        xmin = np.min(x_, 1).reshape(-1, 1)
        ymin = np.min(y_, 1).reshape(-1, 1)
        xmax = np.max(x_, 1).reshape(-1, 1)
        ymax = np.max(y_, 1).reshape(-1, 1)
        h_ = int(ymax - ymin)
        w_ = int(xmax - xmin)
        return h_, w_, angle

    def __call__(self, results):
        img, gt_bboxes, gt_labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        img_h, img_w, _ = results['img_shape']
        if img_h > 1000:
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
                defect = defect_img[y:y+h, x:x+w, :]
                mode = np.random.randint(3)
                # scale
                if mode == 0:
                    defect = self.bbox_scale(defect)
                    h, w = defect.shape[:2]
                # rotate
                elif mode == 1:
                    ctx = x + w * 0.5
                    cty = y + h * 0.5
                    corners = np.hstack((x, y, x+w, y, x, y+h, x+w, y+h))
                    h, w, angle = self.bbox_rotate(corners, h, w)
                    rotated_img = mmcv.imrotate(defect_img, -angle, center=(ctx, cty))
                    x1 = int(ctx - w * 0.5)
                    y1 = int(cty - h * 0.5)
                    x2 = x1 + w
                    y2 = y1 + h
                    defect = rotated_img[y1:y2, x1:x2, :]
                try:
                    xmin = np.random.randint(self.left_border, img_w-self.right_border-w)
                    ymin = np.random.randint(self.top_border, img_h-self.bottom_border-h)
                    xmax = xmin + w
                    ymax = ymin + h
                    img[ymin:ymax, xmin:xmax, :] = cv2.addWeighted(
                        defect, 1, img[ymin:ymax, xmin:xmax, :], 0, 0)
                    gt_bboxes_.append([xmin, ymin, xmax, ymax])
                    gt_labels_.append(label)
                except:
                    continue

            results['img'] = img
            results['gt_bboxes'] = np.array(gt_bboxes_, dtype=np.float32)
            results['gt_labels'] = np.array(gt_labels_, dtype=np.int64)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(gt_json={}, img_path={}, normal_path={}, ' \
                    'keep_ratio={}, max_per_img={}, left_border={}, ' \
                    'right_border={}, top_border={}, bottom_border={}, ' \
                    'max_scale_factor={}, ps_only={})'.format(
            self.gt_json, self.img_path, self.normal_path,
            self.keep_ratio, self.max_per_img, self.left_border,
            self.right_border, self.top_border, self.bottom_border,
            self.max_scale_factor, self.ps_only)
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

