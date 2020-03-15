import numpy as np
from .utils import rotate_with_bboxes, translate_bbox
from ..registry import PIPELINES


@PIPELINES.register_module
class Rotate(object):
    """Rotate image and bboxes by channel."""

    def __init__(self,
                 max_degrees=3,
                 keep_prob=0.5,
                 border_value=128):
        self.max_degrees = max_degrees
        self.keep_prob = keep_prob
        self.border_value = border_value

    def __call__(self, results):
        img, gt_bboxes, gt_labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        h, w, c = img.shape
        gt_dict = {}
        for bbox, label in zip(gt_bboxes, gt_labels):
            xmin, ymin, xmax, ymax = bbox[:4]
            min_y = ymin / h
            min_x = xmin / w
            max_y = ymax / h
            max_x = xmax / w
            gt_dict.setdefault(label, []).append([min_y, min_x, max_y, max_x])

        img_, gt_bboxes_, gt_labels_ = [], [], []
        for i in range(c):
            gray = img[:, :, i]
            bboxes = gt_dict.get(i + 1)
            if bboxes is None:
                img_.append(gray[..., np.newaxis])
            else:
                if np.random.rand() > self.keep_prob:
                    degrees = np.random.randint(-self.max_degrees, self.max_degrees + 1)
                    augmented_gray, augmented_bboxes = rotate_with_bboxes(
                        gray, bboxes, degrees=degrees, replace=self.border_value)
                    if len(augmented_bboxes) == len(bboxes):
                        gray = augmented_gray
                        bboxes = augmented_bboxes
                for bbox in bboxes:
                    ymin = bbox[0] * h
                    xmin = bbox[1] * w
                    ymax = bbox[2] * h
                    xmax = bbox[3] * w
                    gt_bboxes_.append([xmin, ymin, xmax, ymax])
                    gt_labels_.append(i + 1)
                img_.append(gray[..., np.newaxis])

        results['img'] = np.concatenate(img_, axis=-1)
        results['gt_bboxes'] = np.array(gt_bboxes_, dtype=np.float32)
        results['gt_labels'] = np.array(gt_labels_, dtype=np.int64)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(max_degrees={}, keep_prob={}, border_value={})'.format(
            self.max_degrees, self.keep_prob, self.border_value)
        return repr_str


@PIPELINES.register_module
class Translate(object):
    """Translate image and bboxes in X/Y dimension by channel."""

    def __init__(self,
                 max_fraction=0.1,
                 keep_prob=0.5,
                 border_value=128,
                 shift_horizontal=True):
        self.max_fraction = max_fraction
        self.keep_prob = keep_prob
        self.border_value = border_value
        self.shift_horizontal = shift_horizontal

    def __call__(self, results):
        img, gt_bboxes, gt_labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        h, w, c = img.shape
        gt_dict = {}
        for bbox, label in zip(gt_bboxes, gt_labels):
            xmin, ymin, xmax, ymax = bbox[:4]
            min_y = ymin / h
            min_x = xmin / w
            max_y = ymax / h
            max_x = xmax / w
            gt_dict.setdefault(label, []).append([min_y, min_x, max_y, max_x])

        img_, gt_bboxes_, gt_labels_ = [], [], []
        for i in range(c):
            gray = img[:, :, i]
            bboxes = gt_dict.get(i + 1)
            if bboxes is None:
                img_.append(gray[..., np.newaxis])
            else:
                if np.random.rand() > self.keep_prob:
                    fraction = np.random.uniform(-self.max_fraction, self.max_fraction + 1)
                    if self.shift_horizontal:
                        pixels = int(fraction * w)
                    else:
                        pixels = int(fraction * h)
                    augmented_gray, augmented_bboxes = translate_bbox(
                        gray, bboxes, pixels=pixels, replace=self.border_value,
                        shift_horizontal=self.shift_horizontal)
                    if len(augmented_bboxes) == len(bboxes):
                        gray = augmented_gray
                        bboxes = augmented_bboxes
                for bbox in bboxes:
                    ymin = bbox[0] * h
                    xmin = bbox[1] * w
                    ymax = bbox[2] * h
                    xmax = bbox[3] * w
                    gt_bboxes_.append([xmin, ymin, xmax, ymax])
                    gt_labels_.append(i + 1)
                img_.append(gray[..., np.newaxis])

        results['img'] = np.concatenate(img_, axis=-1)
        results['gt_bboxes'] = np.array(gt_bboxes_, dtype=np.float32)
        results['gt_labels'] = np.array(gt_labels_, dtype=np.int64)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(max_fraction={}, keep_prob={}, ' \
                    'border_value={}, shift_horizontal={})'.format(
            self.max_fraction, self.keep_prob,
            self.border_value, self.shift_horizontal)
        return repr_str


@PIPELINES.register_module
class RotateOrTranslate(object):
    """Rotate or Translate image and bboxes by channel."""

    def __init__(self,
                 rotate_max_degrees=3,
                 translate_x_max_fraction=0.01,
                 translate_y_max_fraction=0.01,
                 keep_prob=0.5,
                 rotate_prob=0.5,
                 border_value=128):
        self.rotate_max_degrees = rotate_max_degrees
        self.translate_x_max_fraction = translate_x_max_fraction
        self.translate_y_max_fraction = translate_y_max_fraction
        self.keep_prob = keep_prob
        self.rotate_prob = rotate_prob
        self.border_value = border_value

    def __call__(self, results):
        img, gt_bboxes, gt_labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        h, w, c = img.shape
        gt_dict = {}
        for bbox, label in zip(gt_bboxes, gt_labels):
            xmin, ymin, xmax, ymax = bbox[:4]
            min_y = ymin / h
            min_x = xmin / w
            max_y = ymax / h
            max_x = xmax / w
            gt_dict.setdefault(label, []).append([min_y, min_x, max_y, max_x])

        img_, gt_bboxes_, gt_labels_ = [], [], []
        for i in range(c):
            gray = img[:, :, i]
            bboxes = gt_dict.get(i + 1)
            if bboxes is None:
                img_.append(gray[..., np.newaxis])
            else:
                # do augmentation
                if np.random.rand() > self.keep_prob:
                    # rotate
                    if np.random.rand() <= self.rotate_prob:
                        degrees = np.random.randint(-self.rotate_max_degrees,
                                                    self.rotate_max_degrees + 1)
                        augmented_gray, augmented_bboxes = rotate_with_bboxes(
                            gray, bboxes, degrees=degrees, replace=self.border_value)
                    # translate
                    else:
                        x_fraction = np.random.uniform(-self.translate_x_max_fraction,
                                                       self.translate_x_max_fraction)
                        x_pixels = int(x_fraction * w)
                        y_fraction = np.random.uniform(-self.translate_y_max_fraction,
                                                       self.translate_y_max_fraction)
                        y_pixels = int(y_fraction * h)
                        # translate_x
                        augmented_gray, augmented_bboxes = translate_bbox(
                            gray, bboxes, pixels=x_pixels, replace=self.border_value, shift_horizontal=True)
                        # translate_y
                        augmented_gray, augmented_bboxes = translate_bbox(
                            augmented_gray, augmented_bboxes, pixels=y_pixels, replace=self.border_value, shift_horizontal=False)

                    if len(augmented_bboxes) == len(bboxes):
                        gray = augmented_gray
                        bboxes = augmented_bboxes
                for bbox in bboxes:
                    ymin = bbox[0] * h
                    xmin = bbox[1] * w
                    ymax = bbox[2] * h
                    xmax = bbox[3] * w
                    gt_bboxes_.append([xmin, ymin, xmax, ymax])
                    gt_labels_.append(i + 1)
                img_.append(gray[..., np.newaxis])

        results['img'] = np.concatenate(img_, axis=-1)
        results['gt_bboxes'] = np.array(gt_bboxes_, dtype=np.float32)
        results['gt_labels'] = np.array(gt_labels_, dtype=np.int64)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '''rotate_max_degrees={}, 
                       translate_x_max_fraction={}, 
                       translate_y_max_fraction={},
                       keep_prob={}, rotate_prob={}, 
                       border_value={}'''.format(
            self.rotate_max_degrees, self.translate_x_max_fraction,
            self.translate_y_max_fraction, self.keep_prob,
            self.rotate_prob, self.border_value)
        return repr_str

