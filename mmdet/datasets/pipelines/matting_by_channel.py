import os
import cv2
import mmcv
import numpy as np
from ..registry import PIPELINES


@PIPELINES.register_module
class GroupMatting(object):
    """Matting defect bbox to normal image.
    Args:
        normal_path (str): normal images path
        matting_ratio (float): Whether to keep the original defect image.
        blend_range (tuple[float]): (min_ratio, max_ratio)
    """

    def __init__(self,
                 normal_path,
                 matting_ratio=0.5,
                 blend_range=None):
        self.normal_path = normal_path
        self.matting_ratio = matting_ratio

        if blend_range is None:
            self.blend_ratio = 1
        else:
            assert len(blend_range) == 2
            self.blend_ratio = np.random.uniform(blend_range[0], blend_range[1])

    def normal_groups(self):
        imgs = os.listdir(self.normal_path)
        names = []
        for img in imgs:
            name = img.split('_')[1]
            if name not in names:
                names.append(name)
        return names

    def __call__(self, results):
        img, gt_bboxes, gt_labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        gt_dict = {}
        for bbox, label in zip(gt_bboxes, gt_labels):
            gt_dict.setdefault(label, []).append(bbox)

        if np.random.rand() < self.matting_ratio:
            ind = np.random.randint(len(self.normal_groups()))
            group_name = self.normal_groups()[ind]
            img_ = []
            for i in range(5):
                img_name = 'imgs_' + group_name + '_{}.jpg'.format(str(i))
                gray = cv2.imread(os.path.join(self.normal_path, img_name), 0)
                gray = mmcv.imresize_like(gray, img)

                raw = img[:, :, i]
                bboxes = gt_dict.get(i + 1)
                if bboxes is None:
                    img_.append(gray[..., np.newaxis])
                else:
                    for bbox in bboxes:
                        xmin = int(bbox[0])
                        ymin = int(bbox[1])
                        xmax = int(bbox[2])
                        ymax = int(bbox[3])
                        gray[ymin:ymax, xmin:xmax] = cv2.addWeighted(
                            raw[ymin:ymax, xmin:xmax], self.blend_ratio,
                            gray[ymin:ymax, xmin:xmax], 1 - self.blend_ratio, 0)
                    img_.append(gray[..., np.newaxis])

            results['img'] = np.concatenate(img_, axis=-1)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(normal_path={}, matting_ratio={}, blend_ratio={})'.format(
            self.normal_path, self.matting_ratio, self.blend_ratio)
        return repr_str
